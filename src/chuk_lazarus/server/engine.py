"""
ModelEngine — format-agnostic inference engine.

Wraps a loaded ``UnifiedPipeline`` and exposes async generation against
internal canonical types.  Protocol routers never touch the pipeline directly.

Design rules
------------
- Public surface is fully async.
- ``_generate()`` is the private sync implementation — runs in a thread pool
  so it never blocks the event loop.
- ``astream()`` bridges the synchronous ``_stream_tokens`` generator to an
  async generator via an ``asyncio.Queue``.
- ``_apply_template()`` formats messages (including tool definitions and tool
  result turns) directly through the tokenizer's chat template.
- ``_parse_tool_calls()`` detects ``<tool_call>…</tool_call>`` blocks in
  model output and returns them as ``ToolCall`` instances.
- Thread-safe: a threading.Lock guards model access for non-streaming calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from .schemas.internal import (
    FinishReason,
    InternalChunk,
    InternalRequest,
    InternalResponse,
    InternalUsage,
    StopReason,
    ToolCall,
    ToolCallFunction,
)

if TYPE_CHECKING:
    from chuk_lazarus.inference import UnifiedPipeline

logger = logging.getLogger(__name__)

# Gemma 3 and similar models wrap tool calls in XML-like tags.
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _build_pipeline_config(
    backend: str | None,
    device: str | None,
    engine: str | None = None,
    hot_budget_mib: int | None = None,
):
    """Build an optional UnifiedPipelineConfig from server overrides."""
    if backend is None and device is None and engine is None and hot_budget_mib is None:
        return None

    from chuk_lazarus.inference import UnifiedPipelineConfig
    from chuk_lazarus.inference.backends.types import LazarusBackend
    from chuk_lazarus.inference.unified import EngineMode

    kwargs: dict[str, str | int | LazarusBackend | EngineMode] = {}

    if backend is not None:
        normalized_backend = backend.strip().lower()
        backend_map = {
            "mlx": LazarusBackend.MLX,
            "torch": LazarusBackend.CUDA,
        }
        if normalized_backend not in backend_map:
            raise ValueError(
                f"Unsupported backend override {backend!r}. Expected one of: mlx, torch."
            )
        kwargs["backend"] = backend_map[normalized_backend]
        kwargs["backend_name"] = normalized_backend

    if device is not None:
        kwargs["device"] = device

    if engine is not None:
        kwargs["engine"] = EngineMode(engine)

    if hot_budget_mib is not None:
        kwargs["hot_budget_mib"] = hot_budget_mib

    return UnifiedPipelineConfig(**kwargs)


# ── Streaming token generator ─────────────────────────────────────────────────


def _stream_tokens(model, tokenizer, prompt: str, config):
    """Token-by-token streaming generator compatible with ModelOutput interface.

    Mirrors the autoregressive loop in ``GemmaForCausalLM.generate()``:
      - First call  : ``model(input_ids)``           → ModelOutput (no cache)
      - Subsequent  : ``model(next_token, cache=…)``  → ModelOutput (with cache)

    Yields decoded text chunks as they become non-empty.
    """
    import mlx.core as mx

    from chuk_lazarus.inference.generation import get_stop_tokens

    input_ids = tokenizer.encode(prompt, return_tensors="np")
    input_ids = mx.array(input_ids)
    stop_tokens = set(config.stop_tokens or get_stop_tokens(tokenizer))

    accumulated: list[int] = []

    # Full-prompt prefill
    output = model(input_ids)
    cache = output.cache

    for _ in range(config.max_new_tokens):
        logits = output.logits[:, -1, :]

        if config.temperature == 0:
            next_token = mx.argmax(logits, axis=-1, keepdims=True)
        else:
            probs = mx.softmax(logits / config.temperature, axis=-1)
            next_token = mx.random.categorical(mx.log(probs + 1e-10))
            next_token = mx.expand_dims(next_token, axis=-1)

        mx.eval(next_token)
        token_id = int(next_token[0, 0])

        if token_id in stop_tokens:
            break

        accumulated.append(token_id)

        text = tokenizer.decode(accumulated, skip_special_tokens=True)
        if text:
            yield text
            accumulated = []

        # Next step — single token with cache
        output = model(next_token, cache=cache)
        cache = output.cache
        mx.eval(output.logits)


def _stream_tokens_torch(runtime, prompt: str, config, *, conversation_id: str | None = None):
    """Token streamer for the torch runtime using Hugging Face's text streamer.

    ``conversation_id`` is forwarded to :meth:`TorchInferenceRuntime.stream_generate`
    so the residual-bounded KV-direct path can key its WARM session cache.
    Runtimes without the kwarg (older shims, mocks) gracefully degrade.
    """
    if getattr(runtime, "engine_mode", "standard") != "standard" and hasattr(
        runtime, "stream_generate"
    ):
        if conversation_id is not None:
            try:
                yield from runtime.stream_generate(
                    prompt, config, conversation_id=conversation_id
                )
                return
            except TypeError:
                # Runtime didn't accept the kwarg — fall through to plain call.
                pass
        yield from runtime.stream_generate(prompt, config)
        return

    from transformers import TextIteratorStreamer

    model_inputs = runtime._tokenize_prompt(prompt)
    input_ids = model_inputs["input_ids"]
    generation_kwargs = runtime._generation_kwargs(config, model_inputs, use_cache=True)
    streamer = TextIteratorStreamer(
        runtime._tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    generation_kwargs["streamer"] = streamer

    errors: list[BaseException] = []
    total_window_tokens = int(input_ids.shape[1]) + config.max_new_tokens

    def _run_generate() -> None:
        try:
            with runtime._torch.inference_mode(), runtime._generation_context(total_window_tokens):
                runtime._model.generate(input_ids=input_ids, **generation_kwargs)
        except BaseException as exc:
            errors.append(exc)
            streamer.on_finalized_text("", stream_end=True)

    thread = threading.Thread(target=_run_generate, daemon=True)
    thread.start()

    for text in streamer:
        if text:
            yield text

    thread.join()
    if errors:
        raise errors[0]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _apply_template(tokenizer, request: InternalRequest) -> str:
    """Format the request into a prompt string via the tokenizer's chat template.

    Handles all message roles (system, user, assistant, tool) and injects
    tool definitions when present so the model knows what it can call.
    """
    messages: list[dict] = []
    for msg in request.messages:
        entry: dict = {"role": msg.role.value}

        # content may be None for assistant messages that only contain tool_calls
        entry["content"] = msg.content

        if msg.tool_call_id is not None:
            entry["tool_call_id"] = msg.tool_call_id
        if msg.name is not None:
            entry["name"] = msg.name
        if msg.tool_calls is not None:
            entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(entry)

    kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
    if request.chat_template_kwargs:
        kwargs.update(request.chat_template_kwargs)
        kwargs["tokenize"] = False
        kwargs["add_generation_prompt"] = True
    if request.tools:
        kwargs["tools"] = [t.model_dump() for t in request.tools]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template(messages, **kwargs)
        except Exception:
            logger.warning(
                "apply_chat_template failed with tools; falling back to basic template",
                exc_info=True,
            )
            # Retry without tools kwarg as a fallback
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass

    # Last-resort simple fallback
    parts = []
    for msg in request.messages:
        role_label = msg.role.value.capitalize()
        parts.append(f"{role_label}: {msg.content or ''}")
    return "\n\n".join(parts) + "\n\nAssistant:"


def _parse_tool_calls(text: str) -> tuple[str | None, list[ToolCall] | None]:
    """Detect and extract ``<tool_call>`` blocks from model output.

    Returns ``(cleaned_text, tool_calls)`` where ``cleaned_text`` is the text
    with all tool-call blocks removed (None if nothing is left), and
    ``tool_calls`` is a list of parsed ToolCall instances (None if none found).
    """
    tool_calls: list[ToolCall] = []

    def _extract(match: re.Match) -> str:
        try:
            data = json.loads(match.group(1))
            tool_calls.append(
                ToolCall(
                    function=ToolCallFunction(
                        name=data["name"],
                        arguments=json.dumps(data.get("arguments", {})),
                    )
                )
            )
        except (json.JSONDecodeError, KeyError):
            logger.warning("Could not parse tool_call block: %s", match.group(0))
        return ""

    cleaned = _TOOL_CALL_RE.sub(_extract, text).strip()
    return (cleaned or None, tool_calls or None)


def _map_stop_reason(raw: str) -> FinishReason:
    """Map GenerationResult.stop_reason to FinishReason."""
    if raw in (StopReason.EOS, StopReason.STOP_TOKEN, StopReason.PLUGIN):
        return FinishReason.STOP
    return FinishReason.LENGTH


# ── Engine ────────────────────────────────────────────────────────────────────


class ModelEngine:
    """
    Holds a single loaded model and serves concurrent inference requests.

    Usage::

        engine = await ModelEngine.load("google/gemma-3-4b-it")
        response = await engine.agenerate(request)

        async for chunk in engine.astream(request):
            ...
    """

    def __init__(
        self,
        pipeline: UnifiedPipeline,
        model_id: str,
        source_model_id: str | None = None,
        session_cache_size: int | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._model_id = model_id
        self._source_model_id = source_model_id or model_id
        self._lock = threading.Lock()
        # The WARM residual session cache is only live for the torch
        # ``residual_bounded_kv_direct`` engine. For every other path the
        # attribute stays ``None`` and the runtime behaves as before.
        self._session_cache = None
        self._session_cache_size = session_cache_size
        self._maybe_install_session_cache()

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def source_model_id(self) -> str:
        return self._source_model_id

    @property
    def session_cache(self):
        """The WARM residual session cache if one is installed, else ``None``."""
        return self._session_cache

    # ── Session cache ─────────────────────────────────────────────────────────

    def _maybe_install_session_cache(self) -> None:
        """Lazily build and attach a residual session cache when applicable.

        The cache is only meaningful when the runtime engine mode is
        ``residual_bounded_kv_direct`` — the only path that both bounds the
        hot KV and keeps a boundary residual on GPU. For every other engine
        mode this is a no-op and ``self._session_cache`` stays ``None``.
        """
        runtime = getattr(self._pipeline, "runtime", None) if self._pipeline else None
        if runtime is None:
            return
        engine_mode = getattr(runtime, "engine_mode", None)
        if engine_mode != "residual_bounded_kv_direct":
            return
        attach = getattr(runtime, "attach_session_cache", None)
        if not callable(attach):
            return

        from ._residual_session_cache import DEFAULT_MAX_SESSIONS, ResidualSessionCache

        max_sessions = (
            int(self._session_cache_size)
            if self._session_cache_size is not None
            else DEFAULT_MAX_SESSIONS
        )

        eviction_hook = None
        try:
            from chuk_lazarus.inference.backends._torch_residual_session import (
                make_cuda_eviction_hook,
            )

            eviction_hook = make_cuda_eviction_hook()
        except Exception:  # pragma: no cover - torch import paths
            eviction_hook = None

        self._session_cache = ResidualSessionCache(
            max_sessions=max_sessions, eviction_hook=eviction_hook
        )
        attach(self._session_cache)

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    async def load(
        cls,
        model_id: str,
        verbose: bool = True,
        backend: str | None = None,
        device: str | None = None,
        engine: str | None = None,
        hot_budget_mib: int | None = None,
        served_model_name: str | None = None,
        session_cache_size: int | None = None,
    ) -> ModelEngine:
        """Async factory — loads the model in a thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: cls._load_sync(
                model_id,
                verbose,
                backend=backend,
                device=device,
                engine=engine,
                hot_budget_mib=hot_budget_mib,
                served_model_name=served_model_name,
                session_cache_size=session_cache_size,
            ),
        )

    @classmethod
    def _load_sync(
        cls,
        model_id: str,
        verbose: bool,
        backend: str | None = None,
        device: str | None = None,
        engine: str | None = None,
        hot_budget_mib: int | None = None,
        served_model_name: str | None = None,
        session_cache_size: int | None = None,
    ) -> ModelEngine:
        from chuk_lazarus.inference import UnifiedPipeline

        pipeline_config = _build_pipeline_config(backend, device, engine, hot_budget_mib)
        pipeline = UnifiedPipeline.from_pretrained(
            model_id,
            pipeline_config=pipeline_config,
            verbose=verbose,
        )
        return cls(
            pipeline,
            served_model_name or model_id,
            source_model_id=model_id,
            session_cache_size=session_cache_size,
        )

    # ── Private sync generation ───────────────────────────────────────────────

    def _generate(self, request: InternalRequest) -> InternalResponse:
        """Blocking generation — must be called from a thread pool."""
        from chuk_lazarus.inference.generation import GenerationConfig

        prompt = _apply_template(self._pipeline.tokenizer, request)
        config = GenerationConfig(
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        with self._lock:
            result = self._pipeline.generate(
                prompt, config=config, conversation_id=request.conversation_id
            )

        usage = InternalUsage(
            prompt_tokens=result.stats.input_tokens,
            completion_tokens=result.stats.output_tokens,
            total_tokens=result.stats.input_tokens + result.stats.output_tokens,
        )

        # Detect tool calls in the output
        cleaned_text, tool_calls = _parse_tool_calls(result.text)

        if tool_calls:
            return InternalResponse(
                content=cleaned_text,
                model=self._model_id,
                finish_reason=FinishReason.TOOL_CALLS,
                usage=usage,
                tool_calls=tool_calls,
            )

        return InternalResponse(
            content=result.text,
            model=self._model_id,
            finish_reason=_map_stop_reason(result.stop_reason),
            usage=usage,
        )

    # ── Public async generation ───────────────────────────────────────────────

    async def agenerate(self, request: InternalRequest) -> InternalResponse:
        """Non-streaming generation (async, runs in thread pool)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self._generate(request))

    async def astream(self, request: InternalRequest) -> AsyncIterator[InternalChunk]:
        """
        Streaming generation — yields InternalChunk instances.

        Text chunks are streamed token-by-token via ``_stream_tokens``.
        When the full output contains a ``<tool_call>`` block the accumulated
        text is re-parsed and a final chunk carrying ``tool_calls`` is yielded.
        """
        from chuk_lazarus.inference.generation import GenerationConfig

        prompt = _apply_template(self._pipeline.tokenizer, request)
        config = GenerationConfig(
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[InternalChunk | BaseException | None] = asyncio.Queue()

        def _run_sync() -> None:
            try:
                runtime = self._pipeline.runtime
                if runtime.backend.value == "cuda":
                    token_stream = _stream_tokens_torch(
                        runtime,
                        prompt,
                        config,
                        conversation_id=request.conversation_id,
                    )
                else:
                    token_stream = _stream_tokens(
                        self._pipeline.model,
                        self._pipeline.tokenizer,
                        prompt,
                        config,
                    )

                for text in token_stream:
                    loop.call_soon_threadsafe(queue.put_nowait, InternalChunk(content=text))
            except BaseException as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        thread = threading.Thread(target=_run_sync, daemon=True)
        thread.start()

        accumulated_text = ""

        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            accumulated_text += item.content or ""
            yield item

        # After streaming completes, check for tool calls in accumulated output
        cleaned_text, tool_calls = _parse_tool_calls(accumulated_text)
        if tool_calls:
            yield InternalChunk(
                content=None,
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=tool_calls,
            )
        else:
            yield InternalChunk(content=None, finish_reason=FinishReason.STOP)
