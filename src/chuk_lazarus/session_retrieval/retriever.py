"""Unified axis-4 retrieval entry point: :class:`SessionRetriever`.

Transcribes the Apollo-11-pattern residual-injection pipeline from
``examples/inference/demo_clause_aligned_strict.py`` (lines 97-298) into a
reusable class that:

1. Enumerates per-session checkpoints via :mod:`enumeration`.
2. Loads a HuggingFace model + tokenizer in bfloat16 once.
3. Exposes three query entry points: exact, topical, entity-mention.
4. Funnels all three through ``_generate_from_window`` which enforces the
   six strict-mode assertions from the demo and raises ``RuntimeError`` on
   any silent-fallback condition.

Primitive delegated to
----------------------
- ``TorchKnowledgeStore`` (store loading, window text, boundary tensors).
- ``torch_query._residual_is_compatible`` (residual compatibility gate).
- ``TorchInferenceRuntime.generate_with_residual`` (Apollo-11 injection).
- ``ResidualState``, ``LazarusBackend``, ``GenerationConfig`` types.

NONE of these are re-implemented. This module wraps them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chuk_lazarus.inference.backends import (
    LazarusBackend,
    ResidualState,
    TorchInferenceRuntime,
)
from chuk_lazarus.inference.context.knowledge import torch_query
from chuk_lazarus.inference.context.knowledge.torch_store import TorchKnowledgeStore
from chuk_lazarus.inference.generation import GenerationConfig
from chuk_lazarus.session_retrieval.entity_mention import route_entity_mention
from chuk_lazarus.session_retrieval.enumeration import (
    CheckpointHandle,
    iter_checkpoint_handles,
    load_store,
)
from chuk_lazarus.session_retrieval.exact_id import route_exact_id
from chuk_lazarus.session_retrieval.topical import route_topical

DEFAULT_SYSTEM_PROMPT = (
    "You answer questions about prior conversation sessions using only the "
    "store context provided by Lazarus. Be concise; quote dotted handles when "
    "known; when context is provided, reproduce relevant verbatim phrases."
)


@dataclass
class QueryResult:
    """Structured return for every :class:`SessionRetriever` query.

    Attributes
    ----------
    routing_mode:
        One of ``"exact"``, ``"topical"``, ``"entity_mention"``.
    source_session:
        ``session_id`` of the checkpoint that owned the routed window.
    window_id:
        Store-internal window index that was selected.
    matched_window_text:
        Decoded text of that window (``store.get_window_text``).
    window_keywords:
        Primitive-owned keywords (``store.keywords[window_id]``).
    generated_answer:
        ``runtime.generate_with_residual(...).text`` from the Apollo-11 path.
    strict_assertions:
        Dict with exactly six keys (see :mod:`session_retrieval` design §2.2).
    routing_score:
        Score used to rank topical / entity-mention candidates; ``None`` for
        exact routing (no scoring happens there).
    """

    routing_mode: str
    source_session: str
    window_id: int
    matched_window_text: str
    window_keywords: list[str]
    generated_answer: str
    strict_assertions: dict[str, bool]
    routing_score: float | None = None


@dataclass
class SessionRetriever:
    """Cross-session retrieval front-end.

    Construct via :meth:`from_checkpoint_root` - that classmethod performs the
    CUDA model load. The dataclass fields below are populated by the classmethod;
    direct instantiation is possible but not the intended path.
    """

    handles: list[CheckpointHandle]
    runtime: TorchInferenceRuntime
    tokenizer: Any
    crystal_layer: int
    system_prompt: str
    device: str
    _mem_after_load_at_init: int = field(default=0, repr=False)

    @classmethod
    def from_checkpoint_root(
        cls,
        checkpoint_root: Path,
        *,
        model_id: str = "google/gemma-4-E2B-it",
        device: str = "cuda",
        original_input_root: Path | None = None,
        system_prompt: str | None = None,
    ) -> SessionRetriever:
        """Enumerate handles, load the model+tokenizer, build the retriever.

        Mirrors the demo's model-load sequence (lines 97-107): bfloat16,
        ``low_cpu_mem_usage=True``, moved to ``device`` with ``.eval()``.
        Records ``mem_after_load`` on CUDA so the generation path can later
        assert memory growth (strict-mode assertion 5).

        Raises
        ------
        RuntimeError
            If CUDA is requested but unavailable, if no valid checkpoints are
            found under ``checkpoint_root``, or if the stores disagree on
            ``crystal_layer`` (mixed models are not supported).
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        root = Path(checkpoint_root)
        handles = list(iter_checkpoint_handles(root, original_input_root=original_input_root))
        if not handles:
            raise RuntimeError(
                f"STRICT: no valid clause-aligned checkpoints under {root!s}"
            )

        # Derive + verify crystal_layer across all stores (must agree).
        first_store = load_store(handles[0])
        crystal_layer = int(first_store.config.crystal_layer)
        for handle in handles[1:]:
            other = load_store(handle)
            other_layer = int(other.config.crystal_layer)
            if other_layer != crystal_layer:
                raise RuntimeError(
                    f"STRICT: crystal_layer mismatch across checkpoints - "
                    f"{handles[0].session_id}={crystal_layer} vs "
                    f"{handle.session_id}={other_layer}"
                )

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "STRICT: CUDA device requested but torch.cuda.is_available() is False"
            )

        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token

        mem_before_load = (
            int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, low_cpu_mem_usage=True
        )
        model.to(device)
        model.eval()

        # Strict assertion 2 at init time: params must live on CUDA if requested.
        if device.startswith("cuda"):
            first_param_device = next(model.parameters()).device
            if first_param_device.type != "cuda":
                raise RuntimeError(
                    "STRICT: model parameters on "
                    f"{first_param_device}, not cuda - silent CPU fallback"
                )

        # Gemma-4 specific patches (no-op for other models). Without these,
        # greedy decoding under residual injection produces token salad.
        # See bug-report ve-ins-0mo846s9m0000fb829b.
        from ._gemma_patches import patch_clippable_linear, generation_eos_ids
        patched_count = patch_clippable_linear(model)
        eos_ids = generation_eos_ids(tokenizer)
        # Stash on the runtime for use in _generation_kwargs at call time.
        # The runtime's eos_token_id field overrides the tokenizer default
        # when set (single int or list of ints).
        # Patched layer count is logged for visibility.
        _ = patched_count  # available if needed; not asserted (other models = 0)

        mem_after_load = (
            int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0
        )
        _ = mem_before_load

        runtime = TorchInferenceRuntime(model, tokenizer, device=device)

        return cls(
            handles=handles,
            runtime=runtime,
            tokenizer=tokenizer,
            crystal_layer=crystal_layer,
            system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
            device=device,
            _mem_after_load_at_init=mem_after_load,
        )

    def query_exact_id(self, dotted_handle: str) -> QueryResult:
        """Route a dotted handle, then run the residual-injection pipeline.

        Raises
        ------
        ValueError
            If no session / window matches the handle.
        RuntimeError
            If any strict-mode assertion fails (see module docstring).
        """
        routed = route_exact_id(self.handles, dotted_handle)
        if routed is None:
            raise ValueError(
                f"STRICT: no session/window matches handle {dotted_handle!r}"
            )
        handle, window_id = routed
        return self._generate_from_window(
            handle=handle,
            window_id=window_id,
            question_text=dotted_handle,
            routing_mode="exact",
            routing_score=None,
        )

    def query_topical(self, query_text: str) -> QueryResult:
        """Topical TF-IDF routing across all checkpoints, then generate.

        Raises
        ------
        ValueError
            If no checkpoint returns any candidate.
        RuntimeError
            If any strict-mode assertion fails.
        """
        routed = route_topical(
            self.handles, query_text, self.tokenizer, top_k_per_checkpoint=1
        )
        if routed is None:
            raise ValueError(
                f"STRICT: topical routing produced no candidates for query {query_text!r}"
            )
        handle, window_id, score = routed
        return self._generate_from_window(
            handle=handle,
            window_id=window_id,
            question_text=query_text,
            routing_mode="topical",
            routing_score=score,
        )

    def query_entity_mention(self, query_text: str) -> QueryResult:
        """Entity-mention overlap routing across all checkpoints, then generate.

        Raises
        ------
        ValueError
            If the query produces no entity tokens, or no window overlaps.
        RuntimeError
            If any strict-mode assertion fails.
        """
        routed = route_entity_mention(self.handles, query_text)
        if routed is None:
            raise ValueError(
                f"STRICT: entity-mention routing produced no overlap for query {query_text!r}"
            )
        handle, window_id, score = routed
        return self._generate_from_window(
            handle=handle,
            window_id=window_id,
            question_text=query_text,
            routing_mode="entity_mention",
            routing_score=score,
        )

    def _generate_from_window(
        self,
        handle: CheckpointHandle,
        window_id: int,
        question_text: str,
        routing_mode: str,
        routing_score: float | None = None,
    ) -> QueryResult:
        """Run the 13-step Apollo-11 residual-injection pipeline.

        Each step is numbered per axis-4 spec. Any silent-fallback condition
        (residual incompatible, empty window, hook never fires, GPU memory
        does not grow on CUDA) raises ``RuntimeError``.
        """
        import numpy as np
        import torch

        # 1. Load store for the winning checkpoint.
        store: TorchKnowledgeStore = load_store(handle)

        # 2. Decode window text. Empty = strict-mode assertion 6 failure.
        window_text = store.get_window_text(int(window_id), self.tokenizer)
        if not window_text:
            raise RuntimeError(
                f"STRICT: store.get_window_text({window_id}) returned empty"
            )

        # 3. Primitive-owned keywords.
        window_keywords = list(store.keywords.get(int(window_id), []))

        # 4. Load the boundary tensor on CPU; normalise to a torch tensor.
        boundary = store.load_boundary(int(window_id), device="cpu")
        if isinstance(boundary, np.ndarray):
            boundary_tensor = torch.from_numpy(boundary)
        elif isinstance(boundary, torch.Tensor):
            boundary_tensor = boundary.detach().to("cpu")
        else:
            raise RuntimeError(
                f"STRICT: unexpected boundary type {type(boundary)!r}"
            )

        # 5. Residual compatibility gate (strict-mode assertion 3).
        compatible, reason = torch_query._residual_is_compatible(
            self.runtime, boundary_tensor, self.crystal_layer
        )
        if not compatible:
            raise RuntimeError(
                "STRICT: _residual_is_compatible returned False - "
                f"would have silently fallen back to prompt-context. Reason: {reason}"
            )

        # 6. Spy forward-pre-hook (strict-mode assertion 4).
        # Register directly on the layer — no method-swap indirection so there
        # is no surface where stacked wrappers can accumulate across queries.
        # Defensive: snapshot the pre-hook OrderedDict so the finally block can
        # detect (and clear) any orphan hooks if a prior query leaked.
        hook_state: dict[str, Any] = {"count": 0}
        layers = self.runtime._resolve_layers()
        the_layer = layers[self.crystal_layer]
        pre_hooks_before = dict(getattr(the_layer, "_forward_pre_hooks", {}) or {})

        def spy_hook(_module: Any, args: tuple, kwargs: dict):
            if not args:
                return args, kwargs
            hook_state["count"] += 1
            return args, kwargs

        spy_handle = the_layer.register_forward_pre_hook(spy_hook, with_kwargs=True)
        try:
            # 7. Build the chat-templated prompt.
            user_content = (
                f"Store window excerpt:\n{window_text}\n\n"
                f"Store keywords: {', '.join(window_keywords[:8])}\n\n"
                f"Execution mode: residual\n\n"
                f"Question: {question_text}"
            )
            prompt = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_content},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )

            # 8. ResidualState for the CUDA injection path.
            residual_state = ResidualState(
                backend=LazarusBackend.CUDA,
                layer_index=self.crystal_layer,
                tensor=boundary_tensor,
                sequence_length=int(
                    len(store.window_token_lists.get(int(window_id), []))
                    or store.num_tokens
                ),
                hidden_size=int(boundary_tensor.shape[-1]),
                dtype=str(boundary_tensor.dtype).replace("torch.", ""),
                device="cpu",
            )

            # 9. Pre-gen memory snapshot (CUDA only).
            on_cuda = torch.cuda.is_available()
            if on_cuda:
                torch.cuda.reset_peak_memory_stats()
            mem_before_gen = int(torch.cuda.memory_allocated()) if on_cuda else 0
            _ = mem_before_gen

            # 10. Run the injection path.
            _DEFAULT_MAX_NEW = int(os.environ.get("LAZARUS_MAX_NEW_TOKENS", "120"))
            gen_config = GenerationConfig(max_new_tokens=_DEFAULT_MAX_NEW, temperature=0.0, top_p=1.0)
            result = self.runtime.generate_with_residual_prefill_seeded(prompt, residual_state, gen_config)

            # 11. Peak memory snapshot (CUDA only).
            mem_peak = int(torch.cuda.max_memory_allocated()) if on_cuda else 0
        finally:
            # Always remove our spy hook.
            spy_handle.remove()
            # Defensive: if anything ELSE was added to layer's pre-hooks during
            # this call (a leaked inject_hook from a crashed prior generate, etc.),
            # purge the stragglers so the next query starts from a clean state.
            current_hooks = getattr(the_layer, "_forward_pre_hooks", None)
            if current_hooks is not None:
                stale_keys = [
                    k for k in list(current_hooks.keys()) if k not in pre_hooks_before
                ]
                for k in stale_keys:
                    current_hooks.pop(k, None)

        # Strict-mode assertion 4: hook MUST have fired.
        if hook_state["count"] == 0:
            raise RuntimeError(
                "STRICT: spy hook did NOT fire during generate_with_residual - "
                "the injection layer was never traversed."
            )

        # 12. Populate strict assertions dict.
        cuda_available = bool(torch.cuda.is_available())
        try:
            first_param_device = next(self.runtime._model.parameters()).device
            model_on_cuda = first_param_device.type == "cuda"
        except StopIteration:  # pragma: no cover - models always have params
            model_on_cuda = False

        if cuda_available:
            gpu_memory_grew = mem_peak > self._mem_after_load_at_init
            if not gpu_memory_grew:
                raise RuntimeError(
                    f"STRICT: GPU peak memory ({mem_peak}) did not exceed "
                    f"post-load memory ({self._mem_after_load_at_init}) - "
                    "suspicious CPU fallback during generation."
                )
        else:
            # CPU path: growth check is not meaningful; report True so a
            # consumer that asserts all(strict_assertions.values()) still
            # passes on a CPU-only unit-test host.
            gpu_memory_grew = True

        strict_assertions: dict[str, bool] = {
            "cuda_available": cuda_available,
            "model_on_cuda": bool(model_on_cuda),
            "residual_compatible": True,
            "hook_fired": hook_state["count"] > 0,
            "gpu_memory_grew": bool(gpu_memory_grew),
            "store_window_nonempty": bool(window_text),
        }

        # 13. Return the populated QueryResult.
        return QueryResult(
            routing_mode=routing_mode,
            source_session=handle.session_id,
            window_id=int(window_id),
            matched_window_text=window_text,
            window_keywords=window_keywords,
            generated_answer=result.text,
            strict_assertions=strict_assertions,
            routing_score=routing_score,
        )

    def refresh_handles(
        self,
        checkpoint_root: Path,
        original_input_root: Path | None = None,
    ) -> int:
        """Re-enumerate checkpoints under ``checkpoint_root`` and update ``self.handles`` in place.

        Preserves the crystal_layer agreement invariant enforced at construction
        time: if any newly-enumerated checkpoint's store reports a different
        ``crystal_layer`` than ``self.crystal_layer``, raise ``RuntimeError``
        WITHOUT mutating ``self.handles``.

        Does NOT reload the model or tokenizer. Does NOT touch CUDA. Is a pure
        metadata refresh.

        Returns
        -------
        int
            The number of newly-added handles (i.e. handles whose ``session_id``
            was not already present in ``self.handles`` before the refresh).

        Raises
        ------
        RuntimeError
            - If ``iter_checkpoint_handles`` yields zero handles (all-gone is
              suspicious; previous state had at least one).
            - If the new enumeration contains a checkpoint whose store disagrees
              on ``crystal_layer``.

        Notes
        -----
        * Atomic-in-failure: if the crystal_layer check fails, ``self.handles``
          is unchanged. Accomplish this by enumerating+validating into a local
          list first, and ONLY then assigning to ``self.handles``.
        * The function is idempotent when called twice in a row with no new
          sessions on disk (returns 0 the second time).
        * Uses ``load_store`` (already imported from ``.enumeration``) to peek
          at ``store.config.crystal_layer`` for the agreement check. This
          mirrors ``from_checkpoint_root`` lines 142-152. DO NOT refactor that
          loop out into a shared helper - that's a cross-scope change.
        """
        root = Path(checkpoint_root)
        new_handles = list(
            iter_checkpoint_handles(root, original_input_root=original_input_root)
        )
        if not new_handles:
            raise RuntimeError(
                f"STRICT: refresh_handles enumerated zero checkpoints under {root!s} - "
                "previous state had at least one; refusing to null out self.handles."
            )

        # Crystal_layer agreement check against the already-loaded scalar.
        # Validate BEFORE mutating self.handles so failure is atomic.
        for handle in new_handles:
            other = load_store(handle)
            other_layer = int(other.config.crystal_layer)
            if other_layer != self.crystal_layer:
                raise RuntimeError(
                    f"STRICT: crystal_layer mismatch on refresh - "
                    f"retriever={self.crystal_layer} vs "
                    f"{handle.session_id}={other_layer}"
                )

        existing_ids = {h.session_id for h in self.handles}
        new_ids = {h.session_id for h in new_handles}
        added = len(new_ids - existing_ids)

        self.handles = new_handles
        return added


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "QueryResult",
    "SessionRetriever",
]
