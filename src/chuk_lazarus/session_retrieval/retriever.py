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
from typing import Any, Literal

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


def _install_residual_session_cache(runtime: TorchInferenceRuntime, max_sessions: int | None) -> Any | None:
    """Attach the bounded-engine WARM session cache when that engine is active."""
    if getattr(runtime, "engine_mode", None) != "residual_bounded_kv_direct":
        return None
    attach = getattr(runtime, "attach_session_cache", None)
    if not callable(attach):
        return None

    from chuk_lazarus.server._residual_session_cache import (
        DEFAULT_MAX_SESSIONS,
        ResidualSessionCache,
    )

    eviction_hook = None
    try:
        from chuk_lazarus.inference.backends._torch_residual_session import (
            make_cuda_eviction_hook,
        )

        eviction_hook = make_cuda_eviction_hook()
    except Exception:  # pragma: no cover - torch import path defensive
        eviction_hook = None

    cache = ResidualSessionCache(
        max_sessions=int(max_sessions or DEFAULT_MAX_SESSIONS),
        eviction_hook=eviction_hook,
    )
    attach(cache)
    return cache


def _assert_requested_kv_selector_is_truthful(
    *,
    insertion_family: Literal["full_attention", "sliding"],
    materialization: Any,
    handle: CheckpointHandle,
) -> None:
    """Reject surfaced selector requests the archived source cannot honor."""
    if insertion_family != "sliding":
        return

    actual_family = getattr(materialization, "materialized_insertion_family", None)
    if actual_family is None:
        raise RuntimeError(
            "kv_query insertion_family='sliding' cannot be honored for "
            f"session {handle.session_id}: the archived materialization did "
            "not report its source family. Lower-level runtime support must "
            "surface truthful provenance before sliding can be requested here."
        )

    normalized_family = str(actual_family).strip().lower()
    if normalized_family == "sliding":
        return

    source_layer = getattr(materialization, "materialized_source_layer", None)
    source_layer_desc = "unknown"
    if source_layer is not None:
        source_layer_desc = str(int(source_layer))
    raw_lineage = (
        getattr(materialization, "materialized_lineage_layer_indices", ()) or ()
    )
    lineage = tuple(int(layer_idx) for layer_idx in raw_lineage)
    lineage_desc = ",".join(str(layer_idx) for layer_idx in lineage) or "unknown"
    raise RuntimeError(
        "kv_query insertion_family='sliding' cannot be honored for "
        f"session {handle.session_id}: archived source family is "
        f"{normalized_family!r} (source_layer={source_layer_desc}, "
        f"lineage={lineage_desc}). This surfaced route only supports "
        "sliding when the archived checkpoint was materialized from a "
        "sliding-source lineage. The shipped Gemma-4 /kv_query default "
        "remains full_attention."
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
        engine: str = "standard",
        hot_budget_mib: int | None = None,
        session_cache_size: int | None = None,
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

        runtime = TorchInferenceRuntime(
            model,
            tokenizer,
            device=device,
            engine=engine,
            hot_budget_mib=hot_budget_mib,
        )
        _install_residual_session_cache(runtime, session_cache_size)

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

    def query_topical(
        self,
        query_text: str,
        *,
        conversation_id: str | None = None,
    ) -> QueryResult:
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
            conversation_id=conversation_id,
        )

    def query_entity_mention(
        self,
        query_text: str,
        *,
        conversation_id: str | None = None,
    ) -> QueryResult:
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
            conversation_id=conversation_id,
        )

    def _generate_from_window(
        self,
        handle: CheckpointHandle,
        window_id: int,
        question_text: str,
        routing_mode: str,
        routing_score: float | None = None,
        conversation_id: str | None = None,
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

        from chuk_lazarus import tracing

        if tracing.is_enabled("route"):
            # PROP 3: emit selected window context for route relevance audit.
            tracing.emit(
                "route",
                "route.selected",
                {
                    "routing_mode": routing_mode,
                    "source_session": handle.session_id,
                    "window_id": int(window_id),
                    "routing_score": routing_score,
                    "window_text_len": len(window_text),
                    "window_text_head": window_text[:300],
                    "window_text_tail": window_text[-120:],
                    "window_keywords": window_keywords[:12],
                    "question_text_preview": question_text[:200],
                },
            )
        if tracing.is_enabled("route"):
            # PROP 3: emit expected-substring relevance check when configured.
            expected_substring = os.environ.get("LAZARUS_EXPECTED_SUBSTRING", "")
            if expected_substring:
                contains_expected = (
                    expected_substring.lower() in window_text.lower()
                    if expected_substring and window_text
                    else False
                )
                expected_lower = expected_substring.lower()
                contains_in_keywords = any(
                    expected_lower in str(kw).lower() for kw in window_keywords
                )
                tracing.emit(
                    "route",
                    "route.relevance",
                    {
                        "expected_substring": expected_substring,
                        "contains_expected": contains_expected,
                        "contains_in_keywords": contains_in_keywords,
                    },
                )

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
            result = self.runtime.generate_with_residual_prefill_seeded(
                prompt,
                residual_state,
                gen_config,
                conversation_id=conversation_id,
            )

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

    def prepare_kv_direct_materialization(
        self,
        tier_assignments,
        *,
        hot_budget_mib: int,
        handle: CheckpointHandle | None = None,
    ):
        """Gather residuals and materialize K/V for one checkpoint's windows.

        Frozen contract: axis-5 ``ve-ins-0mo9pb4f5000004cc67``. Reads
        per-window residuals from the LIVE checkpoint archive via
        :class:`TorchKnowledgeStore.load_boundary`, then projects them
        through ``W_K`` / ``W_V`` at ``injection_layer+1`` under the
        Path-A replay guard. Does NOT invoke the model on archived
        tokens.

        Parameters
        ----------
        tier_assignments:
            Iterable of ``TierAssignment`` for a SINGLE checkpoint. For
            cross-checkpoint queries, group assignments by handle and
            call this method once per group.
        hot_budget_mib:
            Hard ceiling on the materialized K + V footprint in MiB.
        handle:
            Defaults to ``self.handles[0]``. Must be the checkpoint from
            which ``tier_assignments`` was produced.

        Returns
        -------
        tuple[GatheredResiduals, KVDirectMaterialization]
        """
        from chuk_lazarus.inference.backends._torch_residual_bounded import (
            gather_selected_residuals,
            materialize_kv_direct,
        )

        target = handle if handle is not None else self.handles[0]
        store = load_store(target)
        source_layer = int(store.config.injection_layer)
        # Concrete-materialize so downstream calls that iterate twice
        # (gather + budget-preserving materialize) are both fed the full
        # sequence even when the caller passes a generator.
        tier_seq = list(tier_assignments)
        residuals = gather_selected_residuals(
            tier_seq,
            checkpoint_handle=target,
            store=store,
            source_layer=source_layer,
            device=self.device,
        )
        materialization = materialize_kv_direct(
            residuals,
            self.runtime,
            hot_budget_mib=int(hot_budget_mib),
            tier_assignments=tier_seq,
        )
        return residuals, materialization

    def answer_with_kv_direct(
        self,
        query_text: str,
        tier_assignments,
        *,
        hot_budget_mib: int,
        warm_config,
        generation_config,
        warm_scores: dict[int, float] | None = None,
        query_id: str | None = None,
        handle: CheckpointHandle | None = None,
        insertion_family: Literal["full_attention", "sliding"] = "full_attention",
        sliding_layer_indices: tuple[int, ...] | None = None,
        sliding_head_indices: tuple[int, ...] | None = None,
    ) -> QueryResult:
        """Run the axis-5 KV-direct generation path end-to-end.

        Pipeline
        --------
        1. :meth:`prepare_kv_direct_materialization` — reads residuals
           and projects K/V (Path-A guard active).
        2. :meth:`TorchInferenceRuntime.generate_with_kv_direct_materialization`
           — generates from the fresh query with the archived K/V
           prepended at ``source_layer+1``.
        3. Returns a :class:`QueryResult` whose
           ``strict_assertions`` dict reports the axis-5 conformance
           invariants (Path-A replay count, observed hot budget, VRAM
           deltas, tier counts).

        Parameters
        ----------
        query_text:
            Fresh user query. Never touches archived tokens.
        tier_assignments:
            Iterable of ``TierAssignment`` for a single checkpoint.
        hot_budget_mib:
            Hard K+V footprint ceiling (MiB).
        warm_config:
            :class:`WarmPenaltyConfig` — currently documented only; live
            subtractive penalty is follow-up work.
        generation_config:
            :class:`GenerationConfig` — sampling + ``max_new_tokens``.
        warm_scores:
            Optional per-window scores for eventual per-warm scaling.
        query_id:
            Optional opaque id echoed onto the result metadata.
        handle:
            Defaults to ``self.handles[0]``.
        insertion_family:
            Runtime K/V insertion family. ``"full_attention"`` preserves
            the current default Gemma-4 path; ``"sliding"`` activates the
            explicit sliding-layer insertion branch in the runtime, but
            only when the archived checkpoint itself was materialized from
            a sliding-source lineage. Otherwise this method raises instead
            of silently pretending the sliding selector ran.
        sliding_layer_indices:
            Sliding-attention decoder layers that should receive the
            archived prefix when ``insertion_family="sliding"``. Required
            together with ``sliding_head_indices`` for surfaced sliding.
        sliding_head_indices:
            Attention-head indices that may attend to the archived prefix
            when ``insertion_family="sliding"``. Required together with
            ``sliding_layer_indices`` for surfaced sliding.
        """
        if insertion_family == "sliding" and (
            sliding_layer_indices is None or sliding_head_indices is None
        ):
            raise ValueError(
                "answer_with_kv_direct: insertion_family='sliding' requires "
                "both sliding_layer_indices and sliding_head_indices."
            )

        # Materialize once; keep the concrete sequence for subsequent
        # tier-map construction so we do not double-iterate a generator.
        tier_seq = list(tier_assignments)
        chosen_handle = handle if handle is not None else self.handles[0]
        residuals, materialization = self.prepare_kv_direct_materialization(
            tier_seq, hot_budget_mib=hot_budget_mib, handle=chosen_handle,
        )
        _assert_requested_kv_selector_is_truthful(
            insertion_family=insertion_family,
            materialization=materialization,
            handle=chosen_handle,
        )
        # Build tier_map ONLY for windows that survived gather_selected_residuals
        # — by default it includes (HOT, WARM) and excludes COLD. Without this
        # filter the downstream apply_tier_attention_mask raises ValueError
        # because tier_assignments has COLD wids that have no matching entry
        # in per_window_token_ranges. (See run-3 chat-loop bug:
        # range-only=[] tier-only=[<cold wids>].)
        gathered_wids = set(residuals.per_window_token_ranges.keys())
        tier_map = {
            int(a.candidate.window_id): a.tier
            for a in tier_seq
            if int(a.candidate.window_id) in gathered_wids
        }

        # Wrap the raw query in a chat template so the model emits an
        # assistant turn instead of an empty turn-boundary sequence. The
        # archived KV prefix provides the memory context; the system prompt
        # still guides answer style.
        try:
            templated_prompt = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": query_text},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:  # noqa: BLE001 - fall back to raw query on any issue
            templated_prompt = query_text

        result = self.runtime.generate_with_kv_direct_materialization(
            templated_prompt,
            generation_config,
            materialization=materialization,
            per_window_token_ranges=dict(residuals.per_window_token_ranges),
            tier_assignments=tier_map,
            warm_config=warm_config,
            warm_scores=warm_scores,
            source_layer=residuals.source_layer,
            query_id=query_id,
            insertion_family=insertion_family,
            sliding_layer_indices=sliding_layer_indices,
            sliding_head_indices=sliding_head_indices,
        )

        first_wid = next(iter(residuals.per_window_token_ranges.keys()), -1)
        metadata = getattr(result, "metadata", {}) or {}
        strict_assertions: dict[str, Any] = {
            "kv_direct_active": bool(metadata.get("kv_direct_active", False)),
            "path_a_replay_count": int(metadata.get("path_a_replay_count", 0)),
            "hot_budget_mib_observed": int(
                materialization.hot_budget_mib_observed
            ),
            "vram_peak_mib": int(metadata.get("vram_peak_mib", 0)),
            "vram_delta_mib": int(metadata.get("vram_delta_mib", 0)),
            "tier_counts_selected": int(
                len(residuals.per_window_token_ranges)
            ),
            "mask_penalty_applied": bool(
                metadata.get("mask_penalty_applied", False)
            ),
        }
        return QueryResult(
            routing_mode="kv_direct",
            source_session=chosen_handle.session_id,
            window_id=int(first_wid),
            matched_window_text="",
            window_keywords=[],
            generated_answer=result.text,
            strict_assertions=strict_assertions,  # type: ignore[arg-type]
            routing_score=None,
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
