"""
Residual-backed bounded KV-direct helpers for the torch/CUDA Qwen runtime.

This module ports the *methodology* of the MLX research engine at
``chuk_lazarus.inference.context.research.bounded_engine.BoundedKVEngine``
(generation_mode ``kv_direct``) onto the Hugging Face torch path used by the
active Qwen serve runtime.

Research methodology (Gemma, MLX)
---------------------------------
Three tiers:

  HOT   — direct K,V cache, bounded by device-memory budget.
  WARM  — residual checkpoints (per-layer, last-position pre-final-norm
          hidden state).  A Markov-sufficient statistic for all prior
          computation: given the residual, no information about prior tokens
          is needed downstream.
  COLD  — token ids, never evicted, 4 bytes/token.

Eviction is *lossless*: when the hot window exceeds budget, the evicted
history is summarised by a residual checkpoint and the hot KV can be rebuilt
via a residual-seeded prefill on the last-N cold tokens.  That rebuild uses
the ``prefill_from_layer``-style trick (MLX) / ``generate_with_residual_
prefill_seeded`` trick (torch) where position 0 of the rebuilt window is
overwritten with the boundary residual via a forward_pre_hook on
``layers[0]``.

What this file provides for the torch Qwen path
-----------------------------------------------
* ``ResidualBoundedState`` — the three-tier state container.
* ``install_boundary_residual_capture_hook`` — always-on forward hook that
  keeps the most recent last-position residual fresh on every forward pass.
* ``install_layer0_residual_seed_hook`` — layer-0 forward_pre_hook that
  replaces ``hidden_states[:, 0, :]`` with the boundary residual on the next
  multi-position forward (ignoring single-position decode steps).
* ``estimate_kv_bytes_per_token`` — config-derived bytes/token estimate so
  the runtime can decide whether the prompt itself exceeds the budget before
  doing a full prefill.

None of these helpers allocate their own GPU memory; they only shape the
forward behaviour of an existing model.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any


def _text_config(config: Any) -> Any:
    """Return ``config.text_config`` for VLM wrappers, else ``config`` itself.

    Gemma 4 / Qwen3.5-MoE conditional-generation wrappers keep the text
    backbone's dimensions under ``text_config``.  The unwrap matches the
    pattern already used in ``UnifiedPipeline._from_pretrained_torch``.
    """
    text_cfg = getattr(config, "text_config", None)
    if text_cfg is None:
        return config
    # VLM wrappers sometimes leave top-level fields populated with None; pick
    # the text backbone when any key field is missing on the wrapper.
    for field_name in ("num_hidden_layers", "num_attention_heads", "hidden_size"):
        if getattr(config, field_name, None) is None:
            return text_cfg
    return config


def _dtype_byte_size(dtype_name: str) -> int:
    """Bytes per element for the common transformer dtypes."""
    normalized = dtype_name.replace("torch.", "").strip().lower()
    return {
        "bfloat16": 2,
        "float16": 2,
        "half": 2,
        "float32": 4,
        "float": 4,
    }.get(normalized, 2)


def estimate_kv_bytes_per_token(model: Any) -> int:
    """Theoretical bytes/token for K + V across all transformer layers.

    Formula::

        bytes_per_token = 2  # K and V
                        * num_kv_heads
                        * head_dim
                        * num_layers
                        * elem_bytes

    For MoE models the KV cache depends only on K/V heads — experts add
    parameter memory but do not multiply KV state.  ``num_key_value_heads``
    defaults to ``num_attention_heads`` when the config does not expose GQA.
    """
    raw_config = getattr(model, "config", None)
    if raw_config is None:
        raise RuntimeError("Model has no config attribute; cannot estimate KV bytes/token.")
    config = _text_config(raw_config)

    num_layers = getattr(config, "num_hidden_layers", None)
    if num_layers is None:
        raise RuntimeError("config.num_hidden_layers is required to estimate KV bytes/token.")
    num_attention_heads = getattr(config, "num_attention_heads", None)
    if num_attention_heads is None:
        raise RuntimeError("config.num_attention_heads is required to estimate KV bytes/token.")
    num_kv_heads = getattr(config, "num_key_value_heads", None) or num_attention_heads
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            raise RuntimeError(
                "config.hidden_size is required to derive head_dim when head_dim is absent."
            )
        head_dim = int(hidden_size) // int(num_attention_heads)

    model_dtype = getattr(model, "dtype", None)
    if model_dtype is None:
        # Fall back to first parameter's dtype.
        try:
            first_param = next(iter(model.parameters()))
            model_dtype = first_param.dtype
        except StopIteration:
            model_dtype = None
    dtype_name = str(model_dtype) if model_dtype is not None else "bfloat16"
    elem_bytes = _dtype_byte_size(dtype_name)

    return int(2 * int(num_kv_heads) * int(head_dim) * int(num_layers) * elem_bytes)


@dataclass
class ResidualBoundedState:
    """Three-tier residual-backed bounded state for one generation pass.

    Fields mirror the research ``ConversationState``:

      cold_token_ids                  — all tokens processed so far (T3).
      boundary_residual               — last captured pre-final-norm last-position
                                         hidden state (T2, Markov-sufficient).
      boundary_residual_position      — absolute token index at which the
                                         residual was captured.
      hot_window_start                — absolute index of first token currently
                                         represented in the hot KV cache.
      hot_token_count                 — number of tokens currently in hot KV.

      prefill_was_split / rehydrate_count — path labels for observability.
    """

    cold_token_ids: list[int] = field(default_factory=list)
    boundary_residual: Any | None = None
    boundary_residual_position: int = -1
    hot_window_start: int = 0
    hot_token_count: int = 0
    prefill_was_split: bool = False
    rehydrate_count: int = 0


class _BoundaryResidualHook:
    """Forward hook that captures the last-position hidden state each call.

    Registered on the *last* transformer layer.  The captured tensor is the
    pre-final-norm residual — i.e. the Markov state of the current forward's
    last token.  Cloned and detached so subsequent forwards do not alias
    gradient-tracked storage.
    """

    def __init__(self) -> None:
        self.tensor: Any | None = None
        self.calls: int = 0

    def __call__(self, _module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if not hasattr(hidden, "shape") or getattr(hidden, "ndim", 0) < 3:
            return
        self.tensor = hidden[:, -1:, :].detach().clone()
        self.calls += 1

    def reset(self) -> None:
        self.tensor = None
        self.calls = 0


def install_boundary_residual_capture_hook(
    last_layer: Any,
) -> tuple[_BoundaryResidualHook, Any]:
    """Register the boundary-residual capture hook on ``last_layer``.

    Returns ``(hook, handle)``.  The caller owns the handle and must call
    ``handle.remove()`` when done.  ``hook.tensor`` will be ``None`` until the
    first forward pass completes.
    """
    hook = _BoundaryResidualHook()
    handle = last_layer.register_forward_hook(hook)
    return hook, handle


class _Layer0ResidualSeedHook:
    """Layer-0 forward_pre_hook that overwrites position 0 with the boundary.

    Single-shot per install: latches after the first multi-position forward
    (``hidden_states.shape[1] > 1``).  Decode-step forwards
    (``shape[1] == 1``) are no-ops and do NOT latch, matching the behaviour
    of ``TorchInferenceRuntime.generate_with_residual_prefill_seeded``.
    """

    def __init__(self, boundary_residual: Any) -> None:
        self._boundary = boundary_residual
        self.injected: bool = False

    def __call__(self, _module, args, kwargs):
        if self.injected or not args:
            return args, kwargs
        hidden_states = args[0]
        if not hasattr(hidden_states, "shape") or getattr(hidden_states, "ndim", 0) < 3:
            return args, kwargs
        if hidden_states.shape[1] <= 1:
            # Single-position decode step — nothing to seed.
            return args, kwargs
        new_hidden = hidden_states.clone()
        boundary = self._boundary.to(
            device=new_hidden.device, dtype=new_hidden.dtype, non_blocking=True
        )
        # Collapse any leading singleton dims: (1, 1, H) or (1, H) -> (H,)
        while getattr(boundary, "ndim", 0) > 1 and boundary.shape[0] == 1:
            boundary = boundary.squeeze(0)
        new_hidden[:, 0, :] = boundary
        self.injected = True
        return (new_hidden, *args[1:]), kwargs


def install_layer0_residual_seed_hook(
    first_layer: Any, boundary_residual: Any
) -> tuple[_Layer0ResidualSeedHook, Any]:
    """Register the residual-seed pre-hook on ``first_layer``.

    Returns ``(hook, handle)``.  The caller owns the handle.
    """
    hook = _Layer0ResidualSeedHook(boundary_residual)
    handle = first_layer.register_forward_pre_hook(hook, with_kwargs=True)
    return hook, handle


# ---------------------------------------------------------------------------
# Axis-5 archived-residual → K/V materialization surface.
#
# Frozen contract: ``ve-ins-0mo9pb4f5000004cc67``. ORCH posture:
# ``ve-ins-0moa09rhm00002c707e``. Canonical scope restatement:
# ``ve-ins-0moa0k13w000012e06b``.
#
# Path-A (replay) prohibition: archived token_ids MUST NEVER be re-embedded
# nor re-forwarded through ``layers[0..injection_layer-1]`` during the
# gather + materialize phases. The only allowed matmul over archived data is
# the single pass through ``layer[injection_layer+1].self_attn.k_proj`` and
# ``.v_proj`` applied to archived residual tensors (Markov-sufficient
# boundary hidden states). Fresh-query forwards outside the gather/materialize
# window are always allowed.
#
# The :class:`PathAReplayGuard` context manager installs
# ``forward_pre_hook`` counters on layers 0..injection_layer-1 and raises
# ``RuntimeError('axis-5 violated: Path A replay detected at layer N')`` on
# first fire. The guard's ``count`` attribute is snapshot onto the returned
# :class:`KVDirectMaterialization` as ``path_a_replay_count`` — always 0 in
# conformance.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GatheredResiduals:
    """Per-window residuals read from the LIVE checkpoint archive.

    Attributes
    ----------
    per_window_residuals:
        Mapping from ``window_id`` to a 1-D ``torch.Tensor`` of shape
        ``[hidden_size]`` loaded via
        :meth:`TorchKnowledgeStore.load_boundary` with ``device='cuda'``.
        The tensor is the Markov-sufficient boundary hidden state captured
        at ``source_layer`` during live indexing — ONE vector per window,
        not per token.
    per_window_token_ranges:
        Mapping from ``window_id`` to a ``(slot_start, slot_end)`` pair on
        the materialized K/V ``N`` axis. Because the archived residual is a
        single-position summary, ``slot_end == slot_start + 1`` for each
        selected window, and ``slot_start`` is the insertion order of the
        window in :func:`gather_selected_residuals`.
    source_layer:
        The decoder-layer index at which the residuals were captured.
        Equal to ``store.config.injection_layer``.
    archive_provenance:
        Mapping from ``window_id`` to the POSIX path of the LIVE
        checkpoint archive file that sourced the residual
        (``<checkpoint_dir>/boundaries/window_NNN.npy``). ORCH decision 4:
        provenance is rooted at the live checkpoint lineage, never at any
        torch_store-internal copy.
    """

    per_window_residuals: dict[int, Any]
    per_window_token_ranges: dict[int, tuple[int, int]]
    source_layer: int
    archive_provenance: dict[int, str]


@dataclass(frozen=True)
class KVDirectMaterialization:
    """K/V tensors projected from archived residuals at ``injection_layer+1``.

    Attributes
    ----------
    K, V:
        ``torch.Tensor`` of shape ``(1, n_kv_heads, N, head_dim)`` on the
        model's device and dtype. ``N`` is the number of HOT+WARM windows
        retained after budget truncation (COLD is excluded at
        :func:`gather_selected_residuals`).
    materialization_mode:
        Literal string
        ``"project_through_W_K_W_V_at_injection_layer"`` — the only
        permitted mode under the axis-5 contract.
    hot_budget_mib_observed:
        Actual ``(K.numel() + V.numel()) * dtype.itemsize / 1024**2``
        rounded to whole MiB. Guaranteed ``<= hot_budget_mib`` configured
        on the runtime.
    path_a_replay_count:
        Snapshot of :class:`PathAReplayGuard`'s fire counter taken AFTER
        the materialization projection. Always ``0`` in conformance — any
        non-zero value means the guard raised before returning.
    materialized_source_layer:
        Actual archived-residual source layer used for the projection.
        This is the truthful ``injection_layer`` provenance and lets the
        runtime reject callers that lie about ``source_layer`` later.
    materialized_insertion_family:
        Runtime family implied by the actual projection target at
        ``materialized_source_layer + 1``. Currently one of
        ``"full_attention"`` or ``"sliding"``.
    materialized_lineage_layer_indices:
        Decoder-layer lineage that can coherently consume the projected
        archived K/V for ``materialized_insertion_family``. For
        ``"sliding"`` this is the source sliding layer plus any
        ``kv_shared_layer_index`` followers; for ``"full_attention"``
        this mirrors the existing propagation lineage rooted at the
        materialized target layer.
    """

    K: Any
    V: Any
    materialization_mode: str
    hot_budget_mib_observed: int
    path_a_replay_count: int
    materialized_source_layer: int | None = None
    materialized_insertion_family: str | None = None
    materialized_lineage_layer_indices: tuple[int, ...] = ()
    per_window_token_ranges: dict[int, tuple[int, int]] | None = None


class PathAReplayGuard:
    """Context manager that detects Path-A replay.

    Installs ``forward_pre_hook`` counters on ``layers[0..injection_layer-1]``
    on ``__enter__`` and removes them on ``__exit__``. On first fire, the
    hook raises ``RuntimeError('axis-5 violated: Path A replay detected at
    layer N')`` where ``N`` is the pre-injection layer index.

    Semantic: any forward over archived-token input under the guarded
    window is a Path-A replay. The fresh-query forward must happen OUTSIDE
    this context so its layer-0 traversal is not flagged.
    """

    def __init__(self, model: Any, injection_layer: int, *, resolve_layers):
        self._model = model
        self._injection_layer = int(injection_layer)
        self._resolve_layers = resolve_layers
        self._handles: list[Any] = []
        self.count: int = 0

    def __enter__(self) -> PathAReplayGuard:
        layers = self._resolve_layers(self._model)
        for idx in range(self._injection_layer):
            layer = layers[idx]

            def _hook(_module, _args, _kwargs, layer_idx=idx):
                self.count += 1
                raise RuntimeError(
                    f"axis-5 violated: Path A replay detected at layer {layer_idx}"
                )

            handle = layer.register_forward_pre_hook(_hook, with_kwargs=True)
            self._handles.append(handle)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False


def _resolve_default_layers(model: Any) -> list[Any]:
    """Module-local mirror of ``TorchInferenceRuntime._resolve_layers``.

    Kept here so :class:`PathAReplayGuard` and :func:`materialize_kv_direct`
    are usable without a runtime instance (the frozen axis-5 contract says
    the guard hooks a model directly).
    """
    candidates = [
        ("model", "language_model", "layers"),
        ("model", "language_model", "model", "layers"),
        ("language_model", "model", "layers"),
        ("language_model", "layers"),
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        (None, "layers"),
    ]
    for path in candidates:
        target = model
        *outer, inner_name = path
        ok = True
        for attr in outer:
            if attr is None:
                continue
            target = getattr(target, attr, None)
            if target is None:
                ok = False
                break
        if not ok:
            continue
        inner = getattr(target, inner_name, None)
        if inner is None:
            continue
        if not hasattr(inner, "__iter__") or not hasattr(inner, "__len__"):
            continue
        try:
            length = len(inner)
        except TypeError:
            continue
        if length == 0:
            continue
        return list(inner)
    raise RuntimeError("Cannot resolve transformer layer stack on model.")


def gather_selected_residuals(
    tier_assignments: Any,
    *,
    checkpoint_handle: Any,
    store: Any,
    source_layer: int,
    include_tiers: Any = None,
    device: str = "cuda",
    residual_mode: str = "boundary",
) -> GatheredResiduals:
    """Gather per-window residuals for selected tiers from the LIVE archive.

    Reads each included window's boundary tensor via
    ``store.load_boundary(window_id, device=device)``. Does NOT invoke the
    model on archived tokens (Path-A prohibition). COLD-tier windows are
    unconditionally skipped; only HOT and WARM are gathered by default.

    Parameters
    ----------
    tier_assignments:
        Iterable of ``TierAssignment`` produced by axis-3 /
        :mod:`chuk_lazarus.session_retrieval.tier_policy`. Each entry
        exposes ``.candidate.window_id`` and ``.tier``.
    checkpoint_handle:
        :class:`CheckpointHandle` providing ``checkpoint_dir`` — used to
        root the provenance paths.
    store:
        :class:`TorchKnowledgeStore` loaded for this handle. Carries
        ``store.config.injection_layer`` for the source-layer consistency
        check and ``store.load_boundary`` for the on-disk read.
    source_layer:
        Decoder-layer index at which the residuals were captured. Must
        equal ``store.config.injection_layer`` — mismatch raises
        ``ValueError``.
    include_tiers:
        Iterable of ``TierLabel`` values to include. Defaults to
        ``(TierLabel.HOT, TierLabel.WARM)``.
    device:
        Device to place loaded tensors on. Must be ``"cuda"`` under the
        GPU-only test policy for this repo.

    Returns
    -------
    GatheredResiduals
        Frozen dataclass describing the gathered tensors and their
        archive provenance.
    """
    from chuk_lazarus.session_retrieval.tier_policy import TierLabel

    residual_mode = str(residual_mode).strip().lower()
    if residual_mode not in {"boundary", "stream"}:
        raise ValueError(
            "gather_selected_residuals: residual_mode must be 'boundary' or "
            f"'stream', got {residual_mode!r}"
        )

    if include_tiers is None:
        include_tiers = (TierLabel.HOT, TierLabel.WARM)
    include_set = set(include_tiers)

    ac_layer = int(store.config.injection_layer)
    if int(source_layer) != ac_layer:
        raise ValueError(
            "gather_selected_residuals: source_layer="
            f"{int(source_layer)} does not match "
            f"store.config.injection_layer={ac_layer}"
        )

    per_window_residuals: dict[int, Any] = {}
    per_window_token_ranges: dict[int, tuple[int, int]] = {}
    archive_provenance: dict[int, str] = {}
    slot = 0
    for assignment in tier_assignments:
        if assignment.tier not in include_set:
            continue
        wid = int(assignment.candidate.window_id)
        if residual_mode == "stream":
            try:
                residual = store.load_residual_stream(wid, device=device)
                provenance_dir = "residual_streams"
            except FileNotFoundError:
                warnings.warn(
                    "gather_selected_residuals: residual stream missing for "
                    f"window {wid}; falling back to boundary residual for "
                    f"checkpoint {checkpoint_handle.checkpoint_dir}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                residual = store.load_boundary(wid, device=device)
                provenance_dir = "boundaries"
            while getattr(residual, "dim", lambda: 0)() > 2 and residual.shape[0] == 1:
                residual = residual.squeeze(0)
            if residual.dim() == 1:
                residual = residual.unsqueeze(0)
            if residual.dim() != 2:
                raise RuntimeError(
                    "gather_selected_residuals: residual stream for window "
                    f"{wid} must be rank-2, got shape {tuple(residual.shape)}"
                )
            slot_count = int(residual.shape[0])
        else:
            residual = store.load_boundary(wid, device=device)
            provenance_dir = "boundaries"
            slot_count = 1
        per_window_residuals[wid] = residual
        per_window_token_ranges[wid] = (slot, slot + slot_count)
        archive_provenance[wid] = str(
            (
                checkpoint_handle.checkpoint_dir
                / provenance_dir
                / f"window_{wid:03d}.npy"
            ).as_posix()
        )
        slot += slot_count

    return GatheredResiduals(
        per_window_residuals=per_window_residuals,
        per_window_token_ranges=per_window_token_ranges,
        source_layer=int(source_layer),
        archive_provenance=archive_provenance,
    )


def _resolve_attention_projections(layer: Any) -> tuple[Any, Any, int, int]:
    """Extract ``(k_proj, v_proj, n_kv_heads, head_dim)`` from a decoder layer.

    Requires split K / V projections (``self_attn.k_proj`` and
    ``self_attn.v_proj``) — fused QKV is not supported by the axis-5
    materialization surface.
    """
    self_attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
    if self_attn is None:
        raise RuntimeError(
            "Decoder layer has no .self_attn or .attention attribute."
        )
    k_proj = getattr(self_attn, "k_proj", None)
    v_proj = getattr(self_attn, "v_proj", None)
    if k_proj is None or v_proj is None:
        raise RuntimeError(
            "Decoder layer's attention does not expose split k_proj/v_proj; "
            "materialize_kv_direct requires split projections."
        )
    attn_config = getattr(self_attn, "config", None)
    if attn_config is not None:
        attn_config = _text_config(attn_config)

    n_kv_heads = getattr(self_attn, "num_key_value_heads", None)
    if n_kv_heads is None and attn_config is not None:
        n_kv_heads = getattr(attn_config, "num_key_value_heads", None)
    if n_kv_heads is None:
        n_kv_heads = getattr(self_attn, "num_heads", None)
    if n_kv_heads is None and attn_config is not None:
        n_kv_heads = getattr(attn_config, "num_attention_heads", None)

    head_dim = getattr(self_attn, "head_dim", None)
    if head_dim is None and attn_config is not None:
        head_dim = getattr(attn_config, "head_dim", None)
    if head_dim is None and attn_config is not None:
        hidden_size = getattr(attn_config, "hidden_size", None)
        num_attention_heads = getattr(attn_config, "num_attention_heads", None)
        if hidden_size is not None and num_attention_heads is not None:
            head_dim = int(hidden_size) // int(num_attention_heads)
    if n_kv_heads is None or head_dim is None:
        raise RuntimeError(
            "Attention module missing num_key_value_heads/head_dim; "
            "materialize_kv_direct cannot reshape K/V."
        )
    return k_proj, v_proj, int(n_kv_heads), int(head_dim)


def _resolve_materialization_projection_layer(
    layers: Any,
    target_idx: int,
) -> tuple[Any, int]:
    """Return the decoder layer whose K/V projections materialize target KV.

    Gemma-4 shared-KV consumer layers omit ``k_proj``/``v_proj`` and point at
    the producer layer through ``kv_shared_layer_index``. Keep this in sync
    with the runtime path in ``torch_runtime.generate_with_kv_direct_materialization``.
    """
    target_layer = layers[int(target_idx)]
    target_self_attn = getattr(target_layer, "self_attn", None) or getattr(
        target_layer, "attention", None
    )
    if target_self_attn is None:
        raise RuntimeError(
            "materialize_kv_direct: target layer has no .self_attn/.attention "
            f"module (target_layer={target_idx})."
        )

    target_is_kv_consumer = bool(
        getattr(target_self_attn, "is_kv_shared_layer", False)
    ) or (
        getattr(target_self_attn, "k_proj", None) is None
        or getattr(target_self_attn, "v_proj", None) is None
    )
    if not target_is_kv_consumer:
        return target_layer, int(target_idx)

    producer_idx_raw = getattr(target_self_attn, "kv_shared_layer_index", None)
    if producer_idx_raw is None:
        raise RuntimeError(
            f"materialize_kv_direct: target layer {target_idx} is a KV-shared "
            "consumer but exposes no kv_shared_layer_index."
        )
    try:
        producer_idx = int(producer_idx_raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"materialize_kv_direct: target layer {target_idx} has invalid "
            f"kv_shared_layer_index={producer_idx_raw!r}."
        ) from exc
    if producer_idx < 0 or producer_idx >= len(layers):
        raise RuntimeError(
            f"materialize_kv_direct: target layer {target_idx} points to "
            f"kv_shared_layer_index={producer_idx}, outside model layer range "
            f"0..{len(layers) - 1}."
        )

    producer_layer = layers[producer_idx]
    producer_self_attn = getattr(producer_layer, "self_attn", None) or getattr(
        producer_layer, "attention", None
    )
    if producer_self_attn is None:
        raise RuntimeError(
            "materialize_kv_direct: KV-shared producer layer has no "
            f".self_attn/.attention module (producer_layer={producer_idx})."
        )
    return producer_layer, producer_idx


def resolve_kv_materialization_provenance(
    model: Any,
    source_layer: int,
    *,
    resolve_layers=_resolve_default_layers,
) -> tuple[str, tuple[int, ...]]:
    """Resolve the truthful family/lineage for a materialized K/V source."""

    layers = resolve_layers(model)
    target_idx = int(source_layer) + 1
    if target_idx >= len(layers):
        raise RuntimeError(
            "resolve_kv_materialization_provenance: source_layer+1 out of range for "
            f"model with {len(layers)} layers (source_layer={source_layer})."
        )

    def _resolve_self_attn(layer) -> Any | None:
        return getattr(layer, "self_attn", None) or getattr(layer, "attention", None)

    target_self_attn = _resolve_self_attn(layers[target_idx])
    if target_self_attn is None:
        raise RuntimeError(
            "resolve_kv_materialization_provenance: target layer has no "
            ".self_attn/.attention module."
        )

    raw_config = getattr(model, "config", None)
    if raw_config is not None and hasattr(raw_config, "get_text_config"):
        text_config = raw_config.get_text_config()
    else:
        text_config = _text_config(raw_config)
    layer_types = list(getattr(text_config, "layer_types", [])) if text_config is not None else []

    def _layer_type_for(layer_idx: int, layer_self_attn: Any | None = None) -> str | None:
        if layer_self_attn is None:
            layer_self_attn = _resolve_self_attn(layers[layer_idx])
        if layer_self_attn is None:
            return None
        layer_type = getattr(layer_self_attn, "layer_type", None)
        if layer_type is not None:
            return str(layer_type)
        if layer_idx < len(layer_types):
            return str(layer_types[layer_idx])
        return None

    def _is_sliding_layer(layer_idx: int, layer_self_attn: Any | None = None) -> bool:
        if layer_self_attn is None:
            layer_self_attn = _resolve_self_attn(layers[layer_idx])
        if layer_self_attn is None:
            return False
        if bool(getattr(layer_self_attn, "is_sliding", False)):
            return True
        return _layer_type_for(layer_idx, layer_self_attn) == "sliding_attention"

    target_layer_type = _layer_type_for(target_idx, target_self_attn)
    if _is_sliding_layer(target_idx, target_self_attn):
        lineage = [target_idx]
        for layer_idx in range(target_idx + 1, len(layers)):
            layer_self_attn = _resolve_self_attn(layers[layer_idx])
            if layer_self_attn is None or not _is_sliding_layer(layer_idx, layer_self_attn):
                continue
            if getattr(layer_self_attn, "kv_shared_layer_index", None) == target_idx:
                lineage.append(layer_idx)
        return "sliding", tuple(dict.fromkeys(int(layer_idx) for layer_idx in lineage))

    lineage = [target_idx]
    for layer_idx in range(target_idx + 1, len(layers)):
        layer_self_attn = _resolve_self_attn(layers[layer_idx])
        if layer_self_attn is None:
            continue
        same_layer_type = _layer_type_for(layer_idx, layer_self_attn) == target_layer_type
        shares_from_target = getattr(layer_self_attn, "kv_shared_layer_index", None) == target_idx
        if same_layer_type or shares_from_target:
            lineage.append(layer_idx)
    return "full_attention", tuple(dict.fromkeys(int(layer_idx) for layer_idx in lineage))


def materialize_kv_direct(
    residuals: GatheredResiduals,
    runtime: Any,
    *,
    hot_budget_mib: int,
    tier_assignments: Any = None,
) -> KVDirectMaterialization:
    """Project gathered residuals through W_K, W_V at ``injection_layer+1``.

    The projection runs inside a :class:`PathAReplayGuard` that hooks
    ``layers[0..injection_layer-1]``. Any hook fire is a Path-A replay
    and raises ``RuntimeError('axis-5 violated: ...')`` immediately.

    Budget pressure evicts WARM before HOT. With
    ``per_slot_bytes = 2 * n_kv_heads * head_dim * dtype.itemsize``::

        max_slots = max(1, hot_budget_mib * 1024 * 1024 // per_slot_bytes)

    When ``len(residuals) > max_slots`` and ``tier_assignments`` is
    provided, WARM windows are truncated from the tail first; HOT windows
    are preserved in full. Without ``tier_assignments`` the truncation is
    a plain tail cut of the ordered residuals.

    Parameters
    ----------
    residuals:
        :class:`GatheredResiduals` produced by
        :func:`gather_selected_residuals`.
    runtime:
        :class:`TorchInferenceRuntime` whose underlying model supplies the
        target attention module and dtype.
    hot_budget_mib:
        Hard ceiling on ``(K.numel() + V.numel()) * dtype.itemsize``
        expressed in MiB.
    tier_assignments:
        Optional iterable of ``TierAssignment`` used to preserve HOT
        windows under budget pressure.

    Returns
    -------
    KVDirectMaterialization
    """
    import torch

    from chuk_lazarus.session_retrieval.tier_policy import TierLabel

    if not residuals.per_window_residuals:
        raise ValueError("materialize_kv_direct: no residuals to materialize.")

    model = runtime._model
    layers = _resolve_default_layers(model)
    injection_layer = int(residuals.source_layer)
    if injection_layer + 1 >= len(layers):
        raise RuntimeError(
            "materialize_kv_direct: injection_layer+1 out of range for "
            f"model with {len(layers)} layers (injection_layer={injection_layer})."
        )
    target_idx = injection_layer + 1
    projection_layer, _projection_layer_idx = _resolve_materialization_projection_layer(
        layers, target_idx
    )
    k_proj, v_proj, n_kv_heads, head_dim = _resolve_attention_projections(
        projection_layer
    )
    materialized_insertion_family, materialized_lineage_layer_indices = (
        resolve_kv_materialization_provenance(
            model,
            injection_layer,
            resolve_layers=_resolve_default_layers,
        )
    )

    first_param = next(model.parameters())
    model_dtype = first_param.dtype
    model_device = first_param.device
    itemsize = int(torch.tensor([], dtype=model_dtype).element_size())

    per_slot_bytes = 2 * int(n_kv_heads) * int(head_dim) * int(itemsize)  # K + V
    budget_bytes = int(hot_budget_mib) * 1024 * 1024
    max_slots = max(1, budget_bytes // per_slot_bytes)

    ordered_wids = list(residuals.per_window_residuals.keys())

    def _slot_count(window_id: int) -> int:
        start, end = residuals.per_window_token_ranges.get(int(window_id), (0, 0))
        return max(0, int(end) - int(start))

    if tier_assignments is not None:
        tier_map: dict[int, Any] = {
            int(a.candidate.window_id): a.tier for a in tier_assignments
        }
        hot_wids = [w for w in ordered_wids if tier_map.get(w) == TierLabel.HOT]
        warm_wids = [w for w in ordered_wids if tier_map.get(w) == TierLabel.WARM]
        cold_wids = [w for w in ordered_wids if tier_map.get(w) == TierLabel.COLD]
        # Preserve ranked tier order and enforce the budget in materialized
        # token slots rather than windows. COLD is naturally last and drops
        # first when the active prefix is too large.
        kept: list[int] = []
        used_slots = 0
        for wid in hot_wids:
            count = _slot_count(wid)
            if count <= 0:
                continue
            kept.append(wid)
            used_slots += count
        for wid in warm_wids + cold_wids:
            count = _slot_count(wid)
            if count <= 0:
                continue
            if used_slots + count > max_slots:
                continue
            kept.append(wid)
            used_slots += count
        ordered_wids = kept
    else:
        kept = []
        used_slots = 0
        for wid in ordered_wids:
            count = _slot_count(wid)
            if count <= 0:
                continue
            if used_slots + count > max_slots:
                continue
            kept.append(wid)
            used_slots += count
        ordered_wids = kept

    if not ordered_wids:
        raise RuntimeError(
            "materialize_kv_direct: budget eviction left zero windows; "
            f"hot_budget_mib={hot_budget_mib} is too small for even one slot."
        )

    chunks = []
    materialized_ranges: dict[int, tuple[int, int]] = {}
    slot_offset = 0
    for wid in ordered_wids:
        tensor = residuals.per_window_residuals[wid].to(
            device=model_device, dtype=model_dtype, non_blocking=True
        )
        while tensor.dim() > 2 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != 2:
            raise RuntimeError(
                "materialize_kv_direct: residual for window "
                f"{wid} must be rank-1 or rank-2, got shape {tuple(tensor.shape)}"
            )
        slot_count = int(tensor.shape[0])
        chunks.append(tensor)
        materialized_ranges[int(wid)] = (slot_offset, slot_offset + slot_count)
        slot_offset += slot_count

    stacked = torch.cat(chunks, dim=0).unsqueeze(0)  # (1, N, hidden)

    guard = PathAReplayGuard(model, injection_layer, resolve_layers=_resolve_default_layers)
    with guard:
        with torch.no_grad():
            k_flat = k_proj(stacked)  # (1, N, n_kv_heads * head_dim)
            v_flat = v_proj(stacked)

    batch = 1
    n_slots = int(stacked.shape[1])
    K = (
        k_flat.view(batch, n_slots, int(n_kv_heads), int(head_dim))
        .transpose(1, 2)
        .contiguous()
    )
    V = (
        v_flat.view(batch, n_slots, int(n_kv_heads), int(head_dim))
        .transpose(1, 2)
        .contiguous()
    )

    total_bytes = (K.numel() + V.numel()) * int(itemsize)
    observed_mib = int(total_bytes // (1024 * 1024))
    if observed_mib > int(hot_budget_mib):
        raise RuntimeError(
            f"materialize_kv_direct: observed {observed_mib} MiB exceeds "
            f"hot_budget_mib={hot_budget_mib}"
        )

    return KVDirectMaterialization(
        K=K,
        V=V,
        materialization_mode="project_through_W_K_W_V_at_injection_layer",
        hot_budget_mib_observed=observed_mib,
        path_a_replay_count=int(guard.count),
        materialized_source_layer=injection_layer,
        materialized_insertion_family=materialized_insertion_family,
        materialized_lineage_layer_indices=materialized_lineage_layer_indices,
        per_window_token_ranges=materialized_ranges,
    )


__all__ = [
    "GatheredResiduals",
    "KVDirectMaterialization",
    "PathAReplayGuard",
    "ResidualBoundedState",
    "estimate_kv_bytes_per_token",
    "gather_selected_residuals",
    "install_boundary_residual_capture_hook",
    "install_layer0_residual_seed_hook",
    "materialize_kv_direct",
    "resolve_kv_materialization_provenance",
]
