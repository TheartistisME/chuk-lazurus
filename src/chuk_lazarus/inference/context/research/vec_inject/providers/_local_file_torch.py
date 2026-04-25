"""LocalVecInjectProvider (torch/CUDA port) — file-backed VecInjectProvider.

Torch-port sibling of ``_local_file.py`` (MLX reference implementation).  The
algorithmic contract — Stage 1 (kspace) → Stage 2 (h4) → Stage 3 (fallback),
adaptive thresholding, distinctive-mask trick — is preserved byte-for-byte;
only the device/runtime bindings differ.

Load once, retrieve fast.

At load time the entire normalised K matrix and all per-fact fields are
copied onto the CUDA device and pinned with torch.cuda.synchronize().  Every
retrieve() call is a single fused CUDA dispatch:

  query forward pass → Q vector (already on device from forward pass)
      ↓
  cosine scores  = flat_k @ q_norm               [CUDA matmul]
      ↓
  top-k indices  = torch.argsort(-scores)[:top_k][CUDA sort]
      ↓
  batch gather   = flat_*[top_idx]               [CUDA gather × 4]
      ↓
  torch.cuda.synchronize()  — one sync, top-k small arrays land in Python

No numpy in the hot path.  numpy is used only at load time for NPZ I/O
and L2-normalisation (before data moves to the CUDA device).

Index files produced by the prefill pipeline:
  --phases vec_inject  → vec_inject.npz        (K vectors + coefficients)
  --phases kvectors    → kv_route_index.npz    (K vectors only, legacy)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel

from .._types import SourceType, VecInjectMatch, VecInjectMeta, VecInjectResult
from ._index_format import (
    KV_ROUTE_FILE,
    VEC_INJECT_FILE,
    VecInjectMetaKey,
    VecInjectWindowKey,
)
from ..prefill_torch import _project_qkv, _resolve_transformer_layers

# ── Internal window summary (diagnostics only — not on the hot path) ──


class _WindowSummary(BaseModel):
    window_id: int
    n_facts: int


# ── Provider ──────────────────────────────────────────────────────────


class LocalVecInjectProvider:
    """File-backed vector injection provider.

    All scoring data lives on the CUDA device after load().

    Attributes
    ----------
    meta                : Index metadata (layers, heads).
    has_injection       : True when injection coefficients are available.
    n_facts             : Total indexed fact positions across all windows.
    confidence_threshold: Minimum Q·K cosine score to trust routing.
                          retrieve() sets routing_confident=False below this
                          value so callers can fall back to window replay.
                          Wrong injection is catastrophically worse than no
                          injection — the 0.05% injected signal dominates the
                          99.95% structural context with no semantic protection.

    Usage
    -----
        # Torch-native path (no MLX kv_gen required):
        provider = await LocalVecInjectProvider.load(checkpoint_dir, raw_model=hf_model)

        # MLX-backed path (legacy — still supported):
        provider = await LocalVecInjectProvider.load(checkpoint_dir, kv_gen)

        result   = await provider.retrieve(query_ids, query_text, top_k=5)
        if result.routing_confident and result.matches:
            h = vec_inject_all(h, result.matches, embed_matrix)
        else:
            # fall back to window replay
    """

    def __init__(
        self,
        meta: VecInjectMeta,
        *,
        # Device-resident arrays (pinned at construction via torch.cuda.synchronize)
        flat_k: torch.Tensor,  # (n_facts, head_dim) float32 L2-normalised
        flat_token_ids: torch.Tensor,  # (n_facts,) int32
        flat_coefs: torch.Tensor,  # (n_facts,) float32
        flat_wid: torch.Tensor,  # (n_facts,) int32
        flat_positions: torch.Tensor,  # (n_facts,) int32
        flat_distinctive: torch.Tensor,  # (n_facts,) int32  1=distinctive, 0=common prefix
        window_summaries: list[_WindowSummary],
        kv_gen: Any | None = None,
        has_injection: bool,
        confidence_threshold: float = 0.15,
        flat_h4: torch.Tensor | None = None,  # (n_facts, hidden_size) float32 L2-normalised
        flat_source_ids: torch.Tensor | None = None,  # (n_facts,) int32
        flat_source_types: torch.Tensor | None = None,  # (n_facts,) uint8 — 0=document, 1=generated
        residual: torch.Tensor | None = None,  # (1, hidden_dim) float32 — running state
        raw_model: Any | None = None,
        # axis-BC PROP K.5 — raw (un-normalised) K and V for KV-direct injection.
        # Optional: legacy callers can construct without these; only kv_for_match() needs them.
        flat_k_raw: torch.Tensor | None = None,  # (n_facts, head_dim) bf16 cuda
        flat_v: torch.Tensor | None = None,  # (n_facts, head_dim) bf16 cuda
        flat_index_map: dict[tuple[int, int], int] | None = None,
    ) -> None:
        self.meta = meta
        self.has_injection = has_injection
        self.confidence_threshold = confidence_threshold
        self._kv_gen = kv_gen
        self._raw_model = raw_model
        self._summaries = window_summaries

        self._flat_k = flat_k
        self._flat_token_ids = flat_token_ids
        self._flat_coefs = flat_coefs
        self._flat_wid = flat_wid
        self._flat_positions = flat_positions
        self._flat_distinctive = flat_distinctive
        self._flat_h4 = flat_h4  # None → Stage 2 not available
        self._flat_source_ids = flat_source_ids
        self._flat_source_types = flat_source_types
        self.residual = residual  # running Markov state
        # axis-BC PROP K.5 KV-direct support (lazily populated by _from_npz)
        self._flat_k_raw = flat_k_raw
        self._flat_v = flat_v
        self._flat_index_map: dict[tuple[int, int], int] = (
            dict(flat_index_map) if flat_index_map is not None else {}
        )

    def close(self) -> None:
        """Release device-resident arrays to free CUDA memory."""
        self._flat_k = torch.zeros((0, 1), dtype=torch.float32, device="cuda")
        self._flat_token_ids = torch.zeros((0,), dtype=torch.int32, device="cuda")
        self._flat_coefs = torch.zeros((0,), dtype=torch.float32, device="cuda")
        self._flat_wid = torch.zeros((0,), dtype=torch.int32, device="cuda")
        self._flat_positions = torch.zeros((0,), dtype=torch.int32, device="cuda")
        self._flat_distinctive = torch.zeros((0,), dtype=torch.int32, device="cuda")
        self._flat_h4 = None
        self._flat_source_ids = None
        self._flat_source_types = None
        self.residual = None
        self._kv_gen = None
        self._raw_model = None
        self._flat_k_raw = None
        self._flat_v = None
        self._flat_index_map = {}

    def __del__(self) -> None:
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    @property
    def n_facts(self) -> int:
        return self._flat_k.shape[0]

    @property
    def head_dim(self) -> int:
        return self._flat_k.shape[1] if self.n_facts > 0 else 0

    # ── axis-BC PROP K.5: KV-direct adapter helper ────────────────────

    def kv_for_match(
        self, match: VecInjectMatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return raw (k_vec, v_vec) for a retrieved match — KV-direct adapter helper.

        /euclid CLAIM:
          For every match m loaded from the NPZ, _flat_k_raw[idx] is the
          PRE-L2-norm K extraction (NPZ key 'w{wid}/k_vecs') and _flat_v[idx]
          is the V extraction (NPZ key 'w{wid}/v_vecs') where idx ==
          _flat_index_map[(m.window_id, m.position)].
        /euclid PASS test:
          tests/inference/context/research/vec_inject/test_axis_BC_kv_direct_adapter.py
          ::test_kv_for_match_helper — provider.kv_for_match(match) returns
          two cuda bf16 tensors of shape (head_dim,).
        /euclid FAIL test:
          KeyError raised when (window_id, position) is not in the index map.

        Returns
        -------
        k_vec : torch.Tensor — shape (head_dim,) cuda bf16, raw (pre-L2-norm)
        v_vec : torch.Tensor — shape (head_dim,) cuda bf16

        Raises
        ------
        RuntimeError : if the provider was constructed without KV-direct payloads
                       (e.g. legacy kv_route_index.npz without v_vecs).
        KeyError     : if the (window_id, position) pair is not registered.
        """
        if self._flat_k_raw is None or self._flat_v is None:
            raise RuntimeError(
                "kv_for_match: provider has no raw K/V payloads. "
                "Rebuild via _from_npz against an NPZ that contains "
                "'w{N}/v_vecs' alongside 'w{N}/k_vecs'."
            )
        key = (int(match.window_id), int(match.position))
        if key not in self._flat_index_map:
            raise KeyError(
                f"kv_for_match: no entry for (window_id={key[0]}, "
                f"position={key[1]}) in the NPZ index map"
            )
        idx = self._flat_index_map[key]
        return self._flat_k_raw[idx], self._flat_v[idx]

    # ── VecInjectProvider interface ───────────────────────────────────

    @property
    def injection_layer(self) -> int:
        return self.meta.injection_layer

    def retrieve_sync(
        self,
        query_ids: list[int],
        query_text: str,
        top_k: int = 5,
    ) -> VecInjectResult:
        """Route query to top-k facts via Q·K cosine scoring on CUDA (sync).

        The full critical path (forward pass → matmul → sort → gather)
        runs as one fused CUDA graph.  Python only touches the final top-k
        scalars after a single torch.cuda.synchronize() sync.

        This is the preferred entry point for in-memory providers — no
        I/O is involved, so async adds no value.

        Parameters
        ----------
        query_ids  : Encoded query tokens.
        query_text : Raw text (available for logging; not used for scoring).
        top_k      : Maximum facts to return.
        """
        if self.n_facts == 0:
            return VecInjectResult(
                injection_layer=self.meta.injection_layer,
                routing_confident=False,
            )

        t0 = time.monotonic()

        # Q vector — torch.Tensor, stays on device, joins the graph
        q_vec = self._query_vec(query_ids)
        q_nrm = q_vec / torch.maximum(
            torch.sqrt(torch.sum(q_vec * q_vec)),
            torch.tensor(1e-9, device="cuda"),
        )

        # Cosine scores — CUDA matmul
        scores = self._flat_k @ q_nrm  # (n_facts,)

        # Top-k overall — CUDA sort (for diagnostics / fallback matches)
        k = min(top_k, self.n_facts)
        top_idx = torch.argsort(-scores)[:k]  # (k,)

        # Best distinctive match — mask non-distinctive positions to -inf so
        # they are never selected.  Non-distinctive tokens (< 4 chars) cause
        # wrong injections because the model's prior for common prefixes
        # overwhelms the small injected coefficient.
        dist_mask = self._flat_distinctive.to(torch.float32)  # 1.0 or 0.0
        scores_dist = scores * dist_mask + (dist_mask - 1.0) * 1e9  # non-dist → -inf
        best_dist_idx_arr = torch.argmax(scores_dist)  # scalar

        # Batch gather
        top_scores = scores[top_idx]
        top_token_ids = self._flat_token_ids[top_idx]
        top_coefs = self._flat_coefs[top_idx]
        top_wids = self._flat_wid[top_idx]
        top_positions = self._flat_positions[top_idx]
        top_distinctive = self._flat_distinctive[top_idx]

        best_dist_score_arr = scores[best_dist_idx_arr]
        best_dist_tok_arr = self._flat_token_ids[best_dist_idx_arr]
        best_dist_coef_arr = self._flat_coefs[best_dist_idx_arr]
        best_dist_wid_arr = self._flat_wid[best_dist_idx_arr]
        best_dist_pos_arr = self._flat_positions[best_dist_idx_arr]

        # Adaptive threshold — mean × 2.0 scales with score distribution.
        # Fixed threshold (e.g. 0.15) becomes useless beyond N≈50: at N=3,625
        # max Q·K scores drop below 1%, killing all injections. The adaptive
        # threshold is relative to the score distribution, not absolute.
        # self.confidence_threshold serves as a minimum floor.
        mean_score_arr = torch.mean(scores)

        # Single sync — everything materialises in one CUDA dispatch
        torch.cuda.synchronize()
        elapsed_ms = (time.monotonic() - t0) * 1000

        mean_score = float(mean_score_arr.item())
        adaptive_threshold = max(self.confidence_threshold, mean_score * 2.0)

        matches = [
            VecInjectMatch(
                token_id=int(top_token_ids[i].item()),
                coefficient=float(top_coefs[i].item()),
                score=float(top_scores[i].item()),
                window_id=int(top_wids[i].item()),
                position=int(top_positions[i].item()),
                distinctive=bool(top_distinctive[i].item()),
            )
            for i in range(k)
        ]

        best_dist_score = float(best_dist_score_arr.item())
        stage1_confident = best_dist_score >= adaptive_threshold

        # ── Stage 1 succeeded — best distinctive K-space match above threshold ──
        if stage1_confident:
            best_dist_match = VecInjectMatch(
                token_id=int(best_dist_tok_arr.item()),
                coefficient=float(best_dist_coef_arr.item()),
                score=best_dist_score,
                window_id=int(best_dist_wid_arr.item()),
                position=int(best_dist_pos_arr.item()),
                distinctive=True,
            )
            return VecInjectResult(
                matches=[best_dist_match] + matches,
                retrieval_ms=elapsed_ms,
                injection_layer=self.meta.injection_layer,
                routing_confident=True,
                top_score=best_dist_score,
                routing_stage="kspace",
            )

        # ── Stage 2: H4 output cosine (same-template entity discrimination) ──
        # Runs when Stage 1 fails the score threshold OR has no distinctive matches.
        # Requires h4_vecs in the index.
        if self._flat_h4 is not None and self.meta.retrieval_layer > 0:
            h4_result = self._route_h4(query_ids, k, elapsed_ms)
            if h4_result is not None:
                return h4_result

        # ── Stage 3: fallback ────────────────────────────────────────
        top_score = float(top_scores[0].item()) if len(top_scores) > 0 else 0.0
        return VecInjectResult(
            matches=matches,
            retrieval_ms=elapsed_ms,
            injection_layer=self.meta.injection_layer,
            routing_confident=False,
            top_score=top_score,
            routing_stage="fallback",
        )

    async def retrieve(
        self,
        query_ids: list[int],
        query_text: str,
        top_k: int = 5,
    ) -> VecInjectResult:
        """Async wrapper around retrieve_sync() for protocol compatibility.

        LocalVecInjectProvider does zero I/O in retrieval — all data is
        pinned on the CUDA device at load time. This method exists only
        to satisfy the async VecInjectProvider protocol. Prefer
        retrieve_sync() in non-async contexts.
        """
        return self.retrieve_sync(query_ids, query_text, top_k)

    # ── Factory ──────────────────────────────────────────────────────

    @classmethod
    async def load(
        cls,
        checkpoint_dir: str | Path,
        kv_gen: Any | None = None,
        *,
        raw_model: Any | None = None,
        prefer_vec_inject: bool = True,
        confidence_threshold: float = 0.15,
    ) -> LocalVecInjectProvider:
        """Load from a checkpoint directory and pin all data on CUDA.

        Prefers vec_inject.npz (full routing + injection) over the legacy
        kv_route_index.npz (routing only, no coefficients).
        Pass prefer_vec_inject=False to force routing-only mode.

        Parameters
        ----------
        kv_gen : Optional MLX-style kv_generator.
            When provided, query-vector synthesis routes through the MLX
            backbone (legacy semantics). When None, the provider expects
            ``raw_model`` and computes Q vectors via native torch forward
            hooks on the HuggingFace model.
        raw_model : Optional HuggingFace causal LM (CUDA-resident).
            Required when ``kv_gen`` is None — the torch-native query path
            registers a forward hook on the retrieval layer to capture the
            hidden state, then re-projects through the layer's QKV weights.
        confidence_threshold : Minimum Q·K cosine score to trust routing.
            retrieve() returns routing_confident=False below this value.
            Default 0.15 is conservative — increase for higher precision
            (more fallbacks), decrease for higher recall (more injections).
        """
        path = Path(checkpoint_dir)
        vec_path = path / VEC_INJECT_FILE
        route_path = path / KV_ROUTE_FILE

        if prefer_vec_inject and vec_path.exists():
            return cls._from_npz(
                vec_path,
                kv_gen,
                has_injection=True,
                confidence_threshold=confidence_threshold,
                raw_model=raw_model,
            )
        if route_path.exists():
            return cls._from_npz(
                route_path,
                kv_gen,
                has_injection=False,
                confidence_threshold=confidence_threshold,
                raw_model=raw_model,
            )

        raise FileNotFoundError(
            f"No vec-inject index found in {checkpoint_dir}. "
            "Run: lazarus context prefill --phases vec_inject --backend torch"
            "  (or --phases kvectors)"
        )

    @classmethod
    def _from_npz(
        cls,
        npz_path: Path,
        kv_gen: Any | None = None,
        *,
        has_injection: bool,
        confidence_threshold: float = 0.15,
        raw_model: Any | None = None,
    ) -> LocalVecInjectProvider:
        """NPZ → numpy (normalise) → CUDA device tensors.

        Uses numpy.load() — assumes the NPZ was written by the torch-side
        prefill pipeline (axis-B deliverable) with numpy-native dtypes.
        bfloat16 void-2 arrays (as produced by MLX's mx.savez) cannot be
        decoded here; a NotImplementedError is raised pointing at axis-B.
        """
        raw = np.load(str(npz_path))

        def _check_dtype(key: str, arr: np.ndarray) -> None:
            # MLX's mx.savez encodes bfloat16 as a void-2 structured dtype that
            # numpy cannot cast.  Detect and redirect callers to the torch-native
            # prefill pipeline (axis-B) which writes numpy-native dtypes.
            if arr.dtype.kind == "V" and arr.dtype.itemsize == 2:
                raise NotImplementedError(
                    f"NPZ key {key!r} uses bfloat16 void-2 encoding "
                    f"(likely written by MLX's mx.savez). The torch port "
                    f"requires numpy-native dtypes — rebuild the index via "
                    f"the torch-side prefill pipeline (axis-B: "
                    f"lazarus context prefill --phases vec_inject --backend torch)."
                )

        def _to_np_f32(key: str) -> np.ndarray:
            arr = raw[key]
            _check_dtype(key, arr)
            return arr.astype(np.float32)

        def _to_np_i32(key: str) -> np.ndarray:
            arr = raw[key]
            _check_dtype(key, arr)
            return arr.astype(np.int32)

        def _scalar(key: str) -> int:
            arr = raw[key]
            _check_dtype(key, arr)
            return int(arr.item())

        all_keys = set(raw.keys())

        meta = VecInjectMeta(
            retrieval_layer=_scalar(VecInjectMetaKey.LAYER),
            kv_head=_scalar(VecInjectMetaKey.KV_HEAD),
            query_head=(
                _scalar(VecInjectMetaKey.QUERY_HEAD)
                if VecInjectMetaKey.QUERY_HEAD in all_keys
                else _scalar(VecInjectMetaKey.KV_HEAD)
            ),
            injection_layer=(
                _scalar(VecInjectMetaKey.INJECT_LAYER)
                if VecInjectMetaKey.INJECT_LAYER in all_keys
                else _scalar(VecInjectMetaKey.LAYER) + 1
            ),
        )

        wids = sorted(
            {
                wid
                for key in all_keys
                if (wid := VecInjectWindowKey.window_id_from_key(key)) is not None
            }
        )

        # Accumulate per-window numpy arrays before device upload
        k_rows: list[np.ndarray] = []
        k_raw_rows: list[np.ndarray] = []  # axis-BC: PRE-L2-norm K for KV-direct
        v_rows: list[np.ndarray] = []  # axis-BC: V vectors for KV-direct
        tok_rows: list[np.ndarray] = []
        coef_rows: list[np.ndarray] = []
        wid_rows: list[np.ndarray] = []
        pos_rows: list[np.ndarray] = []
        dist_rows: list[np.ndarray] = []
        h4_rows: list[np.ndarray] = []  # Stage-2 routing vectors (may be empty)
        summaries: list[_WindowSummary] = []
        # axis-BC: (window_id, position) → flat row index, populated as we walk windows.
        # Schema: NPZ keys are 'w{wid}/k_vecs' and 'w{wid}/v_vecs' (per _index_format.py).
        flat_index_map: dict[tuple[int, int], int] = {}
        running_offset = 0
        all_have_v = True  # tracks whether every window provides v_vecs

        for wid in wids:
            if has_injection and VecInjectWindowKey.k_vecs(wid) in all_keys:
                k = _to_np_f32(VecInjectWindowKey.k_vecs(wid))
                tok = _to_np_i32(VecInjectWindowKey.token_ids(wid))
                coef = _to_np_f32(VecInjectWindowKey.coefs(wid))
                pos = _to_np_i32(VecInjectWindowKey.positions(wid))
                # distinctive flag: 1=safe for 1D injection, 0=needs fallback
                # Legacy indexes without this key: assume all distinctive
                dist_key = VecInjectWindowKey.distinctive(wid)
                dist = (
                    _to_np_i32(dist_key)
                    if dist_key in all_keys
                    else np.ones(len(k), dtype=np.int32)
                )
                # axis-BC: V is the content side of synthetic KV injection.
                v_key = VecInjectWindowKey.v_vecs(wid)
                if v_key in all_keys:
                    v = _to_np_f32(v_key)
                else:
                    all_have_v = False
                    v = np.zeros_like(k)
            else:
                # Legacy kv_route_index.npz: flat "wN" key, no coefficients
                k = _to_np_f32(VecInjectWindowKey.flat(wid))
                n = len(k)
                tok = np.zeros(n, dtype=np.int32)
                coef = np.zeros(n, dtype=np.float32)
                pos = np.arange(n, dtype=np.int32)
                dist = np.ones(n, dtype=np.int32)
                all_have_v = False
                v = np.zeros_like(k)

            n_facts = len(k)
            k_rows.append(k)
            k_raw_rows.append(k.copy())  # PRE-L2-norm — preserved for KV-direct
            v_rows.append(v)
            tok_rows.append(tok)
            coef_rows.append(coef)
            wid_rows.append(np.full(n_facts, wid, dtype=np.int32))
            pos_rows.append(pos)
            dist_rows.append(dist)

            # axis-BC: register (wid, position) → flat row idx for kv_for_match()
            for local_i in range(n_facts):
                flat_index_map[(int(wid), int(pos[local_i]))] = running_offset + local_i
            running_offset += n_facts

            # H4 Stage-2 routing vectors — optional, absent in legacy indexes
            h4_key = VecInjectWindowKey.h4_vecs(wid)
            if h4_key in all_keys:
                h4_rows.append(_to_np_f32(h4_key))

            summaries.append(_WindowSummary(window_id=wid, n_facts=n_facts))

        if not k_rows:
            empty = np.zeros((0, 1), dtype=np.float32)
            empty_i = np.zeros(0, dtype=np.int32)
            empty_f = np.zeros(0, dtype=np.float32)
            flat_k = torch.from_numpy(empty).to("cuda")
            flat_token_ids = torch.from_numpy(empty_i).to("cuda")
            flat_coefs = torch.from_numpy(empty_f).to("cuda")
            flat_wid = torch.from_numpy(empty_i).to("cuda")
            flat_positions = torch.from_numpy(empty_i).to("cuda")
            flat_distinctive = torch.from_numpy(empty_i).to("cuda")
            torch.cuda.synchronize()
            return cls(
                meta=meta,
                flat_k=flat_k,
                flat_token_ids=flat_token_ids,
                flat_coefs=flat_coefs,
                flat_wid=flat_wid,
                flat_positions=flat_positions,
                flat_distinctive=flat_distinctive,
                window_summaries=[],
                kv_gen=kv_gen,
                has_injection=has_injection,
                confidence_threshold=confidence_threshold,
                flat_h4=None,
                raw_model=raw_model,
            )

        # L2-normalise rows on CPU once so CUDA only sees unit vectors
        flat_k_np = np.concatenate(k_rows, axis=0)
        norms = np.linalg.norm(flat_k_np, axis=1, keepdims=True)
        np.clip(norms, 1e-9, None, out=norms)
        flat_k_normed = flat_k_np / norms

        # H4 Stage-2 routing — only present in indexes built after 2026-03-19
        flat_h4: torch.Tensor | None = None
        if h4_rows and len(h4_rows) == len(k_rows):
            # All windows have h4_vecs — safe to concatenate
            flat_h4_np = np.concatenate(h4_rows, axis=0)
            norms_h4 = np.linalg.norm(flat_h4_np, axis=1, keepdims=True)
            np.clip(norms_h4, 1e-9, None, out=norms_h4)
            flat_h4_normed = flat_h4_np / norms_h4
            flat_h4 = torch.from_numpy(flat_h4_normed.astype(np.float32)).to("cuda")

        # Upload to CUDA and pin — single torch.cuda.synchronize() before first retrieve()
        flat_k = torch.from_numpy(flat_k_normed.astype(np.float32)).to("cuda")
        flat_token_ids = torch.from_numpy(np.concatenate(tok_rows).astype(np.int32)).to("cuda")
        flat_coefs = torch.from_numpy(np.concatenate(coef_rows).astype(np.float32)).to("cuda")
        flat_wid = torch.from_numpy(np.concatenate(wid_rows).astype(np.int32)).to("cuda")
        flat_positions = torch.from_numpy(np.concatenate(pos_rows).astype(np.int32)).to("cuda")
        flat_distinctive = torch.from_numpy(np.concatenate(dist_rows).astype(np.int32)).to("cuda")

        # axis-BC PROP K.5: raw K (PRE-L2-norm) and V uploaded as bf16 for KV-direct
        # injection. We cast bf16 here because the downstream torch_runtime path expects
        # bf16 KV pages (recipe step 3-4). When all_have_v is False (legacy or missing
        # v_vecs), flat_v stays None so kv_for_match() raises a clear RuntimeError.
        flat_k_raw_np = np.concatenate(k_raw_rows, axis=0).astype(np.float32)
        flat_k_raw = torch.from_numpy(flat_k_raw_np).to("cuda").to(torch.bfloat16)
        if all_have_v:
            flat_v_np = np.concatenate(v_rows, axis=0).astype(np.float32)
            flat_v = torch.from_numpy(flat_v_np).to("cuda").to(torch.bfloat16)
        else:
            flat_v = None

        torch.cuda.synchronize()

        return cls(
            meta=meta,
            flat_k=flat_k,
            flat_token_ids=flat_token_ids,
            flat_coefs=flat_coefs,
            flat_wid=flat_wid,
            flat_positions=flat_positions,
            flat_distinctive=flat_distinctive,
            window_summaries=summaries,
            kv_gen=kv_gen,
            has_injection=has_injection,
            confidence_threshold=confidence_threshold,
            flat_h4=flat_h4,
            raw_model=raw_model,
            flat_k_raw=flat_k_raw,
            flat_v=flat_v,
            flat_index_map=flat_index_map,
        )

    # ── Internal ─────────────────────────────────────────────────────

    def _query_vec(self, query_ids: list[int]) -> torch.Tensor:
        """Compute Q vector at the retrieval head — returns torch.Tensor on cuda.

        Dispatches to the MLX kv_gen path when available, otherwise to the
        torch-native forward-hook path. No explicit sync here so the caller
        can fuse it into the scoring matmul as a single CUDA dispatch.
        """
        if self._kv_gen is not None:
            return self._query_vec_via_kv_gen(query_ids)
        return self._query_vec_via_hook(query_ids)

    def _query_vec_via_kv_gen(self, query_ids: list[int]) -> torch.Tensor:
        """MLX kv_gen path — preserved verbatim for backward compatibility."""
        backbone = self._kv_gen.backbone
        layer_adapter = backbone.adapted_layers[self.meta.retrieval_layer]

        ids = torch.tensor(query_ids, device="cuda")[None]  # (1, S)
        B, S = ids.shape

        # Forward to retrieval layer — already on CUDA
        h = self._kv_gen.prefill_to_layer(ids, target_layer=self.meta.retrieval_layer)

        # project_qkv applies pre_attn_norm output → q_norm + RoPE in one call
        x = layer_adapter.pre_attn_norm(h[:, -1:, :])  # (1, 1, hidden)
        q, _, _ = layer_adapter.project_qkv(x, B, 1, offset=S - 1)
        # q: (1, nq, 1, head_dim)

        # Joins the downstream scoring graph
        return q[0, self.meta.query_head, 0, :].to(torch.float32)  # (head_dim,)

    def _query_vec_via_hook(self, query_ids: list[int]) -> torch.Tensor:
        """Torch-native path — capture L{rl} output via forward hook, then re-project.

        Mirrors the MLX semantics of `prefill_to_layer(rl)` (= OUTPUT of layer rl)
        followed by re-projection through layer rl's pre-attn norm + QKV + RoPE
        on the LAST position only.
        """
        model = self._raw_model
        if model is None:
            raise RuntimeError(
                "LocalVecInjectProvider has no kv_gen and no raw model — "
                "pass raw_model= to load() when kv_gen=None"
            )
        layers = _resolve_transformer_layers(model)
        rl = self.meta.retrieval_layer
        layer = layers[rl]

        ids = torch.tensor([list(query_ids)], dtype=torch.long, device="cuda")  # (1, S)
        B, S = ids.shape
        attention_mask = torch.ones_like(ids)

        captured: dict[str, torch.Tensor] = {}

        def _capture(_mod, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured["h"] = hidden

        handle = layer.register_forward_hook(_capture)
        try:
            with torch.inference_mode():
                try:
                    model(input_ids=ids, attention_mask=attention_mask, use_cache=False)
                except TypeError:
                    model(input_ids=ids, attention_mask=attention_mask)
        finally:
            handle.remove()

        h = captured.get("h")
        if h is None:
            raise RuntimeError(f"forward hook on layer {rl} did not fire")

        # Mirror MLX: project the LAST position with position_ids=[S-1].
        last_pos = torch.tensor([[S - 1]], dtype=torch.long, device=h.device)
        q, _, _ = _project_qkv(layer, h[:, -1:, :], position_ids=last_pos)
        # q: (1, num_heads, 1, head_dim)
        return q[0, self.meta.query_head, 0, :].to(torch.float32)

    def _query_h4_vec(self, query_ids: list[int]) -> torch.Tensor:
        """Compute H4's attention output at the last query position.

        Dispatches to the MLX kv_gen path when available, otherwise to the
        torch-native forward-hook path.

        Uses h entering L{retrieval_layer} (= output of L{retrieval_layer-1})
        so this reflects the ACTUAL L{retrieval_layer} forward computation,
        not the re-projected state used by _query_vec (K-space Stage 1).

        Returns (hidden_size,) float32 — join into the h4_scores matmul.
        """
        if self._kv_gen is not None:
            return self._query_h4_vec_via_kv_gen(query_ids)
        return self._query_h4_vec_via_hook(query_ids)

    def _query_h4_vec_via_kv_gen(self, query_ids: list[int]) -> torch.Tensor:
        """MLX kv_gen path — preserved verbatim for backward compatibility."""
        backbone = self._kv_gen.backbone
        rl = self.meta.retrieval_layer
        layer_adapter = backbone.adapted_layers[rl]

        ids = torch.tensor(query_ids, device="cuda")[None]  # (1, S)
        B, S = ids.shape

        # Run to L{rl-1} — input to L{rl}
        h_pre = self._kv_gen.prefill_to_layer(ids, target_layer=rl - 1)

        x = layer_adapter.pre_attn_norm(h_pre)  # (1, S, D)
        q, k, v = layer_adapter.project_qkv(x, B, S, offset=0)

        H4 = self.meta.query_head
        kv_h = H4 // layer_adapter.n_rep

        # Last query position attending over all positions
        q_last = q[:, H4, -1:, :]  # (1, 1, dh)
        k_kv = k[:, kv_h, :, :]  # (1, S, dh)
        v_kv = v[:, kv_h, :, :]  # (1, S, dh)

        scores = torch.matmul(q_last, k_kv.permute(0, 2, 1)) * layer_adapter.attn_scale
        attn_w = F.softmax(scores, dim=-1)  # (1, 1, S)
        h4_out = torch.matmul(attn_w, v_kv)[:, 0, :]  # (1, dh)

        # Project through H4's O_proj columns → hidden space (via protocol)
        h4_contrib = layer_adapter.head_output_projection(h4_out, H4)  # (1, D)

        return h4_contrib[0].to(torch.float32)  # (D,)

    def _query_h4_vec_via_hook(self, query_ids: list[int]) -> torch.Tensor:
        """Torch-native path — capture OUTPUT of L{rl-1} (= INPUT to L{rl}) via
        forward hook, then re-project through layer rl's QKV and run a
        last-position attention over the full sequence.
        """
        model = self._raw_model
        if model is None:
            raise RuntimeError(
                "LocalVecInjectProvider has no kv_gen and no raw model — "
                "pass raw_model= to load() when kv_gen=None"
            )
        layers = _resolve_transformer_layers(model)
        rl = self.meta.retrieval_layer
        if rl <= 0:
            raise RuntimeError(
                f"_query_h4_vec requires retrieval_layer > 0, got {rl}"
            )
        capture_layer = layers[rl - 1]  # capture OUTPUT of rl-1 = input to rl
        project_layer = layers[rl]

        ids = torch.tensor([list(query_ids)], dtype=torch.long, device="cuda")
        B, S = ids.shape
        attention_mask = torch.ones_like(ids)

        captured: dict[str, torch.Tensor] = {}

        def _capture(_mod, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured["h"] = hidden

        handle = capture_layer.register_forward_hook(_capture)
        try:
            with torch.inference_mode():
                try:
                    model(input_ids=ids, attention_mask=attention_mask, use_cache=False)
                except TypeError:
                    model(input_ids=ids, attention_mask=attention_mask)
        finally:
            handle.remove()

        h_pre = captured.get("h")
        if h_pre is None:
            raise RuntimeError(f"forward hook on layer {rl - 1} did not fire")

        position_ids = torch.arange(S, dtype=torch.long, device=h_pre.device).unsqueeze(0)
        q, k, v = _project_qkv(project_layer, h_pre, position_ids=position_ids)
        # q: (B, num_heads, S, head_dim); k/v: (B, num_kv_heads, S, head_dim)

        H4 = self.meta.query_head
        num_heads = q.shape[1]
        num_kv_heads = k.shape[1]
        n_rep = max(num_heads // max(num_kv_heads, 1), 1)
        kv_h = min(H4 // n_rep, num_kv_heads - 1)
        head_dim = q.shape[-1]

        q_last = q[:, H4, -1:, :]  # (1, 1, dh)
        k_kv = k[:, kv_h, :, :]  # (1, S, dh)
        v_kv = v[:, kv_h, :, :]  # (1, S, dh)

        attn_scale = 1.0 / (head_dim**0.5)
        scores = torch.matmul(q_last, k_kv.permute(0, 2, 1)) * attn_scale
        attn_w = F.softmax(scores, dim=-1)  # (1, 1, S)
        h4_out = torch.matmul(attn_w, v_kv)[:, 0, :]  # (1, dh)

        # Project through H4's O_proj columns → hidden space.
        o_proj = project_layer.self_attn.o_proj
        o_weight = o_proj.weight  # (hidden, num_heads * head_dim)
        o_slice = o_weight[:, H4 * head_dim : (H4 + 1) * head_dim]  # (D, dh)
        h4_contrib = torch.matmul(h4_out, o_slice.T)  # (1, D)

        return h4_contrib[0].to(torch.float32)

    def _route_h4(
        self,
        query_ids: list[int],
        top_k: int,
        stage1_ms: float,
    ) -> VecInjectResult | None:
        """Stage 2: H4 output cosine routing.

        Returns a VecInjectResult with routing_stage="h4" if confident,
        None if the H4 score distribution doesn't clear the adaptive threshold.
        """
        t1 = time.monotonic()

        q_h4 = self._query_h4_vec(query_ids)
        q_h4_nrm = q_h4 / torch.maximum(
            torch.sqrt(torch.sum(q_h4 * q_h4)),
            torch.tensor(1e-9, device="cuda"),
        )

        # H4 cosine scores against all stored fact H4 vectors (CUDA matmul)
        h4_scores = self._flat_h4 @ q_h4_nrm  # (n_facts,)

        k = min(top_k, self.n_facts)
        top_idx = torch.argsort(-h4_scores)[:k]

        # Best distinctive H4 match — same mask trick as Stage 1
        dist_mask = self._flat_distinctive.to(torch.float32)
        h4_scores_dist = h4_scores * dist_mask + (dist_mask - 1.0) * 1e9
        best_dist_idx_arr = torch.argmax(h4_scores_dist)

        top_h4_scores = h4_scores[top_idx]
        top_token_ids = self._flat_token_ids[top_idx]
        top_coefs = self._flat_coefs[top_idx]
        top_wids = self._flat_wid[top_idx]
        top_positions = self._flat_positions[top_idx]
        top_distinctive = self._flat_distinctive[top_idx]

        best_dist_h4_score_arr = h4_scores[best_dist_idx_arr]
        best_dist_tok_arr = self._flat_token_ids[best_dist_idx_arr]
        best_dist_coef_arr = self._flat_coefs[best_dist_idx_arr]
        best_dist_wid_arr = self._flat_wid[best_dist_idx_arr]
        best_dist_pos_arr = self._flat_positions[best_dist_idx_arr]

        mean_h4_arr = torch.mean(h4_scores)

        torch.cuda.synchronize()
        elapsed_ms = stage1_ms + (time.monotonic() - t1) * 1000

        mean_h4 = float(mean_h4_arr.item())
        h4_threshold = max(self.confidence_threshold, mean_h4 * 2.0)
        best_dist_h4 = float(best_dist_h4_score_arr.item())

        if best_dist_h4 < h4_threshold:
            return None

        best_dist_match = VecInjectMatch(
            token_id=int(best_dist_tok_arr.item()),
            coefficient=float(best_dist_coef_arr.item()),
            score=best_dist_h4,
            window_id=int(best_dist_wid_arr.item()),
            position=int(best_dist_pos_arr.item()),
            distinctive=True,
        )
        top_matches = [
            VecInjectMatch(
                token_id=int(top_token_ids[i].item()),
                coefficient=float(top_coefs[i].item()),
                score=float(top_h4_scores[i].item()),
                window_id=int(top_wids[i].item()),
                position=int(top_positions[i].item()),
                distinctive=bool(top_distinctive[i].item()),
            )
            for i in range(k)
        ]
        return VecInjectResult(
            matches=[best_dist_match] + top_matches,
            retrieval_ms=elapsed_ms,
            injection_layer=self.meta.injection_layer,
            routing_confident=True,
            top_score=best_dist_h4,
            routing_stage="h4",
        )

    # ── Append (living store growth) ─────────────────────────────────

    def append(
        self,
        k_vecs: torch.Tensor,  # (N, k_dim) float32 — L2-normalised
        token_ids: torch.Tensor,  # (N,) int32
        coefs: torch.Tensor,  # (N,) float32
        positions: torch.Tensor,  # (N,) int32
        distinctive: torch.Tensor,  # (N,) int32
        source_id: int,
        source_type: SourceType = SourceType.GENERATED,
        h4_vecs: torch.Tensor | None = None,  # (N, hidden_dim) float32 L2-normed
        window_id: int | None = None,
    ) -> int:
        """Append new entries to the live store. Returns new total fact count.

        All tensors must be on device before calling.
        """
        n_new = k_vecs.shape[0]
        if n_new == 0:
            return self.n_facts

        wid = window_id if window_id is not None else source_id
        wid_arr = torch.full((n_new,), wid, dtype=torch.int32, device="cuda")
        sid_arr = torch.full((n_new,), source_id, dtype=torch.int32, device="cuda")
        stype_arr = torch.full((n_new,), int(source_type), dtype=torch.uint8, device="cuda")

        self._flat_k = torch.cat([self._flat_k, k_vecs])
        self._flat_token_ids = torch.cat([self._flat_token_ids, token_ids])
        self._flat_coefs = torch.cat([self._flat_coefs, coefs])
        self._flat_wid = torch.cat([self._flat_wid, wid_arr])
        self._flat_positions = torch.cat([self._flat_positions, positions])
        self._flat_distinctive = torch.cat([self._flat_distinctive, distinctive])

        if self._flat_source_ids is not None:
            self._flat_source_ids = torch.cat([self._flat_source_ids, sid_arr])
        else:
            # First time — backfill existing entries as source_id=0, document
            old_n = self.n_facts - n_new  # before concat above
            self._flat_source_ids = torch.cat(
                [torch.zeros((old_n,), dtype=torch.int32, device="cuda"), sid_arr]
            )

        if self._flat_source_types is not None:
            self._flat_source_types = torch.cat([self._flat_source_types, stype_arr])
        else:
            old_n = self.n_facts - n_new
            self._flat_source_types = torch.cat(
                [torch.zeros((old_n,), dtype=torch.uint8, device="cuda"), stype_arr]
            )

        if h4_vecs is not None and self._flat_h4 is not None:
            self._flat_h4 = torch.cat([self._flat_h4, h4_vecs])

        torch.cuda.synchronize()

        self._summaries.append(_WindowSummary(window_id=wid, n_facts=n_new))
        return self.n_facts

    def save(self, path: str | Path) -> None:
        """Save the full store (including appended entries) to an npz file.

        Saves in knowledge_store.npz format — flat arrays, not per-window.
        Includes the running residual if set.
        """
        path = Path(path)
        arrays: dict[str, np.ndarray] = {}

        def _to_np(arr: torch.Tensor, dtype) -> np.ndarray:
            return arr.detach().cpu().numpy().astype(dtype)

        arrays["k_vecs"] = _to_np(self._flat_k, np.float32)
        arrays["token_ids"] = _to_np(self._flat_token_ids, np.int32)
        arrays["coefficients"] = _to_np(self._flat_coefs, np.float32)
        arrays["positions"] = _to_np(self._flat_positions, np.int32)
        arrays["distinctive"] = _to_np(self._flat_distinctive, np.int32)
        arrays["source_ids"] = _to_np(
            self._flat_source_ids
            if self._flat_source_ids is not None
            else torch.zeros((self.n_facts,), dtype=torch.int32, device="cuda"),
            np.int32,
        )
        arrays["source_types"] = _to_np(
            self._flat_source_types
            if self._flat_source_types is not None
            else torch.zeros((self.n_facts,), dtype=torch.uint8, device="cuda"),
            np.uint8,
        )
        if self._flat_h4 is not None:
            arrays["h_copy_vecs"] = _to_np(self._flat_h4, np.float32)
        if self.residual is not None:
            arrays["residual"] = _to_np(self.residual, np.float32)

        # Metadata scalars
        arrays[VecInjectMetaKey.LAYER] = np.array(self.meta.retrieval_layer)
        arrays[VecInjectMetaKey.KV_HEAD] = np.array(self.meta.kv_head)
        arrays[VecInjectMetaKey.QUERY_HEAD] = np.array(self.meta.query_head)
        arrays[VecInjectMetaKey.INJECT_LAYER] = np.array(self.meta.injection_layer)

        np.savez(str(path), **arrays)

    # ── Diagnostics ──────────────────────────────────────────────────

    def log_stats(self, file=sys.stderr) -> None:
        h4_note = f"h4={'yes (' + str(self._flat_h4.shape[1]) + 'D)' if self._flat_h4 is not None else 'no'}"
        routing = "S1+S2+S3" if self._flat_h4 is not None else "S1+S3"
        print(
            f"  LocalVecInjectProvider: {self.n_facts} facts × {self.head_dim}D "
            f"across {len(self._summaries)} windows "
            f"(L{self.meta.retrieval_layer} KV-H{self.meta.kv_head}, "
            f"inject L{self.meta.injection_layer}, "
            f"coefs={'yes' if self.has_injection else 'no'}, "
            f"{h4_note}, routing={routing}, "
            f"conf_threshold={self.confidence_threshold:.2f}, "
            f"device=cuda)",
            file=file,
        )
