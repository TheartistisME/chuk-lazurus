# Epic 1: Dual-Backend Bring-Up (MLX + CUDA/torch) — Implementation Spec

Status: Draft
Owner: lazarus-dual-backend-epic1
Target hardware: Apple Silicon (existing) + NVIDIA RTX 5090 (sm_120, Blackwell) readiness

---

## 1. Purpose & Scope

Lazarus currently assumes MLX on Apple Silicon. Epic 1 preserves that path unchanged
while adding a first-class, explicit PyTorch/CUDA backend, including the capability
checks needed to run cleanly on RTX 5090 (Blackwell, `sm_120`).

In scope for Epic 1:

- Explicit backend selection via environment variable and `UnifiedPipelineConfig` field
  (no implicit `platform.system()` sniffing unless nothing else is specified).
- Deferred/lazy import of `mlx.core` / `mlx.nn` everywhere they are currently
  imported at module load — so the package imports cleanly on Linux/CUDA hosts
  without `mlx` installed.
- CUDA device selection + `torch.cuda.get_device_capability()` validation, with a
  clear error on SM/toolkit mismatch and a bf16-preferred dtype policy for
  `sm >= 8.0`.
- `pyproject.toml` split: `mlx` / `torch` / `torch-cuda` / `all` extras so a single
  install does not force the wrong framework.
- Full unit-test coverage of backend selection, with mocks for CUDA-absent CI.

Out of scope: see §10.

---

## 2. Current State Inventory

All paths are relative to the repository root
`/mnt/c/users/jehma/desktop/lazarus/chuk-lazurus/`. Line numbers are verified against
the working tree at the time of writing.

| # | File | Relevant lines | Current behaviour |
|---|------|----------------|--------------------|
| 1 | `src/chuk_lazarus/models_v2/core/backend/base.py` | 16-168 | Abstract `Backend` with `name`, `device`, tensor ops. No device-validation hook. |
| 2 | `src/chuk_lazarus/models_v2/core/backend/registry.py` | 20-50 | `get_backend()` auto-selects MLX on Darwin, else torch. No env override. |
| 3 | `src/chuk_lazarus/models_v2/core/backend/torch_backend.py` | 18-27 | Ctor hardcodes `device="cuda"`, falls back to CPU; no SM check, no device id. |
| 4 | `src/chuk_lazarus/models_v2/core/backend/mlx_backend.py` | 18-28 | Ctor does `import mlx.core as mx` / `mlx.nn as nn` inside `__init__` (already lazy — good). |
| 5 | `src/chuk_lazarus/inference/loader.py` | 23, 38-45 | Top-level `import mlx.core as mx`; `DType.to_mlx()` returns MLX dtype. Breaks import on non-Apple. |
| 6 | `src/chuk_lazarus/inference/unified.py` | 70-87 | `UnifiedPipelineConfig` has no `backend`/`device` field. |
| 7 | `src/chuk_lazarus/inference/context/kv_generator.py` | 41, 46 | Top-level `import mlx.core as mx`; uses `mx.bfloat16` for mask dtype. |
| 8 | `src/chuk_lazarus/introspection/hooks.py` | 30-31 | Top-level `import mlx.core as mx` / `import mlx.nn as nn`. |
| 9 | `src/chuk_lazarus/cli/commands/infer/run.py` | 14-37 | No `--backend`/`--device` flag; `UnifiedPipelineConfig(engine=...)` only. |
| 10 | `src/chuk_lazarus/cli/commands/context/prefill/_vec_inject.py` | 38 | Top-level `import mlx.core as mx`. |
| 11 | `pyproject.toml` | 7-23, 25-58 | `mlx`, `mlx-lm` are *base* deps. `torch>=2.0.0` only appears under `dev`. No `torch-cuda` extra. |
| 12 | `tests/models_v2/core/backend/test_torch_backend.py` | entire file (~270 L) | Exercises math ops; no SM capability / device-id tests. |
| 13 | `tests/models_v2/core/backend/test_mlx_backend.py` | entire file | MLX ops; skipped on non-Apple. |

---

## 3. Target Architecture

### 3.1 Backend selection order

```
1. Explicit argument (Backend instance passed programmatically)
2. UnifiedPipelineConfig.backend  (new field)
3. Env var:  CHUK_BACKEND  in {mlx, torch}
4. Auto-detect: Darwin -> mlx (if importable) else torch
```

Device selection order inside `TorchBackend`:

```
1. Explicit `device=` ctor arg
2. UnifiedPipelineConfig.device  (new field)
3. Env var:  CHUK_DEVICE  in {cuda, cuda:N, cpu, mps}
4. Env var:  CHUK_CUDA_DEVICE_ID  (integer, used when CHUK_DEVICE == "cuda")
5. Auto: cuda if torch.cuda.is_available() else (mps if available else cpu)
```

### 3.2 Environment variables

| Name | Values | Purpose |
|------|--------|---------|
| `CHUK_BACKEND` | `mlx` \| `torch` | Force backend irrespective of platform. |
| `CHUK_DEVICE` | `cuda` \| `cuda:N` \| `cpu` \| `mps` | Force torch device. Ignored by MLX. |
| `CHUK_CUDA_DEVICE_ID` | integer (default `0`) | Preferred GPU when multiple CUDA devices present. Parsed with `try/except ValueError`; invalid values log a warning and fall back to `0`. |
| `CHUK_SKIP_SM_CHECK` | `1` \| `true` \| `yes` \| `on` (case-insensitive) | Bypass the `get_device_capability()` validation (for forward-compat hacking). Any other value (including typos) leaves the check enabled. |

All env vars are consumed with `.strip().lower()` (where applicable) to tolerate
surrounding whitespace and case differences. Unknown values for `CHUK_BACKEND` or
`CHUK_DEVICE` raise `ValueError` with the invalid value and the accepted set in
the message (fail-fast; no silent fallback).

### 3.3 Config keys (`UnifiedPipelineConfig`)

```python
backend: str | None = Field(None, description="'mlx' | 'torch'. None = env/auto.")
device: str | None  = Field(None, description="Torch device spec. None = env/auto.")
cuda_check_sm: bool | None = Field(None,
    description="Validate CUDA SM capability on init. None = consult "
                "CHUK_SKIP_SM_CHECK env var (default: check enabled).")
```

### 3.4 Lazy-import rule

Module-top `import mlx.core` / `import mlx.nn` is **forbidden** in any file that
may be imported on a non-Apple host. The canonical pattern is:

```python
# at use site, inside a function or method, guarded by backend choice
def _to_framework_dtype(dtype: "DType", backend_name: str):
    if backend_name == "mlx":
        import mlx.core as mx
        return {DType.FLOAT16: mx.float16, DType.BFLOAT16: mx.bfloat16,
                DType.FLOAT32: mx.float32}[dtype]
    import torch
    return {DType.FLOAT16: torch.float16, DType.BFLOAT16: torch.bfloat16,
            DType.FLOAT32: torch.float32}[dtype]
```

For type hints only, use `TYPE_CHECKING`:

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import mlx.core as mx  # noqa: F401
```

---

## 4. File-by-File Change Specification

### 4.1 `src/chuk_lazarus/models_v2/core/backend/base.py`

- **Current:** abstract methods for tensor ops only (L16-168).
- **Change:** add an optional hook `validate_device(self) -> None` with a default
  no-op implementation so subclasses can override. This keeps the contract backwards
  compatible.

```python
class Backend(ABC):
    ...
    def validate_device(self) -> None:
        """Raise a RuntimeError if the backend cannot run on the selected device.

        Default: no-op. TorchBackend overrides this to check CUDA capability.
        """
        return None
```

- **Risk:** none — additive default method.

### 4.2 `src/chuk_lazarus/models_v2/core/backend/registry.py`

- **Current:** L20-50 branches on `platform.system() == "Darwin"`.
- **Change:** add env-var override and accept an explicit `name` + `device` kwarg
  pair. Keep the `Darwin -> MLX` default as final fallback.

```python
import os, platform, logging
from .base import Backend
from .types import BackendType

# Module-level singletons. Declared BEFORE the functions that reference them
# so that `global _current_backend, _current_key` inside get_backend / set_backend
# does not hit a NameError at import time.
_current_backend: Backend | None = None
_current_key: tuple | None = None

def _resolve_backend_name() -> str:
    env = os.environ.get("CHUK_BACKEND", "").strip().lower()
    if env == "":
        return "mlx" if platform.system() == "Darwin" else "torch"
    if env not in ("mlx", "torch"):
        raise ValueError(
            f"CHUK_BACKEND={env!r} is invalid; expected one of {{'mlx','torch'}}."
        )
    return env

def get_backend(
    name: str | None = None,
    device: str | None = None,
    *,
    check_sm: bool | None = None,
) -> Backend:
    """Resolve & cache a backend.

    Cache semantics: the singleton is keyed on the RAW ``(chosen_name, device,
    check_sm)`` tuple where ``chosen_name`` is the resolved backend name
    (``_resolve_backend_name`` fills in the default when ``name is None``), and
    ``device`` / ``check_sm`` are stored AS-PASSED (no env-var resolution).
    This means:
      * Two calls with identical args hit the cache.
      * A call with ``check_sm=None`` matches a cached backend keyed on
        ``check_sm=None`` (i.e., a previous call that also omitted it, or a
        ``set_backend`` injection — see below). It does NOT match a prior
        ``get_backend(check_sm=True/False)`` call.
      * Callers who want guaranteed reuse should EITHER retain the returned
        reference OR always pass the same explicit args (including
        ``check_sm``).
    """
    global _current_backend, _current_key
    chosen = (name or _resolve_backend_name()).lower()
    # Raw keying: no env-var resolution on the key side. TorchBackend.__init__
    # still consults the env var when check_sm=None is passed through, so the
    # resulting BACKEND behaves correctly; the cache simply keys on intent
    # ("caller asked for default") rather than outcome ("check turned out to
    # be True"), which keeps set_backend and get_backend mutually consistent.
    key = (chosen, device, check_sm)
    if _current_backend is not None and _current_key == key:
        return _current_backend
    if chosen == "mlx":
        from .mlx_backend import MLXBackend
        backend: Backend = MLXBackend()
    elif chosen == "torch":
        from .torch_backend import TorchBackend
        backend = TorchBackend(device=device, check_sm=check_sm)
    else:
        raise ValueError(
            f"Unknown backend: {chosen!r} (expected 'mlx' or 'torch')."
        )
    try:
        backend.validate_device()
    except Exception:
        # Do NOT mutate the cache on failure. If a prior call had established
        # a valid cached backend, it stays; callers that retry with different
        # args will resolve against the prior cache as-is. Re-raise so the
        # failure surfaces to the caller.
        raise
    # Publish only after validate_device() succeeds so a failed validation
    # never leaves a broken backend cached.
    _current_backend = backend
    _current_key = key
    return backend

def set_backend(backend: Backend) -> None:
    """Install a pre-built backend as the singleton. Signature unchanged.

    The cached key uses ``check_sm=None`` (wildcard for "whatever the injected
    backend was built with"). This means a subsequent ``get_backend()`` call
    that omits the ``check_sm`` arg (i.e., passes ``None``) will MATCH and
    return the injected instance. A call that passes an EXPLICIT ``check_sm``
    value will miss and construct a fresh backend — if reuse is required,
    callers should retain the reference they passed to ``set_backend``.
    """
    global _current_backend, _current_key
    _current_backend = backend
    _current_key = (
        backend.name.value.lower(),
        getattr(backend, "device", None),
        None,  # wildcard; see docstring above
    )

def reset_backend() -> None:
    """Clear the singleton. Signature unchanged."""
    global _current_backend, _current_key
    _current_backend = None
    _current_key = None
```

- **Risk:** the cache is now keyed on `(name, device, check_sm)` — a caller that
  previously relied on "any call returns the cached instance" will see a new
  instance when they pass a different device. This matches user intent and
  eliminates the silent-override bug.
- **Cache semantics clarification:** the cache is a single-slot last-writer-wins
  store, not a dict of backends. `get_backend()` with no args (after a prior
  `get_backend(device="cuda:0")` call) will produce key
  `(resolved_name, None, None)` which does NOT match `(resolved_name, "cuda:0", None)`,
  so a fresh backend is constructed and replaces the cache. Callers that want
  "get the one I built" should either (a) retain the reference returned from
  the first call or (b) always pass the same explicit args. This matches the
  explicit-arg-wins semantics and avoids stealth device swaps.
- **Thread-safety:** `_current_backend` is a process-wide mutable. Epic 1 does
  not introduce concurrent pipeline construction; Epic 2+ threaded inference
  paths must either (a) always read an already-initialized backend or (b) add
  a `threading.Lock` around this module. Documented as a known gap.
- **`set_backend` / `reset_backend`:** existing signatures are preserved; they
  now also update `_current_key` so the next `get_backend()` call with matching
  args reuses the injected instance.

### 4.3 `src/chuk_lazarus/models_v2/core/backend/torch_backend.py`

- **Current:** `__init__(self, device="cuda")` at L18 hardcodes device and falls
  back to CPU silently (L23).
- **Change:** optional `device`; resolve from env; validate SM capability; prefer
  bf16 on sm>=8.0.

```python
import os
from typing import Any
from .base import Backend
from .types import BackendType

_TRUTHY = ("1", "true", "yes", "on")

def _parse_device_id(raw: str, default: int = 0) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError):
        import logging
        logging.getLogger(__name__).warning(
            "CHUK_CUDA_DEVICE_ID=%r is not an integer; falling back to %d", raw, default,
        )
        return default

def _normalize_cuda_device(spec: str) -> str:
    """Return a concrete 'cuda:N' form. 'cuda' alone -> current CUDA device id."""
    if spec == "cuda":
        idx = _parse_device_id(os.environ.get("CHUK_CUDA_DEVICE_ID", "0"))
        return f"cuda:{idx}"
    return spec

class TorchBackend(Backend):
    def __init__(self, device: str | None = None, check_sm: bool | None = None):
        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "PyTorch is required for TorchBackend. "
                "Install with: pip install 'chuk-lazarus[torch]' "
                "or 'chuk-lazarus[torch-cuda]'."
            ) from e
        self._torch = torch
        self._device = self._resolve_device(device)
        self._check_sm = (
            check_sm
            if check_sm is not None
            else os.environ.get("CHUK_SKIP_SM_CHECK", "").strip().lower() not in _TRUTHY
        )
        # NOTE: _preferred_dtype is computed lazily (first access) so the ctor
        # itself does not invoke any capability API. The canonical
        # construction path — get_backend() — calls validate_device()
        # immediately after the ctor (see §4.2), BEFORE any user code can
        # access .preferred_dtype. That guarantees the "torch.cuda
        # unavailable" / "sm_NNN not in this build" messages fire first.
        # Callers who build `TorchBackend(...)` directly (bypassing the
        # registry) are expected to call `.validate_device()` before
        # touching `.preferred_dtype`; this is documented in the docstring
        # of the property below.
        self._preferred_dtype_cache = None

    @property
    def preferred_dtype(self):
        if self._preferred_dtype_cache is None:
            self._preferred_dtype_cache = self._resolve_dtype()
        return self._preferred_dtype_cache

    @staticmethod
    def _validate_device_spec(spec: str, source: str) -> str:
        """Whitelist-check a device string; raise ValueError on malformed input.

        Returns a CANONICAL form so cache keys are stable: `cuda:007` → `cuda:7`,
        so two callers using different-but-equivalent strings share a cache
        slot. Negative indices are rejected at this layer (they cannot be a
        valid non-negative integer id).
        """
        normalized = spec.strip().lower()
        if normalized in ("cpu", "mps", "cuda"):
            return normalized
        if normalized.startswith("cuda:"):
            tail = normalized.split(":", 1)[1]
            if tail.isdigit():  # isdigit() is False for "-1" and for non-numeric
                return f"cuda:{int(tail)}"  # canonicalize leading zeros
        raise ValueError(
            f"{source}={spec!r} is not one of {{cpu, mps, cuda, cuda:N}} "
            f"(where N is a non-negative integer, no leading sign)."
        )

    def _resolve_device(self, explicit: str | None) -> str:
        if explicit:
            # Symmetric with the env var path: explicit args must pass the
            # same whitelist so programmatic callers fail fast instead of
            # getting a cryptic torch error later.
            return self._validate_device_spec(explicit, "device=")
        env = os.environ.get("CHUK_DEVICE", "")
        if env.strip():
            return self._validate_device_spec(env, "CHUK_DEVICE")
        if self._torch.cuda.is_available():
            idx = _parse_device_id(os.environ.get("CHUK_CUDA_DEVICE_ID", "0"))
            return f"cuda:{idx}"
        # MPS is only probed for correctness; defensive getattr is intentional
        # in case of unusual torch builds without a `backends.mps` attribute.
        if getattr(self._torch.backends, "mps", None) and self._torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _resolve_dtype(self):
        t = self._torch
        if self._device.startswith("cuda") and t.cuda.is_available():
            dev = _normalize_cuda_device(self._device)
            major, _ = t.cuda.get_device_capability(dev)
            return t.bfloat16 if major >= 8 else t.float16
        return t.float32

    def validate_device(self) -> None:
        if not self._device.startswith("cuda"):
            return
        t = self._torch
        if not t.cuda.is_available():
            raise RuntimeError(
                f"Requested CUDA device {self._device!r} but torch.cuda is unavailable. "
                f"Install a CUDA-enabled torch (e.g. pip install 'chuk-lazarus[torch-cuda]' "
                f"--extra-index-url https://download.pytorch.org/whl/cu128)."
            )
        dev = _normalize_cuda_device(self._device)
        # Validate device id is within range BEFORE calling get_device_capability
        # so users get a friendly message instead of a torch internal assertion.
        idx = int(dev.split(":")[1]) if ":" in dev else 0
        count = t.cuda.device_count()
        # Negative indices are already rejected by _validate_device_spec so
        # the `idx < 0` branch is unreachable here; we keep the `>= count`
        # check as the sole in-range guard.
        if idx >= count:
            raise RuntimeError(
                f"Requested {self._device!r} but torch.cuda.device_count()={count}."
            )
        if not self._check_sm:
            return
        major, minor = t.cuda.get_device_capability(dev)
        sm = major * 10 + minor
        compiled = {int(a.split("_")[1].rstrip("a"))
                    for a in t.cuda.get_arch_list()
                    if a.startswith("sm_")}
        if not compiled:
            # CUDA-enabled torch without sm_* archs (unusual custom builds);
            # skip the comparison rather than crash on max(). Log a warning
            # so operators know the capability guard was bypassed.
            import logging
            logging.getLogger(__name__).warning(
                "torch.cuda.get_arch_list() returned no sm_* entries; "
                "skipping capability validation for device %s", self._device,
            )
            return
        if sm not in compiled:
            if sm > max(compiled):
                raise RuntimeError(
                    f"CUDA device has capability sm_{sm} but this torch build only "
                    f"supports {sorted(compiled)}. For RTX 5090 / Blackwell (sm_120) install "
                    f"torch>=2.6 built against cu128, or set CHUK_SKIP_SM_CHECK=1."
                )
            if sm < min(compiled):
                raise RuntimeError(
                    f"CUDA device has capability sm_{sm}; this torch build requires "
                    f"sm >= {min(compiled)}. Rebuild torch with older arch support "
                    f"or set CHUK_SKIP_SM_CHECK=1."
                )

    @property
    def name(self) -> BackendType: return BackendType.TORCH
    @property
    def device(self) -> str: return self._device
```

All downstream tensor-creation methods must pass `device=self._device` (already do,
L38-47, L109-114) — no further changes.

- **Risk:** behaviour change if a user previously relied on silent CPU fallback
  when CUDA was missing on a non-Apple host — they now get CPU by default, same as
  before, unless they explicitly set `CHUK_DEVICE=cuda`. Explicit `cuda` with no
  GPU now errors loudly (desired).
- **Notes on `get_device_capability` arg:** torch accepts `int | str | torch.device`.
  The normalized form `"cuda:N"` is always passed (never bare `"cuda"`) so the
  result is deterministic even when the current device differs from device 0.
- **Notes on `sm_90` / `sm_90a` (Hopper):** the `rstrip("a")` in the arch parse
  normalizes sm_90a entries to 90 so the membership check does not spuriously
  reject Hopper when torch was built with the `a` variant. Extend this pattern
  for future arch suffixes (`f`, `t`, etc.) if NVIDIA introduces them.

### 4.4 `src/chuk_lazarus/models_v2/core/backend/mlx_backend.py`

- **Current:** `__init__` already imports MLX lazily (L19-24). Good.
- **Change:** none required beyond adding a no-op `validate_device` (inherits from
  base). Add an explicit guard: if `platform.system() != "Darwin"`, augment the
  ImportError message with a hint pointing to the `[torch]` extra so Linux users
  who accidentally land on this path get a targeted suggestion.

```python
import platform

def __init__(self):
    try:
        import mlx.core as mx
        import mlx.nn as nn
    except ImportError as e:
        if platform.system() != "Darwin":
            raise ImportError(
                "MLX is only available on Apple Silicon (Darwin). The "
                "`[all]` extra installs MLX only under a Darwin environment "
                "marker, so on this host MLX is not present by design. Use "
                "the torch backend: set CHUK_BACKEND=torch (and, if needed, "
                "`pip install 'chuk-lazarus[torch]'`)."
            ) from e
        raise ImportError(
            "MLX is required for MLXBackend. Install with: "
            "pip install 'chuk-lazarus[mlx]' (Apple Silicon only)."
        ) from e
    self._mx, self._nn = mx, nn
```

- **Risk:** none.

### 4.5 `src/chuk_lazarus/inference/loader.py`

- **Current:** L23 `import mlx.core as mx` at module top; L38-45 `DType.to_mlx()`
  references `mx.float16` etc. This file fails to import on a non-Apple host.
- **Change:** move the MLX import into `to_mlx()`; add a symmetric `to_torch()`.
  Guard top-level only with `TYPE_CHECKING`.

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import mlx.core as mx  # noqa: F401
    import torch  # noqa: F401

class DType(str, Enum):
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"

    def to_mlx(self):
        import mlx.core as mx
        return {DType.FLOAT16: mx.float16, DType.FLOAT32: mx.float32,
                DType.BFLOAT16: mx.bfloat16}[self]

    def to_torch(self):
        import torch
        return {DType.FLOAT16: torch.float16, DType.FLOAT32: torch.float32,
                DType.BFLOAT16: torch.bfloat16}[self]
```

Any caller of `to_mlx()` already runs only under the MLX path, so behaviour is
preserved.

- **Risk:** medium — callers that import `mlx.core as mx` *through* this module
  (rare) break. Searchable and fixable.

### 4.6 `src/chuk_lazarus/inference/unified.py`

- **Current:** `UnifiedPipelineConfig` L70-87 has no backend/device fields.
- **Change:** add three fields; thread them into `UnifiedPipeline.from_pretrained`
  by calling
  `get_backend(name=config.backend, device=config.device, check_sm=config.cuda_check_sm)`
  before model construction. The `check_sm` keyword is the glue between the
  config field `cuda_check_sm` and `TorchBackend.__init__`'s `check_sm` parameter
  (see §4.3).

```python
class UnifiedPipelineConfig(BaseModel):
    dtype: DType = Field(DType.BFLOAT16, description="Weight dtype")
    cache_dir: Path | None = Field(None)
    default_system_message: str | None = Field("You are a helpful assistant.")
    default_max_tokens: int = Field(256, ge=1)
    default_temperature: float = Field(0.7, ge=0.0)
    enable_introspection: bool = Field(True)
    introspection_layers: list[int] | None = Field(None)
    engine: EngineMode = Field(EngineMode.STANDARD)

    # New: backend selection
    backend: str | None = Field(None, description="'mlx' | 'torch' | None")
    device: str | None  = Field(None, description="Torch device spec; ignored by MLX")
    cuda_check_sm: bool | None = Field(None,
        description="True = force check; False = skip; None = consult "
                    "CHUK_SKIP_SM_CHECK env var (default: check enabled).")
```

- **Risk:** low — all new fields are optional with defaults that preserve today's
  behaviour on macOS.

### 4.6.1 Epic 2 boundary guard (`inference/unified.py`)

Gate §8 #7 requires the torch selection path to reach the model-load step and
then fail with a specific, recognizable error. Epic 1 installs that guard now
so CI can prove the plumbing works end-to-end without waiting for Epic 2.

Inside `UnifiedPipeline.from_pretrained`, immediately AFTER `get_backend(...)`
returns a `TorchBackend` but BEFORE any weight-loading code executes:

```python
from ..models_v2.core.backend.types import BackendType

backend = get_backend(
    name=config.backend, device=config.device, check_sm=config.cuda_check_sm,
)
if backend.name == BackendType.TORCH:
    raise NotImplementedError(
        "torch model loading lands in Epic 2. "
        "Backend selection plumbing (Epic 1) works; set CHUK_BACKEND=mlx on "
        "Apple Silicon to run models today."
    )
```

Gate §8 #7 asserts a `NotImplementedError` is raised with "torch" and
"Epic 2" as substrings (not exact equality) so the guard is robust to minor
wording adjustments. The guard is removed by Epic 2 when the torch loader
lands.

- **Risk:** low — single guard; deleted in Epic 2.

### 4.7 `src/chuk_lazarus/inference/context/kv_generator.py`

- **Current:** L41 top-level `import mlx.core as mx`; L46 `_MASK_DTYPE = mx.bfloat16`.
- **Change:** remove top-level import; compute `_MASK_DTYPE` lazily via a helper
  keyed on the active backend. Since this module is MLX-only today, keep the MLX
  behaviour but defer the import.

```python
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import mlx.core as mx  # noqa: F401

from .protocols import KVStore, ModelBackboneProtocol

def _mask_dtype():
    import mlx.core as mx
    return mx.bfloat16
```

Callers that reference `_MASK_DTYPE` directly must be changed to call
`_mask_dtype()`. (Epic 2 introduces a torch equivalent; Epic 1 keeps MLX-only
semantics but with deferred imports.)

**Exhaustive caller audit (Epic 1 must update all of these):**

| File | Current usage | New usage |
|------|---------------|-----------|
| `src/chuk_lazarus/inference/context/kv_generator.py` | module-level `_MASK_DTYPE` at L46; referenced in the mask-building helper(s) in this file | replace every `_MASK_DTYPE` token with a call to `_mask_dtype()` |
| `src/chuk_lazarus/inference/context/knowledge/inject.py` | imports / uses `_MASK_DTYPE` (see `grep -rn "_MASK_DTYPE" src/`) | switch to `from ..kv_generator import _mask_dtype` and call at use sites |

Implementation check: the builder MUST run
`grep -rn "_MASK_DTYPE" src/ tests/` and assert zero hits before marking §4.7
complete. Any new hit = audit incomplete.

- **Risk:** medium — module-level constant becomes a function; complete audit
  above is mandatory.
- **Epic 2 note:** the helper currently returns MLX dtype unconditionally.
  Epic 2 will add a backend-aware version that dispatches on the active backend.
  Epic 1 deliberately does NOT guard-on-backend here because every current
  caller is on the MLX path; adding a premature torch branch now would create
  dead code that Epic 2 immediately replaces.

### 4.8 `src/chuk_lazarus/introspection/hooks.py`

- **Current:** L30-31 top-level MLX imports.
- **Change:** move MLX imports into the `HookManager.install()` / per-call
  sites. For typing only, guard with `TYPE_CHECKING`.

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import mlx.core as mx  # noqa: F401
    import mlx.nn as nn  # noqa: F401

class HookManager:
    def __init__(self, ...):
        import mlx.core as mx
        import mlx.nn as nn
        self._mx, self._nn = mx, nn
```

- **Risk:** low — local file; all references become `self._mx` / `self._nn`.

### 4.9 `src/chuk_lazarus/cli/commands/infer/run.py`

- **Current:** L32 builds `UnifiedPipelineConfig(engine=EngineMode(config.engine))`
  only. No `--backend`/`--device` CLI flag.
- **Change:** plumb two new optional CLI args → config.

```python
pipeline_config = UnifiedPipelineConfig(
    engine=EngineMode(config.engine),
    backend=getattr(args, "backend", None),
    device=getattr(args, "device", None),
)
```

The arg parser (`cli/commands/infer/_parser.py` or equivalent) must register:

```python
p.add_argument("--backend", choices=["mlx", "torch"], default=None,
               help="Override backend (default: env CHUK_BACKEND or auto).")
p.add_argument("--device", default=None,
               help="Torch device (cuda, cuda:1, cpu, mps). Ignored by MLX.")
```

- **Risk:** low — new optional flags only.

### 4.10 `src/chuk_lazarus/cli/commands/context/prefill/_vec_inject.py`

- **Current:** L38 top-level `import mlx.core as mx`.
- **Change:** defer import into the function(s) that actually use `mx`.

```python
def _load_vectors(path: Path):
    import mlx.core as mx
    ...
```

- **Risk:** low.

### 4.11 Documentation updates (Epic 1 deliverable)

Two docs must be updated in lockstep with the code changes so the install
matrix and env-var behaviour are discoverable:

**`README.md`** — add a "Backend selection" subsection containing:
- The four env vars (`CHUK_BACKEND`, `CHUK_DEVICE`, `CHUK_CUDA_DEVICE_ID`,
  `CHUK_SKIP_SM_CHECK`) with a one-line description and an example for each.
- The install matrix table from §5 (verbatim, with platform-marker caveat
  spelled out for the `[all]` extra).
- An explicit note that `CHUK_SKIP_SM_CHECK=false` enables the check (i.e.,
  the truthy set is `{1, true, yes, on}` — any other value, including
  `false`, leaves the check on). This prevents the shell-booleans footgun.
- A precedence summary: explicit ctor arg > `UnifiedPipelineConfig.*` field
  > env var > auto-detect.

**`docs/getting-started.md`** — create this file at exactly that path if it
does not already exist. It must contain:
- Three install walkthroughs, one per target in the §5 install matrix:
  (1) Apple Silicon, (2) Linux CPU, (3) Linux CUDA (including the cu128
  `--extra-index-url` for RTX 5090 / Blackwell).
- A minimal "run a pipeline" example using `--backend` and `--device` CLI
  flags, showing the exact command for each install.
- A troubleshooting subsection covering the three most likely errors:
  "torch.cuda is unavailable", "sm_NNN not in this build",
  "MLX is only available on Apple Silicon".
- A one-line pointer to `CHUK_SKIP_SM_CHECK` as the forward-compat escape
  hatch.

- **Risk:** none — docs-only; gate §8 #8 verifies contents.

---

## 5. Dependency Changes (`pyproject.toml`)

Move `mlx`, `mlx-lm`, and `torch` out of the base `dependencies` and `dev` lists
into new extras. Keep `numpy` / `pydantic` / etc. in base.

```diff
 [project]
 name = "chuk-lazarus"
 version = "0.5"
 ...
 dependencies = [
-    "mlx>=0.12.0",
-    "mlx-lm>=0.12.0",
     "transformers>=4.40.1",
     "huggingface-hub>=0.23.0,<0.24.0",
     "pydantic>=2.0.0",
     "pyyaml>=6.0",
     "numpy>=1.24.0",
     "tqdm>=4.65.0",
     "typer>=0.9.0",
     "tabulate>=0.9.0",
     "aiofiles>=23.0.0",
     "matplotlib>=3.10.8",
     "wasmtime>=19.0.0",
     "scikit-learn>=1.7.2",
     "scipy>=1.15.3",
 ]

 [project.optional-dependencies]
+mlx = [
+    "mlx>=0.12.0",
+    "mlx-lm>=0.12.0",
+]
+torch = [
+    "torch>=2.2.0",
+]
+torch-cuda = [
+    # For RTX 5090 / sm_120 install the cu128 wheel via --extra-index-url; the
+    # version floor here ensures the Python-side API is new enough. Pip cannot
+    # itself enforce a cu128 build — see the install matrix below and the
+    # README for the required `--extra-index-url` when targeting Blackwell.
+    "torch>=2.6.0",
+]
+# NOTE: self-referential extras (`chuk-lazarus[mlx,torch]`) are supported by
+# pip >= 21.2 but remain fragile on older toolchains. We expand explicitly to
+# keep installs robust across all supported pip versions.
+all = [
+    "mlx>=0.12.0 ; platform_system == 'Darwin'",
+    "mlx-lm>=0.12.0 ; platform_system == 'Darwin'",
+    "torch>=2.2.0",
+]
 dev = [
     "pytest>=7.0.0",
     "pytest-asyncio>=0.23.0",
     "pytest-cov>=4.0.0",
     "pytest-watch>=4.2.0",
     "ruff>=0.1.0",
     "mypy>=1.0.0",
     "types-tabulate>=0.9.0",
     "types-PyYAML>=6.0.0",
     "types-aiofiles>=23.0.0",
     "bandit>=1.7.0",
     "twine>=4.0.0",
     "build>=1.0.0",
-    "torch>=2.0.0",
 ]
```

Install matrix:

| Target | Command |
|--------|---------|
| Apple Silicon dev | `pip install -e '.[mlx,dev]'` |
| Linux CPU dev | `pip install -e '.[torch,dev]'` |
| Linux + RTX 40xx / Ada | `pip install -e '.[torch-cuda,dev]' --extra-index-url https://download.pytorch.org/whl/cu124` |
| Linux + RTX 5090 / Blackwell | `pip install -e '.[torch-cuda,dev]' --extra-index-url https://download.pytorch.org/whl/cu128` (torch >= 2.6 nightly until stable) |
| Both | `pip install -e '.[all,dev]'` |

> **Gate §8 #6 semantics:** fresh-venv install must succeed for `[mlx]`,
> `[torch]`, and `[all]` from PyPI alone. The `[torch-cuda]` extra is allowed
> to require `--extra-index-url https://download.pytorch.org/whl/cu128` for
> RTX 5090; the gate passes as long as the documented command installs a
> CUDA-capable torch and `python -c "import chuk_lazarus"` returns cleanly.
> `torch>=2.6` with cu128 wheels is the minimum supported floor at the time of
> writing; if the stable release has not shipped cu128 wheels when Epic 1
> lands, the README must note the nightly index URL explicitly.

---

## 6. CUDA Capability Matrix

| GPU family | SM | Min torch | CUDA toolkit | Recommended dtype |
|------------|-----|-----------|--------------|--------------------|
| Pascal (GTX 10xx) | sm_60/61 | 2.0 | 11.8 | float16 |
| Volta (V100) | sm_70 | 2.0 | 11.8 / 12.1 | float16 |
| Turing (T4, RTX 20xx) | sm_75 | 2.0 | 11.8 / 12.1 | float16 |
| Ampere (A100, RTX 30xx) | sm_80 / sm_86 | 2.0 | 11.8 / 12.1 | **bfloat16** |
| Ada (RTX 40xx, L40) | sm_89 | 2.2 | 12.1 / 12.4 | **bfloat16** |
| Hopper (H100, H200) | sm_90 | 2.2 | 12.1 / 12.4 | **bfloat16** |
| Blackwell (B100/B200, RTX 5090) | **sm_120** | **2.6** (nightly or later stable) | **12.8 (cu128)** | **bfloat16** |

> **sm_120 caveat:** the sm_120 / cu128 / torch 2.6 triplet is the
> NVIDIA-documented target for Blackwell consumer GPUs at the time of writing.
> The values are encoded as data in §4.3 error messages and in the dtype
> policy, NOT as hardcoded comparisons. If NVIDIA revises the RTX 5090
> capability tag before Epic 1 ships, the only code impact is the error-message
> string in `validate_device()` ("For RTX 5090 / Blackwell (sm_120) install
> torch>=2.6 built against cu128..."). The validation LOGIC (compare actual
> `sm = major*10+minor` against `get_arch_list()`) is hardware-agnostic and
> does not need to change. Epic 1 PR checklist: verify sm_120 against the
> current NVIDIA docs; if different, update the two error-message strings
> (§4.3 sm_high and the hint line) and the capability table row above.

The `TorchBackend._resolve_dtype()` policy: `bf16` when `major >= 8`, else `fp16`
for CUDA, `fp32` for CPU/MPS. Override via `UnifiedPipelineConfig.dtype`.

---

## 7. Tests Plan

All tests live under `tests/` and run via `pytest`.

### 7.1 `tests/models_v2/core/backend/test_registry.py` (NEW)

- `test_env_override_mlx(monkeypatch)` — set `CHUK_BACKEND=mlx`, stub
  `MLXBackend` (monkeypatch the symbol in the registry module), assert
  `get_backend()` returns the MLX stub on Linux.
- `test_env_override_torch(monkeypatch)` — set `CHUK_BACKEND=torch` on Darwin,
  assert torch path chosen.
- `test_auto_detect_darwin(monkeypatch)` — no env, `platform.system` patched to
  `"Darwin"`, MLX stub returns — expect MLX.
- `test_auto_detect_linux(monkeypatch)` — platform `"Linux"`, expect torch.
- `test_cache_key_mismatch_creates_fresh_backend()` — after `get_backend()`,
  call `get_backend(name="torch", device="cpu")` and verify a fresh instance
  is returned and becomes the new cached singleton. Also assert that a
  second call with identical `(name, device, check_sm)` reuses the cached
  instance (proves the cache works for matching keys).
- `test_set_backend_cache_matches_get_backend()` — inject a pre-built
  `TorchBackend` via `set_backend(backend)`, then call `get_backend(name="torch",
  device=backend.device, check_sm=backend._check_sm)` and assert the returned
  object IS the injected instance (proves `set_backend`'s key extraction
  uses `_check_sm`, not hardcoded `None`).
- `test_unknown_backend_raises()` — `CHUK_BACKEND=jax` → `ValueError`.

### 7.2 `tests/models_v2/core/backend/test_torch_backend.py` (EXTEND)

Add (skipped when `torch` is not installed):

- `test_resolve_device_env(monkeypatch)` — `CHUK_DEVICE=cpu` forces cpu even if
  cuda is available (mock `torch.cuda.is_available`).
- `test_resolve_device_cuda_id(monkeypatch)` — `CHUK_CUDA_DEVICE_ID=3` →
  `device == "cuda:3"`.
- `test_validate_device_raises_on_missing_cuda(monkeypatch)` — explicit
  `device="cuda"`, mock `cuda.is_available() -> False`, expect RuntimeError.
- `test_validate_sm_mismatch(monkeypatch)` — mock `get_device_capability`
  returning `(12, 0)` and `get_arch_list` returning the realistic torch output
  shape `["compute_80","sm_80","compute_86","sm_86","compute_89","sm_89"]`.
  Expect RuntimeError mentioning `sm_120` and `cu128`.
- `test_validate_sm_below_min(monkeypatch)` — capability `(7, 0)` against
  `compiled={80, 86, 89}` → RuntimeError mentioning `sm >= 80`.
- `test_validate_sm_90a_normalized(monkeypatch)` — arch list
  `["sm_90a"]`, capability `(9, 0)` → no raise (the `rstrip("a")` normalization
  is covered).
- `test_validate_empty_compiled(monkeypatch)` — arch list `["compute_80"]`
  only (no `sm_*`) → no raise; the validator skips the comparison.
- `test_validate_invalid_device_id(monkeypatch)` — `device="cuda:99"`,
  `device_count() == 1` → RuntimeError mentioning `device_count=1`.
- `test_skip_sm_check(monkeypatch)` — `CHUK_SKIP_SM_CHECK=1`; no raise.
- `test_skip_sm_check_case_variants(monkeypatch)` — `"TRUE"`, `"Yes"`, `"on"`,
  `" 1 "` all skip; `"yse"` (typo) does NOT skip.
- `test_cuda_device_id_parse_invalid(monkeypatch)` — `CHUK_CUDA_DEVICE_ID=abc`
  → resolves to `"cuda:0"` and logs a warning (captured via `caplog`).
- `test_dtype_policy_bf16(monkeypatch)` — capability `(8, 0)` →
  `backend.preferred_dtype is torch.bfloat16`. Test skeleton (canonical — all
  torch-backend tests that need CUDA semantics follow this shape):

  ```python
  def test_dtype_policy_bf16(monkeypatch):
      import torch
      # IMPORTANT: patch torch.cuda BEFORE constructing the backend so the
      # ctor's _resolve_device sees is_available()=True and chooses cuda:0,
      # and so the lazy preferred_dtype call later sees the mocked capability.
      monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
      monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
      monkeypatch.setattr(torch.cuda, "get_device_capability",
                          lambda dev=None: (8, 0))
      monkeypatch.setattr(torch.cuda, "get_arch_list",
                          lambda: ["sm_80", "compute_80"])
      backend = TorchBackend(device="cuda", check_sm=False)
      assert backend.preferred_dtype is torch.bfloat16
  ```
- `test_dtype_policy_fp16(monkeypatch)` — capability `(7, 5)` → `torch.float16`
  (same skeleton; mock `is_available=True`, `device_count=1`).
- `test_dtype_memoized(monkeypatch)` — access `.preferred_dtype` twice; the
  second access uses a mock that raises on call, proving the value is cached
  and not recomputed.
- `test_dtype_lazy_no_cuda_call_in_init(monkeypatch)` — mock `is_available=True`,
  `device_count=1`, and replace `torch.cuda.get_device_capability` with a
  `MagicMock` that raises if called. Construct `TorchBackend(device="cuda",
  check_sm=False)`; assert the mock was NOT invoked during `__init__`; then
  reset the mock to return `(8, 0)`, access `.preferred_dtype`, and assert the
  mock WAS called exactly once (proves the capability API is deferred until
  first property access, not merely skipped because of a non-cuda device).
- `test_resolve_device_invalid_raises(monkeypatch)` — `CHUK_DEVICE=gpu`
  (invalid) must raise `ValueError` with the literal `"gpu"` in the message
  and the accepted-set `{cpu, mps, cuda, cuda:N}` hint.
- `test_explicit_device_invalid_raises()` — `TorchBackend(device="invalid")`
  must raise `ValueError` (symmetric with env var validation).
- `test_cuda_device_id_empty_falls_back(monkeypatch)` — `CHUK_CUDA_DEVICE_ID=""`
  → parsed as invalid, warning logged, falls back to `"cuda:0"`.
- `test_reject_negative_device_id()` — `TorchBackend(device="cuda:-1")` raises
  `ValueError` at construction time (rejected by `_validate_device_spec`, since
  `"-1".isdigit()` is `False`). Error message must include `"cuda:-1"` and the
  accepted-set hint.
- `test_device_spec_canonicalized()` — `TorchBackend(device="cuda:007")` ends
  up with `.device == "cuda:7"`; two backends built from `"cuda:007"` and
  `"cuda:7"` share the same cache key in §7.1's cache tests.
- `test_validate_device_success_path(monkeypatch)` — happy path:
  `is_available=True`, `device_count=2`, `get_device_capability=(8, 0)`,
  `get_arch_list=["sm_80","compute_80"]`. Build `TorchBackend(device="cuda:1",
  check_sm=True)`, call `.validate_device()` — must return cleanly without
  raising. Protects against regressions where a future guard over-rejects.

All CUDA interactions use `monkeypatch.setattr(backend._torch.cuda, ...)`; no
real GPU required. For stubs of `MLXBackend` / `TorchBackend` used in §7.1,
the stub class MUST implement `validate_device()` (even as a no-op) and a
`name` property returning the matching `BackendType` — otherwise registry-side
attribute lookups fail.

### 7.3 `tests/inference/test_loader_backend.py` (NEW)

- `test_loader_imports_without_mlx(monkeypatch)` — ensure `sys.modules["mlx"]`
  absent via `monkeypatch.setitem(sys.modules, "mlx", None)`, then
  `importlib.import_module("chuk_lazarus.inference.loader")` must succeed.
- `test_dtype_to_torch()` — `DType.BFLOAT16.to_torch() is torch.bfloat16`.
- `test_dtype_to_mlx_lazy(monkeypatch)` — when MLX is unavailable, `to_mlx()`
  raises `ImportError`; when available, returns `mx.bfloat16`.

### 7.4 `tests/inference/test_unified_backend.py` (NEW)

- `test_config_accepts_backend_field()` — `UnifiedPipelineConfig(backend="torch",
  device="cpu")` round-trips via `model_dump()`.
- `test_config_defaults_preserve_macos()` — defaults keep `backend is None`,
  `device is None`.
- `test_pipeline_uses_configured_backend(monkeypatch)` — patch `get_backend`
  in `unified.py` with a `MagicMock` returning a stub backend; patch
  `UnifiedPipeline.from_pretrained`'s model-loading step to a no-op; build a
  minimal pipeline from `UnifiedPipelineConfig(backend="torch", device="cpu",
  cuda_check_sm=False)`. Assert `get_backend` was called once with
  `name="torch", device="cpu", check_sm=False` (keyword match).
- `test_cuda_check_sm_env_consulted_when_config_none(monkeypatch)` — set
  `CHUK_SKIP_SM_CHECK=1` and leave `UnifiedPipelineConfig.cuda_check_sm` as
  its default (`None`); build the pipeline with a mocked `get_backend`, and
  assert the call received `check_sm=None`, so the TorchBackend ctor's env
  var branch (§4.3) runs and disables the check. This verifies §3.1's
  precedence order (explicit > config > env > default) is preserved.
- `test_config_serialization_roundtrip()` — dump an existing (pre-Epic-1)
  config YAML that lacks `backend`/`device`/`cuda_check_sm` and load it via
  `UnifiedPipelineConfig.model_validate` — all three new fields must default
  without error (schema-migration safety).

### 7.5 `tests/inference/test_kv_generator_backend.py` (NEW)

- `test_module_imports_without_mlx(monkeypatch)` — same pattern as loader.
- `test_mask_dtype_matches_mlx(monkeypatch)` — when mlx importable,
  `_mask_dtype() == mx.bfloat16`.
- `test_inject_module_imports_without_mlx(monkeypatch)` — importing
  `chuk_lazarus.inference.context.knowledge.inject` with `sys.modules["mlx"]`
  absent must succeed (covers the second `_MASK_DTYPE` caller listed in §4.7).
- `test_no_stale_mask_dtype_references()` — `subprocess.run(["grep","-rn",
  "_MASK_DTYPE","src/","tests/"])` returns non-zero exit (no matches). This
  is the automated enforcement of the §4.7 caller audit.

### 7.6 `tests/cli/test_infer_backend.py` (NEW)

- `test_cli_parses_backend_flag()` — argparse parses `--backend torch --device
  cpu`; resulting `Namespace` has the attributes.
- `test_cli_threads_into_config(monkeypatch)` — patch `UnifiedPipeline.from_pretrained`
  and assert it is called with `pipeline_config.backend == "torch"`.

### 7.7 Optional real-GPU smoke test

`tests/models_v2/core/backend/test_cuda_smoke.py` (new, gated):

```python
import os, pytest
torch = pytest.importorskip("torch")
if not (torch.cuda.is_available() and os.environ.get("CHUK_CUDA_SMOKE") == "1"):
    pytest.skip("real CUDA smoke test disabled", allow_module_level=True)
```

Allocates a `TorchBackend(device="cuda")`, runs `matmul`, `softmax`,
`scaled_dot_product_attention`, asserts output shapes and dtypes.

---

## 8. Quality Gates

A change is considered shippable when **all** of the following pass:

1. `pytest -m "not slow"` on macOS with `pip install -e '.[mlx,dev]'` — MLX tests
   pass, torch tests skip cleanly.
2. `pytest -m "not slow"` on Linux CPU-only with `pip install -e '.[torch,dev]'`
   — torch tests pass, MLX tests skip cleanly, **no `ImportError` on any module
   import**.
3. Optional: `CHUK_CUDA_SMOKE=1 pytest tests/models_v2/core/backend/test_cuda_smoke.py`
   on a CUDA host passes.
4. `ruff check src tests` is clean.
5. `mypy src` has no *new* errors vs. main.
6. Fresh-venv install works for each extra: `[mlx]`, `[torch]`, `[torch-cuda]`,
   `[all]`. Verify with `python -c "import chuk_lazarus"`.
7. CLI plumbing gate (Epic-1-only, does NOT require Epic 2 model code):
   - `lazarus infer --help` shows the new `--backend` and `--device` flags.
   - Run on a Linux CPU host (CUDA not available — gate #2's environment is
     sufficient) so `validate_device()` trivially passes for `device="cpu"`:
     `CHUK_BACKEND=torch CHUK_DEVICE=cpu lazarus infer <args>` resolves to a
     `TorchBackend` instance with `.device == "cpu"` and `validate_device()`
     returning cleanly (the cuda-branch short-circuits at §4.3 start).
   - The subsequent Epic 2 boundary guard (§4.6.1) must then raise a
     `NotImplementedError` whose message contains BOTH the substrings
     `"torch"` and `"Epic 2"` (case-sensitive substring match — the exact
     message in §4.6.1 satisfies this, but wording tweaks that keep both
     substrings also pass).
   - Gate passes iff: (a) the `--help` output lists the flags, (b) a
     `TorchBackend` is constructed for `device="cpu"`, (c) the
     `NotImplementedError` above is raised. Any earlier failure (import,
     config parsing, backend selection, device validation) fails the gate.
   - Gate is explicitly NOT run on a CUDA host; the CUDA path is covered by
     gate #3.
8. Docs (§4.11) updated: `README.md` has the Backend-selection section and
   `docs/getting-started.md` walks the install matrix and new CLI flags.
9. CI matrix: at least one macOS arm64 runner (`macos-14` or self-hosted
   Apple Silicon) for gate #1, and one Linux x86_64 runner for gate #2. Intel
   macOS runners cannot satisfy gate #1 because MLX requires Apple Silicon;
   if no arm64 runner is available, gate #1 must be run manually on an M-series
   developer machine and the result attached to the Epic 1 PR.

---

## 9. Rollout & Backward Compatibility

- **Default behaviour on macOS** is unchanged: no env var set, no config field
  set → registry auto-detects MLX exactly as today.
- **Default behaviour on Linux** today is "ImportError on import". After Epic 1
  the package imports cleanly and auto-selects torch/CPU.
- **Public API stability.** The documented public surface
  (`get_backend()` / `set_backend()` / `reset_backend()`, `UnifiedPipelineConfig`,
  CLI flags) is preserved; new keyword args are additive with defaults that
  match today's behaviour.
- **Removed re-exports (breaking for undocumented callers):** the incidental
  top-level `mx` symbol that leaked out of `chuk_lazarus.inference.loader` (and
  the `_MASK_DTYPE` constant in `chuk_lazarus.inference.context.kv_generator`)
  are *not* part of the documented API but may have been imported ad-hoc. These
  are removed without a deprecation shim because (a) they are not in any
  module's `__all__`, (b) they were never referenced in README or getting-started
  docs, and (c) any direct import of `mlx` symbols through this package breaks
  the lazy-import rule in §3.4. Callers should `import mlx.core as mx` directly
  and gate the import on `get_backend().name == BackendType.MLX`. An audit
  (`grep -rn "from chuk_lazarus.inference.loader import mx"` and similar) is
  part of the Epic 1 PR checklist; any hit outside the files touched by §4 is
  treated as a public-API break and must be discussed before merge.
- **Environment defaults:** shipping README / `docs/getting-started.md` updates
  are an Epic 1 deliverable so users know about `CHUK_BACKEND`.

---

## 10. Out of Scope (Epic 2+)

- Porting model weight loaders (`inference/loader.py`, `models_v2/families/*`) to
  produce `torch.Tensor` weights. Epic 1 lands the selection plumbing only;
  actually running a model on torch is Epic 2.
- MLX ↔ torch weight conversion / numerical-parity harness.
- Multi-GPU (DDP / tensor parallel) on torch.
- Quantization parity (MLX 4-bit vs. bitsandbytes / AWQ on torch).
- JAX / XLA backend (sketched in `base.py` docstring but not planned).
- `torch.compile` / CUDA Graphs tuning for Blackwell.
