"""Shared backend-dispatch helpers for the ``introspection`` package.

Owned by EWS-5. EWS-6 and EWS-7 import these symbols but must not edit
this module. The helpers are deliberately small and allocation-free where
possible so they can sit on hot paths inside hooks, patchers, and logit
lenses without changing MLX behaviour.

Public surface (see §EWS-5 acceptance in docs/refactor/dual-backend-cuda-all-buckets/03-workstreams.md):
    - ``to_backend_tensor(x, backend=None)``: numpy/scalar/tensor -> native backend tensor.
    - ``from_backend_tensor(x, backend=None)``: backend tensor -> numpy ndarray.
    - ``backend_matmul(a, b, backend=None)``: matmul routed through the active backend.
    - ``register_hook(backend, module, hook)``: attach a forward-hook that fires
      once per forward pass; returns a callable that detaches the hook.

All helpers preserve lazy-import semantics — MLX is only imported when the
backend is MLX, and torch is only imported when the backend is torch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

    from chuk_lazarus.models_v2.core.backend.base import Backend


__all__ = [
    "to_backend_tensor",
    "from_backend_tensor",
    "backend_matmul",
    "register_hook",
    "lazy_mx",
    "lazy_nn",
]


class _LazyModule:
    """Proxy that defers ``import <name>`` to first attribute access.

    Lets introspection framework files keep writing ``mx.array(...)`` and
    ``nn.MultiHeadAttention`` at module scope without importing MLX unless
    the MLX code path actually runs. Imports under the torch backend stay
    clean (see ``tests/introspection/test_*_backend.py``).
    """

    def __init__(self, name: str) -> None:
        self.__dict__["_name"] = name
        self.__dict__["_mod"] = None

    def _resolve(self) -> Any:
        mod = self.__dict__["_mod"]
        if mod is None:
            import importlib

            mod = importlib.import_module(self.__dict__["_name"])
            self.__dict__["_mod"] = mod
        return mod

    def __getattr__(self, attr: str) -> Any:
        if attr.startswith("_") and attr not in ("_resolve",):
            raise AttributeError(attr)
        return getattr(self._resolve(), attr)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<lazy {self.__dict__['_name']}>"


lazy_mx = _LazyModule("mlx.core")
lazy_nn = _LazyModule("mlx.nn")


def _resolve_backend(backend: "Backend | None") -> "Backend":
    if backend is not None:
        return backend
    from chuk_lazarus.models_v2.core.backend import get_backend

    return get_backend()


def to_backend_tensor(x: Any, backend: "Backend | None" = None) -> Any:
    """Materialise *x* as a tensor on the active backend.

    Accepts Python scalars, numpy arrays, or an already-native tensor. No-op
    when *x* is already on the requested backend.
    """

    be = _resolve_backend(backend)
    name = be.name

    if name == "mlx":
        import mlx.core as mx

        if isinstance(x, mx.array):
            return x
        return mx.array(x)
    if name == "torch":
        import numpy as _np
        import torch

        if isinstance(x, torch.Tensor):
            return x.to(be.device) if str(x.device) != be.device else x
        if isinstance(x, _np.ndarray):
            return torch.from_numpy(x).to(be.device)
        return torch.as_tensor(x, device=be.device)
    raise NotImplementedError(f"to_backend_tensor: unsupported backend {name!r}")


def from_backend_tensor(x: Any, backend: "Backend | None" = None) -> "np.ndarray":
    """Return *x* as a numpy ndarray regardless of originating backend."""

    import numpy as np

    be = _resolve_backend(backend)
    name = be.name

    if name == "mlx":
        import mlx.core as mx

        if isinstance(x, np.ndarray):
            return x
        if isinstance(x, mx.array):
            return np.array(x)
        return np.asarray(x)
    if name == "torch":
        import torch

        if isinstance(x, np.ndarray):
            return x
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)
    raise NotImplementedError(f"from_backend_tensor: unsupported backend {name!r}")


def backend_matmul(a: Any, b: Any, backend: "Backend | None" = None) -> Any:
    """Matrix multiplication routed through the active backend.

    Inputs are coerced via :func:`to_backend_tensor` so callers can pass
    numpy arrays or native tensors interchangeably.
    """

    be = _resolve_backend(backend)
    a_t = to_backend_tensor(a, backend=be)
    b_t = to_backend_tensor(b, backend=be)
    return be.matmul(a_t, b_t)


def register_hook(backend: "Backend | Any", module: Any, hook: Callable[..., Any]) -> Callable[[], None]:
    """Register *hook* on *module* and return a detach callable.

    The torch arm uses ``module.register_forward_hook`` directly. The MLX arm
    implements a wrapper: MLX modules are callable rather than PyTorch
    ``nn.Module`` instances with a hook registry, so we monkey-patch
    ``__call__`` and restore it from the detach callable.

    *backend* may be a ``Backend`` instance or the backend name string — the
    helper accepts both to reduce friction at call sites.
    """

    name = getattr(backend, "name", backend)
    if isinstance(name, str):
        backend_name = name
    else:  # pragma: no cover - defensive
        backend_name = str(name)

    if backend_name == "torch":
        handle = module.register_forward_hook(hook)

        def _detach_torch() -> None:
            handle.remove()

        return _detach_torch

    if backend_name == "mlx":
        if not callable(module):
            raise TypeError(
                "MLX register_hook requires a callable module (e.g. nn.Module); "
                f"got {type(module).__name__}"
            )
        original_call = module.__call__

        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            output = original_call(*args, **kwargs)
            try:
                hook(module, args, output)
            except Exception:  # noqa: BLE001 - hooks must not break forward
                pass
            return output

        try:
            module.__call__ = _wrapped  # type: ignore[method-assign]
        except (AttributeError, TypeError):
            # Some MLX modules use __call__ from class; fall back to bound attr
            object.__setattr__(module, "__call__", _wrapped)

        def _detach_mlx() -> None:
            try:
                module.__call__ = original_call  # type: ignore[method-assign]
            except (AttributeError, TypeError):
                object.__setattr__(module, "__call__", original_call)

        return _detach_mlx

    raise NotImplementedError(f"register_hook: unsupported backend {backend_name!r}")
