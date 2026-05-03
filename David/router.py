"""Importable wrapper for the temporary space-named central router module."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from typing import Any, List


_MODULE_PATH = Path(__file__).with_name("central router.py")
_SPEC = spec_from_file_location("_david_central_router", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load central router module from {_MODULE_PATH}")

_MODULE = module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

__all__ = tuple(getattr(_MODULE, "__all__", ()))

globals().update({name: getattr(_MODULE, name) for name in __all__})


def __getattr__(name: str) -> Any:
    return getattr(_MODULE, name)


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(dir(_MODULE)))
