from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src"
PKG = SRC / "chuk_lazarus"


def _install_namespace(name: str, path: Path) -> None:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]  # type: ignore[attr-defined]
        sys.modules[name] = module
        return

    if not hasattr(module, "__path__"):
        module.__path__ = [str(path)]  # type: ignore[attr-defined]


def stub_module(name: str, **attrs: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _prime_namespaces() -> None:
    _install_namespace("chuk_lazarus", PKG)
    _install_namespace("chuk_lazarus.training", PKG / "training")
    _install_namespace("chuk_lazarus.training.trainers", PKG / "training" / "trainers")
    _install_namespace("chuk_lazarus.training.losses", PKG / "training" / "losses")
    _install_namespace("chuk_lazarus.training.utils", PKG / "training" / "utils")
    _install_namespace("chuk_lazarus.utils", PKG / "utils")
    _install_namespace("chuk_lazarus.cli", PKG / "cli")
    _install_namespace("chuk_lazarus.cli.commands", PKG / "cli" / "commands")
    _install_namespace("chuk_lazarus.cli.commands.train", PKG / "cli" / "commands" / "train")
    _install_namespace("chuk_lazarus.cli._parsers", PKG / "cli" / "_parsers")
    _install_namespace("chuk_lazarus.models_v2", PKG / "models_v2")
    _install_namespace("chuk_lazarus.models_v2.adapters", PKG / "models_v2" / "adapters")


def load_repo_module(
    dotted_name: str,
    relative_path: str,
    *,
    stubs: dict[str, types.ModuleType] | None = None,
) -> types.ModuleType:
    _prime_namespaces()
    for name, module in (stubs or {}).items():
        sys.modules[name] = module

    sys.modules.pop(dotted_name, None)
    spec = importlib.util.spec_from_file_location(dotted_name, SRC / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module {dotted_name} from {relative_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = module
    spec.loader.exec_module(module)
    return module


def torch_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        tokens = [2 + (ord(ch) % 29) for ch in text]
        return tokens or [self.eos_token_id]

    def decode(self, ids: list[int]) -> str:
        chars = []
        for token in ids:
            if token <= self.eos_token_id:
                continue
            chars.append(chr(97 + ((token - 2) % 26)))
        return "".join(chars)


class TinyTorchLM:  # pragma: no cover - exercised through trainer tests
    def __new__(cls, vocab_size: int = 32, hidden_size: int = 24):
        import torch

        class _Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(vocab_size, hidden_size)
                self.proj = torch.nn.Linear(hidden_size, vocab_size)

            def forward(self, input_ids, cache=None):
                hidden = self.embed(input_ids.long())
                return self.proj(hidden)

        return _Model()


def clone_tiny_lm(model):
    import copy

    return copy.deepcopy(model)


class StaticBatchDataset:
    def __init__(self, batches: list[dict[str, Any]]) -> None:
        self._batches = batches

    def __len__(self) -> int:
        return len(self._batches)

    def iter_batches(self, batch_size: int, shuffle: bool, pad_token_id: int):
        yield from self._batches
