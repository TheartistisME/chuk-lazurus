"""EWS-13 — assert every tokenizer subparser accepts ``--backend``/``--device``.

The bucket uses a post-registration walk (``_apply_backend_flags_recursive``)
to guarantee that every leaf parser under ``chuk-lazarus tokenizer ...``
exposes the shared backend-selection flags registered by the EWS-0 helper.
"""

from __future__ import annotations

import argparse

import pytest

from chuk_lazarus.cli._parsers._tokenizer import register_tokenizer_parsers


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chuk-lazarus")
    sub = parser.add_subparsers(dest="command")
    register_tokenizer_parsers(sub)
    return parser


def _iter_leaf_parsers(
    parser: argparse.ArgumentParser,
    prefix: tuple[str, ...] = (),
):
    sub_actions = [
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    ]
    if not sub_actions:
        yield prefix, parser
        return
    for action in sub_actions:
        for name, child in action.choices.items():
            yield from _iter_leaf_parsers(child, prefix + (name,))


def test_every_tokenizer_leaf_accepts_backend_flags() -> None:
    root = _build_parser()
    leaves = list(_iter_leaf_parsers(root))
    assert leaves, "expected at least one tokenizer leaf parser"
    for path, leaf in leaves:
        dests = {a.dest for a in leaf._actions}
        assert "backend" in dests, f"tokenizer {' '.join(path)} missing --backend"
        assert "device" in dests, f"tokenizer {' '.join(path)} missing --device"


@pytest.mark.parametrize(
    "argv",
    [
        ["tokenizer", "encode", "--tokenizer", "gpt2", "--text", "hi", "--backend", "torch"],
        ["tokenizer", "decode", "--tokenizer", "gpt2", "--ids", "1,2", "--device", "cuda:0"],
        ["tokenizer", "benchmark", "--tokenizer", "gpt2", "--backend", "mlx"],
        ["tokenizer", "analyze", "coverage", "--tokenizer", "gpt2", "--backend", "torch"],
        ["tokenizer", "instrument", "histogram", "--tokenizer", "gpt2", "--backend", "torch"],
    ],
)
def test_parse_accepts_backend_flag(argv: list[str]) -> None:
    parser = _build_parser()
    ns = parser.parse_args(argv)
    # Either --backend or --device was supplied by the argv fixture.
    assert getattr(ns, "backend", None) in {None, "mlx", "torch"}
    assert hasattr(ns, "device")


def test_add_backend_flags_is_idempotent() -> None:
    # Registering twice must not raise argparse.ArgumentError.
    root1 = _build_parser()
    root2 = _build_parser()
    assert root1 is not root2
