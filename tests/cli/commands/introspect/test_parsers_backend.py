"""EWS-7 backend-flag regression for every ``introspect`` subcommand.

Asserts the shared ``add_backend_flags`` helper (EWS-0 single source of
truth) is wired onto every leaf subparser under ``chuk-lazarus introspect``
with the Epic 2 flag contract (``--backend`` from ``BACKEND_CHOICES``,
``--device`` optional).
"""

from __future__ import annotations

import argparse

import pytest

from chuk_lazarus.cli._parsers._introspect import register_introspect_parsers
from chuk_lazarus.cli.commands._base import BACKEND_CHOICES


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chuk-lazarus")
    sub = parser.add_subparsers(dest="command")
    register_introspect_parsers(sub)
    return parser


def _leaf_subparsers(parser: argparse.ArgumentParser):
    subs = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    if not subs:
        yield (), parser
        return
    for sa in subs:
        for name, sp in sa.choices.items():
            for path, leaf in _leaf_subparsers(sp):
                yield (name, *path), leaf


def _all_leaves():
    parser = _build_parser()
    # Skip the outer "command" subparser action — start from the introspect subtree.
    subs = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)][0]
    introspect = subs.choices["introspect"]
    return list(_leaf_subparsers(introspect))


LEAVES = _all_leaves()


def test_has_leaves() -> None:
    assert LEAVES, "expected at least one introspect leaf subparser"


@pytest.mark.parametrize("path,leaf", LEAVES, ids=[" ".join(p) for p, _ in LEAVES])
def test_backend_flag_registered(path, leaf) -> None:
    dests = {a.dest for a in leaf._actions}
    assert "backend" in dests, f"introspect {' '.join(path)} missing --backend"
    action = next(a for a in leaf._actions if a.dest == "backend")
    assert tuple(action.choices) == BACKEND_CHOICES


@pytest.mark.parametrize("path,leaf", LEAVES, ids=[" ".join(p) for p, _ in LEAVES])
def test_device_flag_registered(path, leaf) -> None:
    dests = {a.dest for a in leaf._actions}
    assert "device" in dests, f"introspect {' '.join(path)} missing --device"


def test_backend_defaults_to_none() -> None:
    # Pick one representative subcommand; default should be ``None`` so that
    # env-var resolution (``CHUK_BACKEND``) still wins when the flag is omitted.
    leaf = dict(LEAVES)[("ablate",)]
    action = next(a for a in leaf._actions if a.dest == "backend")
    assert action.default is None


def test_rejects_unknown_backend() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["introspect", "ablate", "--model", "x", "--backend", "jax"]
        )
