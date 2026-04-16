"""Lazy command dispatch helpers for introspect parsers."""

from __future__ import annotations

import asyncio
from importlib import import_module


def _load_command(name: str):
    module = import_module("chuk_lazarus.cli.commands.introspect")
    return getattr(module, name)


def run_command(name: str):
    def _runner(args):
        return _load_command(name)(args)

    return _runner


def run_async_command(name: str):
    def _runner(args):
        return asyncio.run(_load_command(name)(args))

    return _runner
