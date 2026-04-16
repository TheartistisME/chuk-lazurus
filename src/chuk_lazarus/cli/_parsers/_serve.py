"""Serve subcommand parser — thin adapter onto ``chuk_lazarus.server.cli``.

EWS-0 seeded this file; EWS-8 now owns it and wires ``add_backend_flags``
into the subparser after ``add_serve_parser`` has registered the serve
arguments.  ``chuk_lazarus.server.cli._add_serve_args`` itself also calls
``add_backend_flags`` (idempotent) so the ``lazarus-serve`` standalone
entrypoint honours the same flags.
"""

from __future__ import annotations


def register_serve_parser(subparsers) -> None:
    """Register the ``serve`` subcommand if server dependencies are installed.

    Silently returns when the optional ``[server]`` extra is not installed —
    mirroring the behaviour of the lazy registrar in ``cli.main``.
    """

    try:
        from chuk_lazarus.server.cli import add_serve_parser
    except ImportError:
        return None

    serve_parser = add_serve_parser(subparsers)

    # Idempotent re-registration — guarantees ``--backend`` / ``--device``
    # are wired onto the ``serve`` subparser even if future refactors drop
    # the call site inside ``server.cli._add_serve_args``.
    from chuk_lazarus.cli.commands._base import add_backend_flags

    add_backend_flags(serve_parser)
    return None


__all__ = ["register_serve_parser"]
