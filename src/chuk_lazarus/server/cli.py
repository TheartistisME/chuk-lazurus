"""
CLI entry points for the Lazarus inference server.

Two entry paths
---------------
1. ``lazarus serve`` — integrated into the main lazarus CLI.
   Uses ``add_serve_parser()`` to register the subcommand, then
   ``run_serve_cmd()`` as the handler.

2. ``lazarus-serve`` — standalone script registered in pyproject.toml.
   Calls ``main()`` directly with flat argparse args.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import types

from ._compat import SERVER_DEPENDENCY_MESSAGE


def _missing_uvicorn_dependency(*_args, **_kwargs):
    raise ImportError(SERVER_DEPENDENCY_MESSAGE)


# ── Startup debug dump ────────────────────────────────────────────────────────


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KiB"
    if n < 1024 ** 3:
        return f"{n / 1024 ** 2:.1f} MiB"
    return f"{n / 1024 ** 3:.2f} GiB"


def _dump_startup_debug(app, engine, args) -> None:
    """Print a vLLM-style startup banner: routes, runtime, engine, budgets.

    Called once after ``create_app`` so the operator has a full picture of
    what the server is actually serving before uvicorn takes over the stdout.
    Every line is a plain ``print`` (not ``logger.info``) so it appears even
    when the process is launched via ``ensure-qwen-vllm.sh`` which redirects
    stdout to ``~/.pi/agent/run/qwen-vllm.log``.
    """
    print("=" * 78, flush=True)
    print("[startup-debug] registered routes", flush=True)
    # FastAPI keeps every mounted endpoint in app.router.routes.  Print each
    # with its method set — mirrors the vLLM launcher output.
    try:
        for route in app.router.routes:
            methods = sorted(getattr(route, "methods", None) or [])
            path = getattr(route, "path", "?")
            if methods:
                print(
                    f"[startup-debug]   Route: {path}, Methods: {', '.join(methods)}",
                    flush=True,
                )
            else:
                # Mount points (/docs, /redoc, static, etc.) have no methods.
                print(f"[startup-debug]   Mount: {path}", flush=True)
    except Exception as exc:  # pragma: no cover — never block startup on log failure
        print(f"[startup-debug]   <failed to enumerate routes: {exc}>", flush=True)

    # Runtime + engine introspection.
    try:
        pipeline = getattr(engine, "_pipeline", None)
        runtime = getattr(pipeline, "runtime", None)
        model = getattr(pipeline, "model", None)
        model_config = getattr(pipeline, "config", None) or getattr(model, "config", None)
        family = getattr(pipeline, "family_type", None)
        engine_mode = getattr(runtime, "engine_mode", None)
        device = getattr(runtime, "_device", None)
        hot_budget_bytes = getattr(runtime, "_hot_budget_bytes", None)
        session_cache = getattr(runtime, "session_cache", None)
        session_cap = getattr(session_cache, "max_sessions", None) if session_cache is not None else None

        print("[startup-debug] runtime + engine", flush=True)
        print(f"[startup-debug]   served_model_id    : {engine.model_id}", flush=True)
        print(
            f"[startup-debug]   source_model_id    : {getattr(engine, 'source_model_id', engine.model_id)}",
            flush=True,
        )
        print(f"[startup-debug]   family             : {getattr(family, 'value', family)}", flush=True)
        print(f"[startup-debug]   device             : {device}", flush=True)
        print(f"[startup-debug]   backend            : {getattr(runtime, 'backend', None)}", flush=True)
        print(
            f"[startup-debug]   engine_mode        : "
            f"{getattr(engine_mode, 'value', engine_mode)}",
            flush=True,
        )
        print(
            f"[startup-debug]   hot_budget_bytes   : "
            f"{_fmt_bytes(hot_budget_bytes) if hot_budget_bytes else 'unset'}",
            flush=True,
        )
        print(f"[startup-debug]   session_cache_cap  : {session_cap if session_cap is not None else 'disabled'}", flush=True)

        # Config-derived KV sizing (only meaningful for bounded engines).
        if hot_budget_bytes and model is not None:
            try:
                from chuk_lazarus.inference.backends._torch_residual_bounded import (
                    estimate_kv_bytes_per_token,
                )

                kv_per_tok = estimate_kv_bytes_per_token(model)
                max_hot_tokens = int(hot_budget_bytes) // max(kv_per_tok, 1)
                print(
                    f"[startup-debug]   kv_bytes/token     : {_fmt_bytes(kv_per_tok)}  "
                    f"(config-derived)",
                    flush=True,
                )
                print(
                    f"[startup-debug]   max_hot_tokens     : {max_hot_tokens:,}  "
                    f"(= hot_budget ÷ kv_bytes/token)",
                    flush=True,
                )
            except Exception as exc:  # pragma: no cover
                print(f"[startup-debug]   kv estimate        : unavailable ({exc})", flush=True)

        # Text-backbone dims for quick sanity.
        if model_config is not None:
            text_cfg = getattr(model_config, "text_config", None) or model_config
            for attr in (
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "hidden_size",
                "head_dim",
            ):
                val = getattr(text_cfg, attr, None)
                if val is not None:
                    print(f"[startup-debug]   config.{attr:<14}: {val}", flush=True)

        # Parameter count if available.
        param_count = None
        try:
            param_count = sum(int(p.numel()) for p in model.parameters())
        except Exception:
            pass
        if param_count:
            print(f"[startup-debug]   param_count        : {param_count:,}", flush=True)
    except Exception as exc:  # pragma: no cover
        print(f"[startup-debug] <runtime introspection failed: {exc}>", flush=True)

    print(
        f"[startup-debug] cli args               : model={args.model} "
        f"backend={getattr(args, 'backend', None)} device={getattr(args, 'device', None)} "
        f"engine={getattr(args, 'engine', None)} hot_budget_mib={getattr(args, 'hot_budget_mib', None)} "
        f"session_cache_size={getattr(args, 'session_cache_size', None)} "
        f"max_tokens={getattr(args, 'max_tokens', None)} api_key_set={bool(getattr(args, 'api_key', None))}",
        flush=True,
    )
    print("=" * 78, flush=True)


try:
    import uvicorn as _uvicorn  # noqa: F401
except ImportError:
    _uvicorn = types.ModuleType("uvicorn")
    _uvicorn.Config = _missing_uvicorn_dependency
    _uvicorn.Server = _missing_uvicorn_dependency
    sys.modules.setdefault("uvicorn", _uvicorn)

# ── Shared serve logic ────────────────────────────────────────────────────────


def _parse_protocols(raw: str) -> list:
    """Parse a comma-separated protocol string into Protocol enum values."""
    from .app import Protocol

    names = [p.strip().lower() for p in raw.split(",") if p.strip()]
    protocols: list[Protocol] = []
    for name in names:
        try:
            protocols.append(Protocol(name))
        except ValueError:
            valid = ", ".join(p.value for p in Protocol)
            print(f"ERROR: Unknown protocol '{name}'. Valid options: {valid}", file=sys.stderr)
            sys.exit(1)
    return protocols


async def _serve_async(args: argparse.Namespace) -> None:
    """Load the model and start the server (async)."""
    try:
        import uvicorn
    except ImportError:
        print(f"ERROR: {SERVER_DEPENDENCY_MESSAGE}", file=sys.stderr)
        sys.exit(1)
    if (
        getattr(uvicorn, "Config", None) is _missing_uvicorn_dependency
        or getattr(uvicorn, "Server", None) is _missing_uvicorn_dependency
    ):
        print(f"ERROR: {SERVER_DEPENDENCY_MESSAGE}", file=sys.stderr)
        sys.exit(1)

    from .app import create_app
    from .engine import ModelEngine

    protocols = _parse_protocols(args.protocols)
    api_key: str | None = getattr(args, "api_key", None)
    backend: str | None = getattr(args, "backend", None)
    device: str | None = getattr(args, "device", None)
    engine_mode: str | None = getattr(args, "engine", None)
    hot_budget_mib: int | None = getattr(args, "hot_budget_mib", None)
    served_model_name: str | None = getattr(args, "served_model_name", None)
    max_tokens: int = getattr(args, "max_tokens", 512)
    session_cache_size: int | None = getattr(args, "session_cache_size", None)

    print(f"\nLoading model: {args.model}")
    print("=" * 60)
    engine = await ModelEngine.load(
        args.model,
        verbose=True,
        backend=backend,
        device=device,
        engine=engine_mode,
        hot_budget_mib=hot_budget_mib,
        served_model_name=served_model_name,
        session_cache_size=session_cache_size,
    )

    try:
        app = create_app(
            engine,
            protocols=protocols,
            api_key=api_key,
            default_max_tokens=max_tokens,
        )
    except ImportError:
        print(f"ERROR: {SERVER_DEPENDENCY_MESSAGE}", file=sys.stderr)
        sys.exit(1)

    host: str = args.host
    port: int = args.port

    print("\n" + "=" * 60)
    print("Lazarus inference server ready")
    print(f"  Source    : {args.model}")
    print(f"  Served as : {engine.model_id}")
    runtime_engine = getattr(engine._pipeline, "engine_mode", None)
    print(f"  Engine    : {runtime_engine.value if runtime_engine is not None else 'standard'}")
    if hot_budget_mib is not None:
        print(f"  Hot KV    : {hot_budget_mib} MiB")
    print(f"  Protocols : {', '.join(p.value for p in protocols)}")
    print(f"  Base URL  : http://{host}:{port}")
    if "openai" in [p.value for p in protocols]:
        print(f"  OpenAI URL: http://{host}:{port}/v1")
    if api_key:
        print("  Auth      : Bearer token enabled")
    print("=" * 60 + "\n")

    # vLLM-style startup debug dump — routes, runtime, engine, budgets.
    _dump_startup_debug(app, engine, args)

    # ``access_log=True`` + ``log_level="info"`` guarantees per-request access
    # lines (``POST /v1/chat/completions 200 OK``) — matches vLLM's uvicorn
    # configuration so the two stacks produce comparable request-level logs.
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True,
    )
    server = uvicorn.Server(config)
    await server.serve()


def run_serve_cmd(args: argparse.Namespace) -> None:
    """Synchronous handler registered with the main CLI (args.func)."""
    asyncio.run(_serve_async(args))


# ── Subparser registration (for main CLI integration) ─────────────────────────


def add_serve_parser(subparsers) -> argparse.ArgumentParser:
    """Register the ``serve`` subcommand on an existing argparse subparsers."""
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the OpenAI-compatible inference server",
    )
    _add_serve_args(serve_parser)
    serve_parser.set_defaults(func=run_serve_cmd)
    return serve_parser


def _add_serve_args(parser: argparse.ArgumentParser) -> None:
    """Add all serve arguments to an argparse parser."""
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model HuggingFace ID or local path",
    )
    parser.add_argument(
        "--backend",
        choices=("mlx", "torch"),
        default=None,
        help="Optional inference backend override: mlx or torch",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional runtime device override, for example mps or cuda:0",
    )
    parser.add_argument(
        "--engine",
        choices=(
            "standard",
            "kv_direct",
            "bounded_kv_direct",
            "residual_bounded_kv_direct",
        ),
        default=None,
        help=(
            "Optional torch engine override: standard, kv_direct, "
            "bounded_kv_direct, or residual_bounded_kv_direct (bounded hot KV "
            "with residual-seeded rebuild for the Qwen serve path)."
        ),
    )
    parser.add_argument(
        "--hot-budget-mib",
        dest="hot_budget_mib",
        type=int,
        default=None,
        help=(
            "Hot-tier KV budget in MiB — applies to bounded_kv_direct and "
            "residual_bounded_kv_direct."
        ),
    )
    parser.add_argument(
        "--served-model-name",
        dest="served_model_name",
        default=None,
        help="Optional model alias exposed via /v1/models and response payloads",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8080,
        help="Port (default: 8080)",
    )
    parser.add_argument(
        "--protocols",
        default="openai",
        help="Comma-separated protocols to enable: openai,ollama,anthropic (default: openai)",
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        default=None,
        help="Optional bearer token — if set, all requests must include Authorization: Bearer <key>",
    )
    parser.add_argument(
        "--max-tokens",
        dest="max_tokens",
        type=int,
        default=512,
        help="Default max_tokens when callers do not specify (default: 512)",
    )
    parser.add_argument(
        "--session-cache-size",
        dest="session_cache_size",
        type=int,
        default=None,
        help=(
            "Max number of concurrent conversations kept in the WARM "
            "residual session cache (residual_bounded_kv_direct engine "
            "only). Each entry pins one hot KV budget worth of GPU memory. "
            "Default: 8."
        ),
    )


# ── Standalone entry point ────────────────────────────────────────────────────


def main() -> None:
    """Entry point for the ``lazarus-serve`` standalone script."""
    parser = argparse.ArgumentParser(
        prog="lazarus-serve",
        description="Lazarus inference server — OpenAI-compatible local LLM API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  lazarus-serve --model google/gemma-3-1b-it
  lazarus-serve --model ./checkpoints/my-model --port 8080 --api-key secret
  lazarus-serve --model gemma-3-1b --protocols openai,ollama

mcp-cli usage:
  mcp-cli --provider openai --base-url http://localhost:8080/v1 --model <model-id>
        """,
    )
    _add_serve_args(parser)
    args = parser.parse_args()
    asyncio.run(_serve_async(args))
