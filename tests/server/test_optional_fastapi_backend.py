"""Import-clean regression coverage when FastAPI is unavailable."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
PACKAGE_ROOT = SRC_ROOT / "chuk_lazarus"


def _blocked_fastapi_code(body: str) -> str:
    return textwrap.dedent(
        f"""
        import importlib
        import importlib.abc
        import pathlib
        import sys

        src_root = pathlib.Path({str(SRC_ROOT)!r})
        package_root = pathlib.Path({str(PACKAGE_ROOT)!r})

        sys.path.insert(0, str(src_root))
        sys.path.insert(0, str(package_root))

        class _BlockFastAPI(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "fastapi" or fullname.startswith("fastapi."):
                    raise ModuleNotFoundError(fullname)
                return None

        sys.meta_path.insert(0, _BlockFastAPI())

        {body}
        """
    )


def _run_without_fastapi(body: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SRC_ROOT), str(PACKAGE_ROOT), env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    return subprocess.run(
        [sys.executable, "-c", _blocked_fastapi_code(body)],
        capture_output=True,
        text=True,
        env=env,
    )


def test_server_modules_import_cleanly_without_fastapi() -> None:
    res = _run_without_fastapi(
        """
        modules = [
            "chuk_lazarus.server",
            "chuk_lazarus.server.app",
            "chuk_lazarus.server.cli",
            "chuk_lazarus.server.routers",
            "chuk_lazarus.server.routers.openai",
            "chuk_lazarus.server.routers.ollama",
            "chuk_lazarus.server.routers.anthropic",
            "chuk_lazarus.server.schemas",
            "chuk_lazarus.server.schemas.openai",
            "server",
            "server.app",
            "server.cli",
            "server.routers",
            "server.routers.openai",
            "server.routers.ollama",
            "server.routers.anthropic",
            "server.schemas",
            "server.schemas.openai",
        ]

        for module_name in modules:
            module = importlib.import_module(module_name)
            assert module.__name__ == module_name

        from chuk_lazarus.server import ModelEngine, Protocol, create_app
        from server import Protocol as bare_protocol
        from server.app import create_app as bare_create_app

        assert ModelEngine.__name__ == "ModelEngine"
        assert Protocol.OPENAI.value == "openai"
        assert bare_protocol.OPENAI.value == "openai"
        assert callable(create_app)
        assert callable(bare_create_app)
        assert "fastapi" not in sys.modules
        """
    )

    assert res.returncode == 0, (
        "server modules are not import-clean without FastAPI\n"
        f"stdout={res.stdout}\nstderr={res.stderr}"
    )


def test_create_app_raises_clear_error_without_fastapi() -> None:
    res = _run_without_fastapi(
        """
        from chuk_lazarus.server.app import create_app

        class DummyEngine:
            model_id = "fake/model"

        try:
            create_app(DummyEngine())
        except ImportError as exc:
            message = str(exc)
        else:
            raise AssertionError("create_app() should fail without FastAPI")

        assert "chuk-lazarus[server]" in message
        assert "uv add" in message
        """
    )

    assert res.returncode == 0, (
        "create_app does not raise the expected optional dependency error\n"
        f"stdout={res.stdout}\nstderr={res.stderr}"
    )


def test_router_proxy_raises_clear_error_without_fastapi() -> None:
    res = _run_without_fastapi(
        """
        from chuk_lazarus.server.routers.openai import router

        try:
            _ = router.routes
        except ImportError as exc:
            message = str(exc)
        else:
            raise AssertionError("router proxy should fail when FastAPI is missing")

        assert "chuk-lazarus[server]" in message
        """
    )

    assert res.returncode == 0, (
        "router placeholder does not defer FastAPI import cleanly\n"
        f"stdout={res.stdout}\nstderr={res.stderr}"
    )
