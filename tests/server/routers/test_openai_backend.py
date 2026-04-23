"""EWS-8: lazy-import assertion for server/routers/openai.py."""

from __future__ import annotations

import subprocess
import sys

import pytest


def test_openai_router_does_not_import_mlx():
    code = (
        "import sys, importlib\n"
        "importlib.import_module('chuk_lazarus.server.routers.openai')\n"
        "bad = [k for k in sys.modules if k in ('mlx', 'mlx_lm')]\n"
        "assert not bad, bad\n"
    )
    res = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert res.returncode == 0, res.stderr


def test_openai_router_exports_router():
    pytest.importorskip(
        "fastapi",
        reason="route export assertions require the optional server dependency",
    )

    from chuk_lazarus.server.routers.openai import router

    assert router is not None
    paths = {r.path for r in router.routes}
    assert any("chat/completions" in p for p in paths)


def test_chat_completion_request_accepts_structured_user_content():
    from chuk_lazarus.server.schemas.openai import ChatCompletionRequest

    request = ChatCompletionRequest.model_validate(
        {
            "model": "qwen3.6-35b-local",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hi"},
                        {"type": "text", "text": "there"},
                    ],
                }
            ],
            "chat_template_kwargs": {"enable_thinking": True},
        }
    )

    internal = request.to_internal(default_max_tokens=321)

    assert internal.messages[0].content == "hi\nthere"
    assert internal.chat_template_kwargs == {"enable_thinking": True}
    assert internal.max_tokens == 321
