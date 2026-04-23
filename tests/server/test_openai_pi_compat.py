from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from chuk_lazarus.server.app import Protocol, create_app
from chuk_lazarus.server.schemas.internal import (
    FinishReason,
    InternalResponse,
    InternalUsage,
)
from chuk_lazarus.server.schemas.openai import ChatCompletionRequest


def test_chat_request_accepts_pi_style_content_parts():
    request = ChatCompletionRequest.model_validate(
        {
            "model": "qwen3.6-35b-local",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                        {"type": "input_text", "text": "world"},
                    ],
                }
            ],
            "chat_template_kwargs": {"enable_thinking": False},
            "stream_options": {"include_usage": True},
            "store": False,
        }
    )

    internal = request.to_internal(default_max_tokens=256)

    assert internal.messages[0].content == "hello\n[image]\nworld"
    assert internal.chat_template_kwargs == {"enable_thinking": False}
    assert internal.max_tokens == 256


def test_chat_request_accepts_developer_role_alias():
    request = ChatCompletionRequest.model_validate(
        {
            "model": "qwen3.6-35b-local",
            "messages": [{"role": "developer", "content": "system guidance"}],
        }
    )

    internal = request.to_internal(default_max_tokens=128)

    assert internal.messages[0].role.value == "system"
    assert internal.messages[0].content == "system guidance"


def test_chat_completions_route_accepts_pi_style_payload():
    testclient = pytest.importorskip("fastapi.testclient")
    TestClient = testclient.TestClient

    engine = MagicMock()
    engine.model_id = "qwen3.6-35b-local"
    engine.agenerate = AsyncMock(
        return_value=InternalResponse(
            content="ok",
            model="qwen3.6-35b-local",
            finish_reason=FinishReason.STOP,
            usage=InternalUsage(prompt_tokens=4, completion_tokens=1, total_tokens=5),
        )
    )

    app = create_app(engine, protocols=[Protocol.OPENAI], api_key="EMPTY")

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer EMPTY"},
            json={
                "model": "qwen3.6-35b-local",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "hey"}],
                    }
                ],
                "chat_template_kwargs": {"enable_thinking": False},
                "stream": False,
            },
        )

    assert response.status_code == 200
    engine.agenerate.assert_awaited_once()
    internal_request = engine.agenerate.await_args.args[0]
    assert internal_request.messages[0].content == "hey"
    assert internal_request.chat_template_kwargs == {"enable_thinking": False}
