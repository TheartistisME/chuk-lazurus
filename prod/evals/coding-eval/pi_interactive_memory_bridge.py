#!/usr/bin/env python3
"""OpenAI-compatible Pi provider backed by interactive_memory_chat.MemoryChat.

This bridge exists for one very specific test shape: let Pi keep its normal
coding-tool loop while every model turn is produced by the Lazarus interactive
memory runtime. In other words:

    Pi tools/read-edit-bash loop <-> OpenAI chat API shim <-> MemoryChat

The shim intentionally imports and drives ``scripts/interactive_memory_chat.py``
instead of using ``lazarus-serve`` so A/B coding evals exercise the same
Gemma-4, store, memory-mode, and telemetry path as the interactive REPL.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import re
import sys
import threading
import time
from contextlib import redirect_stderr, redirect_stdout
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from uuid import uuid4


def find_repo_root(start: Path) -> Path:
    for parent in start.resolve().parents:
        if (parent / "pyproject.toml").exists() and (parent / "scripts").is_dir():
            return parent
    raise RuntimeError(f"could not find repo root from {start}")


REPO_ROOT = find_repo_root(Path(__file__))
SRC_ROOT = REPO_ROOT / "src"
CHAT_SCRIPT = REPO_ROOT / "scripts" / "interactive_memory_chat.py"
TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
FENCE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def load_interactive_memory_chat() -> Any:
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    spec = importlib.util.spec_from_file_location("interactive_memory_chat", CHAT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {CHAT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["interactive_memory_chat"] = module
    spec.loader.exec_module(module)
    return module


def text_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif "content" in block:
                    parts.append(str(block.get("content", "")))
        return "\n".join(part for part in parts if part)
    return str(content)


def parse_tools(raw_tools: Any) -> list[dict[str, Any]]:
    tools = []
    if not isinstance(raw_tools, list):
        return tools
    for item in raw_tools:
        if not isinstance(item, dict):
            continue
        function = item.get("function") if item.get("type") == "function" else item
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not name:
            continue
        tools.append(
            {
                "name": str(name),
                "description": str(function.get("description", "")),
                "parameters": function.get("parameters") or {},
            }
        )
    return tools


def message_summary(messages: list[dict[str, Any]], *, limit: int = 10) -> str:
    rendered = []
    for msg in messages[-limit:]:
        role = str(msg.get("role", "unknown"))
        content = text_content(msg.get("content"))
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            rendered_calls = []
            for call in tool_calls:
                fn = call.get("function", {}) if isinstance(call, dict) else {}
                rendered_calls.append(
                    {
                        "name": fn.get("name"),
                        "arguments": fn.get("arguments"),
                    }
                )
            content = (content + "\n" if content else "") + (
                "assistant tool calls: " + json.dumps(rendered_calls)
            )
        if role == "tool":
            name = msg.get("name") or msg.get("tool_call_id") or "tool"
            content = f"{name}: {content}"
        rendered.append(f"{role.upper()}:\n{content}".strip())
    return "\n\n".join(rendered)


def build_memory_prompt(
    payload: dict[str, Any],
    *,
    target_file: str | None,
) -> str:
    messages = payload.get("messages") or []
    if not isinstance(messages, list):
        messages = []
    tools = parse_tools(payload.get("tools"))
    tool_text = json.dumps(tools, indent=2, sort_keys=True)
    target_note = target_file or "the requested target file"
    return (
        "You are the model brain inside Pi, a coding agent with tools. "
        "You are NOT allowed to claim you edited files yourself. If a file "
        "needs to change, call a Pi tool.\n\n"
        "Use this exact tool-call syntax when you need a tool:\n"
        "<tool_call>{\"name\":\"write\",\"arguments\":{\"path\":\"solution.py\","
        "\"content\":\"...\"}}</tool_call>\n\n"
        "For shell checks, call bash:\n"
        "<tool_call>{\"name\":\"bash\",\"arguments\":{\"command\":\"python3 "
        "visible_tests.py\",\"timeout\":60}}</tool_call>\n\n"
        "For targeted edits, call edit with path and edits[].oldText/newText. "
        "When replacing the full fixture solution, prefer write.\n\n"
        f"Target file for this eval: {target_note}\n\n"
        "Available Pi tools as supplied by Pi in this request:\n"
        f"{tool_text}\n\n"
        "Conversation from Pi:\n"
        f"{message_summary(messages)}\n\n"
        "Return exactly one tool call when another tool action is needed. "
        "When the coding task is done, return a short final answer with no "
        "tool_call block."
    )


def parse_tool_call(text: str) -> dict[str, Any] | None:
    match = TOOL_RE.search(text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or "name" not in parsed:
        return None
    arguments = parsed.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"raw": arguments}
    if not isinstance(arguments, dict):
        arguments = {"value": arguments}
    return {"name": str(parsed["name"]), "arguments": arguments}


def extract_python_solution(text: str, function_name: str | None) -> str | None:
    for candidate in reversed(FENCE_RE.findall(text)):
        if function_name is None or f"def {function_name}" in candidate:
            return candidate.strip() + "\n"
    if function_name:
        index = text.find(f"def {function_name}")
        if index >= 0:
            return text[index:].strip() + "\n"
    return None


def cleaned_content(text: str) -> str:
    return TOOL_RE.sub("", text).strip()


class MemoryChatBackend:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.lock = threading.Lock()
        self.imc = load_interactive_memory_chat()
        self.chat = self.imc.MemoryChat(
            store_root=args.store_root,
            model_path=args.model_path,
            max_new_tokens=args.max_new_tokens,
            memory_mode=args.memory_mode,
            device=args.device,
            generation_engine=args.generation_engine,
            hot_budget_mib=args.hot_budget_mib,
            session_cache_size=args.session_cache_size,
            memory_profile=args.memory_profile,
        )
        self.chat.load_model()
        self._plant_lessons_if_needed()
        self.chat.maybe_load_retriever()
        self.chat.start_new_session()

    def _plant_lessons_if_needed(self) -> None:
        lessons_path = self.args.store_root / "lessons.md"
        if not lessons_path.exists():
            return
        if any((self.args.store_root / "checkpoints").glob("*/torch_store/manifest.json")):
            return
        lessons = []
        for raw_line in lessons_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            lessons.append(line[2:].strip() if line.startswith("- ") else line)
        if not lessons:
            return

        self.chat.start_new_session()
        from chuk_lazarus.inference.chat import Role

        for index, lesson in enumerate(lessons, start=1):
            text = (
                f"Project coding memory lesson {index}: {lesson}\n"
                "This is durable project behavior for future coding tasks."
            )
            self.chat.history.add_user(text)
            turn = self.chat.session.begin_turn(Role.USER, text)
            self.chat.session.finish_turn(turn)
            self.chat._capture_turn_text_live(turn)
        self.chat.save_current_session(rebuild_retriever=True)

    def generate(self, prompt: str) -> tuple[str, Any]:
        with self.lock:
            buffer = io.StringIO()
            with redirect_stdout(buffer), redirect_stderr(buffer):
                if self.chat.retriever is None or self.chat.memory_mode == "off":
                    meta = self.chat.plain_chat_turn(prompt)
                else:
                    meta = self.chat.recall_chat_turn(prompt)
            answer = str(getattr(meta, "generated_answer", "") or "")
            if not answer:
                answer = buffer.getvalue().strip()
            self._write_telemetry(meta)
            return answer, meta

    def _write_telemetry(self, meta: Any) -> None:
        telemetry_file = self.args.telemetry_file
        if telemetry_file is None:
            return
        telemetry_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ts": time.time(),
            "memory_used": bool(getattr(meta, "memory_used", False)),
            "mode": getattr(meta, "mode", None),
            "routing_mode": getattr(meta, "routing_mode", None),
            "source_session": getattr(meta, "source_session", None),
            "window_id": getattr(meta, "window_id", None),
            "kv_direct_active": bool(getattr(meta, "kv_direct_active", False)),
            "no_silent_fallback": bool(getattr(meta, "no_silent_fallback", False)),
        }
        with telemetry_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def make_tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"call_{uuid4().hex[:16]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, separators=(",", ":")),
        },
    }


def serialize_nonstream_response(
    *,
    model: str,
    content: str | None,
    tool_call: dict[str, Any] | None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant"}
    finish_reason = "stop"
    if tool_call is not None:
        message["content"] = None
        message["tool_calls"] = [tool_call]
        finish_reason = "tool_calls"
    else:
        message["content"] = content or ""
    return {
        "id": f"chatcmpl_{uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
    }


def stream_chunks(model: str, content: str | None, tool_call: dict[str, Any] | None) -> list[dict[str, Any]]:
    chunk_id = f"chatcmpl_{uuid4().hex}"
    created = int(time.time())

    def chunk(delta: dict[str, Any], finish_reason: str | None = None) -> dict[str, Any]:
        return {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    chunks = [chunk({"role": "assistant"})]
    if tool_call is not None:
        chunks.append(
            chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": tool_call["id"],
                            "type": "function",
                            "function": {
                                "name": tool_call["function"]["name"],
                                "arguments": tool_call["function"]["arguments"],
                            },
                        }
                    ]
                }
            )
        )
        chunks.append(chunk({}, "tool_calls"))
    else:
        chunks.append(chunk({"content": content or ""}))
        chunks.append(chunk({}, "stop"))
    return chunks


class BridgeHandler(BaseHTTPRequestHandler):
    backend: MemoryChatBackend
    model_id: str
    target_file: str | None
    function_name: str | None

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API.
        if self.path in {"/health", "/v1/health"}:
            self._json({"status": "ok"})
            return
        if self.path == "/v1/models":
            self._json(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.model_id,
                            "object": "model",
                            "created": 0,
                            "owned_by": "lazarus",
                        }
                    ],
                }
            )
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API.
        if self.path != "/v1/chat/completions":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except Exception as exc:  # noqa: BLE001 - bad client payload.
            self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return

        prompt = build_memory_prompt(payload, target_file=self.target_file)
        answer, _meta = self.backend.generate(prompt)
        parsed = parse_tool_call(answer)
        recent_messages = payload.get("messages") if isinstance(payload, dict) else []
        latest_role = (
            str(recent_messages[-1].get("role", ""))
            if isinstance(recent_messages, list) and recent_messages
            else ""
        )
        if parsed is None and latest_role != "tool":
            code = extract_python_solution(answer, self.function_name)
            if code is not None and self.target_file:
                parsed = {"name": "write", "arguments": {"path": self.target_file, "content": code}}

        tool_call = (
            make_tool_call(parsed["name"], parsed["arguments"])
            if parsed is not None
            else None
        )
        content = cleaned_content(answer) if tool_call is None else None
        model = str(payload.get("model") or self.model_id)
        if payload.get("stream"):
            self._sse(stream_chunks(model, content, tool_call))
        else:
            self._json(
                serialize_nonstream_response(
                    model=model,
                    content=content,
                    tool_call=tool_call,
                )
            )

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[pi-memory-bridge] {self.address_string()} {fmt % args}", flush=True)

    def _json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _sse(self, chunks: list[dict[str, Any]]) -> None:
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.end_headers()
        for payload in chunks:
            self.wfile.write(b"data: ")
            self.wfile.write(json.dumps(payload).encode("utf-8"))
            self.wfile.write(b"\n\n")
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pi OpenAI-compatible provider backed by MemoryChat.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--model-id", default="lazarus-memory-gemma4")
    parser.add_argument("--model-path", default=os.environ.get("LAZARUS_MODEL"))
    parser.add_argument("--max-new-tokens", type=int, default=800)
    parser.add_argument("--memory-profile", default="plug_and_play")
    parser.add_argument("--memory-mode", default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--generation-engine", default="standard")
    parser.add_argument("--hot-budget-mib", type=int, default=None)
    parser.add_argument("--session-cache-size", type=int, default=None)
    parser.add_argument("--target-file", default=None)
    parser.add_argument("--function-name", default=None)
    parser.add_argument("--telemetry-file", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    backend = MemoryChatBackend(args)
    BridgeHandler.backend = backend
    BridgeHandler.model_id = args.model_id
    BridgeHandler.target_file = args.target_file
    BridgeHandler.function_name = args.function_name
    server = ThreadingHTTPServer((args.host, args.port), BridgeHandler)
    print(
        f"[pi-memory-bridge] serving {args.model_id} at "
        f"http://{args.host}:{args.port}/v1",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
