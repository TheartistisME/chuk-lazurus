"""CLI: ``python -m chuk_lazarus.session_close.cli --session-json P --out-dir Q``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chuk_lazarus.chat_loop.session import TurnRecord
from chuk_lazarus.session_close.wind_down import emit_session


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m chuk_lazarus.session_close.cli",
        description=(
            "Emit AUS3000-shaped per-turn-chunk JSON files from a saved chat-loop "
            "session JSON dump."
        ),
    )
    parser.add_argument(
        "--session-json",
        required=True,
        type=Path,
        help=(
            "Path to a JSON file of shape "
            "{\"session_id\": str, \"turns\": [TurnRecord-json-dict, ...]}"
        ),
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Directory under which <session_id>/<NNN>_<ti>_<ci>.json files are written.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    session_json_path: Path = args.session_json
    out_dir: Path = args.out_dir

    with session_json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise SystemExit(
            f"Expected a JSON object in {session_json_path}, got {type(payload).__name__}"
        )

    session_id = payload.get("session_id")
    turns_raw = payload.get("turns")
    if not isinstance(session_id, str) or not session_id:
        raise SystemExit("Missing or empty 'session_id' in session JSON.")
    if not isinstance(turns_raw, list):
        raise SystemExit("Missing or non-list 'turns' in session JSON.")

    turns = [TurnRecord.model_validate(d) for d in turns_raw]

    written = emit_session(turns, session_id, out_dir)

    session_dir = Path(out_dir) / session_id
    print(f"Wrote {len(written)} record(s) to {session_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
