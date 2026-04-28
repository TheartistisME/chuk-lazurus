#!/usr/bin/env python
"""Emit machine-readable baseline-vs-utility selector evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chuk_lazarus.session_retrieval.selector_eval import (
    DEFAULT_FIXTURE_PATH,
    run_selector_eval,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=DEFAULT_FIXTURE_PATH,
        help="Selector eval fixture JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the JSON report.",
    )
    args = parser.parse_args()
    report = run_selector_eval(args.fixture)
    blob = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(blob, encoding="utf-8")
    print(blob, end="")
    return 0 if bool(report["pass"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
