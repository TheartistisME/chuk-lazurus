"""Thin example wrapper around :func:`chuk_lazarus.cross_session_demo.run_demo`.

Usage::

    python examples/cross_session_demo.py \\
        --inputs-root /tmp/xsd/inputs \\
        --checkpoints-root /tmp/xsd/checkpoints \\
        --report-out /tmp/xsd/report.json

All real work happens inside the package; this script exists so the
demo can be run without remembering the ``-m`` invocation form.
"""

from __future__ import annotations

import sys

from chuk_lazarus.cross_session_demo.cli import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
