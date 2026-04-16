"""
Async helpers for warming residual state files into memory.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from .storage import load_residual_state
from .types import ResidualState


async def prefetch_residual_state(path: str | Path) -> ResidualState:
    """Warm a residual state file into system RAM, then parse it off-thread."""
    input_path = Path(path)

    try:
        import aiofiles
    except ImportError:
        aiofiles = None

    if aiofiles is not None:
        async with aiofiles.open(input_path, "rb") as handle:
            await handle.read(1)

    return await asyncio.to_thread(load_residual_state, input_path)
