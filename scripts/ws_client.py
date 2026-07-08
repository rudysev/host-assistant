"""Shared WebSocket client helpers for smoke_test and test_client."""

from __future__ import annotations

import asyncio
import json
import ssl
from typing import Any


def connect_kwargs(url: str) -> dict[str, Any]:
    """Extra ``websockets.connect`` kwargs — ``wss://`` uses TLS without cert verification (LAN dev)."""
    if url.lower().startswith("wss://"):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return {"ssl": ctx}
    return {}


async def wait_for_frame_type(
    ws: Any,
    frame_type: str,
    *,
    deadline: float,
    seen: set[str] | None = None,
) -> dict[str, Any]:
    """Receive until a JSON control frame with ``type == frame_type`` arrives."""
    loop = asyncio.get_running_loop()
    while loop.time() < deadline:
        remaining = deadline - loop.time()
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=min(remaining, 5.0))
        except asyncio.TimeoutError:
            continue

        if isinstance(msg, (bytes, bytearray)):
            continue

        frame = json.loads(msg)
        kind = frame.get("type")
        if kind and seen is not None:
            seen.add(kind)
        if kind == "error":
            raise RuntimeError(frame.get("message", "host error"))
        if kind == frame_type:
            return frame

    raise TimeoutError(f"timed out waiting for {frame_type!r}")
