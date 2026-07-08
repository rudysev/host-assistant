"""Smoke test against a running host — uses user_text (skips STT) for a fast path.

    # terminal 1
    python -m host_assistant

    # terminal 2
    python -m scripts.smoke_test [wss://localhost:8080]

Exits 0 when the host returns ready, model_generating, output_transcript, and
turn_complete for a short typed turn. Much faster than streaming a WAV through STT.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging

from host_assistant.logging_config import configure_logging
from scripts.ws_client import connect_kwargs, wait_for_frame_type

log = logging.getLogger(__name__)

REQUIRED_FRAME_TYPES = frozenset({"ready", "model_generating", "output_transcript", "turn_complete"})

SETUP = {
    "type": "setup",
    "systemPrompt": "You are a helpful assistant. Reply in one short sentence.",
    "tools": [],
}

USER_TURN = {
    "type": "user_text",
    "text": "Say hello in one short sentence.",
}


def smoke_succeeded(seen: set[str]) -> bool:
    """True when every required host→app control frame was observed."""
    return REQUIRED_FRAME_TYPES <= seen


async def run_smoke(url: str, *, timeout_s: float = 120.0) -> set[str]:
    import websockets

    seen: set[str] = set()
    deadline = asyncio.get_running_loop().time() + timeout_s

    async with websockets.connect(url, max_size=None, open_timeout=10, **connect_kwargs(url)) as ws:
        await ws.send(json.dumps(SETUP))
        ready = await wait_for_frame_type(ws, "ready", deadline=deadline, seen=seen)
        log.info("<- %s", ready)

        await ws.send(json.dumps(USER_TURN))

        while asyncio.get_running_loop().time() < deadline:
            remaining = deadline - asyncio.get_running_loop().time()
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=min(remaining, 5.0))
            except asyncio.TimeoutError:
                continue

            if isinstance(msg, (bytes, bytearray)):
                continue

            frame = json.loads(msg)
            kind = frame.get("type")
            if kind:
                seen.add(kind)
                log.info("<- %s", frame)
            if kind == "error":
                raise RuntimeError(frame.get("message", "host error"))
            if smoke_succeeded(seen):
                return seen

    return seen


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description="Smoke-test a running host-assistant instance.")
    parser.add_argument("url", nargs="?", default="wss://localhost:8080")
    parser.add_argument("--timeout", type=float, default=120.0, help="seconds to wait for a full turn")
    args = parser.parse_args(argv)

    try:
        seen = asyncio.run(run_smoke(args.url, timeout_s=args.timeout))
    except Exception as e:  # noqa: BLE001 - smoke script reports any failure to the user
        log.error("smoke test failed: %s", e)
        return 1

    if smoke_succeeded(seen):
        log.info("smoke test passed")
        return 0

    missing = sorted(REQUIRED_FRAME_TYPES - seen)
    log.error("smoke test failed: missing frame types: %s", ", ".join(missing))
    log.error("seen: %s", sorted(seen))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
