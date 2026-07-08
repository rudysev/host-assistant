"""WebSocket test client — exercises the host without the Android app.

Streams a 16 kHz mono WAV up as binary PCM frames (like the app's mic), collects control frames,
auto-answers any portal-assistant tool_call with a stub, and writes the model's speech to reply.pcm (24 kHz mono).

    python -m scripts.test_client sample.wav [wss://localhost:8080]

Make a suitable WAV with:  ffmpeg -i in.m4a -ac 1 -ar 16000 -f wav sample.wav
Play the reply with:       ffplay -f s16le -ar 24000 -ac 1 reply.pcm
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import wave
from dataclasses import dataclass, field
from typing import Any

from host_assistant.logging_config import configure_logging
from scripts.ws_client import connect_kwargs, wait_for_frame_type

log = logging.getLogger(__name__)

FRAME_MS = 20
REQUIRED_AFTER_AUDIO = frozenset({"input_transcript", "turn_complete"})


@dataclass
class ClientSessionResult:
    control_frames: list[dict[str, Any]] = field(default_factory=list)
    audio_bytes: int = 0
    seen_types: set[str] = field(default_factory=set)

    def succeeded(self) -> bool:
        return "ready" in self.seen_types and REQUIRED_AFTER_AUDIO <= self.seen_types


async def run_client_session(
    wav_path: str,
    url: str,
    *,
    wait_s: float = 120.0,
    reply_path: str = "reply.pcm",
) -> ClientSessionResult:
    import websockets

    with wave.open(wav_path, "rb") as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1:
            raise ValueError("need 16 kHz mono WAV")
        pcm = w.readframes(w.getnframes())

    chunk = int(16000 * 2 * FRAME_MS / 1000)
    result = ClientSessionResult()
    deadline = asyncio.get_running_loop().time() + wait_s

    async with websockets.connect(url, max_size=None, **connect_kwargs(url)) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "setup",
                    "systemPrompt": "You are a helpful assistant. Keep answers short.",
                    "tools": [
                        {
                            "name": "portal.set_volume",
                            "description": "Set the media volume.",
                            "parameters": {
                                "type": "object",
                                "properties": {"level_percent": {"type": "number"}},
                                "required": ["level_percent"],
                            },
                        }
                    ],
                }
            )
        )
        ready = await wait_for_frame_type(ws, "ready", deadline=deadline, seen=result.seen_types)
        result.control_frames.append(ready)

        with open(reply_path, "wb") as reply:

            async def receive() -> None:
                while True:
                    msg = await ws.recv()
                    if isinstance(msg, (bytes, bytearray)):
                        reply.write(msg)
                        result.audio_bytes += len(msg)
                        continue
                    obj = json.loads(msg)
                    kind = obj.get("type")
                    if kind:
                        result.seen_types.add(kind)
                    result.control_frames.append(obj)
                    if kind == "tool_call":
                        results = [
                            {"id": c["id"], "name": c["name"], "response": {"ok": True}}
                            for c in obj["calls"]
                        ]
                        await ws.send(json.dumps({"type": "tool_result", "results": results}))

            recv_task = asyncio.create_task(receive())
            try:
                for i in range(0, len(pcm), chunk):
                    await ws.send(pcm[i : i + chunk])
                    await asyncio.sleep(FRAME_MS / 1000)

                while asyncio.get_running_loop().time() < deadline:
                    if result.succeeded():
                        break
                    await asyncio.sleep(0.2)
            finally:
                recv_task.cancel()
                try:
                    await recv_task
                except asyncio.CancelledError:
                    pass

    return result


async def _run_and_log(wav_path: str, url: str, *, wait_s: float) -> ClientSessionResult:
    result = await run_client_session(wav_path, url, wait_s=wait_s)
    for frame in result.control_frames:
        log.info("<- %s", frame)
    log.info("wrote reply.pcm (%s audio bytes)", result.audio_bytes)
    return result


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description="Exercise the host with a WAV file.")
    parser.add_argument("wav")
    parser.add_argument("url", nargs="?", default="wss://localhost:8080")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)

    try:
        result = asyncio.run(_run_and_log(args.wav, args.url, wait_s=args.timeout))
    except Exception as e:  # noqa: BLE001 - CLI reports failures to the user
        log.error("test client failed: %s", e)
        return 1

    if not result.succeeded():
        missing = sorted(REQUIRED_AFTER_AUDIO - result.seen_types)
        log.error("test client incomplete: missing frame types: %s", ", ".join(missing))
        log.error("seen: %s", sorted(result.seen_types))
        return 1

    log.info("test client passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
