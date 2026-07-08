"""Kokoro TTS helpers — synthesis warm-up to cut first-turn TTFB."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipecat.services.kokoro.tts import KokoroTTSService

_WARM_PHRASE = "Hello."


async def warm_kokoro(tts: KokoroTTSService) -> None:
    """Run a one-line synthesis so the first real turn skips cold ONNX startup."""
    async for _frame in tts.run_tts(_WARM_PHRASE, "warm"):
        pass
