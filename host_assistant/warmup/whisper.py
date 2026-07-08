"""Whisper STT helpers — model warm-up to cut first-turn ASR latency."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipecat.services.whisper.stt import WhisperSTTServiceMLX

# Enough silence to trigger mlx-whisper load + one decode pass without speech.
_WARM_SILENCE_SECS = 0.1


async def warm_whisper(stt: WhisperSTTServiceMLX, *, sample_rate: int) -> None:
    """Transcribe a short silent clip so the first real turn skips cold MLX load."""
    n_samples = max(1, int(sample_rate * _WARM_SILENCE_SECS))
    silence = b"\x00\x00" * n_samples
    async for _frame in stt.run_stt(silence):
        pass
