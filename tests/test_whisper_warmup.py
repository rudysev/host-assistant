"""Unit tests for Whisper warm-up helpers."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from host_assistant.warmup.whisper import warm_whisper


class WarmWhisperTests(unittest.IsolatedAsyncioTestCase):
    async def test_warm_whisper_transcribes_silence_once(self) -> None:
        stt = MagicMock()
        seen: list[bytes] = []

        async def fake_run_stt(audio: bytes):
            seen.append(audio)
            return
            yield  # pragma: no cover - makes this an async generator

        stt.run_stt = fake_run_stt
        await warm_whisper(stt, sample_rate=16000)

        self.assertEqual(len(seen), 1)
        self.assertEqual(len(seen[0]), 3200)  # 0.1s @ 16 kHz mono int16


if __name__ == "__main__":
    unittest.main()
