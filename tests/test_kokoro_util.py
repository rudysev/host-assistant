"""Unit tests for kokoro_util helpers."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from host_assistant.warmup.kokoro import warm_kokoro


class WarmKokoroTests(unittest.IsolatedAsyncioTestCase):
    async def test_warm_kokoro_synthesizes_once(self) -> None:
        tts = MagicMock()

        async def fake_run_tts(text: str, context_id: str):
            yield MagicMock()
            assert text == "Hello."
            assert context_id == "warm"

        tts.run_tts = fake_run_tts
        await warm_kokoro(tts)


if __name__ == "__main__":
    unittest.main()
