"""Unit tests for test_client result helpers."""

from __future__ import annotations

import unittest

from scripts.test_client import ClientSessionResult


class ClientSessionResultTests(unittest.TestCase):
    def test_succeeded_requires_ready_and_audio_turn_frames(self) -> None:
        result = ClientSessionResult(
            seen_types={"ready", "input_transcript", "turn_complete"},
        )
        self.assertTrue(result.succeeded())

    def test_fails_without_input_transcript(self) -> None:
        result = ClientSessionResult(seen_types={"ready", "turn_complete"})
        self.assertFalse(result.succeeded())


if __name__ == "__main__":
    unittest.main()
