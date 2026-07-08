"""Unit tests for smoke_test helpers (no running host required)."""

from __future__ import annotations

import unittest

from scripts.smoke_test import REQUIRED_FRAME_TYPES, smoke_succeeded


class SmokeSucceededTests(unittest.TestCase):
    def test_requires_all_frame_types(self) -> None:
        self.assertTrue(smoke_succeeded(set(REQUIRED_FRAME_TYPES)))

    def test_fails_when_ready_missing(self) -> None:
        self.assertFalse(smoke_succeeded({"model_generating", "output_transcript", "turn_complete"}))

    def test_fails_when_empty(self) -> None:
        self.assertFalse(smoke_succeeded(set()))


if __name__ == "__main__":
    unittest.main()
