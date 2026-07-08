"""Unit tests for ollama_util helpers."""

from __future__ import annotations

import unittest

from host_assistant.warmup.ollama import ollama_native_base


class OllamaNativeBaseTests(unittest.TestCase):
    def test_strips_v1_suffix(self) -> None:
        self.assertEqual(
            ollama_native_base("http://localhost:11434/v1"),
            "http://localhost:11434",
        )

    def test_leaves_bare_base_unchanged(self) -> None:
        self.assertEqual(
            ollama_native_base("http://localhost:11434"),
            "http://localhost:11434",
        )


if __name__ == "__main__":
    unittest.main()
