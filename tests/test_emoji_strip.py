"""Unit tests for emoji_strip.strip_emoji (stdlib only — no Pipecat required)."""

from __future__ import annotations

import unittest

from host_assistant.text.emoji_strip import strip_emoji


class StripEmojiTests(unittest.TestCase):
    def test_removes_weather_emoji_and_collapses_whitespace(self) -> None:
        self.assertEqual(strip_emoji("Hello 🌤️ world"), "Hello world")

    def test_leading_emoji_strips_cleanly(self) -> None:
        self.assertEqual(strip_emoji("🌤️ It's 72°F"), "It's 72°F")

    def test_preserves_ordinary_symbols(self) -> None:
        self.assertEqual(strip_emoji("72°F and 82%. Hope"), "72°F and 82%. Hope")

    def test_strips_misc_symbols_and_dingbats(self) -> None:
        self.assertEqual(strip_emoji("☀ sunny"), "sunny")
        self.assertEqual(strip_emoji("✅ done"), "done")

    def test_strips_flag_sequences(self) -> None:
        self.assertEqual(strip_emoji("🇺🇸 flag"), "flag")

    def test_strips_zwj_family_emoji(self) -> None:
        self.assertEqual(strip_emoji("👨‍👩‍👧 family"), "family")

    def test_leaves_markdown_and_plain_text(self) -> None:
        self.assertEqual(strip_emoji("**bold**"), "**bold**")
        self.assertEqual(strip_emoji("plain text"), "plain text")

    def test_emoji_only_returns_empty(self) -> None:
        self.assertEqual(strip_emoji("🎉"), "")
        self.assertEqual(strip_emoji("  🎉  "), "")

    def test_trailing_emoji(self) -> None:
        self.assertEqual(strip_emoji("only emoji 🎉"), "only emoji")

    def test_preserves_non_emoji_unicode(self) -> None:
        self.assertEqual(strip_emoji("Numbers ①②③"), "Numbers ①②③")
        self.assertEqual(strip_emoji("™ registered"), "™ registered")
        self.assertEqual(strip_emoji("→ arrow"), "→ arrow")


if __name__ == "__main__":
    unittest.main()
