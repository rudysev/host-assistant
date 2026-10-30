"""Unit tests for PostToolAnswerFallback: a post-tool LLM pass always has something to say."""

from __future__ import annotations

import unittest

from pipecat.frames.frames import (
    FunctionCallsStartedFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)

from host_assistant.pipeline.answer_fallback import FALLBACK_ANSWER, PostToolAnswerFallback


class PostToolAnswerFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fallback = PostToolAnswerFallback()

    def _run(self, frames) -> list:
        out = []
        for frame in frames:
            out.extend(self.fallback._frames_for(frame))
        return out

    def _spoken(self, frames) -> list[str]:
        return [f.text for f in self._run(frames) if isinstance(f, LLMTextFrame)]

    def _tool_round(self) -> list:
        return [LLMFullResponseStartFrame(), FunctionCallsStartedFrame(function_calls=[]), LLMFullResponseEndFrame()]

    def test_silent_post_tool_pass_says_the_fallback_before_its_end(self) -> None:
        out = self._run(self._tool_round() + [LLMFullResponseStartFrame(), LLMFullResponseEndFrame()])
        self.assertIsInstance(out[-2], LLMTextFrame)
        self.assertEqual(out[-2].text, FALLBACK_ANSWER)
        self.assertIsInstance(out[-1], LLMFullResponseEndFrame)

    def test_spoken_post_tool_pass_is_unchanged(self) -> None:
        frames = self._tool_round() + [LLMFullResponseStartFrame(), LLMTextFrame("Timer set."), LLMFullResponseEndFrame()]
        self.assertEqual(self._spoken(frames), ["Timer set."])

    def test_blank_or_emoji_only_text_counts_as_silent(self) -> None:
        frames = self._tool_round() + [
            LLMFullResponseStartFrame(),
            LLMTextFrame("  "),
            LLMTextFrame("⏰"),
            LLMFullResponseEndFrame(),
        ]
        self.assertEqual(self._spoken(frames), ["  ", "⏰", FALLBACK_ANSWER])

    def test_silent_turn_without_a_tool_is_left_alone(self) -> None:
        self.assertEqual(self._spoken([LLMFullResponseStartFrame(), LLMFullResponseEndFrame()]), [])

    def test_chained_tool_rounds_answer_only_the_last_pass(self) -> None:
        frames = self._tool_round() + self._tool_round() + [LLMFullResponseStartFrame(), LLMFullResponseEndFrame()]
        self.assertEqual(self._spoken(frames), [FALLBACK_ANSWER])

    def test_the_next_turn_starts_fresh(self) -> None:
        answered = self._tool_round() + [LLMFullResponseStartFrame(), LLMTextFrame("Done."), LLMFullResponseEndFrame()]
        self._run(answered)
        self.assertEqual(self._spoken([LLMFullResponseStartFrame(), LLMFullResponseEndFrame()]), [])

    def test_silent_answer_after_a_spoken_tool_call_pass_stays_silent(self) -> None:
        frames = [
            LLMFullResponseStartFrame(),
            LLMTextFrame("You're welcome. Goodbye!"),
            FunctionCallsStartedFrame(function_calls=[]),
            LLMFullResponseEndFrame(),
            LLMFullResponseStartFrame(),
            LLMFullResponseEndFrame(),
        ]
        self.assertEqual(self._spoken(frames), ["You're welcome. Goodbye!"])

    def test_interruption_drops_the_pending_answer(self) -> None:
        self._run(self._tool_round() + [InterruptionFrame()])
        self.assertEqual(self._spoken([LLMFullResponseStartFrame(), LLMFullResponseEndFrame()]), [])


if __name__ == "__main__":
    unittest.main()
