"""Unit tests for ControlBridge control-frame mapping."""

from __future__ import annotations

import asyncio
import json
import unittest

from pipecat.frames.frames import (
    AggregationType,
    BotStoppedSpeakingFrame,
    ErrorFrame,
    FunctionCallsStartedFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TranscriptionFrame,
    TTSTextFrame,
)

from host_assistant.pipeline.bridge import ControlBridge


def _tts(text: str) -> TTSTextFrame:
    return TTSTextFrame(text=text, aggregated_by=AggregationType.SENTENCE)


class ControlBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bridge = ControlBridge()

    def _controls(self, frame) -> list[dict]:
        self.bridge._track_frame(frame)
        return [json.loads(msg) for msg in self.bridge._to_controls(frame)]

    def test_llm_start_emits_model_generating(self) -> None:
        controls = self._controls(LLMFullResponseStartFrame())
        self.assertEqual(controls, [{"type": "model_generating"}])

    def test_error_frame_emits_no_controls(self) -> None:
        """Pipeline errors are handled by the worker handler in __main__, not here."""
        controls = self._controls(ErrorFrame(error="boom"))
        self.assertEqual(controls, [])

    def test_transcription_frame_emits_no_controls(self) -> None:
        """input_transcript is owned by InputTranscriptEmitter upstream of the user aggregator."""
        controls = self._controls(TranscriptionFrame(text="hello", user_id="u", timestamp="0"))
        self.assertEqual(controls, [])

    def test_llm_end_without_tts_emits_turn_complete(self) -> None:
        self._controls(LLMFullResponseStartFrame())
        controls = self._controls(LLMFullResponseEndFrame())
        self.assertEqual(controls, [{"type": "turn_complete"}])

    def test_tool_pass_defers_turn_complete_until_after_tts(self) -> None:
        self._controls(LLMFullResponseStartFrame())
        self.assertEqual(self._controls(FunctionCallsStartedFrame(function_calls=[])), [{"type": "model_generating"}])
        self.assertEqual(self._controls(LLMFullResponseEndFrame()), [])
        self._controls(_tts("Done."))
        self.assertEqual(self._controls(LLMFullResponseEndFrame()), [])
        self.assertEqual(self._controls(BotStoppedSpeakingFrame()), [{"type": "turn_complete"}])

    def test_host_tool_second_llm_end_defers_until_playback(self) -> None:
        """Host tools run a second LLM pass; turn_complete must not race ahead of Kokoro."""
        self._controls(LLMFullResponseStartFrame())
        self._controls(FunctionCallsStartedFrame(function_calls=[]))
        self.assertEqual(self._controls(LLMFullResponseEndFrame()), [])
        self._controls(LLMFullResponseStartFrame())  # follow-up pass — must not reset tool latch
        self.assertEqual(self._controls(LLMFullResponseEndFrame()), [])
        self._controls(_tts("It is sunny."))
        self.assertEqual(self._controls(BotStoppedSpeakingFrame()), [{"type": "turn_complete"}])

    def test_host_tool_tts_before_second_llm_end_still_waits_for_playback(self) -> None:
        self._controls(LLMFullResponseStartFrame())
        self._controls(FunctionCallsStartedFrame(function_calls=[]))
        self.assertEqual(self._controls(LLMFullResponseEndFrame()), [])
        self._controls(LLMFullResponseStartFrame())
        self._controls(_tts("It is sunny."))
        self.assertEqual(self._controls(LLMFullResponseEndFrame()), [])
        self.assertEqual(self._controls(BotStoppedSpeakingFrame()), [{"type": "turn_complete"}])

    def test_host_tool_silent_follow_up_emits_turn_complete(self) -> None:
        """Post-tool LLM pass with no TTS must not leave the app waiting forever."""
        self._controls(LLMFullResponseStartFrame())
        self._controls(FunctionCallsStartedFrame(function_calls=[]))
        self._controls(LLMFullResponseEndFrame())
        self._controls(LLMFullResponseStartFrame())
        self._controls(LLMFullResponseEndFrame())
        asyncio.run(self.bridge._finish_silent_post_tool_turn())
        self.assertTrue(self.bridge._turn_complete_sent)

    def test_multi_tool_turn_defers_until_final_silent_end(self) -> None:
        self._controls(LLMFullResponseStartFrame())
        self._controls(FunctionCallsStartedFrame(function_calls=[]))
        self.assertEqual(self._controls(LLMFullResponseEndFrame()), [])
        self._controls(LLMFullResponseStartFrame())
        self._controls(FunctionCallsStartedFrame(function_calls=[]))
        self.assertEqual(self._controls(LLMFullResponseEndFrame()), [])
        self._controls(LLMFullResponseStartFrame())
        self._controls(LLMFullResponseEndFrame())
        asyncio.run(self.bridge._finish_silent_post_tool_turn())
        self.assertTrue(self.bridge._turn_complete_sent)

    def test_reset_turn_state_clears_bot_speaking(self) -> None:
        self.bridge._bot_speaking = True
        self.bridge.reset_turn_state()
        self.assertFalse(self.bridge._bot_speaking)
        self.assertEqual(self._controls(InterruptionFrame()), [])

    def test_llm_heartbeat_starts_and_stops(self) -> None:
        async def _run() -> None:
            self.bridge._ensure_llm_heartbeat()
            task = self.bridge._llm_heartbeat_task
            self.assertIsNotNone(task)
            self.assertFalse(task.done())
            self.bridge._stop_llm_heartbeat()
            await asyncio.sleep(0)
            self.assertTrue(task.done())

        asyncio.run(_run())

    def test_sync_turn_completed_clears_tool_latches(self) -> None:
        self.bridge._tool_round_trip = True
        self.bridge._awaiting_post_tool_speech = True
        self.bridge.sync_turn_completed()
        self.assertFalse(self.bridge._tool_round_trip)
        self.assertFalse(self.bridge._awaiting_post_tool_speech)
        self.assertTrue(self.bridge._turn_complete_sent)
        self.assertFalse(self.bridge._pending_silent_complete)

    def test_interrupted_only_when_bot_was_speaking(self) -> None:
        self.bridge._bot_speaking = True
        self.assertEqual(self._controls(InterruptionFrame()), [{"type": "interrupted"}])
        self.bridge._bot_speaking = False
        self.assertEqual(self._controls(InterruptionFrame()), [])

    def test_output_transcript(self) -> None:
        controls = self._controls(_tts("Hi"))
        self.assertEqual(controls, [{"type": "output_transcript", "text": "Hi "}])


if __name__ == "__main__":
    unittest.main()
