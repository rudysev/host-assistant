"""Unit tests for InputTranscriptEmitter."""

from __future__ import annotations

import json
import unittest
from unittest.mock import AsyncMock

from pipecat.frames.frames import TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection

from host_assistant.pipeline.input_transcript import InputTranscriptEmitter


class InputTranscriptEmitterTests(unittest.IsolatedAsyncioTestCase):
    async def test_emits_input_transcript_and_forwards_frame(self) -> None:
        push_message = AsyncMock()
        emitter = InputTranscriptEmitter(push_message)
        forwarded: list[TranscriptionFrame] = []

        async def capture_push(frame, direction):  # noqa: ANN001 - test shim
            forwarded.append(frame)

        emitter.push_frame = capture_push  # type: ignore[method-assign]

        frame = TranscriptionFrame(text="hello", user_id="u", timestamp="0")
        await emitter.process_frame(frame, FrameDirection.DOWNSTREAM)

        push_message.assert_awaited_once()
        self.assertEqual(
            json.loads(push_message.await_args.args[0]),
            {"type": "input_transcript", "text": "hello"},
        )
        self.assertIs(forwarded[0], frame)

    async def test_skips_empty_transcription(self) -> None:
        push_message = AsyncMock()
        emitter = InputTranscriptEmitter(push_message)
        forwarded: list[TranscriptionFrame] = []
        emitter.push_frame = AsyncMock(side_effect=lambda frame, direction: forwarded.append(frame))  # type: ignore[method-assign]

        frame = TranscriptionFrame(text="", user_id="u", timestamp="0")
        await emitter.process_frame(frame, FrameDirection.DOWNSTREAM)

        push_message.assert_not_awaited()
        self.assertIs(forwarded[0], frame)


if __name__ == "__main__":
    unittest.main()
