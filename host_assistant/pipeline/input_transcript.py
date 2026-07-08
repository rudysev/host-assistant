"""Emit input_transcript control frames when STT produces a TranscriptionFrame.

The user context aggregator consumes TranscriptionFrame without forwarding it, so
ControlBridge (downstream of LLM/TTS) never sees user speech. This processor sits
between STT and the user aggregator to push transcripts to the app immediately.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from host_assistant import protocol


class InputTranscriptEmitter(FrameProcessor):
    def __init__(self, push_message: Callable[[str], Awaitable[None]]) -> None:
        super().__init__()
        self._push_message = push_message

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and frame.text:
            await self._push_message(protocol.input_transcript(frame.text))

        await self.push_frame(frame, direction)
