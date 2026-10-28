"""Emit input_transcript control frames when STT produces a TranscriptionFrame.

The user context aggregator consumes TranscriptionFrame without forwarding it, so
ControlBridge (downstream of LLM/TTS) never sees user speech. This processor sits
between STT and the user aggregator to push transcripts to the app immediately.

It also relays the aggregator's VAD start/stop (pushed upstream, back through here) as
user_started_speaking / user_stopped_speaking: the app's no-speech window then counts from
when the user stopped speaking, not from a transcript that lands after VAD stop + Whisper.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from pipecat.frames.frames import (
    Frame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
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
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            await self._push_message(protocol.user_started_speaking())
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            await self._push_message(protocol.user_stopped_speaking())

        await self.push_frame(frame, direction)
