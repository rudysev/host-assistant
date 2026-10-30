"""PostToolAnswerFallback — a post-tool LLM pass always says something.

The app follows Gemini 3.8's tool-turn order: the tool-call turn completes before the app's result, then the answer
is a turn of its own, and the app keeps the turn until that answer's audio arrives. A post-tool pass with no speech
(Qwen sometimes returns nothing after a tool result) would leave the app waiting for its 30s stall, so this sits
between the LLM and TTS and says :data:`FALLBACK_ANSWER` in that pass instead. Chained tool rounds are answered once,
by the last pass; a turn with no tool call is left alone.

A turn that already spoke before its tool call needs no fallback: its speech completes the turn after playback
(ControlBridge), and the answer pass may be silent on purpose. Qwen signs off, then calls portal.end_conversation,
whose result says "If you already said goodbye, say nothing more" — a fallback there spoke "Done." after the goodbye.
"""

from __future__ import annotations

from pipecat.frames.frames import (
    Frame,
    FunctionCallsStartedFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from host_assistant.text.emoji_strip import strip_emoji

FALLBACK_ANSWER = "Done."


class PostToolAnswerFallback(FrameProcessor):
    def __init__(self) -> None:
        super().__init__()
        self._reset()

    def _reset(self) -> None:
        # A tool was called this turn and no pass has answered it yet.
        self._tool_round = False
        # The open LLM pass follows a tool round: it is the answer.
        self._answer_pass = False
        # Any pass of this turn has spoken, the tool-call pass included.
        self._said = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return
        for out in self._frames_for(frame):
            await self.push_frame(out, direction)

    def _frames_for(self, frame: Frame) -> list[Frame]:
        """The frames to push for [frame]: itself, preceded by the fallback when it ends a silent answer pass."""
        if isinstance(frame, LLMFullResponseStartFrame):
            if self._tool_round:
                self._answer_pass = True
            else:
                self._reset()
        elif isinstance(frame, FunctionCallsStartedFrame):
            self._tool_round = True
            self._answer_pass = False
        elif isinstance(frame, LLMTextFrame):
            if strip_emoji(frame.text).strip():
                self._said = True
        elif isinstance(frame, LLMFullResponseEndFrame):
            if self._answer_pass:
                silent = not self._said
                self._reset()
                if silent:
                    return [LLMTextFrame(FALLBACK_ANSWER), frame]
        elif isinstance(frame, InterruptionFrame):
            self._reset()
        return [frame]
