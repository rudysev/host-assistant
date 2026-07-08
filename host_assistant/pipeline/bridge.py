"""ControlBridge — turns Pipecat lifecycle/transcript frames into our host->app control frames.

It sits just before the output transport, observes frames flowing by, and for the interesting ones emits
a ``OutputTransportMessageUrgentFrame`` carrying our JSON (which PortalSerializer sends as a text frame).
Audio and everything else pass through untouched.

The output transport pushes ``Bot{Started,Stopped}SpeakingFrame`` **upstream**, so we see them here and use
them to (a) emit ``turn_complete`` and (b) gate ``interrupted`` to *real* barge-ins — Pipecat broadcasts an
interruption at the start of every user turn, which is not a barge-in unless the bot was already speaking.
"""

from __future__ import annotations

import asyncio

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    FunctionCallsStartedFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    OutputTransportMessageUrgentFrame,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from host_assistant import protocol
from host_assistant.text.emoji_strip import strip_emoji

DEFAULT_TURN_ERROR = "Something went wrong while generating a response."

# Re-emit model_generating while an LLM response is open so Portal's 30s dead-air stall stays fresh.
_LLM_HEARTBEAT_SECS = 10.0


class ControlBridge(FrameProcessor):
    def __init__(self) -> None:
        super().__init__()
        self._bot_speaking = False
        self._turn_had_tts = False
        self._turn_complete_sent = False
        # True once the model calls a tool this turn — stays set across the follow-up LLM pass until
        # playback finishes, so a second LLMFullResponseStartFrame does not reset the deferral.
        self._tool_round_trip = False
        # True during the post-tool LLM pass (synthesis after tool results). Cleared when another
        # tool call starts in the same turn.
        self._awaiting_post_tool_speech = False
        self._pending_silent_complete = False
        self._silent_complete_task: asyncio.Task | None = None
        self._llm_heartbeat_task: asyncio.Task | None = None

    def reset_turn_state(self) -> None:
        """Clear per-turn tracking (e.g. when the client disconnects)."""
        self._cancel_silent_complete_task()
        self._stop_llm_heartbeat()
        self._bot_speaking = False
        self._turn_had_tts = False
        self._turn_complete_sent = False
        self._tool_round_trip = False
        self._awaiting_post_tool_speech = False
        self._pending_silent_complete = False

    def sync_turn_completed(self) -> None:
        """Sync latch state after turn_complete was sent outside this bridge (e.g. pipeline error)."""
        self._cancel_silent_complete_task()
        self._stop_llm_heartbeat()
        self._turn_complete_sent = True
        self._tool_round_trip = False
        self._awaiting_post_tool_speech = False
        self._turn_had_tts = False
        self._bot_speaking = False
        self._pending_silent_complete = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        self._track_frame(frame)

        if isinstance(frame, (LLMFullResponseStartFrame, FunctionCallsStartedFrame)):
            self._ensure_llm_heartbeat()

        if self._pending_silent_complete:
            self._pending_silent_complete = False
            self._schedule_silent_complete()

        for control in self._to_controls(frame):
            await self.push_frame(OutputTransportMessageUrgentFrame(message=control), FrameDirection.DOWNSTREAM)

        await self.push_frame(frame, direction)

    def _cancel_silent_complete_task(self) -> None:
        if self._silent_complete_task is not None and not self._silent_complete_task.done():
            self._silent_complete_task.cancel()
        self._silent_complete_task = None

    def _stop_llm_heartbeat(self) -> None:
        if self._llm_heartbeat_task is not None and not self._llm_heartbeat_task.done():
            self._llm_heartbeat_task.cancel()
        self._llm_heartbeat_task = None

    def _ensure_llm_heartbeat(self) -> None:
        if self._llm_heartbeat_task is not None and not self._llm_heartbeat_task.done():
            return
        self._llm_heartbeat_task = asyncio.create_task(self._llm_heartbeat_loop())

    async def _llm_heartbeat_loop(self) -> None:
        """Push model_generating every 10s while the LLM response is still open."""
        try:
            while True:
                await asyncio.sleep(_LLM_HEARTBEAT_SECS)
                await self.push_frame(
                    OutputTransportMessageUrgentFrame(message=protocol.model_generating()),
                    FrameDirection.DOWNSTREAM,
                )
        except asyncio.CancelledError:
            raise

    def _schedule_silent_complete(self) -> None:
        self._cancel_silent_complete_task()
        self._silent_complete_task = asyncio.create_task(self._finish_silent_post_tool_turn())

    async def _finish_silent_post_tool_turn(self) -> None:
        """Complete a post-tool turn that produced no TTS, after one event-loop tick for late TTSTextFrame."""
        await asyncio.sleep(0)
        try:
            if self._turn_complete_sent or self._turn_had_tts:
                return
            if not (self._tool_round_trip and self._awaiting_post_tool_speech):
                return
            for control in self._complete_turn_if_needed():
                await self.push_frame(
                    OutputTransportMessageUrgentFrame(message=control),
                    FrameDirection.DOWNSTREAM,
                )
        except asyncio.CancelledError:
            raise

    def _track_frame(self, frame: Frame) -> None:
        if isinstance(frame, LLMFullResponseStartFrame):
            if self._tool_round_trip:
                # Follow-up LLM pass after a tool call — keep the tool-turn latch, only reset TTS tracking.
                self._turn_had_tts = False
                self._awaiting_post_tool_speech = True
            else:
                self.reset_turn_state()
        elif isinstance(frame, FunctionCallsStartedFrame):
            self._tool_round_trip = True
            self._awaiting_post_tool_speech = False
            self._cancel_silent_complete_task()
        elif isinstance(frame, LLMFullResponseEndFrame):
            self._stop_llm_heartbeat()
            if (
                self._tool_round_trip
                and self._awaiting_post_tool_speech
                and not self._turn_had_tts
            ):
                self._pending_silent_complete = True
        elif isinstance(frame, InterruptionFrame):
            self._stop_llm_heartbeat()
        elif isinstance(frame, TTSTextFrame):
            if strip_emoji(frame.text).rstrip():
                self._turn_had_tts = True
                self._cancel_silent_complete_task()
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            self._stop_llm_heartbeat()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False

    def _to_controls(self, frame: Frame) -> list[str]:
        controls: list[str] = []

        if isinstance(frame, (LLMFullResponseStartFrame, FunctionCallsStartedFrame)):
            # Must precede the first TTS audio frame — the app's reducer keys off model_generating.
            controls.append(protocol.model_generating())
        elif isinstance(frame, LLMFullResponseEndFrame):
            if not self._tool_round_trip and not self._turn_had_tts:
                controls.extend(self._complete_turn_if_needed())
        elif isinstance(frame, BotStoppedSpeakingFrame):
            controls.extend(self._complete_turn_if_needed())
        elif isinstance(frame, InterruptionFrame):
            if self._bot_speaking:
                controls.append(protocol.interrupted())
        elif isinstance(frame, TTSTextFrame):
            text = strip_emoji(frame.text).rstrip()
            if text:
                controls.append(protocol.output_transcript(text + " "))

        return controls

    def _complete_turn_if_needed(self) -> list[str]:
        if self._turn_complete_sent:
            return []
        self._cancel_silent_complete_task()
        self._stop_llm_heartbeat()
        self._turn_complete_sent = True
        self._tool_round_trip = False
        self._awaiting_post_tool_speech = False
        self._pending_silent_complete = False
        return [protocol.turn_complete()]
