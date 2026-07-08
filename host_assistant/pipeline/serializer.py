"""The bespoke Pipecat frame serializer for our LAN wire protocol.

This is the *only* novel piece of the host — it maps between Pipecat frames and our binary-PCM / JSON
control protocol (see protocol.py + README.md). Everything else is stock Pipecat.

Split of concerns:
  - **audio** ↔ raw PCM binary frames (in: InputAudioRawFrame; out: Output/TTS audio → bytes).
  - **inbound control** (setup / user_text / tool_result) is handled by the shared ``Session``.
  - **outbound control** (ready / transcripts / turn events / tool_call / error) is pushed *into* the
    pipeline as ``TransportMessageUrgentFrame`` by ``bridge.ControlBridge`` and just passed through here.

# VERIFY: if a Pipecat upgrade breaks the transport, confirm frame class imports below.
"""

from __future__ import annotations

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    TTSAudioRawFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer

from host_assistant.config import CONFIG
from host_assistant.pipeline.session import Session


class PortalSerializer(FrameSerializer):
    """Serialize Pipecat frames to our wire, and deserialize our wire to Pipecat frames."""

    def __init__(self, session: Session):
        super().__init__()
        self._session = session

    async def serialize(self, frame: Frame) -> str | bytes | None:
        # Model speech → raw 24 kHz PCM binary frame. (An output resampler in the pipeline ensures 24 kHz.)
        if isinstance(frame, (TTSAudioRawFrame, OutputAudioRawFrame)):
            return frame.audio
        # Control JSON that ControlBridge (or Session) encoded into a transport message → send as text.
        # Our messages are always JSON strings; other transport messages (e.g. the pipeline's RTVI dicts)
        # aren't part of our protocol, so drop them — never hand the socket a non-str/bytes payload.
        if isinstance(frame, (OutputTransportMessageUrgentFrame, OutputTransportMessageFrame)):
            msg = frame.message
            return msg if isinstance(msg, (str, bytes, bytearray)) else None
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        # Binary frame = raw 16 kHz mic PCM.
        if isinstance(data, (bytes, bytearray)):
            return InputAudioRawFrame(
                audio=bytes(data),
                sample_rate=CONFIG.input_sample_rate,
                num_channels=1,
            )

        # Text frame = JSON control. setup / tool_result are session state (no frame injected); user_text
        # becomes an LLM turn. Session owns the details so this stays a thin adapter.
        return await self._session.on_client_text(data)
