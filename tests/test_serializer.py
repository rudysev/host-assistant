"""Unit tests for PortalSerializer."""

from __future__ import annotations

import json
import unittest
from unittest.mock import AsyncMock, MagicMock

from pipecat.frames.frames import (
    InputAudioRawFrame,
    OutputTransportMessageUrgentFrame,
    TTSAudioRawFrame,
)

from host_assistant.pipeline.serializer import PortalSerializer


class PortalSerializerTests(unittest.IsolatedAsyncioTestCase):
    async def test_serializes_tts_audio_as_bytes(self) -> None:
        serializer = PortalSerializer(MagicMock())
        audio = b"\x00\x01"
        result = await serializer.serialize(TTSAudioRawFrame(audio=audio, sample_rate=24000, num_channels=1))
        self.assertEqual(result, audio)

    async def test_serializes_control_message_as_text(self) -> None:
        serializer = PortalSerializer(MagicMock())
        message = json.dumps({"type": "ready"})
        result = await serializer.serialize(OutputTransportMessageUrgentFrame(message=message))
        self.assertEqual(result, message)

    async def test_drops_non_str_control_messages(self) -> None:
        serializer = PortalSerializer(MagicMock())
        result = await serializer.serialize(OutputTransportMessageUrgentFrame(message={"type": "ready"}))
        self.assertIsNone(result)

    async def test_deserializes_binary_pcm(self) -> None:
        serializer = PortalSerializer(MagicMock())
        frame = await serializer.deserialize(b"\x00\x01")
        self.assertIsInstance(frame, InputAudioRawFrame)
        self.assertEqual(frame.audio, b"\x00\x01")

    async def test_deserializes_user_text_via_session(self) -> None:
        session = MagicMock()
        session.on_client_text = AsyncMock(return_value=None)
        serializer = PortalSerializer(session)
        await serializer.deserialize(json.dumps({"type": "user_text", "text": "hi"}))
        session.on_client_text.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
