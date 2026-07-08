"""Unit tests for protocol.py (stdlib only — no Pipecat required)."""

from __future__ import annotations

import json
import unittest

from host_assistant import protocol


class ProtocolEncoderTests(unittest.TestCase):
    def test_ready(self) -> None:
        self.assertEqual(json.loads(protocol.ready()), {"type": "ready"})

    def test_input_transcript(self) -> None:
        self.assertEqual(
            json.loads(protocol.input_transcript("hello")),
            {"type": "input_transcript", "text": "hello"},
        )

    def test_output_transcript(self) -> None:
        self.assertEqual(
            json.loads(protocol.output_transcript("hi there ")),
            {"type": "output_transcript", "text": "hi there "},
        )

    def test_model_generating(self) -> None:
        self.assertEqual(json.loads(protocol.model_generating()), {"type": "model_generating"})

    def test_turn_complete(self) -> None:
        self.assertEqual(json.loads(protocol.turn_complete()), {"type": "turn_complete"})

    def test_interrupted(self) -> None:
        self.assertEqual(json.loads(protocol.interrupted()), {"type": "interrupted"})

    def test_tool_call(self) -> None:
        calls = [{"id": "abc", "name": "portal.set_volume", "args": {"level_percent": 30}}]
        self.assertEqual(
            json.loads(protocol.tool_call(calls)),
            {"type": "tool_call", "calls": calls},
        )

    def test_error(self) -> None:
        self.assertEqual(
            json.loads(protocol.error("something broke")),
            {"type": "error", "message": "something broke"},
        )


class ProtocolParseTests(unittest.TestCase):
    def test_parse_setup(self) -> None:
        raw = json.dumps({"type": "setup", "systemPrompt": "hi", "tools": []})
        self.assertEqual(protocol.parse(raw), {"type": "setup", "systemPrompt": "hi", "tools": []})

    def test_parse_user_text(self) -> None:
        raw = json.dumps({"type": "user_text", "text": "weather?"})
        self.assertEqual(protocol.parse(raw), {"type": "user_text", "text": "weather?"})

    def test_parse_tool_result(self) -> None:
        raw = json.dumps({"type": "tool_result", "results": [{"id": "1", "name": "x", "response": {}}]})
        self.assertIsNotNone(protocol.parse(raw))
        self.assertEqual(protocol.parse(raw)["type"], "tool_result")

    def test_parse_invalid_json(self) -> None:
        self.assertIsNone(protocol.parse("not json"))

    def test_parse_missing_type(self) -> None:
        self.assertIsNone(protocol.parse(json.dumps({"text": "no type"})))

    def test_parse_non_object(self) -> None:
        self.assertIsNone(protocol.parse(json.dumps(["array"])))


class ToolRoutingTests(unittest.TestCase):
    def test_host_executed_tools(self) -> None:
        self.assertTrue(protocol.is_host_tool("web_search"))
        self.assertTrue(protocol.is_host_tool("get_weather"))
        self.assertFalse(protocol.is_host_tool("portal.set_volume"))
        self.assertFalse(protocol.is_host_tool("com.portal.kasa.set_plug"))

    def test_runs_on_portal_assistant_by_default(self) -> None:
        self.assertTrue(protocol.runs_on_portal_assistant("portal.set_volume"))
        self.assertTrue(protocol.runs_on_portal_assistant("com.example.any.tool"))
        self.assertFalse(protocol.runs_on_portal_assistant("web_search"))
        self.assertFalse(protocol.runs_on_portal_assistant("get_weather"))


if __name__ == "__main__":
    unittest.main()
