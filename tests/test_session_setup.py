"""Unit tests for session setup and tool declaration handling."""

from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from host_assistant.pipeline.session import (
    HOST_SYSTEM_SUFFIX,
    OLLAMA_NO_THINK_PREFIX,
    Session,
    filter_tool_declarations,
    parse_tool_results,
)


class FilterToolDeclarationsTests(unittest.TestCase):
    def test_skips_invalid_entries(self) -> None:
        tools = [{"name": "portal.ok"}, "bad", {"description": "no name"}]
        with self.assertLogs("host_assistant.pipeline.session", level="WARNING"):
            valid = filter_tool_declarations(tools)
        self.assertEqual(valid, [{"name": "portal.ok"}])


class ParseToolResultsTests(unittest.TestCase):
    def test_rejects_non_list_results(self) -> None:
        with self.assertLogs("host_assistant.pipeline.session", level="WARNING") as logs:
            self.assertEqual(parse_tool_results("bad"), [])
        self.assertTrue(any("must be a list" in m for m in logs.output))

    def test_skips_non_object_entries(self) -> None:
        with self.assertLogs("host_assistant.pipeline.session", level="WARNING"):
            valid = parse_tool_results([{"id": "1"}, "bad", 42])
        self.assertEqual(valid, [{"id": "1"}])


class SessionSetupTests(unittest.IsolatedAsyncioTestCase):
    async def test_setup_replaces_system_prompt_on_repeat(self) -> None:
        session = Session()
        context = MagicMock()
        session.configure(llm=MagicMock(), context=context, push_message=AsyncMock())

        await session._on_setup({"systemPrompt": "new", "tools": []})

        context.set_messages.assert_called_once_with(
            [
                {
                    "role": "system",
                    "content": OLLAMA_NO_THINK_PREFIX + "new" + HOST_SYSTEM_SUFFIX,
                }
            ]
        )

    async def test_setup_clears_prior_conversation(self) -> None:
        session = Session()
        context = MagicMock()
        session.configure(llm=MagicMock(), context=context, push_message=AsyncMock())

        await session._on_setup({"systemPrompt": "fresh", "tools": []})

        context.set_messages.assert_called_once_with(
            [
                {
                    "role": "system",
                    "content": OLLAMA_NO_THINK_PREFIX + "fresh" + HOST_SYSTEM_SUFFIX,
                }
            ]
        )

    async def test_setup_sends_ready_once(self) -> None:
        session = Session()
        push = AsyncMock()
        session.configure(llm=MagicMock(), context=MagicMock(), push_message=push)

        await session._on_setup({"systemPrompt": "hi", "tools": []})
        await session._on_setup({"systemPrompt": "hi", "tools": []})

        ready_calls = [c for c in push.await_args_list if "ready" in c.args[0]]
        self.assertEqual(len(ready_calls), 1)

    async def test_setup_registers_external_provider_tools(self) -> None:
        session = Session()
        llm = MagicMock()
        context = MagicMock()
        session.configure(llm=llm, context=context, push_message=AsyncMock())

        kasa_tool = {
            "name": "com.portal.kasa.set_plug",
            "description": "Turn a smart plug on or off by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plug_name": {"type": "string"},
                    "on": {"type": "boolean"},
                },
                "required": ["plug_name", "on"],
            },
        }
        await session._on_setup({"systemPrompt": "hi", "tools": [kasa_tool]})

        registered = {call.args[0] for call in llm.register_function.call_args_list}
        self.assertIn("com.portal.kasa.set_plug", registered)

    async def test_cancelled_portal_tool_returns_error_dict(self) -> None:
        session = Session()
        push = AsyncMock()
        session.configure(llm=MagicMock(), context=MagicMock(), push_message=push)

        async def run_tool() -> dict:
            task = asyncio.create_task(session._call_portal_tool("portal.set_volume", {"level_percent": 1}))
            await asyncio.sleep(0)
            session.reset()
            return await task

        result = await run_tool()
        self.assertEqual(result, {"error": "portal tool 'portal.set_volume' cancelled"})

    async def test_portal_tool_timeout_returns_error_dict(self) -> None:
        session = Session()
        push = AsyncMock()
        session.configure(llm=MagicMock(), context=MagicMock(), push_message=push)

        with self.assertLogs("host_assistant.pipeline.session", level="WARNING") as logs:
            with patch("host_assistant.pipeline.session.PORTAL_TOOL_TIMEOUT_S", 0.01):
                result = await session._call_portal_tool("portal.set_volume", {"level_percent": 1})

        self.assertEqual(result, {"error": "portal tool 'portal.set_volume' timed out"})
        self.assertTrue(any("timed out" in m for m in logs.output))
        self.assertEqual(session._pending, {})

    async def test_reset_unregisters_tools(self) -> None:
        session = Session()
        llm = MagicMock()
        context = MagicMock()
        session.configure(llm=llm, context=context, push_message=AsyncMock())
        session._registered_tool_names = {"portal.set_volume", "get_weather"}

        session.reset()

        self.assertEqual(
            {call.args[0] for call in llm.unregister_function.call_args_list},
            {"portal.set_volume", "get_weather"},
        )
        context.set_tools.assert_called_once()
        self.assertEqual(session._registered_tool_names, set())

    async def test_re_setup_unregisters_removed_tools(self) -> None:
        session = Session()
        llm = MagicMock()
        context = MagicMock()
        session.configure(llm=llm, context=context, push_message=AsyncMock())
        session._registered_tool_names = {"portal.old_tool"}

        await session._on_setup(
            {
                "systemPrompt": "hi",
                "tools": [
                    {
                        "name": "portal.new_tool",
                        "description": "new",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ],
            }
        )

        llm.unregister_function.assert_called_once_with("portal.old_tool")
        registered = {call.args[0] for call in llm.register_function.call_args_list}
        self.assertIn("portal.new_tool", registered)
        self.assertIn("get_weather", registered)
        self.assertNotIn("portal.old_tool", session._registered_tool_names)


if __name__ == "__main__":
    unittest.main()
