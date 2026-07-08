"""Unit tests for Session portal-tool result correlation."""

from __future__ import annotations

import asyncio
import unittest

from host_assistant.pipeline.session import Session


class SessionToolResultTests(unittest.IsolatedAsyncioTestCase):
    async def test_resolve_tool_results_fulfills_pending_future(self) -> None:
        session = Session()
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        session._pending["call-1"] = fut

        session._resolve_tool_results(
            [{"id": "call-1", "name": "portal.set_volume", "response": {"ok": True}}]
        )

        self.assertTrue(fut.done())
        self.assertEqual(await fut, {"ok": True})

    async def test_resolve_ignores_unknown_call_id(self) -> None:
        session = Session()
        session._resolve_tool_results(
            [{"id": "missing", "name": "portal.set_volume", "response": {"ok": True}}]
        )
        self.assertEqual(session._pending, {})

    async def test_resolve_does_not_overwrite_completed_future(self) -> None:
        session = Session()
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        fut.set_result({"first": True})
        session._pending["call-1"] = fut

        session._resolve_tool_results(
            [{"id": "call-1", "name": "portal.set_volume", "response": {"second": True}}]
        )

        self.assertEqual(await fut, {"first": True})

    async def test_reset_cancels_pending_futures(self) -> None:
        session = Session()
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        session._pending["call-1"] = fut

        session.reset()

        self.assertTrue(fut.cancelled())
        self.assertEqual(session._pending, {})
        self.assertFalse(session._ready_sent)

    async def test_invalid_control_frame_is_logged(self) -> None:
        session = Session()
        with self.assertLogs("host_assistant.pipeline.session", level="WARNING") as logs:
            result = await session.on_client_text("not json")
        self.assertIsNone(result)
        self.assertTrue(any("invalid control frame" in m for m in logs.output))

    async def test_unknown_control_type_is_logged(self) -> None:
        session = Session()
        with self.assertLogs("host_assistant.pipeline.session", level="WARNING") as logs:
            result = await session.on_client_text('{"type": "future_feature"}')
        self.assertIsNone(result)
        self.assertTrue(any("unknown type" in m for m in logs.output))

    async def test_malformed_tool_result_does_not_crash(self) -> None:
        session = Session()
        with self.assertLogs("host_assistant.pipeline.session", level="WARNING") as logs:
            result = await session.on_client_text('{"type": "tool_result", "results": "bad"}')
        self.assertIsNone(result)
        self.assertTrue(any("must be a list" in m for m in logs.output))


if __name__ == "__main__":
    unittest.main()
