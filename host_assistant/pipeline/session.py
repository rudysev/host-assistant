"""Per-connection session state: setup, tool routing, and the portal-assistant control channel.

Holds the glue that isn't pure frame conversion:
  - **setup**: apply the app's system prompt; register tools from setup with the LLM. Tools in
    :data:`protocol.HOST_EXECUTED_TOOLS` run here; every other declared tool round-trips to
    portal-assistant.
  - **portal-tool round-trip**: when the model calls a non-host tool, push ``tool_call`` to the app
    and await ``tool_result`` (correlated by call id).
  - **host tools**: ``web_search`` and ``get_weather`` execute locally and return straight to the model.

The system prompt is sent once per WebSocket session (in ``setup``) and kept in ``LLMContext`` for all
turns in that conversation — it is not re-sent on every user turn. A new connection calls ``reset()``,
then ``setup`` seeds the prompt again for the new conversation. Re-sending ``setup`` on the same
connection clears prior turns and replaces the system prompt.

# VERIFY: if a Pipecat upgrade breaks tool calling, confirm register_function / ToolsSchema /
# set_tools and LLMMessagesAppendFrame against your installed version.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Awaitable, Callable

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import Frame, LLMMessagesAppendFrame

from host_assistant import protocol
from host_assistant.config import CONFIG, web_search_available
from host_assistant.tools.host_tools import (
    GET_WEATHER_DECLARATION,
    WEB_SEARCH_DECLARATION,
    run_get_weather,
    run_web_search,
)

PORTAL_TOOL_TIMEOUT_S = 8.0

# Appended to every setup system prompt — Ollama/Qwen-specific; host tool routing lives here too.
OLLAMA_NO_THINK_PREFIX = "/no_think\n"

HOST_SYSTEM_SUFFIX = (
    "\n\nHost guidance: Tools marked [Host] in their description run on this host; "
    "every other tool runs on the Portal. "
    "For turn-on/off or set requests, call the matching external action tool with the "
    "name the user gave — not portal.open_app or a list_* tool. "
    "If get_weather already returned a forecast for this place in the conversation, "
    "reuse it for same-place day follow-ups (match weekday labels in the summary). "
    "Keep every spoken reply to at most three short sentences."
)

log = logging.getLogger(__name__)


def parse_tool_results(raw: Any) -> list[dict[str, Any]]:
    """Return valid tool results from a ``tool_result`` frame, skipping malformed entries."""
    if not isinstance(raw, list):
        log.warning("ignoring tool_result: results must be a list, got %r", type(raw).__name__)
        return []
    valid: list[dict[str, Any]] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            log.warning(
                "skipping tool result %d: expected an object, got %r",
                index,
                type(entry).__name__,
            )
            continue
        valid.append(entry)
    return valid


def filter_tool_declarations(app_tools: list[Any]) -> list[dict[str, Any]]:
    """Return valid tool declarations from portal-assistant setup, skipping malformed entries."""
    valid: list[dict[str, Any]] = []
    for index, decl in enumerate(app_tools):
        if not isinstance(decl, dict):
            log.warning("skipping tool %d: expected an object, got %r", index, type(decl).__name__)
            continue
        name = decl.get("name")
        if not isinstance(name, str) or not name:
            log.warning("skipping tool %d: missing or invalid name", index)
            continue
        valid.append(decl)
    return valid


class Session:
    def __init__(self) -> None:
        self._llm = None
        self._context = None
        # Sends a JSON control string to the app (wired by host_assistant.__main__ to enqueue a TransportMessageUrgentFrame).
        self._push_message: Callable[[str], Awaitable[None]] | None = None
        # In-flight portal-assistant tool calls, keyed by the id we sent in the tool_call frame.
        self._pending: dict[str, asyncio.Future] = {}
        self._ready_sent = False
        self._registered_tool_names: set[str] = set()

    def configure(self, llm, context, push_message: Callable[[str], Awaitable[None]]) -> None:
        self._llm = llm
        self._context = context
        self._push_message = push_message

    def reset(self) -> None:
        """Reset per-connection state. Called on client connect and disconnect."""
        self._ready_sent = False
        for fut in self._pending.values():
            if not fut.done():
                fut.cancel()
        self._pending.clear()
        self._unregister_all_tools()
        if self._context is not None:
            self._context.set_messages([])

    # ---- inbound: app -> host text control frames ----

    async def on_client_text(self, text: str) -> Frame | None:
        msg = protocol.parse(text)
        if msg is None:
            preview = text[:120] + ("..." if len(text) > 120 else "")
            log.warning("ignored invalid control frame: %r", preview)
            return None
        kind = msg.get("type")
        if kind == protocol.SETUP:
            await self._on_setup(msg)
            return None
        if kind == protocol.USER_TEXT:
            return LLMMessagesAppendFrame(
                messages=[{"role": "user", "content": msg.get("text", "")}],
                run_llm=True,
            )
        if kind == protocol.TOOL_RESULT:
            self._resolve_tool_results(parse_tool_results(msg.get("results", [])))
            return None
        log.warning("ignored control frame with unknown type: %r", kind)
        return None

    async def _on_setup(self, msg: dict[str, Any]) -> None:
        system_prompt = msg.get("systemPrompt", "")
        app_tools = filter_tool_declarations(msg.get("tools", []) or [])

        if self._context is not None:
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages = [
                    {
                        "role": "system",
                        "content": OLLAMA_NO_THINK_PREFIX + system_prompt + HOST_SYSTEM_SUFFIX,
                    }
                ]
            self._context.set_messages(messages)

        declarations = list(app_tools)
        declarations.append(GET_WEATHER_DECLARATION)
        if web_search_available():
            declarations.append(WEB_SEARCH_DECLARATION)
        self._register_tools(declarations)
        log.info(
            "session setup: %d app tools, host tools: %s",
            len(app_tools),
            ", ".join(sorted(protocol.HOST_EXECUTED_TOOLS)),
        )

        if not self._ready_sent:
            self._ready_sent = True
            await self._send(protocol.ready())

    def _unregister_all_tools(self) -> None:
        if self._llm is not None:
            for name in self._registered_tool_names:
                self._llm.unregister_function(name)
        self._registered_tool_names.clear()
        if self._context is not None:
            self._context.set_tools(ToolsSchema(standard_tools=[]))

    def _register_tools(self, declarations: list[dict[str, Any]]) -> None:
        if self._llm is None or self._context is None:
            return

        new_names = {decl["name"] for decl in declarations}
        for name in self._registered_tool_names - new_names:
            self._llm.unregister_function(name)

        schemas: list[FunctionSchema] = []
        for decl in declarations:
            params = decl.get("parameters") or {}
            name = decl["name"]
            schemas.append(
                FunctionSchema(
                    name=name,
                    description=decl.get("description", ""),
                    properties=params.get("properties", {}),
                    required=params.get("required", []),
                )
            )
            if protocol.is_host_tool(name):
                if name == WEB_SEARCH_DECLARATION["name"]:
                    self._llm.register_function(name, self._make_web_search_handler())
                elif name == GET_WEATHER_DECLARATION["name"]:
                    self._llm.register_function(name, self._make_get_weather_handler())
            elif protocol.runs_on_portal_assistant(name):
                self._llm.register_function(name, self._make_portal_handler(name))

        self._registered_tool_names = new_names
        self._context.set_tools(ToolsSchema(standard_tools=schemas))

    def _make_web_search_handler(self):
        async def handler(params) -> None:  # params: FunctionCallParams
            result = await run_web_search(params.arguments)
            await params.result_callback(result)

        return handler

    def _make_get_weather_handler(self):
        async def handler(params) -> None:  # params: FunctionCallParams
            result = await run_get_weather(params.arguments)
            await params.result_callback(result)

        return handler

    def _make_portal_handler(self, name: str):
        async def handler(params) -> None:  # params: FunctionCallParams
            result = await self._call_portal_tool(name, params.arguments)
            await params.result_callback(result)

        return handler

    async def _call_portal_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Forward a tool call to portal-assistant and wait for its result (id-correlated)."""
        call_id = uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[call_id] = fut
        log.info("portal tool call → app: %s(%s)", name, args or {})
        await self._send(protocol.tool_call([{"id": call_id, "name": name, "args": args or {}}]))
        try:
            result = await asyncio.wait_for(fut, timeout=PORTAL_TOOL_TIMEOUT_S)
            log.info("portal tool result ← app: %s → %s", name, result)
            return result
        except asyncio.TimeoutError:
            log.warning("portal tool timed out after %.0fs: %s", PORTAL_TOOL_TIMEOUT_S, name)
            return {"error": f"portal tool '{name}' timed out"}
        except asyncio.CancelledError:
            return {"error": f"portal tool '{name}' cancelled"}
        finally:
            self._pending.pop(call_id, None)

    def _resolve_tool_results(self, results: list[dict[str, Any]]) -> None:
        for r in results:
            fut = self._pending.get(r.get("id", ""))
            if fut is not None and not fut.done():
                fut.set_result(r.get("response", {}))

    async def _send(self, message: str) -> None:
        if self._push_message is not None:
            await self._push_message(message)
