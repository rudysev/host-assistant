"""Benchmark weather-query latency against a running host-assistant instance.

Skips STT (uses user_text) and records timestamps for key host→app frames plus
first audio byte. Also supports component-only timing (Ollama + Tavily) without
the full pipeline.

    # terminal 1
    python -m host_assistant

    # terminal 2
    python -m scripts.benchmark_weather
    python -m scripts.benchmark_weather --components-only
    python -m scripts.benchmark_weather --runs 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from host_assistant.config import CONFIG
from host_assistant.logging_config import configure_logging
from host_assistant.pipeline.session import HOST_SYSTEM_SUFFIX, OLLAMA_NO_THINK_PREFIX
from host_assistant.tools.host_tools import (
    GET_WEATHER_DECLARATION,
    WEB_SEARCH_DECLARATION,
    run_get_weather,
    run_web_search,
)
from scripts.ws_client import connect_kwargs

log = logging.getLogger(__name__)

# Mirrors the portal-assistant setup from production logs (trimmed tool list).
PORTAL_SETUP = {
    "type": "setup",
    "systemPrompt": (
        "Role: Warm, friendly display voice assistant. Never ask the user to say a wake word "
        "or goodbye (conversations end automatically).\n\n"
        "Tool Usage Rules:\n"
        "- Google Search: Use web_search for real-time/current info (news, stocks, sports, prices, "
        "hours, recent events). Base answers on results.\n"
        "- Weather: Use get_weather (faster than web_search). Pass the city from Device Context when known.\n"
        "- Time/Date: Use portal.get_time.\n"
        "- Timers: Use portal.set_timer (convert phrasing to duration_seconds; pass name as label, "
        "e.g. 'pasta') and portal.cancel_timer (by label). Use portal.list_timers to check remaining "
        "time (match by label); never guess time left from the set_timer response.\n"
        "- Volume: portal.set_volume (0-100; 100=max), portal.adjust_volume (up/down 1 step), "
        "portal.set_mute, portal.get_volume.\n"
        "- Brightness: portal.set_brightness (0-100; 0=min visible), portal.adjust_brightness "
        "(up/down 1 step), portal.get_brightness.\n"
        "- Do Not Disturb: portal.set_do_not_disturb (on/off), portal.get_do_not_disturb.\n"
        "- Music (portal.play_music): Plays on the user's default music app. Put request in query "
        "(infer and append artist for known songs, e.g. 'Bohemian Rhapsody Queen'). Set app ONLY if "
        "explicitly named (e.g. TIDAL). Set type (song/artist/album/playlist) ONLY if explicitly "
        "stated; otherwise omit.\n"
        "- Media Controls: portal.media_control (play/pause/next/previous), portal.set_repeat "
        "(one [current song], all [album/playlist], off), portal.now_playing.\n"
        "- Apps (portal.open_app): Launch by name. If uninstalled, offer returned close matches "
        "(do not guess). Use portal.play_music instead to play a specific song.\n\n"
        "Device Context:\n"
        "Time: Thursday, July 9, 2026 at 12:44 PM (America/Los_Angeles).\n"
        "Location: Mountain View, California."
    ),
    "tools": [
        {
            "name": "portal.get_time",
            "description": "Get the current time.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "portal.set_volume",
            "description": "Set media volume (0-100).",
            "parameters": {
                "type": "object",
                "properties": {"level_percent": {"type": "number"}},
                "required": ["level_percent"],
            },
        },
    ],
}

WEATHER_TURN = {"type": "user_text", "text": "What's the weather?"}


@dataclass
class E2EResult:
    t_ready: float | None = None
    t_user_sent: float | None = None
    t_model_generating: float | None = None
    t_first_audio: float | None = None
    t_turn_complete: float | None = None
    t_first_output_transcript: float | None = None
    frames: list[dict[str, Any]] = field(default_factory=list)
    audio_bytes: int = 0

    def summary(self) -> dict[str, float | None]:
        base = self.t_user_sent
        if base is None:
            return {}
        return {
            "to_model_generating_s": _delta(base, self.t_model_generating),
            "to_first_audio_s": _delta(base, self.t_first_audio),
            "to_first_transcript_s": _delta(base, self.t_first_output_transcript),
            "to_turn_complete_s": _delta(base, self.t_turn_complete),
            "audio_bytes": float(self.audio_bytes),
        }


def _delta(start: float, end: float | None) -> float | None:
    return None if end is None else round(end - start, 3)


async def run_e2e(url: str, *, timeout_s: float = 180.0) -> E2EResult:
    import websockets

    result = E2EResult()
    deadline = asyncio.get_running_loop().time() + timeout_s

    async with websockets.connect(url, max_size=None, open_timeout=10, **connect_kwargs(url)) as ws:
        await ws.send(json.dumps(PORTAL_SETUP))
        await _recv_until(ws, "ready", result, deadline)
        result.t_ready = time.perf_counter()

        await ws.send(json.dumps(WEATHER_TURN))
        result.t_user_sent = time.perf_counter()

        while asyncio.get_running_loop().time() < deadline:
            remaining = deadline - asyncio.get_running_loop().time()
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=min(remaining, 2.0))
            except asyncio.TimeoutError:
                if result.t_turn_complete is not None:
                    break
                continue

            now = time.perf_counter()
            if isinstance(msg, (bytes, bytearray)):
                result.audio_bytes += len(msg)
                if result.t_first_audio is None:
                    result.t_first_audio = now
                continue

            frame = json.loads(msg)
            result.frames.append(frame)
            kind = frame.get("type")
            if kind == "model_generating" and result.t_model_generating is None:
                result.t_model_generating = now
            elif kind == "output_transcript" and result.t_first_output_transcript is None:
                result.t_first_output_transcript = now
            elif kind == "turn_complete":
                result.t_turn_complete = now
                break
            elif kind == "error":
                raise RuntimeError(frame.get("message", "host error"))

    return result


async def _recv_until(ws: Any, frame_type: str, result: E2EResult, deadline: float) -> None:
    loop = asyncio.get_running_loop()
    while loop.time() < deadline:
        remaining = deadline - loop.time()
        msg = await asyncio.wait_for(ws.recv(), timeout=min(remaining, 5.0))
        if isinstance(msg, (bytes, bytearray)):
            continue
        frame = json.loads(msg)
        result.frames.append(frame)
        if frame.get("type") == frame_type:
            return
        if frame.get("type") == "error":
            raise RuntimeError(frame.get("message", "host error"))
    raise TimeoutError(f"timed out waiting for {frame_type!r}")


async def benchmark_components() -> dict[str, float]:
    """Time Ollama tool-routing + tool execution + Ollama response without the full pipeline."""
    timings: dict[str, float] = {}
    system = OLLAMA_NO_THINK_PREFIX + PORTAL_SETUP["systemPrompt"] + HOST_SYSTEM_SUFFIX
    user = WEATHER_TURN["text"]

    # Pass 1: does the model call a tool?
    t0 = time.perf_counter()
    tool_name, tool_args = await _ollama_tool_pass(system, user)
    timings["llm_pass1_tool_decision_s"] = round(time.perf_counter() - t0, 3)
    timings["tool_called"] = tool_name or "none"

    if not tool_name:
        return timings

    # Tool execution
    t0 = time.perf_counter()
    if tool_name == "get_weather":
        tool_result = await run_get_weather(tool_args)
    else:
        tool_result = await run_web_search(tool_args)
    timings["tool_execution_s"] = round(time.perf_counter() - t0, 3)

    # Pass 2: spoken answer from tool results
    t0 = time.perf_counter()
    await _ollama_response_pass(system, user, tool_name, tool_args, tool_result)
    timings["llm_pass2_response_s"] = round(time.perf_counter() - t0, 3)
    timings["llm_total_s"] = round(
        timings["llm_pass1_tool_decision_s"] + timings["llm_pass2_response_s"], 3
    )
    timings["components_total_s"] = round(
        timings["llm_pass1_tool_decision_s"]
        + timings["tool_execution_s"]
        + timings["llm_pass2_response_s"],
        3,
    )
    _ = (GET_WEATHER_DECLARATION, WEB_SEARCH_DECLARATION)
    return timings


async def _ollama_tool_pass(system: str, user: str) -> tuple[str | None, dict[str, Any]]:
    """One chat completion with tools; return tool name + args if the model calls one."""
    payload = {
        "model": CONFIG.ollama_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": GET_WEATHER_DECLARATION["name"],
                    "description": GET_WEATHER_DECLARATION["description"],
                    "parameters": GET_WEATHER_DECLARATION["parameters"],
                },
            },
            {
                "type": "function",
                "function": {
                    "name": WEB_SEARCH_DECLARATION["name"],
                    "description": WEB_SEARCH_DECLARATION["description"],
                    "parameters": WEB_SEARCH_DECLARATION["parameters"],
                },
            },
        ],
        "max_tokens": CONFIG.ollama_max_tokens,
        "stream": False,
    }
    data = await _ollama_chat(payload)
    message = data["choices"][0]["message"]
    tool_calls = message.get("tool_calls") or []
    for call in tool_calls:
        fn = call.get("function") or {}
        name = fn.get("name")
        if name in ("web_search", "get_weather"):
            return name, json.loads(fn.get("arguments") or "{}")
    return None, {}


async def _ollama_response_pass(
    system: str,
    user: str,
    tool_name: str,
    tool_args: dict[str, Any],
    tool_result: dict[str, Any],
) -> str:
    payload = {
        "model": CONFIG.ollama_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "bench_call",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_args),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "bench_call",
                "content": json.dumps(tool_result),
            },
        ],
        "max_tokens": CONFIG.ollama_max_tokens,
        "stream": False,
    }
    data = await _ollama_chat(payload)
    return data["choices"][0]["message"].get("content") or ""


async def _ollama_chat(payload: dict[str, Any]) -> dict[str, Any]:
    import urllib.request

    def _post() -> dict[str, Any]:
        req = urllib.request.Request(
            f"{CONFIG.ollama_base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer ollama"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read())

    return await asyncio.to_thread(_post)


def _log_e2e(label: str, result: E2EResult) -> None:
    summary = result.summary()
    log.info("=== %s ===", label)
    for key, val in summary.items():
        log.info("  %s: %s", key, val)
    frame_types = [f.get("type") for f in result.frames]
    log.info("  frames: %s", frame_types)


def _log_components(label: str, timings: dict[str, float]) -> None:
    log.info("=== %s (components) ===", label)
    for key, val in timings.items():
        log.info("  %s: %s", key, val)


async def _main_async(args: argparse.Namespace) -> int:
    log.info("config: model=%s max_tokens=%s", CONFIG.ollama_model, CONFIG.ollama_max_tokens)

    if args.components_only:
        for i in range(args.runs):
            timings = await benchmark_components()
            _log_components(f"run {i + 1}/{args.runs}", timings)
        return 0

    for i in range(args.runs):
        result = await run_e2e(args.url, timeout_s=args.timeout)
        _log_e2e(f"run {i + 1}/{args.runs}", result)
        if args.components and i == 0:
            timings = await benchmark_components()
            _log_components("component breakdown", timings)
    return 0


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description="Benchmark weather-query latency.")
    parser.add_argument("url", nargs="?", default="wss://localhost:8080")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--components-only", action="store_true")
    parser.add_argument("--components", action="store_true", help="Also run component timing after E2E")
    args = parser.parse_args(argv)

    try:
        return asyncio.run(_main_async(args))
    except Exception as e:  # noqa: BLE001
        log.error("benchmark failed: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
