"""The wire protocol shared with the Android app's `LocalBackend`.

Deliberately vendor-neutral: binary WS frames carry raw PCM (16 kHz up / 24 kHz down); text WS frames
are JSON control messages keyed by ``type``. These helpers are the single place the JSON shapes live, so
they can't drift from the app. See README.md for the full table.
"""

from __future__ import annotations

import json
from typing import Any

# host -> app
READY = "ready"
INPUT_TRANSCRIPT = "input_transcript"
OUTPUT_TRANSCRIPT = "output_transcript"
MODEL_GENERATING = "model_generating"
TURN_COMPLETE = "turn_complete"
INTERRUPTED = "interrupted"
TOOL_CALL = "tool_call"
ERROR = "error"

# app -> host
SETUP = "setup"
USER_TEXT = "user_text"
TOOL_RESULT = "tool_result"

# Explicit allowlist: only these tool names execute on the host. Every other tool in the setup
# frame is forwarded to portal-assistant via ``tool_call`` / ``tool_result``.
HOST_EXECUTED_TOOLS = frozenset({"web_search", "get_weather"})


# ---- host -> app encoders (return a JSON string to send as a text frame) ----

def ready() -> str:
    return json.dumps({"type": READY})


def input_transcript(text: str) -> str:
    return json.dumps({"type": INPUT_TRANSCRIPT, "text": text})


def output_transcript(text: str) -> str:
    return json.dumps({"type": OUTPUT_TRANSCRIPT, "text": text})


def model_generating() -> str:
    return json.dumps({"type": MODEL_GENERATING})


def turn_complete() -> str:
    return json.dumps({"type": TURN_COMPLETE})


def interrupted() -> str:
    return json.dumps({"type": INTERRUPTED})


def tool_call(calls: list[dict[str, Any]]) -> str:
    # calls: [{"id","name","args": {...}}]
    return json.dumps({"type": TOOL_CALL, "calls": calls})


def error(message: str) -> str:
    return json.dumps({"type": ERROR, "message": message})


# ---- app -> host decoding ----

def parse(text: str) -> dict[str, Any] | None:
    """Parse a text control frame into a dict, or None if it isn't JSON with a ``type``."""
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return None
    return obj if isinstance(obj, dict) and "type" in obj else None


def is_host_tool(name: str) -> bool:
    """True when this tool runs on the host (the explicit allowlist above)."""
    return name in HOST_EXECUTED_TOOLS


def runs_on_portal_assistant(name: str) -> bool:
    """True for any setup tool not in :data:`HOST_EXECUTED_TOOLS` (the default)."""
    return not is_host_tool(name)
