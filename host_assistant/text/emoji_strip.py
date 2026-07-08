"""Pure emoji stripping — no Pipecat dependency so it can be unit-tested in isolation."""

from __future__ import annotations

import re

_EMOJI_RE = re.compile(
    "["
    "\U0001f000-\U0001faff"  # emoji, pictographs & symbols (weather, faces, celebration, …)
    "\U00002600-\U000027bf"  # misc symbols (☀ ⛅ ☂) + dingbats (✨ ✅ ➤)
    "\U0001f1e6-\U0001f1ff"  # regional indicators (flags)
    "\U00002300-\U000023ff"  # misc technical (⏰ ⌛)
    "\U00002b00-\U00002bff"  # stars / arrows (⭐)
    "\U0000fe00-\U0000fe0f"  # variation selectors (emoji presentation)
    "\U0000200d"             # zero-width joiner (emoji sequences)
    "]+",
    flags=re.UNICODE,
)


def strip_emoji(text: str) -> str:
    """Remove emoji/pictographs; keep ordinary symbols (°, %, $, …).

    Collapses whitespace left behind by removed emoji so "Hello 🌤️ world" becomes
    "Hello world", not "Hello  world".
    """
    cleaned = _EMOJI_RE.sub("", text)
    return re.sub(r"\s+", " ", cleaned).strip()
