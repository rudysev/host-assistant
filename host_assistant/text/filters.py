"""Pipecat text filters built on shared cleanup helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pipecat.utils.text.base_text_filter import BaseTextFilter

from host_assistant.text.emoji_strip import strip_emoji

__all__ = ["EmojiTextFilter", "strip_emoji"]


class EmojiTextFilter(BaseTextFilter):
    """A Pipecat text filter that strips emoji before TTS synthesis."""

    async def update_settings(self, settings: Mapping[str, Any]) -> None:
        pass

    async def filter(self, text: str) -> str:
        return strip_emoji(text)

    async def handle_interruption(self) -> None:
        pass

    async def reset_interruption(self) -> None:
        pass
