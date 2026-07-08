"""Ollama helpers — model warm-up to cut first-turn latency."""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from typing import Any


def ollama_native_base(openai_base_url: str) -> str:
    """Map ``http://host:11434/v1`` → ``http://host:11434`` for Ollama's native API."""
    return openai_base_url.removesuffix("/v1")


def _warm_ollama_sync(base_url: str, model: str) -> None:
    url = f"{ollama_native_base(base_url)}/api/generate"
    payload: dict[str, Any] = {
        "model": model,
        "prompt": "hi",
        "stream": False,
        "options": {"num_predict": 1},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        resp.read()


async def warm_ollama(base_url: str, model: str) -> None:
    """Load the model into Ollama with a one-token generate (runs off the event loop)."""
    await asyncio.to_thread(_warm_ollama_sync, base_url, model)
