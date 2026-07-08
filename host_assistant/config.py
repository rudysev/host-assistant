"""Runtime config for the local voice host, from environment / .env (dev-only)."""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

log = logging.getLogger(__name__)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_PATH = PACKAGE_ROOT / ".env"

_env_load_warning: str | None = None


def _load_dotenv() -> None:
    """Load ``.env`` from the package root; fall back to cwd with a clear warning."""
    global _env_load_warning
    if DEFAULT_ENV_PATH.is_file():
        load_dotenv(DEFAULT_ENV_PATH)
        return

    cwd_env = Path.cwd() / ".env"
    if cwd_env.is_file():
        load_dotenv(cwd_env)
        _env_load_warning = (
            f"loaded .env from cwd ({cwd_env}), not package root ({DEFAULT_ENV_PATH}) — "
            "run from the repo root or place .env next to host_assistant/"
        )
        return

    _env_load_warning = (
        f"no .env found at {DEFAULT_ENV_PATH} (cwd={Path.cwd()}) — "
        "using environment variables and defaults"
    )


_load_dotenv()


class ConfigError(Exception):
    """Raised when a required environment variable has an invalid value."""


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float, *, minimum: float | None = None) -> float:
    raw = os.getenv(name)
    if raw is None:
        value = default
    else:
        try:
            value = float(raw)
        except ValueError as exc:
            raise ConfigError(f"{name} must be a number (got {raw!r})") from exc
    if minimum is not None and value < minimum:
        raise ConfigError(f"{name} must be >= {minimum} (got {value})")
    return value


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None:
        value = default
    else:
        try:
            value = int(raw)
        except ValueError as exc:
            raise ConfigError(f"{name} must be an integer (got {raw!r})") from exc
    if minimum is not None and value < minimum:
        raise ConfigError(f"{name} must be >= {minimum} (got {value})")
    return value


def _env_choice(name: str, default: str, *, choices: set[str]) -> str:
    value = _env_str(name, default).lower()
    if value not in choices:
        raise ConfigError(f"{name} must be one of {sorted(choices)} (got {value!r})")
    return value


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return default if raw is None else raw


@dataclass(frozen=True)
class Config:
    host: str
    port: int
    ollama_base_url: str
    ollama_model: str
    ollama_max_tokens: int
    warm_ollama_at_startup: bool
    warm_kokoro_at_startup: bool
    warm_tavily_at_startup: bool
    warm_whisper_at_startup: bool
    web_search_answer_only: bool
    web_search_backend: str
    whisper_model: str
    kokoro_voice: str
    tavily_api_key: str
    input_sample_rate: int
    output_sample_rate: int
    vad_stop_secs: float


def load_config() -> Config:
    return Config(
        host=_env_str("HOST", "0.0.0.0"),
        port=_env_int("PORT", 8080, minimum=1),
        ollama_base_url=_env_str("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        ollama_model=_env_str("OLLAMA_MODEL", "qwen3.5:9b"),
        ollama_max_tokens=_env_int("OLLAMA_MAX_TOKENS", 1024, minimum=1),
        warm_ollama_at_startup=_env_bool("WARM_OLLAMA_AT_STARTUP", True),
        warm_kokoro_at_startup=_env_bool("WARM_KOKORO_AT_STARTUP", True),
        warm_tavily_at_startup=_env_bool("WARM_TAVILY_AT_STARTUP", True),
        warm_whisper_at_startup=_env_bool("WARM_WHISPER_AT_STARTUP", True),
        web_search_answer_only=_env_bool("WEB_SEARCH_ANSWER_ONLY", True),
        web_search_backend=_env_choice(
            "WEB_SEARCH_BACKEND", "tavily", choices={"tavily", "duckduckgo"}
        ),
        whisper_model=_env_str("WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo"),
        kokoro_voice=_env_str("KOKORO_VOICE", "af_heart"),
        tavily_api_key=_env_str("TAVILY_API_KEY", ""),
        input_sample_rate=_env_int("INPUT_SAMPLE_RATE", 16000, minimum=1),
        output_sample_rate=_env_int("OUTPUT_SAMPLE_RATE", 24000, minimum=1),
        vad_stop_secs=_env_float("VAD_STOP_SECS", 0.8, minimum=0.1),
    )


try:
    CONFIG = load_config()
except ConfigError as exc:
    from host_assistant.logging_config import configure_logging

    configure_logging(level=logging.ERROR)
    log.error("config error: %s", exc)
    raise SystemExit(1) from exc


def require_openssl() -> None:
    """TLS is mandatory — without ``openssl`` we cannot mint the cert and the app cannot connect."""
    if shutil.which("openssl") is None:
        raise ConfigError(
            "openssl not found — required to generate the TLS certificate "
            "(install openssl via your OS package manager if needed)"
        )


def web_search_available() -> bool:
    """True when the host can advertise ``web_search`` to the model."""
    if CONFIG.web_search_backend == "duckduckgo":
        return True
    return bool(CONFIG.tavily_api_key)


def startup_warnings() -> list[str]:
    """Non-fatal config issues to print at boot."""
    warnings: list[str] = []
    if _env_load_warning:
        warnings.append(_env_load_warning)
    if CONFIG.web_search_backend == "tavily" and not CONFIG.tavily_api_key:
        warnings.append("TAVILY_API_KEY not set — web_search will not be advertised to the model")
    return warnings
