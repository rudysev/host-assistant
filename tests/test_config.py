"""Unit tests for config startup validation and parsing."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from host_assistant import config


def _minimal_config(**overrides) -> config.Config:
    base = dict(
        host="0.0.0.0",
        port=8080,
        ollama_base_url="http://localhost:11434/v1",
        ollama_model="qwen3.5:9b",
        ollama_max_tokens=128,
        warm_ollama_at_startup=True,
        warm_kokoro_at_startup=True,
        warm_tavily_at_startup=True,
        warm_whisper_at_startup=True,
        web_search_answer_only=True,
        web_search_backend="tavily",
        whisper_model="mlx-community/whisper-large-v3-turbo",
        kokoro_voice="af_heart",
        tavily_api_key="",
        input_sample_rate=16000,
        output_sample_rate=24000,
        vad_stop_secs=0.8,
    )
    base.update(overrides)
    return config.Config(**base)


class StartupWarningsTests(unittest.TestCase):
    def test_warns_when_tavily_backend_without_key(self) -> None:
        with patch.object(
            config,
            "CONFIG",
            _minimal_config(tavily_api_key="", web_search_backend="tavily", host="0.0.0.0"),
        ):
            with patch.object(config, "_env_load_warning", None):
                warnings = config.startup_warnings()
        self.assertEqual(len(warnings), 1)
        self.assertIn("TAVILY_API_KEY", warnings[0])

    def test_no_warnings_when_tavily_configured(self) -> None:
        with patch.object(
            config,
            "CONFIG",
            _minimal_config(tavily_api_key="tvly-test", web_search_backend="tavily", host="0.0.0.0"),
        ):
            with patch.object(config, "_env_load_warning", None):
                self.assertEqual(config.startup_warnings(), [])

    def test_no_warnings_for_duckduckgo_without_tavily_key(self) -> None:
        with patch.object(
            config,
            "CONFIG",
            _minimal_config(tavily_api_key="", web_search_backend="duckduckgo", host="0.0.0.0"),
        ):
            with patch.object(config, "_env_load_warning", None):
                self.assertEqual(config.startup_warnings(), [])

    def test_warns_when_env_file_missing(self) -> None:
        with patch.object(config, "_env_load_warning", "no .env found at /repo/.env"):
            with patch.object(config, "CONFIG", _minimal_config(host="0.0.0.0")):
                warnings = config.startup_warnings()
        self.assertTrue(any("no .env found" in w for w in warnings))


class WebSearchAvailableTests(unittest.TestCase):
    def test_duckduckgo_available_without_key(self) -> None:
        with patch.object(
            config,
            "CONFIG",
            _minimal_config(tavily_api_key="", web_search_backend="duckduckgo"),
        ):
            self.assertTrue(config.web_search_available())

    def test_tavily_requires_key(self) -> None:
        with patch.object(
            config,
            "CONFIG",
            _minimal_config(tavily_api_key="", web_search_backend="tavily"),
        ):
            self.assertFalse(config.web_search_available())


class RequireOpensslTests(unittest.TestCase):
    def test_raises_when_openssl_missing(self) -> None:
        with patch("host_assistant.config.shutil.which", return_value=None):
            with self.assertRaises(config.ConfigError) as ctx:
                config.require_openssl()
        self.assertIn("openssl", str(ctx.exception))

    def test_passes_when_openssl_present(self) -> None:
        with patch("host_assistant.config.shutil.which", return_value="/usr/bin/openssl"):
            config.require_openssl()


class ConfigParsingTests(unittest.TestCase):
    def test_default_host_binds_all_interfaces(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            cfg = config.load_config()
        self.assertEqual(cfg.host, "0.0.0.0")

    def test_invalid_port_raises_config_error(self) -> None:
        with patch.dict(os.environ, {"PORT": "not-a-port"}, clear=False):
            with self.assertRaises(config.ConfigError):
                config.load_config()

    def test_invalid_web_search_backend_raises_config_error(self) -> None:
        with patch.dict(os.environ, {"WEB_SEARCH_BACKEND": "google"}, clear=False):
            with self.assertRaises(config.ConfigError):
                config.load_config()

    def test_negative_sample_rate_raises_config_error(self) -> None:
        with patch.dict(os.environ, {"INPUT_SAMPLE_RATE": "0"}, clear=False):
            with self.assertRaises(config.ConfigError):
                config.load_config()

    def test_default_vad_stop_secs(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            cfg = config.load_config()
        self.assertEqual(cfg.vad_stop_secs, 0.8)

    def test_invalid_vad_stop_secs_raises_config_error(self) -> None:
        with patch.dict(os.environ, {"VAD_STOP_SECS": "fast"}, clear=False):
            with self.assertRaises(config.ConfigError):
                config.load_config()

    def test_vad_stop_secs_below_minimum_raises_config_error(self) -> None:
        with patch.dict(os.environ, {"VAD_STOP_SECS": "0.05"}, clear=False):
            with self.assertRaises(config.ConfigError):
                config.load_config()


if __name__ == "__main__":
    unittest.main()
