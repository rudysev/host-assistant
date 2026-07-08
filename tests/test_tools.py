"""Unit tests for tools.run_web_search."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from host_assistant.tools import host_tools
from host_assistant.tools.host_tools import (
    _compact_weather_payload,
    run_get_weather,
    run_web_search,
    warm_tavily,
)


class WebSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_query_returns_error(self) -> None:
        result = await run_web_search({"query": "  "})
        self.assertEqual(result, {"error": "empty query"})

    @patch("host_assistant.tools.host_tools._tavily")
    @patch("host_assistant.tools.host_tools.CONFIG")
    async def test_search_returns_answer_only_when_configured(
        self, mock_config: MagicMock, mock_tavily: MagicMock
    ) -> None:
        mock_config.web_search_answer_only = True
        mock_config.web_search_backend = "tavily"
        mock_tavily.return_value.search.return_value = {
            "answer": "sunny",
            "results": [{"title": "Weather", "content": "x" * 300}],
        }
        result = await run_web_search({"query": "weather"})
        self.assertEqual(result, {"answer": "sunny"})

    @patch("host_assistant.tools.host_tools._tavily")
    @patch("host_assistant.tools.host_tools.CONFIG")
    async def test_search_returns_compact_payload(self, mock_config: MagicMock, mock_tavily: MagicMock) -> None:
        mock_config.web_search_answer_only = False
        mock_config.web_search_backend = "tavily"
        mock_tavily.return_value.search.return_value = {
            "answer": "sunny",
            "results": [{"title": "Weather", "content": "x" * 300}],
        }
        result = await run_web_search({"query": "weather"})
        self.assertEqual(result["answer"], "sunny")
        self.assertEqual(len(result["results"][0]["content"]), 120)

    @patch("host_assistant.tools.host_tools._duckduckgo_search")
    @patch("host_assistant.tools.host_tools.CONFIG")
    async def test_duckduckgo_backend(self, mock_config: MagicMock, mock_ddg: MagicMock) -> None:
        mock_config.web_search_backend = "duckduckgo"
        mock_ddg.return_value = {"answer": "headline"}
        result = await run_web_search({"query": "news"})
        self.assertEqual(result, {"answer": "headline"})
        mock_ddg.assert_called_once_with("news")

    @patch("host_assistant.tools.host_tools._warm_tavily_sync")
    @patch("host_assistant.tools.host_tools.CONFIG")
    async def test_warm_tavily_skips_without_key(self, mock_config: MagicMock, mock_warm: MagicMock) -> None:
        mock_config.web_search_backend = "tavily"
        mock_config.tavily_api_key = ""
        await warm_tavily()
        mock_warm.assert_not_called()

    @patch("host_assistant.tools.host_tools._warm_tavily_sync")
    @patch("host_assistant.tools.host_tools.CONFIG")
    async def test_warm_tavily_runs_with_key(self, mock_config: MagicMock, mock_warm: MagicMock) -> None:
        mock_config.web_search_backend = "tavily"
        mock_config.tavily_api_key = "tvly-test"
        await warm_tavily()
        mock_warm.assert_called_once()


class GetWeatherTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_location_returns_error(self) -> None:
        result = await run_get_weather({"location": "  "})
        self.assertEqual(result, {"error": "empty location"})

    @patch("host_assistant.tools.host_tools._fetch_weather")
    @patch("host_assistant.tools.host_tools._geocode")
    async def test_get_weather_returns_structured_payload(
        self, mock_geocode: MagicMock, mock_fetch: MagicMock
    ) -> None:
        mock_geocode.return_value = {
            "name": "Mountain View, California, United States",
            "latitude": 37.4,
            "longitude": -122.1,
            "timezone": "America/Los_Angeles",
        }
        mock_fetch.return_value = {
            "location": "Mountain View, California, United States",
            "current": {"temp_f": 68, "conditions": "clear", "wind_mph": 6.2},
            "today": {"high_f": 79, "low_f": 57, "rain_chance_pct": 0},
            "daily": [
                {
                    "date": "2026-07-10",
                    "high_f": 79,
                    "low_f": 57,
                    "rain_chance_pct": 0,
                    "conditions": "clear",
                },
                {
                    "date": "2026-07-11",
                    "high_f": 81,
                    "low_f": 58,
                    "rain_chance_pct": 10,
                    "conditions": "partly cloudy",
                },
            ],
        }
        result = await run_get_weather({"location": "Mountain View, California"})
        self.assertIn("summary", result)
        self.assertIn("68", result["summary"])
        self.assertIn("Sat 2026-07-11", result["summary"])
        self.assertNotIn("daily", result)


class CompactWeatherPayloadTests(unittest.TestCase):
    def test_folds_daily_forecast_into_summary(self) -> None:
        payload = _compact_weather_payload(
            {
                "location": "Mountain View, California, United States",
                "current": {"temp_f": 56, "conditions": "clear", "wind_mph": 8},
                "today": {"high_f": 75, "low_f": 54, "rain_chance_pct": 1},
                "daily": [
                    {
                        "date": "2026-07-10",
                        "high_f": 75,
                        "low_f": 54,
                        "rain_chance_pct": 1,
                        "conditions": "clear",
                    },
                    {
                        "date": "2026-07-11",
                        "high_f": 72,
                        "low_f": 52,
                        "rain_chance_pct": 5,
                        "conditions": "overcast",
                    },
                ],
            }
        )
        self.assertIn("summary", payload)
        self.assertNotIn("daily", payload)
        # 2026-07-10 is Friday, 2026-07-11 is Saturday
        self.assertIn("Fri 2026-07-10: 75/54°F clear rain 1%", payload["summary"])
        self.assertIn("Sat 2026-07-11: 72/52°F overcast rain 5%", payload["summary"])

    def test_weekday_label_skips_bad_dates(self) -> None:
        self.assertEqual(host_tools._weekday_label("not-a-date"), "")
        self.assertEqual(host_tools._weekday_label("2026-07-12"), "Sun")


class GeocodeCacheTests(unittest.TestCase):
    def test_cache_evicts_oldest_at_cap(self) -> None:
        host_tools._geocode_cache.clear()
        for i in range(130):
            host_tools._geocode_cache_put(f"city{i}", {"name": f"city{i}"})
        self.assertEqual(len(host_tools._geocode_cache), 128)
        self.assertNotIn("city0", host_tools._geocode_cache)
        self.assertNotIn("city1", host_tools._geocode_cache)
        self.assertIn("city129", host_tools._geocode_cache)


class DailyForecastRowsTests(unittest.TestCase):
    def test_maps_open_meteo_daily_arrays(self) -> None:
        rows = host_tools._daily_forecast_rows(
            {
                "time": ["2026-07-10", "2026-07-11"],
                "temperature_2m_max": [75.0, 72.0],
                "temperature_2m_min": [54.0, 52.0],
                "precipitation_probability_max": [1, 5],
                "weather_code": [0, 3],
            }
        )
        self.assertEqual(rows[0]["conditions"], "clear")
        self.assertEqual(rows[1]["high_f"], 72.0)
        self.assertEqual(rows[1]["conditions"], "overcast")


if __name__ == "__main__":
    unittest.main()
