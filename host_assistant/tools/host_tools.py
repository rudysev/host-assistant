"""Host-executed tools for the local model.

Only ``web_search`` and ``get_weather`` (see :data:`protocol.HOST_EXECUTED_TOOLS`) run here.
Every other tool in the setup frame is declared by portal-assistant and executed on the Portal
via a ``tool_call`` round-trip (see :mod:`session`).

``get_weather`` uses Open-Meteo (free, no API key) and is much faster than Tavily for weather turns —
the model's second pass gets a tiny structured payload instead of page snippets.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from datetime import date
from typing import Any

from tavily import TavilyClient

from host_assistant.config import CONFIG

# Fixed English labels so weekday follow-ups ("Sunday?") map without date arithmetic.
_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

# Declarations use the same neutral JSON Schema shape portal-assistant sends for its tools.
GET_WEATHER_DECLARATION: dict[str, Any] = {
    "name": "get_weather",
    "description": (
        "[Host] Get current weather plus a 7-day daily forecast for a city or place. "
        "Runs on the LAN host (Open-Meteo), not on the Portal. "
        "Prefer this over web_search for weather. Daily lines include weekday labels "
        "(e.g. Sun 2026-07-12). If a prior get_weather result for the same place is "
        "already in the conversation and covers the asked day, answer from that — "
        "do not call again."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City and region/country, e.g. 'Mountain View, California'.",
            },
        },
        "required": ["location"],
    },
}

WEB_SEARCH_DECLARATION: dict[str, Any] = {
    "name": "web_search",
    "description": (
        "[Host] Search the web for current information (news, stocks, sports, prices, facts you don't know). "
        "Runs on the LAN host, not on the Portal. "
        "Do not use for weather — use get_weather instead. Returns short result snippets."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
        },
        "required": ["query"],
    },
}

_client: TavilyClient | None = None
_GEOCODE_CACHE_MAX = 128
_geocode_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()


def _geocode_cache_get(key: str) -> dict[str, Any] | None:
    if key not in _geocode_cache:
        return None
    _geocode_cache.move_to_end(key)
    return _geocode_cache[key]


def _geocode_cache_put(key: str, value: dict[str, Any]) -> None:
    _geocode_cache[key] = value
    _geocode_cache.move_to_end(key)
    while len(_geocode_cache) > _GEOCODE_CACHE_MAX:
        _geocode_cache.popitem(last=False)


def _weekday_label(iso_date: str) -> str:
    """Return Mon–Sun for an ISO date, or '' if unparseable."""
    try:
        return _WEEKDAYS[date.fromisoformat(iso_date).weekday()]
    except ValueError:
        return ""


def _compact_weather_payload(full: dict[str, Any]) -> dict[str, Any]:
    """Return a tiny tool result so the post-tool LLM prefill stays fast."""
    current = full.get("current") or {}
    today = full.get("today") or {}
    lines = [
        f"{full.get('location')}: now {current.get('temp_f')}°F and {current.get('conditions')}, "
        f"wind {current.get('wind_mph')} mph; today high {today.get('high_f')}°F, "
        f"low {today.get('low_f')}°F, rain chance {today.get('rain_chance_pct')}%."
    ]
    for day in full.get("daily") or []:
        iso = day.get("date") or ""
        weekday = _weekday_label(iso) if isinstance(iso, str) else ""
        date_label = f"{weekday} {iso}" if weekday else iso
        lines.append(
            f"{date_label}: {day.get('high_f')}/{day.get('low_f')}°F "
            f"{day.get('conditions')} rain {day.get('rain_chance_pct')}%"
        )
    return {"summary": " ".join(lines)}


def _tavily() -> TavilyClient:
    global _client
    if _client is None:
        if not CONFIG.tavily_api_key:
            raise RuntimeError("TAVILY_API_KEY is not set — web_search is unavailable.")
        _client = TavilyClient(api_key=CONFIG.tavily_api_key)
    return _client


def _compact_search_payload(answer: str | None, results: list[dict[str, Any]]) -> dict[str, Any]:
    if CONFIG.web_search_answer_only:
        return {"answer": answer}
    return {"answer": answer, "results": results}


def _tavily_search(query: str) -> dict[str, Any]:
    resp = _tavily().search(
        query=query,
        search_depth="basic",
        max_results=3,
        include_answer=True,
    )
    answer = resp.get("answer")
    results = [
        {"title": r.get("title"), "content": (r.get("content") or "")[:120]}
        for r in resp.get("results", [])
    ]
    return _compact_search_payload(answer, results)


def _duckduckgo_search(query: str) -> dict[str, Any]:
    from duckduckgo_search import DDGS

    with DDGS() as ddgs:
        hits = list(ddgs.text(query, max_results=3))
    results = [
        {"title": r.get("title"), "content": (r.get("body") or "")[:120]}
        for r in hits
    ]
    answer = results[0]["content"] if results else None
    return _compact_search_payload(answer, results)


def _warm_tavily_sync() -> None:
    _tavily_search("weather")


async def warm_tavily() -> None:
    """Prime the Tavily client and connection with a tiny search (runs off the event loop)."""
    if CONFIG.web_search_backend != "tavily" or not CONFIG.tavily_api_key:
        return
    await asyncio.to_thread(_warm_tavily_sync)


async def warm_web_search() -> None:
    """Warm whichever web-search backend is configured."""
    if CONFIG.web_search_backend == "tavily":
        await warm_tavily()
    elif CONFIG.web_search_backend == "duckduckgo":

        def _warm() -> None:
            _duckduckgo_search("weather")

        await asyncio.to_thread(_warm)


_WMO_DESCRIPTIONS: dict[int, str] = {
    0: "clear",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "foggy",
    48: "foggy",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    80: "light showers",
    81: "showers",
    82: "heavy showers",
    95: "thunderstorms",
}


def _http_json(url: str, *, timeout: float = 10.0) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "host-assistant/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _parse_location(location: str) -> tuple[str, str | None]:
    parts = [part.strip() for part in location.split(",") if part.strip()]
    if len(parts) >= 2:
        return parts[0], parts[1]
    return location.strip(), None


def _matches_region(hit: dict[str, Any], region_hint: str | None) -> bool:
    if not region_hint:
        return True
    hint = region_hint.casefold()
    for key in ("admin1", "country"):
        value = hit.get(key)
        if isinstance(value, str) and hint in value.casefold():
            return True
    return False


def _geocode(location: str) -> dict[str, Any]:
    key = location.casefold()
    cached = _geocode_cache_get(key)
    if cached is not None:
        return cached

    city, region_hint = _parse_location(location)
    candidates = [location]
    if city != location:
        candidates.append(city)
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            query = urllib.parse.urlencode(
                {"name": candidate, "count": 10, "language": "en", "format": "json"}
            )
            data = _http_json(f"https://geocoding-api.open-meteo.com/v1/search?{query}")
            results = data.get("results") or []
            if not results:
                continue
            hit = next((r for r in results if _matches_region(r, region_hint)), results[0])
            label = ", ".join(
                part
                for part in (hit.get("name"), hit.get("admin1"), hit.get("country"))
                if part
            )
            place = {
                "name": label,
                "latitude": hit["latitude"],
                "longitude": hit["longitude"],
                "timezone": hit.get("timezone", "auto"),
            }
            _geocode_cache_put(key, place)
            return place
        except (urllib.error.URLError, KeyError, TypeError) as e:
            last_error = e
    raise ValueError(f"unknown location: {location!r}") from last_error


def _daily_forecast_rows(daily: dict[str, Any]) -> list[dict[str, Any]]:
    dates = daily.get("time") or []
    highs = daily.get("temperature_2m_max") or []
    lows = daily.get("temperature_2m_min") or []
    rain = daily.get("precipitation_probability_max") or []
    codes = daily.get("weather_code") or []
    rows: list[dict[str, Any]] = []
    for i, date in enumerate(dates):
        code = int(codes[i]) if i < len(codes) else -1
        rows.append(
            {
                "date": date,
                "high_f": highs[i] if i < len(highs) else None,
                "low_f": lows[i] if i < len(lows) else None,
                "rain_chance_pct": rain[i] if i < len(rain) else None,
                "conditions": _WMO_DESCRIPTIONS.get(code, "unknown"),
            }
        )
    return rows


def _fetch_weather(place: dict[str, Any]) -> dict[str, Any]:
    params = urllib.parse.urlencode(
        {
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "timezone": place["timezone"],
            "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m,wind_direction_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "forecast_days": 7,
        }
    )
    data = _http_json(f"https://api.open-meteo.com/v1/forecast?{params}")
    current = data.get("current") or {}
    daily = data.get("daily") or {}
    code = int(current.get("weather_code", -1))
    days = _daily_forecast_rows(daily)
    today = days[0] if days else {
        "high_f": None,
        "low_f": None,
        "rain_chance_pct": None,
        "conditions": "unknown",
    }
    return {
        "location": place["name"],
        "current": {
            "temp_f": current.get("temperature_2m"),
            "humidity_pct": current.get("relative_humidity_2m"),
            "conditions": _WMO_DESCRIPTIONS.get(code, "unknown"),
            "wind_mph": current.get("wind_speed_10m"),
            "wind_dir_deg": current.get("wind_direction_10m"),
        },
        "today": {
            "high_f": today.get("high_f"),
            "low_f": today.get("low_f"),
            "rain_chance_pct": today.get("rain_chance_pct"),
            "conditions": today.get("conditions"),
        },
        "daily": days,
    }


async def run_get_weather(args: dict[str, Any]) -> dict[str, Any]:
    location = (args or {}).get("location", "").strip()
    if not location:
        return {"error": "empty location"}

    def _run() -> dict[str, Any]:
        try:
            place = _geocode(location)
            return _compact_weather_payload(_fetch_weather(place))
        except (urllib.error.URLError, ValueError, KeyError, TypeError) as e:
            return {"error": f"get_weather failed: {e}"}

    return await asyncio.to_thread(_run)


async def run_web_search(args: dict[str, Any]) -> dict[str, Any]:
    """Execute a web search off the event loop. Backend is ``tavily`` (default) or ``duckduckgo``."""
    query = (args or {}).get("query", "").strip()
    if not query:
        return {"error": "empty query"}

    def _search() -> dict[str, Any]:
        if CONFIG.web_search_backend == "duckduckgo":
            return _duckduckgo_search(query)
        return _tavily_search(query)

    try:
        return await asyncio.to_thread(_search)
    except Exception as e:  # noqa: BLE001 - surface any search failure to the model as data
        return {"error": f"web_search failed: {e}"}
