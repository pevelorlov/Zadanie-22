"""MCP-сервер прогноза погоды Open-Meteo (API-ключ не нужен)."""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer

from mcp_servers.common.http_json import get_json


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_FIELDS = (
    "temperature_2m,apparent_temperature,relative_humidity_2m,"
    "precipitation_probability,precipitation,rain,showers,snowfall,"
    "weather_code,visibility,wind_speed_10m,wind_gusts_10m"
)
DAILY_FIELDS = (
    "weather_code,temperature_2m_max,temperature_2m_min,"
    "apparent_temperature_max,apparent_temperature_min,sunrise,sunset,"
    "precipitation_sum,precipitation_probability_max,wind_speed_10m_max,"
    "wind_gusts_10m_max,uv_index_max"
)

server = MCPServer(
    name="open-meteo-mcp",
    title="Open-Meteo Weather",
    description="Геокодирование, текущая погода и прогноз без API-ключа.",
    version="1.0.0",
)


def _coordinates(latitude: float, longitude: float) -> None:
    if not -90 <= latitude <= 90:
        raise ValueError("latitude должна быть от -90 до 90.")
    if not -180 <= longitude <= 180:
        raise ValueError("longitude должна быть от -180 до 180.")


def _forecast(latitude: float, longitude: float, **params: Any) -> dict[str, Any]:
    _coordinates(latitude, longitude)
    return get_json(FORECAST_URL, {"latitude": latitude, "longitude": longitude, "timezone": "auto", **params})


@server.tool(structured_output=True)
def find_location(query: str, count: int = 5, language: str = "ru", country_code: str | None = None) -> dict[str, Any]:
    """Найти населённый пункт и координаты по названию. Сначала вызови этот инструмент, если координаты неизвестны."""
    query = query.strip()
    if len(query) < 2:
        raise ValueError("query должен содержать хотя бы 2 символа.")
    count = max(1, min(int(count), 10))
    payload = get_json(GEOCODING_URL, {
        "name": query, "count": count, "language": language, "format": "json",
        "countryCode": country_code.strip().upper() if country_code else None,
    })
    locations = []
    for item in payload.get("results") or []:
        locations.append({
            key: item.get(key)
            for key in ("id", "name", "latitude", "longitude", "elevation", "timezone", "country", "country_code", "admin1", "admin2")
            if item.get(key) is not None
        })
    return {"query": query, "count": len(locations), "locations": locations}


@server.tool(structured_output=True)
def get_current_weather(latitude: float, longitude: float) -> dict[str, Any]:
    """Получить текущую погоду по координатам: температуру, ощущаемую температуру, осадки, ветер и код погоды."""
    return _forecast(
        latitude, longitude,
        current="temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,rain,showers,snowfall,weather_code,cloud_cover,wind_speed_10m,wind_direction_10m,wind_gusts_10m",
    )


@server.tool(structured_output=True)
def get_hourly_forecast(latitude: float, longitude: float, hours: int = 24) -> dict[str, Any]:
    """Получить почасовой прогноз на ближайшие 1–96 часов по координатам."""
    hours = max(1, min(int(hours), 96))
    payload = _forecast(latitude, longitude, hourly=HOURLY_FIELDS, forecast_hours=hours)
    return {key: payload.get(key) for key in ("latitude", "longitude", "timezone", "timezone_abbreviation", "utc_offset_seconds", "hourly_units", "hourly")}


@server.tool(structured_output=True)
def get_daily_forecast(latitude: float, longitude: float, days: int = 7) -> dict[str, Any]:
    """Получить дневной прогноз на 1–16 дней по координатам."""
    days = max(1, min(int(days), 16))
    payload = _forecast(latitude, longitude, daily=DAILY_FIELDS, forecast_days=days)
    return {key: payload.get(key) for key in ("latitude", "longitude", "timezone", "timezone_abbreviation", "utc_offset_seconds", "daily_units", "daily")}


@server.tool(structured_output=True)
def get_precipitation_window(
    latitude: float,
    longitude: float,
    hours: int = 48,
    probability_threshold: int = 30,
) -> dict[str, Any]:
    """Найти ближайший непрерывный период вероятных осадков и его пик в следующие 1–168 часов."""
    hours = max(1, min(int(hours), 168))
    probability_threshold = max(0, min(int(probability_threshold), 100))
    payload = _forecast(
        latitude, longitude,
        hourly="precipitation_probability,precipitation,rain,showers,snowfall,weather_code",
        forecast_hours=hours,
    )
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    probabilities = hourly.get("precipitation_probability") or []
    amounts = hourly.get("precipitation") or []
    wet = [
        index for index in range(min(len(times), len(probabilities), len(amounts)))
        if (probabilities[index] or 0) >= probability_threshold or (amounts[index] or 0) > 0
    ]
    if not wet:
        return {
            "found": False, "checked_hours": hours, "probability_threshold": probability_threshold,
            "message": "В выбранном интервале вероятные осадки не найдены.",
        }
    start = wet[0]
    end = start
    wet_set = set(wet)
    while end + 1 in wet_set:
        end += 1
    peak = max(range(start, end + 1), key=lambda index: (probabilities[index] or 0, amounts[index] or 0))
    return {
        "found": True,
        "checked_hours": hours,
        "probability_threshold": probability_threshold,
        "window": {
            "start": times[start], "end": times[end],
            "peak_time": times[peak],
            "peak_probability_percent": probabilities[peak],
            "peak_precipitation_mm": amounts[peak],
            "total_precipitation_mm": round(sum(float(value or 0) for value in amounts[start:end + 1]), 2),
        },
    }


if __name__ == "__main__":
    server.run("stdio")
