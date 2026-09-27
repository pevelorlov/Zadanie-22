"""MCP-сервер данных о качестве воздуха OpenAQ v3."""

from __future__ import annotations

import os
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from mcp_servers.common.http_json import get_json


BASE_URL = "https://api.openaq.org/v3"
server = MCPServer(name="openaq-mcp", title="OpenAQ Air Quality", version="1.0.0")


def _headers() -> dict[str, str]:
    api_key = os.getenv("OPENAQ_API_KEY", "").strip()
    if not api_key:
        raise ToolError("Для OpenAQ v3 задайте бесплатный OPENAQ_API_KEY в файле .env.")
    return {"X-API-Key": api_key}


def _get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        return get_json(f"{BASE_URL}{path}", params, headers=_headers())
    except ToolError:
        raise
    except RuntimeError as error:
        raise ToolError(str(error)) from error


@server.tool(structured_output=True)
def find_air_quality_locations(
    latitude: float,
    longitude: float,
    radius_meters: int = 25000,
    limit: int = 10,
) -> dict[str, Any]:
    """Найти ближайшие станции OpenAQ по координатам в радиусе до 25 км."""
    radius_meters = max(1, min(int(radius_meters), 25000))
    limit = max(1, min(int(limit), 100))
    payload = _get("/locations", {
        "coordinates": f"{latitude},{longitude}", "radius": radius_meters, "limit": limit,
    })
    locations = []
    for item in payload.get("results") or []:
        locations.append({
            "id": item.get("id"), "name": item.get("name"), "locality": item.get("locality"),
            "country": (item.get("country") or {}).get("name"), "coordinates": item.get("coordinates"),
            "distance_meters": item.get("distance"), "sensors": item.get("sensors"),
        })
    return {"count": len(locations), "locations": locations}


@server.tool(structured_output=True)
def get_location_details(location_id: int) -> dict[str, Any]:
    """Получить описание станции OpenAQ и список её датчиков."""
    payload = _get(f"/locations/{int(location_id)}")
    return {"location": (payload.get("results") or [None])[0]}


@server.tool(structured_output=True)
def get_latest_measurements(location_id: int) -> dict[str, Any]:
    """Получить последние доступные измерения всех датчиков выбранной станции OpenAQ."""
    payload = _get(f"/locations/{int(location_id)}/latest")
    return {"location_id": int(location_id), "measurements": payload.get("results") or []}


if __name__ == "__main__":
    server.run("stdio")
