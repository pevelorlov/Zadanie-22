"""Небольшой HTTP-клиент для публичных JSON API без новых зависимостей."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def get_json(
    url: str,
    params: dict[str, Any] | None = None,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 20.0,
) -> dict[str, Any]:
    query = urlencode({key: value for key, value in (params or {}).items() if value is not None})
    target = f"{url}?{query}" if query else url
    request = Request(
        target,
        headers={"Accept": "application/json", "User-Agent": "deepseek-agent-mcp/1.0", **(headers or {})},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"API вернул HTTP {error.code}: {body}") from error
    except URLError as error:
        raise RuntimeError(f"Не удалось подключиться к API: {error.reason}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError("API вернул некорректный JSON.") from error
    if not isinstance(payload, dict):
        raise RuntimeError("API вернул JSON неожиданного формата.")
    return payload
