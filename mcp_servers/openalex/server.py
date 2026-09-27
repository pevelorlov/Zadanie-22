"""MCP-сервер поиска научных работ и авторов OpenAlex."""

from __future__ import annotations

import os
from typing import Any

from mcp.server.mcpserver import MCPServer

from mcp_servers.common.http_json import get_json


BASE_URL = "https://api.openalex.org"
server = MCPServer(name="openalex-mcp", title="OpenAlex Research", version="1.0.0")


def _params(values: dict[str, Any]) -> dict[str, Any]:
    api_key = os.getenv("OPENALEX_API_KEY")
    if api_key:
        values["api_key"] = api_key
    return values


def _work(item: dict[str, Any]) -> dict[str, Any]:
    primary = item.get("primary_location") or {}
    source = primary.get("source") or {}
    return {
        "id": item.get("id"), "doi": item.get("doi"), "title": item.get("display_name"),
        "publication_year": item.get("publication_year"), "type": item.get("type"),
        "cited_by_count": item.get("cited_by_count"), "open_access": item.get("open_access"),
        "source": source.get("display_name"), "landing_page_url": primary.get("landing_page_url"),
        "authors": [entry.get("author", {}).get("display_name") for entry in (item.get("authorships") or [])[:10]],
    }


@server.tool(structured_output=True)
def search_works(query: str, limit: int = 5, year_from: int | None = None) -> dict[str, Any]:
    """Найти научные работы по теме; при необходимости ограничить начальным годом публикации."""
    query = query.strip()
    if not query:
        raise ValueError("query не должен быть пустым.")
    limit = max(1, min(int(limit), 25))
    params: dict[str, Any] = {"search": query, "per-page": limit}
    if year_from is not None:
        params["filter"] = f"from_publication_date:{int(year_from)}-01-01"
    payload = get_json(f"{BASE_URL}/works", _params(params))
    works = [_work(item) for item in payload.get("results") or []]
    return {"query": query, "count": len(works), "works": works}


@server.tool(structured_output=True)
def get_work(work_id: str) -> dict[str, Any]:
    """Получить одну научную работу по OpenAlex ID, DOI или URL DOI."""
    work_id = work_id.strip()
    if not work_id:
        raise ValueError("work_id не должен быть пустым.")
    payload = get_json(f"{BASE_URL}/works/{work_id}", _params({}))
    return _work(payload)


@server.tool(structured_output=True)
def search_authors(query: str, limit: int = 5) -> dict[str, Any]:
    """Найти авторов по имени и вернуть OpenAlex ID, организацию и показатели цитирования."""
    query = query.strip()
    if not query:
        raise ValueError("query не должен быть пустым.")
    limit = max(1, min(int(limit), 25))
    payload = get_json(f"{BASE_URL}/authors", _params({"search": query, "per-page": limit}))
    authors = []
    for item in payload.get("results") or []:
        institutions = item.get("last_known_institutions") or []
        authors.append({
            "id": item.get("id"), "name": item.get("display_name"), "orcid": item.get("orcid"),
            "works_count": item.get("works_count"), "cited_by_count": item.get("cited_by_count"),
            "institutions": [institution.get("display_name") for institution in institutions],
        })
    return {"query": query, "count": len(authors), "authors": authors}


if __name__ == "__main__":
    server.run("stdio")
