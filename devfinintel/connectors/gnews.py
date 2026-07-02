"""Optional-key GNews connector."""

from __future__ import annotations

import os
from typing import Any, Callable

from devfinintel.connectors.common import build_url, connector_result, fetch_json, missing_key_result


GNEWS_URL = "https://gnews.io/api/v4/search"


def fetch_gnews_signals(
    *,
    query: str,
    limit: int = 25,
    timeout_seconds: int = 20,
    fetcher: Callable[..., dict[str, Any]] = fetch_json,
) -> dict[str, Any]:
    """Fetch GNews records when a key is configured."""

    api_key = os.getenv("GNEWS_API_KEY")
    if not api_key:
        return missing_key_result("GNews", "GNEWS_API_KEY")
    url = build_url(GNEWS_URL, {"q": query, "lang": "en", "max": min(max(limit, 1), 100), "apikey": api_key})
    safe_url = build_url(GNEWS_URL, {"q": query, "lang": "en", "max": min(max(limit, 1), 100)})
    try:
        payload = fetcher(url, timeout_seconds=timeout_seconds)
        rows = []
        for item in payload.get("articles", []):
            rows.append(
                {
                    "raw_source_id": item.get("url", ""),
                    "title": item.get("title", ""),
                    "date": item.get("publishedAt", ""),
                    "url": item.get("url", ""),
                    "source": (item.get("source") or {}).get("name", "GNews"),
                    "summary": item.get("description") or item.get("content") or "",
                    "evidence_text": f"{item.get('title', '')}. {item.get('description') or item.get('content') or ''}",
                    "metadata": {"query": query},
                }
            )
        return connector_result(
            source_name="GNews",
            source_type="news",
            source_status="live/optional-key",
            records=rows,
            query=query,
            url=safe_url,
            metadata={"returned": len(rows), "secret_visible": "no"},
        )
    except Exception as exc:
        return connector_result(
            source_name="GNews",
            source_type="news",
            source_status="failed",
            query=query,
            url=safe_url,
            errors=[str(exc)],
            metadata={"secret_visible": "no"},
        )
