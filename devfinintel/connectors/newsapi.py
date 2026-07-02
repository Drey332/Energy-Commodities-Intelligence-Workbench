"""Optional-key NewsAPI connector."""

from __future__ import annotations

import os
from typing import Any, Callable

from devfinintel.connectors.common import build_url, connector_result, fetch_json, missing_key_result


NEWSAPI_URL = "https://newsapi.org/v2/everything"


def fetch_newsapi_signals(
    *,
    query: str,
    lookback_days: int = 7,
    limit: int = 25,
    timeout_seconds: int = 20,
    fetcher: Callable[..., dict[str, Any]] = fetch_json,
) -> dict[str, Any]:
    """Fetch NewsAPI records when a key is configured."""

    api_key = os.getenv("NEWSAPI_API_KEY") or os.getenv("DEVFIN_NEWS_API_KEY")
    if not api_key:
        return missing_key_result("NewsAPI", "NEWSAPI_API_KEY")
    url = build_url(
        NEWSAPI_URL,
        {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": min(max(limit, 1), 100),
        },
    )
    try:
        payload = fetcher(url, timeout_seconds=timeout_seconds, headers={"X-Api-Key": api_key})
        rows = []
        for item in payload.get("articles", []):
            rows.append(
                {
                    "raw_source_id": item.get("url", ""),
                    "title": item.get("title", ""),
                    "date": item.get("publishedAt", ""),
                    "url": item.get("url", ""),
                    "source": (item.get("source") or {}).get("name", "NewsAPI"),
                    "summary": item.get("description") or item.get("content") or "",
                    "evidence_text": f"{item.get('title', '')}. {item.get('description') or item.get('content') or ''}",
                    "metadata": {"query": query, "lookback_days": lookback_days},
                }
            )
        return connector_result(
            source_name="NewsAPI",
            source_type="news",
            source_status="live/optional-key",
            records=rows,
            query=query,
            url=NEWSAPI_URL,
            metadata={"returned": len(rows), "secret_visible": "no"},
        )
    except Exception as exc:
        return connector_result(
            source_name="NewsAPI",
            source_type="news",
            source_status="failed",
            query=query,
            url=NEWSAPI_URL,
            errors=[str(exc)],
            metadata={"secret_visible": "no"},
        )
