"""Keyless GDELT connector for global news monitoring."""

from __future__ import annotations

from typing import Any, Callable

from devfinintel.connectors.common import build_url, connector_result, fetch_json
from devfinintel.news import DEFAULT_NEWS_QUERY


GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"


def fetch_gdelt_signals(
    *,
    query: str = DEFAULT_NEWS_QUERY,
    country: str = "",
    commodity: str = "",
    topic: str = "",
    lookback_days: int = 7,
    limit: int = 50,
    timeout_seconds: int = 20,
    fetcher: Callable[..., dict[str, Any]] = fetch_json,
) -> dict[str, Any]:
    """Fetch article rows from GDELT's public DOC API."""

    search_query = " ".join(part for part in [query, country, commodity, topic, "Africa"] if part).strip()
    url = build_url(
        GDELT_DOC_API,
        {
            "query": search_query,
            "mode": "ArtList",
            "format": "json",
            "maxrecords": min(max(limit, 1), 250),
            "sort": "HybridRel",
            "timespan": f"{max(1, min(lookback_days, 90))}d",
        },
    )
    try:
        payload = fetcher(url, timeout_seconds=timeout_seconds)
        articles = payload.get("articles", []) if isinstance(payload, dict) else []
        rows = [normalize_gdelt_article(article, search_query) for article in articles]
        return connector_result(
            source_name="GDELT",
            source_type="news",
            source_status="live/keyless",
            records=rows,
            query=search_query,
            url=url,
            metadata={"lookback_days": lookback_days, "returned": len(rows)},
        )
    except Exception as exc:
        return connector_result(
            source_name="GDELT",
            source_type="news",
            source_status="failed",
            query=search_query,
            url=url,
            errors=[str(exc)],
            metadata={"lookback_days": lookback_days},
        )


def normalize_gdelt_article(article: dict[str, Any], query: str) -> dict[str, Any]:
    """Map a GDELT article to raw connector fields."""

    title = str(article.get("title", "") or "Untitled GDELT article")
    return {
        "raw_source_id": str(article.get("url", "")) or title,
        "title": title,
        "date": str(article.get("seendate", "") or article.get("publishedAt", "")),
        "url": str(article.get("url", "")),
        "source": str(article.get("domain", "") or article.get("sourceCountry", "") or "GDELT"),
        "language": str(article.get("language", "")),
        "summary": str(article.get("snippet", "") or article.get("title", "")),
        "tone": str(article.get("tone", "")),
        "evidence_text": f"{title}. {article.get('snippet', '')}",
        "metadata": {
            "query": query,
            "source_country": article.get("sourceCountry", ""),
            "image": article.get("socialimage", ""),
        },
    }
