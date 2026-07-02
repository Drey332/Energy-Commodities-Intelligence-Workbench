"""Keyless ReliefWeb connector for climate, conflict, and humanitarian risk."""

from __future__ import annotations

from typing import Any, Callable

from devfinintel.connectors.common import build_url, connector_result, fetch_json


RELIEFWEB_REPORTS_API = "https://api.reliefweb.int/v1/reports"


def fetch_reliefweb_signals(
    *,
    query: str = "Africa drought flood conflict displacement food security infrastructure energy mining climate",
    country: str = "",
    topic: str = "",
    lookback_days: int = 30,
    limit: int = 25,
    timeout_seconds: int = 20,
    fetcher: Callable[..., dict[str, Any]] = fetch_json,
) -> dict[str, Any]:
    """Fetch public ReliefWeb reports related to risk context."""

    search_query = " ".join(part for part in [query, country, topic] if part).strip()
    url = build_url(
        RELIEFWEB_REPORTS_API,
        {
            "appname": "devfinintel",
            "profile": "list",
            "preset": "latest",
            "limit": min(max(limit, 1), 100),
            "query[value]": search_query,
        },
    )
    try:
        payload = fetcher(url, timeout_seconds=timeout_seconds)
        records = [normalize_reliefweb_item(item, search_query) for item in payload.get("data", [])]
        return connector_result(
            source_name="ReliefWeb",
            source_type="risk_context",
            source_status="live/keyless",
            records=records,
            query=search_query,
            url=url,
            metadata={"lookback_days": lookback_days, "returned": len(records)},
        )
    except Exception as exc:
        return connector_result(
            source_name="ReliefWeb",
            source_type="risk_context",
            source_status="failed",
            query=search_query,
            url=url,
            errors=[str(exc)],
            metadata={"lookback_days": lookback_days},
        )


def normalize_reliefweb_item(item: dict[str, Any], query: str) -> dict[str, Any]:
    """Map a ReliefWeb item to raw connector fields."""

    fields = item.get("fields", {}) if isinstance(item, dict) else {}
    countries = fields.get("country") or []
    themes = fields.get("theme") or []
    sources = fields.get("source") or []
    title = str(fields.get("title", "") or "Untitled ReliefWeb report")
    return {
        "raw_source_id": str(item.get("id", "")) or title,
        "title": title,
        "date": str((fields.get("date") or {}).get("created", "")),
        "url": str(fields.get("url", "")),
        "source": "; ".join(source.get("name", "") for source in sources if isinstance(source, dict)) or "ReliefWeb",
        "country": "; ".join(country.get("name", "") for country in countries if isinstance(country, dict)),
        "theme_tags": "; ".join(theme.get("name", "") for theme in themes if isinstance(theme, dict)),
        "summary": str(fields.get("body", "") or fields.get("headline", "") or title),
        "evidence_text": f"{title}. {fields.get('body', '') or fields.get('headline', '')}",
        "metadata": {"query": query, "reliefweb_url": fields.get("url_alias", "")},
    }
