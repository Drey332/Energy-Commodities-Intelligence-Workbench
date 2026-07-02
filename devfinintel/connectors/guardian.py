"""Optional-key Guardian Open Platform connector."""

from __future__ import annotations

import os
from typing import Any, Callable

from devfinintel.connectors.common import build_url, connector_result, fetch_json, missing_key_result


GUARDIAN_URL = "https://content.guardianapis.com/search"


def fetch_guardian_signals(
    *,
    query: str,
    limit: int = 25,
    timeout_seconds: int = 20,
    fetcher: Callable[..., dict[str, Any]] = fetch_json,
) -> dict[str, Any]:
    """Fetch Guardian records when a key is configured."""

    api_key = os.getenv("GUARDIAN_API_KEY")
    if not api_key:
        return missing_key_result("Guardian", "GUARDIAN_API_KEY")
    url = build_url(
        GUARDIAN_URL,
        {
            "q": query,
            "page-size": min(max(limit, 1), 50),
            "order-by": "newest",
            "show-fields": "trailText,bodyText",
            "api-key": api_key,
        },
    )
    safe_url = build_url(
        GUARDIAN_URL,
        {
            "q": query,
            "page-size": min(max(limit, 1), 50),
            "order-by": "newest",
            "show-fields": "trailText,bodyText",
        },
    )
    try:
        payload = fetcher(url, timeout_seconds=timeout_seconds)
        rows = []
        for item in (payload.get("response") or {}).get("results", []):
            fields = item.get("fields") or {}
            rows.append(
                {
                    "raw_source_id": item.get("id", ""),
                    "title": item.get("webTitle", ""),
                    "date": item.get("webPublicationDate", ""),
                    "url": item.get("webUrl", ""),
                    "source": "The Guardian",
                    "summary": fields.get("trailText") or fields.get("bodyText", "")[:500],
                    "evidence_text": f"{item.get('webTitle', '')}. {fields.get('trailText') or fields.get('bodyText', '')[:500]}",
                    "metadata": {"query": query, "section": item.get("sectionName", "")},
                }
            )
        return connector_result(
            source_name="Guardian",
            source_type="news",
            source_status="live/optional-key",
            records=rows,
            query=query,
            url=safe_url,
            metadata={"returned": len(rows), "secret_visible": "no"},
        )
    except Exception as exc:
        return connector_result(
            source_name="Guardian",
            source_type="news",
            source_status="failed",
            query=query,
            url=safe_url,
            errors=[str(exc)],
            metadata={"secret_visible": "no"},
        )
