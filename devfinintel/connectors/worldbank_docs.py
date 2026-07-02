"""Keyless World Bank Documents connector."""

from __future__ import annotations

from typing import Any, Callable

from devfinintel.connectors.common import build_url, connector_result, fetch_json
from devfinintel.news import DEFAULT_NEWS_QUERY


WORLD_BANK_DOCS_API = "https://search.worldbank.org/api/v2/wds"


def fetch_worldbank_document_signals(
    *,
    query: str = DEFAULT_NEWS_QUERY,
    country: str = "",
    commodity: str = "",
    topic: str = "",
    limit: int = 25,
    timeout_seconds: int = 20,
    fetcher: Callable[..., dict[str, Any]] = fetch_json,
) -> dict[str, Any]:
    """Search public World Bank documents and return institutional signals."""

    search_query = " ".join(part for part in [query, country, commodity, topic] if part).strip()
    url = build_url(
        WORLD_BANK_DOCS_API,
        {
            "format": "json",
            "rows": min(max(limit, 1), 100),
            "qterm": search_query,
            "fl": "docna,display_title,docty,docdt,txturl,url,abstracts,admreg,majtheme,countryshortname",
        },
    )
    try:
        payload = fetcher(url, timeout_seconds=timeout_seconds)
        documents = payload.get("documents", {}) if isinstance(payload, dict) else {}
        rows = [normalize_document(doc_id, doc, search_query) for doc_id, doc in documents.items() if isinstance(doc, dict)]
        return connector_result(
            source_name="World Bank Documents",
            source_type="institutional_report",
            source_status="live/keyless",
            records=rows,
            query=search_query,
            url=url,
            metadata={"returned": len(rows)},
        )
    except Exception as exc:
        return connector_result(
            source_name="World Bank Documents",
            source_type="institutional_report",
            source_status="failed",
            query=search_query,
            url=url,
            errors=[str(exc)],
        )


def normalize_document(doc_id: str, doc: dict[str, Any], query: str) -> dict[str, Any]:
    """Map a World Bank document search result to raw connector fields."""

    title = str(doc.get("display_title") or doc.get("docna") or "Untitled World Bank document")
    abstract = extract_abstract(doc.get("abstracts"))
    url = str(doc.get("txturl") or doc.get("url") or "")
    return {
        "raw_source_id": str(doc_id),
        "title": title,
        "date": str(doc.get("docdt", "")),
        "url": url,
        "source": "World Bank Documents",
        "country": str(doc.get("countryshortname", "")),
        "document_type": str(doc.get("docty", "")),
        "theme_tags": str(doc.get("majtheme", "")),
        "summary": abstract or title,
        "evidence_text": f"{title}. {abstract}",
        "metadata": {
            "query": query,
            "region": doc.get("admreg", ""),
            "document_type": doc.get("docty", ""),
        },
    }


def extract_abstract(value: Any) -> str:
    """Extract a readable abstract from World Bank's variable JSON shape."""

    if isinstance(value, dict):
        return " ".join(str(item) for item in value.values() if item)
    if isinstance(value, list):
        return " ".join(str(item) for item in value if item)
    return str(value or "")
