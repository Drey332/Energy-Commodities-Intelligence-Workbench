"""News intake for the Africa energy and commodities workbench.

The app should be useful even when the user has no paid API key or network
access. This module therefore uses a provider adapter pattern: try configured
news APIs first, then fall back to clearly marked sample monitoring items.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from devfinintel.extraction import detect_countries, unique_preserve_order
from devfinintel.knowledge import (
    detect_commodities,
    detect_event_types,
    detect_relevance,
    detect_resource_sectors,
    detect_risk_flags,
    detect_sentiment_tone,
    recommend_action,
)
from devfinintel.utils import stable_id, utc_now_iso


DEFAULT_NEWS_QUERY = (
    "Africa energy commodities oil gas mining critical minerals electricity "
    "investment infrastructure climate governance"
)


SAMPLE_ARTICLES = [
    {
        "title": "Nigeria gas infrastructure financing remains central to power-sector reliability debate",
        "source": "Sample monitoring feed",
        "published_at": "2026-06-27T09:00:00Z",
        "summary": (
            "Officials and investors continue to frame gas pipelines, grid reliability, "
            "and tariff reform as linked constraints for Nigeria's electricity sector."
        ),
        "url": "sample://nigeria-gas-power-finance",
    },
    {
        "title": "DRC copper and cobalt value-chain plans raise financing, governance, and local-processing questions",
        "source": "Sample monitoring feed",
        "published_at": "2026-06-25T12:00:00Z",
        "summary": (
            "Critical-minerals investment signals remain strong, but policy analysts "
            "flag infrastructure, transparency, conflict, and community-benefit risks."
        ),
        "url": "sample://drc-copper-cobalt-value-chain",
    },
    {
        "title": "South Africa power-market reforms keep renewables, grid investment, and coal transition in focus",
        "source": "Sample monitoring feed",
        "published_at": "2026-06-23T15:30:00Z",
        "summary": (
            "Energy-market restructuring is increasing attention on transmission "
            "investment, procurement delays, private generation, and just-transition finance."
        ),
        "url": "sample://south-africa-power-reform",
    },
    {
        "title": "Mozambique LNG outlook remains tied to security, local benefits, and fiscal expectations",
        "source": "Sample monitoring feed",
        "published_at": "2026-06-21T08:30:00Z",
        "summary": (
            "Gas-export prospects continue to create development-finance opportunities "
            "while conflict, resettlement, and revenue-governance risks require monitoring."
        ),
        "url": "sample://mozambique-lng-security",
    },
    {
        "title": "Zambia copper-sector momentum highlights power supply and debt-management constraints",
        "source": "Sample monitoring feed",
        "published_at": "2026-06-19T10:15:00Z",
        "summary": (
            "Copper output ambitions are linked to electricity availability, logistics, "
            "debt conditions, and investor confidence in mining policy."
        ),
        "url": "sample://zambia-copper-power-debt",
    },
    {
        "title": "Ghana gold and cocoa revenues renew focus on export exposure and fiscal buffers",
        "source": "Sample monitoring feed",
        "published_at": "2026-06-18T11:45:00Z",
        "summary": (
            "Commodity-price exposure remains relevant for fiscal planning, rural "
            "livelihoods, and finance records that track export-linked development risk."
        ),
        "url": "sample://ghana-gold-cocoa-fiscal",
    },
]


def fetch_news(
    *,
    query: str = DEFAULT_NEWS_QUERY,
    country: str = "",
    commodity: str = "",
    topic: str = "",
    limit: int = 20,
    days: int = 30,
    timeout_seconds: int = 20,
) -> dict[str, Any]:
    """Fetch live news when configured, otherwise return marked sample news.

    Supported environment variables:

    - ``NEWSAPI_API_KEY`` or ``DEVFIN_NEWS_API_KEY`` for NewsAPI.
    - ``GNEWS_API_KEY`` for GNews.

    The return shape is provider-neutral so the Streamlit app does not need to
    know which adapter supplied the articles.
    """

    search_query = build_query(query=query, country=country, commodity=commodity, topic=topic)
    errors: list[str] = []
    articles: list[dict[str, Any]] = []
    provider = "sample"

    newsapi_key = os.getenv("NEWSAPI_API_KEY") or os.getenv("DEVFIN_NEWS_API_KEY")
    gnews_key = os.getenv("GNEWS_API_KEY")

    if newsapi_key:
        try:
            articles = fetch_newsapi(
                search_query,
                api_key=newsapi_key,
                limit=limit,
                days=days,
                timeout_seconds=timeout_seconds,
            )
            provider = "newsapi"
        except Exception as exc:  # pragma: no cover - network/API dependent.
            errors.append(f"NewsAPI failed: {exc}")

    if not articles and gnews_key:
        try:
            articles = fetch_gnews(
                search_query,
                api_key=gnews_key,
                limit=limit,
                timeout_seconds=timeout_seconds,
            )
            provider = "gnews"
        except Exception as exc:  # pragma: no cover - network/API dependent.
            errors.append(f"GNews failed: {exc}")

    if not articles:
        articles = sample_articles(search_query, country=country, commodity=commodity, topic=topic, limit=limit)
        provider = "sample"

    normalized = [normalize_article(article, provider=provider, search_query=search_query) for article in articles]
    return {
        "provider": provider,
        "status": "live" if provider != "sample" else "sample",
        "query": search_query,
        "generated_at": utc_now_iso(),
        "warning": sample_warning(errors) if provider == "sample" else "",
        "errors": errors,
        "articles": normalized[:limit],
        "what_changed": summarize_changes(normalized[:limit]),
    }


def build_query(*, query: str, country: str, commodity: str, topic: str) -> str:
    """Combine user filters into a search query."""

    parts = [query or DEFAULT_NEWS_QUERY, country, commodity, topic, "Africa"]
    return " ".join(part for part in parts if part).strip()


def fetch_newsapi(
    query: str,
    *,
    api_key: str,
    limit: int,
    days: int,
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    """Fetch articles from NewsAPI's everything endpoint."""

    from_date = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
    params = urlencode(
        {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "from": from_date,
            "pageSize": min(max(limit, 1), 100),
        }
    )
    request = Request(
        f"https://newsapi.org/v2/everything?{params}",
        headers={"X-Api-Key": api_key, "User-Agent": "devfinintel/1.0"},
    )
    payload = json.loads(urlopen(request, timeout=timeout_seconds).read().decode("utf-8"))
    if payload.get("status") != "ok":
        raise RuntimeError(payload.get("message", "NewsAPI returned a non-ok response."))
    rows = []
    for item in payload.get("articles", []):
        rows.append(
            {
                "title": item.get("title", ""),
                "source": (item.get("source") or {}).get("name", "NewsAPI"),
                "published_at": item.get("publishedAt", ""),
                "summary": item.get("description") or item.get("content") or "",
                "url": item.get("url", ""),
            }
        )
    return rows


def fetch_gnews(
    query: str,
    *,
    api_key: str,
    limit: int,
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    """Fetch articles from GNews search."""

    params = urlencode(
        {
            "q": query,
            "lang": "en",
            "max": min(max(limit, 1), 100),
            "apikey": api_key,
        }
    )
    request = Request(
        f"https://gnews.io/api/v4/search?{params}",
        headers={"User-Agent": "devfinintel/1.0"},
    )
    payload = json.loads(urlopen(request, timeout=timeout_seconds).read().decode("utf-8"))
    rows = []
    for item in payload.get("articles", []):
        rows.append(
            {
                "title": item.get("title", ""),
                "source": (item.get("source") or {}).get("name", "GNews"),
                "published_at": item.get("publishedAt", ""),
                "summary": item.get("description") or item.get("content") or "",
                "url": item.get("url", ""),
            }
        )
    return rows


def sample_articles(
    query: str,
    *,
    country: str,
    commodity: str,
    topic: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Return relevant sample articles for offline demos."""

    filter_text = " ".join([query, country, commodity, topic]).lower()
    scored = []
    for article in SAMPLE_ARTICLES:
        article_text = f"{article['title']} {article['summary']}".lower()
        score = sum(1 for token in meaningful_terms(filter_text) if token in article_text)
        scored.append((score, article))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [article for _, article in scored[:limit]]


def normalize_article(article: dict[str, Any], *, provider: str, search_query: str) -> dict[str, Any]:
    """Add monitoring metadata and deterministic classification fields."""

    title = str(article.get("title", "") or "Untitled article")
    summary = str(article.get("summary", "") or "")
    text = f"{title}. {summary}"
    countries = detect_countries(text)
    sectors = detect_resource_sectors(text)
    commodities = detect_commodities(text)
    risk_flags = detect_risk_flags(text)
    event_types = detect_event_types(text)
    tone = detect_sentiment_tone(text)
    event_type = "; ".join(event_types)
    return {
        "article_id": stable_id("news", provider, article.get("url", ""), title),
        "provider": provider,
        "title": title,
        "source": str(article.get("source", provider)),
        "published_at": str(article.get("published_at", "")),
        "summary": summary,
        "url": str(article.get("url", "")),
        "country": "; ".join(countries) if countries else "Regional Africa",
        "sector": "; ".join(sectors) if sectors else "Not classified",
        "commodity": "; ".join(commodities) if commodities else "Not specified",
        "topic_tags": "; ".join(unique_preserve_order(sectors + commodities + event_types)),
        "event_type": event_type or "Monitoring signal",
        "sentiment_tone": tone,
        "risk_flags": "; ".join(risk_flags),
        "relevance": detect_relevance("monitoring_digest", f"{search_query} {text}"),
        "recommended_action": recommend_action("monitoring_digest", risk_flags, event_type, text),
        "raw_text": text,
    }


def summarize_changes(articles: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Create a compact monitoring summary from article classifications."""

    if not articles:
        return [
            {
                "signal": "No current articles available",
                "why_it_matters": "The app needs configured news keys, approved feeds, or reviewed local records.",
                "evidence": "No articles returned.",
            }
        ]

    summaries = []
    for article in articles[:5]:
        country = article.get("country") or "Regional Africa"
        commodity = article.get("commodity") or "resources"
        tone = article.get("sentiment_tone") or "neutral"
        risks = article.get("risk_flags") or "no explicit risk flags"
        summaries.append(
            {
                "signal": f"{country}: {article.get('event_type', 'Monitoring signal')}",
                "why_it_matters": f"Tone is {tone}; commodity focus is {commodity}; risks: {risks}.",
                "evidence": f"{article.get('source', '')}: {article.get('title', '')}",
            }
        )
    return summaries


def sample_warning(errors: list[str]) -> str:
    """Explain why sample news is being used."""

    message = (
        "Using sample monitoring news because no configured news API returned live articles. "
        "Set NEWSAPI_API_KEY, DEVFIN_NEWS_API_KEY, or GNEWS_API_KEY for live search."
    )
    if errors:
        message += " " + " ".join(errors[:2])
    return message


def meaningful_terms(text: str) -> list[str]:
    """Tokenize enough for transparent matching without extra dependencies."""

    stopwords = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "about",
        "into",
        "africa",
        "african",
        "energy",
        "commodities",
    }
    return [
        token
        for token in "".join(char.lower() if char.isalnum() else " " for char in text).split()
        if len(token) >= 3 and token not in stopwords
    ]
