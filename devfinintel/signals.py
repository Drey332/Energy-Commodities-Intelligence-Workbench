"""Normalize connector outputs into common monitoring signals."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from devfinintel.extraction import detect_countries, unique_preserve_order
from devfinintel.knowledge import (
    detect_event_types,
    detect_resource_sectors,
    detect_risk_flags,
    detect_sentiment_tone,
    region_for_country,
)
from devfinintel.utils import stable_id, utc_now_iso


COMMODITY_TERMS = {
    "oil": "Oil",
    "crude": "Oil",
    "fuel": "Refined fuel",
    "petrol": "Refined fuel",
    "diesel": "Refined fuel",
    "gasoline": "Refined fuel",
    "gas": "Gas",
    "lng": "LNG",
    "electricity": "Electricity",
    "power": "Power",
    "renewable": "Renewables",
    "solar": "Solar",
    "wind": "Wind",
    "hydro": "Hydropower",
    "mining": "Mining",
    "mineral": "Critical minerals",
    "critical minerals": "Critical minerals",
    "cobalt": "Cobalt",
    "copper": "Copper",
    "lithium": "Lithium",
    "gold": "Gold",
    "uranium": "Uranium",
    "cocoa": "Cocoa",
    "commodity": "Commodities",
}

SECTOR_TERMS = {
    "oil": "Oil and gas",
    "gas": "Oil and gas",
    "lng": "Oil and gas",
    "fuel": "Fuel markets",
    "electricity": "Power",
    "power": "Power",
    "grid": "Power",
    "renewable": "Renewables",
    "solar": "Renewables",
    "wind": "Renewables",
    "mining": "Mining",
    "mineral": "Mining",
    "infrastructure": "Infrastructure",
    "transport": "Infrastructure",
    "finance": "Development finance",
    "investment": "Investment",
    "climate": "Climate risk",
    "conflict": "Conflict risk",
    "regulation": "Regulation",
}

EVENT_TERMS = {
    "approved": "financing",
    "financing": "financing",
    "loan": "financing",
    "grant": "financing",
    "investment": "investment",
    "invest": "investment",
    "launched": "project launch",
    "launch": "project launch",
    "construction": "project implementation",
    "delay": "delay",
    "delayed": "delay",
    "suspended": "delay",
    "policy": "policy reform",
    "regulation": "regulation",
    "tariff": "regulation",
    "prices": "market pressure",
    "price": "market pressure",
    "exports": "trade/export",
    "export": "trade/export",
    "conflict": "conflict",
    "dispute": "dispute",
    "drought": "climate shock",
    "flood": "climate shock",
}

RELEVANCE_TERMS = {
    "energy",
    "oil",
    "gas",
    "lng",
    "electricity",
    "power",
    "renewable",
    "mining",
    "critical",
    "mineral",
    "cobalt",
    "copper",
    "lithium",
    "gold",
    "uranium",
    "fuel",
    "commodity",
    "infrastructure",
    "investment",
    "finance",
    "climate",
    "conflict",
    "regulation",
}


def normalize_signal(
    raw: dict[str, Any],
    *,
    source_type: str,
    source_name: str,
    source_status: str,
    default_region: str = "Africa",
) -> dict[str, Any]:
    """Normalize one raw connector row into a monitoring signal."""

    evidence_text = str(raw.get("evidence_text") or raw.get("summary") or raw.get("title") or "")
    title = str(raw.get("title") or "Untitled signal")
    text = f"{title}. {evidence_text} {raw.get('country', '')} {raw.get('theme_tags', '')}"
    countries = classify_country(text, raw.get("country", ""))
    commodities = classify_commodity(text)
    sectors = classify_sector(text)
    event_type = classify_event_type(text)
    risk_flags = classify_risk_flags(text)
    score = compute_relevance_score(text, source_type=source_type, source_status=source_status)
    primary_country = countries[0] if countries else "Regional Africa"
    return {
        "signal_id": generate_signal_id(raw, source_name=source_name, source_type=source_type),
        "source_type": source_type,
        "source_name": source_name,
        "source_status": source_status,
        "title": title,
        "date": str(raw.get("date", "")),
        "url": str(raw.get("url", "")),
        "country": "; ".join(countries) if countries else "Regional Africa",
        "region": region_for_country(primary_country) if primary_country != "Regional Africa" else default_region,
        "commodity": "; ".join(commodities) if commodities else "Not specified",
        "sector": "; ".join(sectors) if sectors else "Not classified",
        "event_type": event_type,
        "summary": clean_summary(str(raw.get("summary") or evidence_text or title)),
        "tone": str(raw.get("tone") or detect_sentiment_tone(text)),
        "risk_flags": "; ".join(risk_flags),
        "relevance_score": score,
        "evidence_text": clean_summary(evidence_text or title),
        "retrieved_at": utc_now_iso(),
        "raw_source_id": str(raw.get("raw_source_id") or raw.get("url") or title),
        "metadata": raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {},
    }


def normalize_connector_result(result: dict[str, Any], *, default_region: str = "Africa") -> list[dict[str, Any]]:
    """Normalize all records from one connector result."""

    return [
        normalize_signal(
            record,
            source_type=result.get("source_type", "unknown"),
            source_name=result.get("source_name", "Unknown source"),
            source_status=result.get("source_status", "unknown"),
            default_region=default_region,
        )
        for record in result.get("records", [])
    ]


def generate_signal_id(raw: dict[str, Any], *, source_name: str, source_type: str) -> str:
    """Create a stable signal ID."""

    return stable_id(source_type, source_name, raw.get("raw_source_id", ""), raw.get("url", ""), raw.get("title", ""))


def classify_country(text: str, explicit_country: Any = "") -> list[str]:
    """Detect African countries from explicit source fields and evidence text."""

    candidates = []
    if explicit_country:
        candidates.extend(split_labels(str(explicit_country)))
    candidates.extend(detect_countries(text))
    normalized = []
    aliases = {
        "DRC": "Democratic Republic of the Congo",
        "Cote d'Ivoire": "Côte d'Ivoire",
        "Côte d’Ivoire": "Côte d'Ivoire",
    }
    for value in candidates:
        normalized.append(aliases.get(value, value))
    unique = unique_preserve_order(normalized)
    lowered_text = text.lower()
    return [
        country
        for country in unique
        if not any(
            country != other
            and country.lower() in other.lower()
            and other.lower() in lowered_text
            for other in unique
        )
    ]


def classify_commodity(text: str) -> list[str]:
    """Detect commodities and energy resources."""

    lower = text.lower()
    return unique_preserve_order(label for term, label in COMMODITY_TERMS.items() if term in lower)


def classify_sector(text: str) -> list[str]:
    """Detect broad sectors."""

    lower = text.lower()
    sectors = [label for term, label in SECTOR_TERMS.items() if term in lower]
    sectors.extend(detect_resource_sectors(text))
    return unique_preserve_order(sectors)


def classify_event_type(text: str) -> str:
    """Detect one broad event type."""

    lower = text.lower()
    labels = [label for term, label in EVENT_TERMS.items() if term in lower]
    labels.extend(detect_event_types(text))
    return unique_preserve_order(labels)[0] if labels else "monitoring signal"


def classify_risk_flags(text: str) -> list[str]:
    """Detect risk flags."""

    return detect_risk_flags(text)


def compute_relevance_score(text: str, *, source_type: str, source_status: str) -> float:
    """Score relevance to African energy and commodities."""

    tokens = set(tokenize(text))
    hits = len(tokens & RELEVANCE_TERMS)
    source_bonus = 0.15 if source_type in {"news", "institutional_report", "dataset_indicator"} else 0.0
    status_bonus = 0.1 if "live" in source_status else 0.0
    score = min(1.0, hits / 6 + source_bonus + status_bonus)
    return round(score, 3)


def deduplicate_signals(signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate signals by URL or normalized title."""

    seen = set()
    output = []
    for signal in signals:
        key = signal.get("url") or normalize_title(signal.get("title", ""))
        if key in seen:
            continue
        seen.add(key)
        output.append(signal)
    return output


def split_labels(value: str) -> list[str]:
    """Split semicolon/comma labels."""

    return [item.strip() for item in re.split(r"[;,]", value) if item.strip()]


def tokenize(text: str) -> list[str]:
    """Tokenize text for scoring."""

    return re.findall(r"[a-zA-Z][a-zA-Z0-9']+", text.lower())


def normalize_title(title: str) -> str:
    """Normalize a title for deduplication."""

    return " ".join(tokenize(title))[:160]


def clean_summary(text: str, limit: int = 700) -> str:
    """Make connector text safe for tables and briefs."""

    cleaned = re.sub(r"<[^>]+>", " ", text)
    cleaned = " ".join(cleaned.split())
    return cleaned[:limit].rstrip()


def signal_summary(signals: list[dict[str, Any]]) -> dict[str, Any]:
    """Return quick signal counts for status panels."""

    def top_split(field: str, limit: int = 10) -> list[dict[str, Any]]:
        counts: Counter[str] = Counter()
        for signal in signals:
            for value in split_labels(str(signal.get(field, ""))):
                if value and value not in {"Not specified", "Not classified"}:
                    counts[value] += 1
        return [{"value": value, "count": count} for value, count in counts.most_common(limit)]

    return {
        "signals": len(signals),
        "sources": dict(Counter(signal.get("source_name", "unknown") for signal in signals)),
        "countries": dict(Counter(signal.get("country", "Regional Africa") for signal in signals)),
        "event_types": dict(Counter(signal.get("event_type", "monitoring signal") for signal in signals)),
        "top_countries": top_split("country"),
        "top_sectors": top_split("sector"),
        "top_commodities": top_split("commodity"),
        "top_risk_flags": top_split("risk_flags"),
    }
