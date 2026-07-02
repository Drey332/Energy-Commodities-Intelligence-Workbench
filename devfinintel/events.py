"""Cluster normalized monitoring signals into developments."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

from devfinintel.signals import deduplicate_signals, split_labels, tokenize
from devfinintel.utils import stable_id, utc_now_iso


def cluster_signals(signals: list[dict[str, Any]], *, date_window_days: int = 7) -> list[dict[str, Any]]:
    """Group related signals into event clusters."""

    unique_signals = deduplicate_signals(signals)
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for signal in unique_signals:
        country = first_label(signal.get("country", "Regional Africa"))
        topic = first_label(signal.get("commodity", "")) or first_label(signal.get("sector", "")) or "General"
        event_type = signal.get("event_type") or "monitoring signal"
        period = date_bucket(signal.get("date", ""), date_window_days=date_window_days)
        groups[(country, topic, event_type, period)].append(signal)

    clusters = []
    for key, grouped in groups.items():
        clusters.append(build_cluster(key, grouped))
    clusters.sort(
        key=lambda row: (risk_sort_value(row["risk_level"]), row["source_count"], row["latest_update"]),
        reverse=True,
    )
    return clusters


def build_cluster(key: tuple[str, str, str, str], signals: list[dict[str, Any]]) -> dict[str, Any]:
    """Build one event-cluster row."""

    country, topic, event_type, period = key
    countries = sorted({label for signal in signals for label in split_labels(signal.get("country", "")) if label})
    commodities = sorted({label for signal in signals for label in split_labels(signal.get("commodity", "")) if label and label != "Not specified"})
    sectors = sorted({label for signal in signals for label in split_labels(signal.get("sector", "")) if label and label != "Not classified"})
    risk_flags = sorted({label for signal in signals for label in split_labels(signal.get("risk_flags", "")) if label})
    source_names = sorted({signal.get("source_name", "") for signal in signals if signal.get("source_name")})
    latest_update = max((signal.get("date", "") for signal in signals), default="")
    relevance = sum(float(signal.get("relevance_score", 0.0) or 0.0) for signal in signals) / max(len(signals), 1)
    risk_level = classify_risk_level(signals, risk_flags)
    event_title = cluster_title(country=country, topic=topic, event_type=event_type)
    evidence_summary = summarize_evidence(signals)
    return {
        "event_id": stable_id("event-cluster", country, topic, event_type, period, ";".join(s["signal_id"] for s in signals)),
        "event_title": event_title,
        "countries": "; ".join(countries) if countries else country,
        "commodities": "; ".join(commodities) if commodities else topic,
        "sectors": "; ".join(sectors),
        "event_type": event_type,
        "risk_level": risk_level,
        "what_changed": what_changed(country, topic, event_type, signals),
        "why_it_matters": why_it_matters(topic, event_type, risk_flags),
        "supporting_signal_ids": [signal["signal_id"] for signal in signals],
        "source_count": len(source_names),
        "signal_count": len(signals),
        "latest_update": latest_update,
        "confidence_level": confidence_level(len(signals), len(source_names), relevance),
        "evidence_summary": evidence_summary,
        "risk_flags": "; ".join(risk_flags),
        "sources": "; ".join(source_names),
        "period": period,
        "generated_at": utc_now_iso(),
    }


def summarize_monitoring_cycle(
    *,
    source_results: list[dict[str, Any]],
    signals: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    fallback_used: bool,
) -> list[dict[str, str]]:
    """Create a compact 'what changed' summary for the workbench."""

    if clusters:
        return [
            {
                "development": cluster["event_title"],
                "what_changed": cluster["what_changed"],
                "why_it_matters": cluster["why_it_matters"],
                "confidence": cluster["confidence_level"],
            }
            for cluster in clusters[:6]
        ]
    failed = [row["source_name"] for row in source_results if row.get("source_status") == "failed"]
    message = "No event clusters were produced."
    if fallback_used:
        message += " Fallback sample signals were used."
    if failed:
        message += f" Failed sources: {', '.join(failed[:4])}."
    return [
        {
            "development": "Insufficient clustered evidence",
            "what_changed": message,
            "why_it_matters": "The analyst should add sources, adjust the query, or check network/API configuration.",
            "confidence": "low",
        }
    ]


def clusters_to_evidence_rows(clusters: list[dict[str, Any]], signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten clusters and supporting signals into an evidence table."""

    signals_by_id = {signal["signal_id"]: signal for signal in signals}
    rows = []
    for cluster in clusters:
        for signal_id in cluster.get("supporting_signal_ids", []):
            signal = signals_by_id.get(signal_id)
            if not signal:
                continue
            rows.append(
                {
                    "event_id": cluster["event_id"],
                    "event_title": cluster["event_title"],
                    "signal_id": signal_id,
                    "source_type": signal.get("source_type", ""),
                    "source_name": signal.get("source_name", ""),
                    "title": signal.get("title", ""),
                    "date": signal.get("date", ""),
                    "country": signal.get("country", ""),
                    "commodity": signal.get("commodity", ""),
                    "sector": signal.get("sector", ""),
                    "event_type": signal.get("event_type", ""),
                    "risk_flags": signal.get("risk_flags", ""),
                    "relevance_score": signal.get("relevance_score", 0.0),
                    "url": signal.get("url", ""),
                    "evidence_text": signal.get("evidence_text", ""),
                }
            )
    return rows


def cluster_title(*, country: str, topic: str, event_type: str) -> str:
    """Create a compact cluster title."""

    return f"{country} {topic} {event_type}".replace("  ", " ").strip().title()


def what_changed(country: str, topic: str, event_type: str, signals: list[dict[str, Any]]) -> str:
    """Summarize what changed."""

    titles = [signal.get("title", "") for signal in signals[:3] if signal.get("title")]
    base = f"{len(signals)} signal(s) mention {topic} in {country} around {event_type}."
    if titles:
        base += " Examples: " + "; ".join(titles[:2]) + "."
    return base


def why_it_matters(topic: str, event_type: str, risk_flags: list[str]) -> str:
    """Explain policy/finance relevance."""

    reasons = []
    if topic.lower() in {"oil", "gas", "lng", "refined fuel"}:
        reasons.append("energy security, fiscal revenue, imports, and inflation")
    elif topic.lower() in {"cobalt", "copper", "lithium", "critical minerals", "mining"}:
        reasons.append("industrial value chains, export revenue, infrastructure, and governance")
    elif topic.lower() in {"electricity", "power", "renewables", "solar", "wind"}:
        reasons.append("energy access, grid reliability, private investment, and transition finance")
    else:
        reasons.append("development finance, investment planning, and country risk")
    if event_type in {"delay", "conflict", "dispute", "market pressure"}:
        reasons.append("implementation risk and monitoring urgency")
    if risk_flags:
        reasons.append("risk flags: " + ", ".join(risk_flags[:4]))
    return "; ".join(reasons) + "."


def classify_risk_level(signals: list[dict[str, Any]], risk_flags: list[str]) -> str:
    """Classify cluster risk from flags and tone."""

    negative = sum(1 for signal in signals if signal.get("tone") in {"negative", "mixed"})
    if len(risk_flags) >= 2 or negative >= 2:
        return "high"
    if risk_flags or negative:
        return "medium"
    return "low"


def confidence_level(signal_count: int, source_count: int, relevance: float) -> str:
    """Classify cluster confidence from source diversity and relevance."""

    if signal_count >= 3 and source_count >= 2 and relevance >= 0.5:
        return "high"
    if signal_count >= 2 or relevance >= 0.35:
        return "medium"
    return "low"


def summarize_evidence(signals: list[dict[str, Any]]) -> str:
    """Create a short source/evidence summary."""

    source_counts = Counter(signal.get("source_name", "Unknown source") for signal in signals)
    titles = [signal.get("title", "") for signal in signals if signal.get("title")]
    return f"Sources: {dict(source_counts)}. Evidence examples: {'; '.join(titles[:3])}."


def first_label(value: str) -> str:
    """Return the first semicolon-delimited label."""

    labels = split_labels(value)
    if labels:
        if labels[0] in {"Not specified", "Not classified"} and len(labels) > 1:
            return labels[1]
        return labels[0]
    return ""


def date_bucket(value: str, *, date_window_days: int) -> str:
    """Bucket a date into a week-ish period."""

    parsed = parse_date(value)
    if not parsed:
        return "undated"
    if date_window_days <= 7:
        return f"{parsed.isocalendar().year}-W{parsed.isocalendar().week:02d}"
    return f"{parsed.year}-{parsed.month:02d}"


def parse_date(value: str) -> datetime | None:
    """Parse common connector date strings."""

    if not value:
        return None
    raw = str(value).replace("Z", "+00:00")
    for candidate in (raw, raw[:19], raw[:10]):
        try:
            return datetime.fromisoformat(candidate)
        except Exception:
            continue
    try:
        return datetime.strptime(str(value)[:8], "%Y%m%d")
    except Exception:
        return None


def risk_sort_value(level: str) -> int:
    """Sort risk levels."""

    return {"low": 1, "medium": 2, "high": 3}.get(level, 0)


def keyword_overlap(left: str, right: str) -> float:
    """Compute transparent title overlap for tests and future clustering."""

    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
