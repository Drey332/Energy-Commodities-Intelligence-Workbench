"""Country-level intelligence products built from reviewed evidence records.

The evidence store answers "where did this come from?" and the knowledge table
answers "what fact did we extract?" This module answers the next analyst
question: "what should I pay attention to, compare, or do next for this
country?"

The functions are deterministic on purpose. For an institutional knowledge
platform, action recommendations and source comparisons should be transparent
enough for a reviewer to challenge and edit.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from devfinintel.models import ActionItem
from devfinintel.utils import stable_id, utc_now_iso


OFFICIAL_SOURCE_TERMS = {
    "afdb",
    "africa pulse",
    "african development bank",
    "biofin",
    "eiti",
    "government",
    "iea",
    "international energy agency",
    "imf",
    "ministry",
    "oecd",
    "undp",
    "united nations",
    "world bank",
}


PUBLIC_SIGNAL_TERMS = {
    "blog",
    "media",
    "news",
    "press",
    "reuters",
    "social",
    "twitter",
    "x.com",
}


RISK_PRIORITY = {
    "corruption",
    "conflict",
    "debt",
    "governance",
    "social license",
    "implementation delay",
}


def build_country_intelligence(
    *,
    country: str,
    knowledge_records: list[dict[str, Any]],
    coverage_matrix: list[dict[str, Any]],
    action_items: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return one country workspace payload for the UI and CLI."""

    country_records = records_for_country(knowledge_records, country)
    country_actions = [item for item in action_items if item.get("country") == country]
    comparisons = build_official_public_comparisons(country, knowledge_records)
    stakeholders = build_stakeholder_map(country, country_records)
    coverage = [row for row in coverage_matrix if row.get("country") == country]
    open_actions = [item for item in country_actions if item.get("status") in {"open", "in_progress"}]
    high_actions = [
        item for item in open_actions if item.get("priority") in {"urgent", "high"}
    ]

    snapshot = {
        "country": country,
        "records": len(country_records),
        "open_actions": len(open_actions),
        "high_priority_actions": len(high_actions),
        "stakeholders": len(stakeholders),
        "comparisons": len(comparisons),
        "usable_coverage_cells": sum(1 for row in coverage if row.get("status") == "usable_records"),
        "country_specific_gaps": sum(
            1 for row in coverage if row.get("specific_downloaded_sources", 0) == 0
        ),
        "top_sectors": top_counts(country_records, "sector"),
        "top_commodities": top_counts(country_records, "commodity"),
        "top_partners": top_counts(country_records, "partner"),
        "risk_flags": top_split_counts(country_records, "risk_flags"),
        "recommended_actions": top_counts(country_records, "recommended_action"),
    }
    return {
        "snapshot": snapshot,
        "actions": sorted_actions(country_actions),
        "stakeholders": stakeholders,
        "comparisons": comparisons,
        "coverage": coverage,
        "records": sorted(country_records, key=lambda record: record.get("updated_at", ""), reverse=True),
    }


def build_action_items(
    *,
    knowledge_records: list[dict[str, Any]],
    source_backlog_rows: list[dict[str, Any]] | None = None,
    limit: int = 500,
) -> list[ActionItem]:
    """Create action items from knowledge records and coverage gaps."""

    actions: list[ActionItem] = []
    for record in knowledge_records:
        country = normalized_country(record)
        if not country:
            continue
        action_type = infer_action_type(record)
        priority = infer_priority(record)
        now = utc_now_iso()
        action_id = stable_id("action", record.get("record_id", ""), action_type, country)
        actions.append(
            ActionItem(
                action_id=action_id,
                country=country,
                action_type=action_type,
                priority=priority,
                status="open",
                title=action_title(record, action_type),
                rationale=action_rationale(record),
                source_record_id=str(record.get("record_id", "")),
                source_title=str(record.get("source_title") or record.get("title", "")),
                source_page=record.get("source_page"),
                source_path=str(record.get("source_path", "")),
                due_bucket="this_week" if priority in {"urgent", "high"} else "next_review",
                created_at=now,
                updated_at=now,
            )
        )

    for row in source_backlog_rows or []:
        country = str(row.get("country", "")).strip()
        if not country:
            continue
        topic = str(row.get("topic", "")).strip() or "source coverage"
        now = utc_now_iso()
        actions.append(
            ActionItem(
                action_id=stable_id("source-gap", country, row.get("topic_id", ""), row.get("status", "")),
                country=country,
                action_type="add_country_source",
                priority="medium" if row.get("status") == "regional_source_ready" else "high",
                status="open",
                title=f"{country}: add country-specific source for {topic}",
                rationale=str(row.get("recommended_next_source", "")),
                source_record_id="",
                source_title="Coverage matrix",
                source_page=None,
                source_path="",
                due_bucket="next_review",
                created_at=now,
                updated_at=now,
            )
        )

    return unique_action_items(actions)[:limit]


def build_official_public_comparisons(
    country: str,
    knowledge_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compare official baselines with monitoring/public signals for a country."""

    country_records = records_for_country(knowledge_records, country)
    regional_records = [record for record in knowledge_records if record.get("country") == "Africa"]
    baselines = [
        record
        for record in country_records + regional_records
        if source_category(record) == "official"
        and record.get("record_type") in {"partner_profile", "finance_resource_record", "case_study_card"}
    ]
    signals = [
        record
        for record in country_records
        if record.get("record_type") == "monitoring_digest" or source_category(record) != "official"
    ]

    comparisons: list[dict[str, Any]] = []
    for signal in signals:
        baseline = best_baseline_match(signal, baselines)
        relation = relation_between(signal, baseline)
        comparisons.append(
            {
                "country": country,
                "topic": match_key(signal),
                "official_baseline": baseline.get("title", "No official baseline found") if baseline else "No official baseline found",
                "official_source": baseline.get("source_title", "") if baseline else "",
                "public_or_monitoring_signal": signal.get("title", ""),
                "signal_source": signal.get("source_title", ""),
                "relation": relation,
                "risk_flags": signal.get("risk_flags", ""),
                "tone": signal.get("sentiment_tone", ""),
                "recommended_action": signal.get("recommended_action", ""),
                "evidence_page": signal.get("source_page"),
                "source_record_id": signal.get("record_id", ""),
            }
        )
    return comparisons[:50]


def build_stakeholder_map(country: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate actors and partners into a compact country stakeholder map."""

    stakeholders: dict[str, dict[str, Any]] = {}
    for record in records:
        for name in stakeholder_names(record):
            entry = stakeholders.setdefault(
                name,
                {
                    "country": country,
                    "stakeholder": name,
                    "role": stakeholder_role(name),
                    "records": 0,
                    "sectors": Counter(),
                    "commodities": Counter(),
                    "risk_flags": Counter(),
                    "recommended_actions": Counter(),
                    "latest_source": "",
                },
            )
            entry["records"] += 1
            add_counter(entry["sectors"], record.get("sector"))
            add_counter(entry["commodities"], record.get("commodity"))
            for flag in split_values(record.get("risk_flags", "")):
                add_counter(entry["risk_flags"], flag)
            add_counter(entry["recommended_actions"], record.get("recommended_action"))
            if not entry["latest_source"]:
                entry["latest_source"] = record.get("source_title", "")

    rows: list[dict[str, Any]] = []
    for entry in stakeholders.values():
        rows.append(
            {
                "country": entry["country"],
                "stakeholder": entry["stakeholder"],
                "role": entry["role"],
                "records": entry["records"],
                "sectors": compact_counter(entry["sectors"]),
                "commodities": compact_counter(entry["commodities"]),
                "risk_flags": compact_counter(entry["risk_flags"]),
                "recommended_actions": compact_counter(entry["recommended_actions"]),
                "latest_source": entry["latest_source"],
            }
        )
    return sorted(rows, key=lambda row: (-int(row["records"]), row["stakeholder"]))[:100]


def records_for_country(records: list[dict[str, Any]], country: str) -> list[dict[str, Any]]:
    """Return records that explicitly belong to a country."""

    return [record for record in records if record.get("country") == country]


def source_category(record: dict[str, Any]) -> str:
    """Classify source type for official-vs-public comparison."""

    text = " ".join(
        str(record.get(key, ""))
        for key in ("source_title", "source_path", "partner", "actors")
    ).lower()
    if any(term in text for term in PUBLIC_SIGNAL_TERMS):
        return "public"
    if any(term in text for term in OFFICIAL_SOURCE_TERMS):
        return "official"
    return "local_or_uncategorized"


def best_baseline_match(
    signal: dict[str, Any],
    baselines: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Find the closest official baseline for a monitoring signal."""

    signal_key = match_key(signal)
    for baseline in baselines:
        if baseline.get("record_id") == signal.get("record_id"):
            continue
        if match_key(baseline) == signal_key:
            return baseline
    return baselines[0] if baselines else None


def relation_between(signal: dict[str, Any], baseline: dict[str, Any] | None) -> str:
    """Label how a monitoring signal relates to an official baseline."""

    if not baseline:
        return "new signal without official baseline"
    risk_flags = str(signal.get("risk_flags", "")).lower()
    tone = str(signal.get("sentiment_tone", "")).lower()
    if risk_flags and tone in {"mixed", "negative"}:
        return "qualifies official baseline with risk signal"
    if risk_flags:
        return "adds risk context to official baseline"
    if signal.get("amount") and not baseline.get("amount"):
        return "adds finance detail to official baseline"
    return "corroborates active country theme"


def infer_action_type(record: dict[str, Any]) -> str:
    """Normalize a record recommendation into a work queue action type."""

    recommendation = str(record.get("recommended_action", "")).lower()
    if "risk" in recommendation:
        return "review_risk_flags"
    if "bulletin" in recommendation:
        return "add_to_bulletin"
    if "finance record" in recommendation:
        return "create_finance_record"
    if "partner profile" in recommendation:
        return "update_partner_profile"
    if record.get("record_type") == "monitoring_digest":
        return "monitor_signal"
    return "review_record"


def infer_priority(record: dict[str, Any]) -> str:
    """Set action priority using transparent risk/relevance rules."""

    risk_text = str(record.get("risk_flags", "")).lower()
    relevance = str(record.get("relevance", "")).lower()
    status = str(record.get("review_status", "")).lower()
    if any(flag in risk_text for flag in RISK_PRIORITY) and relevance == "high":
        return "urgent"
    if status == "review" or risk_text:
        return "high"
    if relevance == "high":
        return "medium"
    return "low"


def action_title(record: dict[str, Any], action_type: str) -> str:
    """Create a compact action title for dashboards."""

    country = normalized_country(record) or "Unscoped"
    label = action_type.replace("_", " ")
    return f"{country}: {label} - {record.get('title', '')}"


def action_rationale(record: dict[str, Any]) -> str:
    """Explain why an action item exists."""

    parts = [
        f"Recommended action: {record.get('recommended_action', 'review')}",
        f"relevance={record.get('relevance', '')}",
        f"tone={record.get('sentiment_tone', '')}",
    ]
    if record.get("risk_flags"):
        parts.append(f"risk={record.get('risk_flags')}")
    if record.get("source_title"):
        parts.append(f"source={record.get('source_title')}, p. {record.get('source_page')}")
    return "; ".join(part for part in parts if part)


def normalized_country(record: dict[str, Any]) -> str:
    """Return a country suitable for action routing."""

    country = str(record.get("country", "")).strip()
    if country in {"", "Not specified"}:
        return ""
    return country


def stakeholder_names(record: dict[str, Any]) -> list[str]:
    """Return stakeholder names from partner and actor fields."""

    values = split_values(record.get("actors", ""))
    partner = str(record.get("partner", "")).strip()
    if partner:
        values.append(partner)
    return sorted(set(value for value in values if value))


def stakeholder_role(name: str) -> str:
    """Assign a broad stakeholder role."""

    lower = name.lower()
    if "world bank" in lower or "afdb" in lower or "monetary" in lower or lower == "ifi":
        return "IFI"
    if "government" in lower or "ministry" in lower:
        return "Government"
    if "company" in lower or "investor" in lower or "private" in lower:
        return "Private sector"
    if "civil society" in lower or "community" in lower:
        return "Civil society / community"
    if "un" in lower:
        return "UN system"
    return "Stakeholder"


def match_key(record: dict[str, Any]) -> str:
    """Topic key used for coarse official-vs-signal matching."""

    for key in ("sector", "theme", "commodity"):
        value = str(record.get(key, "")).strip()
        if value:
            return value
    return "General"


def top_counts(records: list[dict[str, Any]], key: str, limit: int = 5) -> list[dict[str, Any]]:
    """Return top values for one record key."""

    counter: Counter[str] = Counter()
    for record in records:
        add_counter(counter, record.get(key))
    return counter_rows(counter, limit)


def top_split_counts(records: list[dict[str, Any]], key: str, limit: int = 5) -> list[dict[str, Any]]:
    """Return top semicolon-delimited values for one key."""

    counter: Counter[str] = Counter()
    for record in records:
        for value in split_values(record.get(key, "")):
            add_counter(counter, value)
    return counter_rows(counter, limit)


def add_counter(counter: Counter[str], value: Any) -> None:
    """Increment a counter for a non-empty display value."""

    text = str(value or "").strip()
    if text:
        counter[text] += 1


def counter_rows(counter: Counter[str], limit: int) -> list[dict[str, Any]]:
    """Turn a counter into rows for dashboards."""

    return [{"value": value, "count": count} for value, count in counter.most_common(limit)]


def compact_counter(counter: Counter[str], limit: int = 4) -> str:
    """Return a semicolon-separated summary of a counter."""

    return "; ".join(value for value, _count in counter.most_common(limit))


def split_values(value: Any) -> list[str]:
    """Split semicolon-delimited fields into clean values."""

    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in str(value or "").split(";") if part.strip()]


def sorted_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort actions by status and priority."""

    priority_rank = {"urgent": 0, "high": 1, "medium": 2, "low": 3}
    status_rank = {"open": 0, "in_progress": 1, "blocked": 2, "done": 3, "dismissed": 4}
    return sorted(
        actions,
        key=lambda action: (
            status_rank.get(str(action.get("status", "")), 9),
            priority_rank.get(str(action.get("priority", "")), 9),
            str(action.get("updated_at", "")),
        ),
    )


def unique_action_items(actions: list[ActionItem]) -> list[ActionItem]:
    """Deduplicate action items while preserving first occurrence."""

    seen: set[str] = set()
    result: list[ActionItem] = []
    for action in actions:
        if action.action_id in seen:
            continue
        seen.add(action.action_id)
        result.append(action)
    return result
