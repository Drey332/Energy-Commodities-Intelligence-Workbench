"""Monitoring intake, event normalization, and insight summaries.

The monitoring layer is intentionally separate from document retrieval. Reports
and PDFs produce evidence records; news, press releases, RSS feeds, and reviewed
source updates produce dated monitoring events. Both can be compared later, but
they should not be mixed at ingestion time.
"""

from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from html import unescape
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from devfinintel.extraction import detect_countries, unique_preserve_order
from devfinintel.knowledge import (
    detect_event_types,
    detect_relevance,
    detect_resource_sectors,
    detect_risk_flags,
    detect_sentiment_tone,
    detect_commodities,
    region_for_country,
    recommend_action,
)
from devfinintel.models import MonitoringEvent, MonitoringSource
from devfinintel.utils import stable_id, utc_now_iso


MONITORING_SOURCE_FIELDS = [
    "source_id",
    "name",
    "publisher",
    "url",
    "source_type",
    "scope",
    "topics",
    "countries",
    "credibility_tier",
    "refresh_cadence",
    "status",
    "last_checked_at",
    "notes",
    "created_at",
    "updated_at",
]


DEFAULT_MONITORING_SOURCES = [
    MonitoringSource(
        source_id="manual-reviewed-monitoring",
        name="Reviewed local monitoring items",
        publisher="Local analyst workspace",
        url="local://knowledge-records",
        source_type="manual",
        scope="reviewed local evidence",
        topics="energy; oil; gas; mining; infrastructure; finance; governance",
        countries="ALL_AFRICA",
        credibility_tier="reviewed_workspace",
        refresh_cadence="on_demand",
        status="active",
        last_checked_at="",
        notes="Converts generated monitoring and finance/resource records into dated monitoring events.",
        created_at="",
        updated_at="",
    ),
    MonitoringSource(
        source_id="world-bank-africa-news",
        name="World Bank Africa news and press updates",
        publisher="World Bank",
        url="https://www.worldbank.org/en/region/afr/whats-new",
        source_type="web_page",
        scope="official institutional news",
        topics="development finance; energy; infrastructure; jobs; private sector",
        countries="ALL_AFRICA",
        credibility_tier="official",
        refresh_cadence="weekly",
        status="registered",
        last_checked_at="",
        notes="Registered official source. Add an RSS/API endpoint when available for automated refresh.",
        created_at="",
        updated_at="",
    ),
    MonitoringSource(
        source_id="afdb-news",
        name="African Development Bank news",
        publisher="African Development Bank",
        url="https://www.afdb.org/en/news-and-events",
        source_type="web_page",
        scope="official institutional news",
        topics="development finance; energy; infrastructure; agriculture; private sector",
        countries="ALL_AFRICA",
        credibility_tier="official",
        refresh_cadence="weekly",
        status="registered",
        last_checked_at="",
        notes="Registered official source. Keep as governed intake even when collection remains manual.",
        created_at="",
        updated_at="",
    ),
    MonitoringSource(
        source_id="eiti-news",
        name="EITI news and country updates",
        publisher="Extractive Industries Transparency Initiative",
        url="https://eiti.org/news",
        source_type="web_page",
        scope="official extractives transparency updates",
        topics="oil; gas; mining; governance; transparency",
        countries="ALL_AFRICA",
        credibility_tier="official",
        refresh_cadence="weekly",
        status="registered",
        last_checked_at="",
        notes="Useful for extractives governance and disclosure monitoring.",
        created_at="",
        updated_at="",
    ),
]


def initialize_monitoring_sources(path: Path, overwrite: bool = False) -> Path:
    """Create the governed monitoring-source registry."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not path.exists():
        save_monitoring_sources(path, DEFAULT_MONITORING_SOURCES)
    return path


def load_monitoring_sources(path: Path) -> list[MonitoringSource]:
    """Load monitoring sources from CSV."""

    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [monitoring_source_from_row(row) for row in reader]


def save_monitoring_sources(path: Path, sources: list[MonitoringSource]) -> None:
    """Persist monitoring-source rows in a reviewer-editable CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    now = utc_now_iso()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MONITORING_SOURCE_FIELDS)
        writer.writeheader()
        for source in sources:
            row = source.__dict__.copy()
            row["created_at"] = row["created_at"] or now
            row["updated_at"] = now
            writer.writerow(row)


def monitoring_source_from_row(row: dict[str, Any]) -> MonitoringSource:
    """Create a MonitoringSource from a CSV or SQLite row."""

    return MonitoringSource(
        source_id=str(row.get("source_id", "")),
        name=str(row.get("name", "")),
        publisher=str(row.get("publisher", "")),
        url=str(row.get("url", "")),
        source_type=str(row.get("source_type", "")),
        scope=str(row.get("scope", "")),
        topics=str(row.get("topics", "")),
        countries=str(row.get("countries", "")),
        credibility_tier=str(row.get("credibility_tier", "")),
        refresh_cadence=str(row.get("refresh_cadence", "")),
        status=str(row.get("status", "")),
        last_checked_at=str(row.get("last_checked_at", "")),
        notes=str(row.get("notes", "")),
        created_at=str(row.get("created_at", "")),
        updated_at=str(row.get("updated_at", "")),
    )


def collect_monitoring_sources(
    sources: list[MonitoringSource],
    *,
    limit: int = 100,
    timeout_seconds: int = 30,
) -> tuple[list[MonitoringEvent], list[dict[str, Any]], list[MonitoringSource]]:
    """Fetch supported live sources and normalize them into events.

    Only feed-like sources are fetched automatically. Web pages remain governed
    registry rows until a source-specific connector is added.
    """

    events: list[MonitoringEvent] = []
    results: list[dict[str, Any]] = []
    updated_sources: list[MonitoringSource] = []
    now = utc_now_iso()
    for source in sources:
        if source.status not in {"active", "registered"}:
            updated_sources.append(source)
            continue
        if source.source_type.lower() not in {"rss", "atom", "feed"}:
            results.append(
                {
                    "source_id": source.source_id,
                    "status": "skipped",
                    "message": f"{source.source_type} sources require a source-specific connector or manual review.",
                }
            )
            updated_sources.append(source)
            continue
        try:
            feed_text = fetch_text(source.url, timeout_seconds=timeout_seconds)
            entries = parse_feed_entries(feed_text)
            for entry in entries[: max(limit - len(events), 0)]:
                events.extend(events_from_text_entry(source, entry, now))
            results.append(
                {
                    "source_id": source.source_id,
                    "status": "collected",
                    "message": f"Parsed {len(entries)} feed entries.",
                }
            )
            updated_sources.append(
                MonitoringSource(
                    **{
                        **source.__dict__,
                        "status": "active",
                        "last_checked_at": now,
                        "updated_at": now,
                    }
                )
            )
        except Exception as exc:  # pragma: no cover - network-dependent.
            results.append({"source_id": source.source_id, "status": "error", "message": str(exc)})
            updated_sources.append(
                MonitoringSource(
                    **{
                        **source.__dict__,
                        "status": "error",
                        "last_checked_at": now,
                        "updated_at": now,
                    }
                )
            )
        if len(events) >= limit:
            break
    return unique_events(events)[:limit], results, updated_sources


def monitoring_events_from_knowledge_records(records: list[dict[str, Any]], limit: int = 1000) -> list[MonitoringEvent]:
    """Normalize existing knowledge records into dated monitoring events."""

    events: list[MonitoringEvent] = []
    now = utc_now_iso()
    for record in records:
        if record.get("record_type") not in {"monitoring_digest", "finance_resource_record", "partner_profile", "case_study_card"}:
            continue
        text = knowledge_record_text(record)
        country = str(record.get("country") or "Regional Africa")
        events.append(
            build_monitoring_event(
                source_id="manual-reviewed-monitoring",
                source_name=str(record.get("source_title") or "Reviewed local evidence"),
                source_url=str(record.get("source_path", "")),
                source_category="local_reviewed_record",
                title=str(record.get("title", "")),
                url=str(record.get("source_path", "")),
                published_at=str(record.get("event_date") or record.get("updated_at") or now),
                collected_at=now,
                country=country,
                text=text,
                source_record_id=str(record.get("record_id", "")),
                fallback_sector=str(record.get("sector", "")),
                fallback_commodity=str(record.get("commodity", "")),
                fallback_actors=str(record.get("actors") or record.get("partner") or ""),
                fallback_event_type=str(record.get("event_type", "")),
                fallback_tone=str(record.get("sentiment_tone", "")),
                fallback_risks=str(record.get("risk_flags", "")),
                fallback_relevance=str(record.get("relevance", "")),
                fallback_action=str(record.get("recommended_action", "")),
                confidence=float(record.get("confidence") or 0.7),
            )
        )
        if len(events) >= limit:
            break
    return unique_events(events)


def build_monitoring_event(
    *,
    source_id: str,
    source_name: str,
    source_url: str,
    source_category: str,
    title: str,
    url: str,
    published_at: str,
    collected_at: str,
    country: str,
    text: str,
    source_record_id: str = "",
    fallback_sector: str = "",
    fallback_commodity: str = "",
    fallback_actors: str = "",
    fallback_event_type: str = "",
    fallback_tone: str = "",
    fallback_risks: str = "",
    fallback_relevance: str = "",
    fallback_action: str = "",
    confidence: float | None = None,
) -> MonitoringEvent:
    """Create one normalized monitoring event from raw text."""

    sectors = detect_resource_sectors(text)
    commodities = detect_commodities(text)
    risks = split_labels(fallback_risks) or detect_risk_flags(text)
    event_types = split_labels(fallback_event_type) or detect_event_types(text)
    event_type = first_or_default(event_types, "Source update")
    relevance = fallback_relevance or detect_relevance("monitoring_digest", text)
    tone = fallback_tone or detect_sentiment_tone(text)
    action = fallback_action or recommend_action("monitoring_digest", risks, event_type, text)
    outcome = infer_outcome(event_type, text, risks)
    event_id = stable_id("monitoring-event", source_id, title, url, country, published_at, source_record_id)
    return MonitoringEvent(
        event_id=event_id,
        source_id=source_id,
        source_name=source_name,
        source_url=source_url,
        source_category=source_category,
        title=compact_text(title, 160),
        url=url,
        published_at=published_at,
        collected_at=collected_at,
        country=country,
        region=region_for_country(country) if country != "Regional Africa" else "Africa",
        sector=first_or_default(sectors, fallback_sector or "Not classified"),
        commodity=first_or_default(commodities, fallback_commodity),
        actors=fallback_actors or infer_actors(text),
        event_type=event_type,
        outcome=outcome,
        sentiment_tone=tone,
        risk_flags="; ".join(risks),
        relevance=relevance,
        confidence=confidence if confidence is not None else confidence_for_source(source_category, relevance, risks),
        summary=compact_text(text, 320),
        recommended_action=action,
        source_record_id=source_record_id,
        status="new",
        raw_text=compact_text(text, 2500),
    )


def events_from_text_entry(
    source: MonitoringSource,
    entry: dict[str, str],
    collected_at: str,
) -> list[MonitoringEvent]:
    """Normalize one feed entry into one event per detected country."""

    text = " ".join([entry.get("title", ""), entry.get("summary", "")]).strip()
    countries = detect_countries(text) or ["Regional Africa"]
    events = []
    for country in countries:
        events.append(
            build_monitoring_event(
                source_id=source.source_id,
                source_name=source.name,
                source_url=source.url,
                source_category=source_category(source),
                title=entry.get("title", ""),
                url=entry.get("url", source.url),
                published_at=entry.get("published_at", ""),
                collected_at=collected_at,
                country=country,
                text=text,
            )
        )
    return events


def build_monitoring_insights(events: list[dict[str, Any]], actions: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Summarize monitoring events into visual analytics and analyst insights."""

    actions = actions or []
    open_actions_by_country = Counter(
        action.get("country", "")
        for action in actions
        if action.get("status") in {"open", "in_progress"}
    )
    risk_events = [event for event in events if event.get("risk_flags")]
    high_relevance = [event for event in events if event.get("relevance") == "high"]
    mixed_or_negative = [
        event for event in events if str(event.get("sentiment_tone", "")).lower() in {"mixed", "negative"}
    ]
    snapshot = {
        "events": len(events),
        "countries": len({event.get("country") for event in events if event.get("country")}),
        "risk_events": len(risk_events),
        "high_relevance_events": len(high_relevance),
        "mixed_or_negative_events": len(mixed_or_negative),
        "open_actions": sum(open_actions_by_country.values()),
    }
    country_rows = build_country_event_rows(events, open_actions_by_country)
    timeline_rows = build_timeline_rows(events)
    top_risks = top_split_counts(events, "risk_flags")
    top_outcomes = top_counts(events, "outcome")
    insight_cards = build_insight_cards(country_rows, top_risks, top_outcomes, events)
    return {
        "snapshot": snapshot,
        "country_rows": country_rows,
        "timeline_rows": timeline_rows,
        "top_countries": top_counts(events, "country"),
        "top_sectors": top_counts(events, "sector"),
        "top_commodities": top_counts(events, "commodity"),
        "top_event_types": top_counts(events, "event_type"),
        "top_outcomes": top_outcomes,
        "top_risks": top_risks,
        "top_tones": top_counts(events, "sentiment_tone"),
        "insight_cards": insight_cards,
    }


def build_monitoring_agent_run(
    *,
    events: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    insights: dict[str, Any],
) -> dict[str, Any]:
    """Return an analyst-grade monitoring run summary.

    This is intentionally deterministic. In institutional settings, an
    "agent" should make its triage rules visible: source health, signal priority,
    watchlist logic, and recommended next actions all need to be inspectable.
    """

    source_health = build_source_health(sources)
    triage_queue = build_signal_triage(events)
    watchlist = build_country_watchlist(insights.get("country_rows", []), triage_queue)
    briefing = build_situation_brief(
        snapshot=insights.get("snapshot", {}),
        source_health=source_health,
        triage_queue=triage_queue,
        watchlist=watchlist,
        top_risks=insights.get("top_risks", []),
        top_outcomes=insights.get("top_outcomes", []),
    )
    return {
        "generated_at": utc_now_iso(),
        "mode": "approved sources plus reviewed local evidence",
        "snapshot": insights.get("snapshot", {}),
        "source_health": source_health,
        "watchlist": watchlist,
        "triage_queue": triage_queue,
        "briefing": briefing,
        "next_actions": build_agent_next_actions(source_health, triage_queue, watchlist, actions),
    }


def build_source_health(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize monitoring-source readiness and automation state."""

    rows = []
    for source in sources:
        source_type = str(source.get("source_type", "")).lower()
        status = str(source.get("status", ""))
        last_checked = str(source.get("last_checked_at", ""))
        if source_type in {"rss", "atom", "feed"}:
            automation_state = "automated_feed"
        elif source_type == "manual":
            automation_state = "local_reviewed_records"
        else:
            automation_state = "connector_needed"

        if status == "error":
            health = "error"
        elif automation_state == "connector_needed":
            health = "registered_manual"
        elif last_checked:
            health = "checked"
        elif status in {"active", "registered"}:
            health = "ready"
        else:
            health = "inactive"

        rows.append(
            {
                "source_id": source.get("source_id", ""),
                "name": source.get("name", ""),
                "publisher": source.get("publisher", ""),
                "source_type": source.get("source_type", ""),
                "status": status,
                "automation_state": automation_state,
                "health": health,
                "last_checked_at": last_checked,
                "credibility_tier": source.get("credibility_tier", ""),
                "topics": source.get("topics", ""),
            }
        )
    return rows


def build_signal_triage(events: list[dict[str, Any]], limit: int = 50) -> list[dict[str, Any]]:
    """Rank events by operational urgency."""

    rows = []
    for event in events:
        priority, score = event_priority(event)
        rows.append(
            {
                "priority": priority,
                "score": score,
                "country": event.get("country", ""),
                "sector": event.get("sector", ""),
                "commodity": event.get("commodity", ""),
                "outcome": event.get("outcome", ""),
                "risk_flags": event.get("risk_flags", ""),
                "tone": event.get("sentiment_tone", ""),
                "relevance": event.get("relevance", ""),
                "recommended_action": event.get("recommended_action", ""),
                "source_name": event.get("source_name", ""),
                "published_at": event.get("published_at", ""),
                "title": event.get("title", ""),
                "event_id": event.get("event_id", ""),
            }
        )
    return sorted(rows, key=lambda row: (-row["score"], row["country"], row["title"]))[:limit]


def event_priority(event: dict[str, Any]) -> tuple[str, int]:
    """Return transparent priority and score for one monitoring event."""

    score = 0
    if event.get("relevance") == "high":
        score += 3
    elif event.get("relevance") == "medium":
        score += 1
    if event.get("risk_flags"):
        score += 3
    if str(event.get("sentiment_tone", "")).lower() in {"mixed", "negative"}:
        score += 2
    outcome = str(event.get("outcome", "")).lower()
    if any(term in outcome for term in ("implementation risk", "dispute", "conflict")):
        score += 3
    elif any(term in outcome for term in ("financing", "investment", "policy")):
        score += 2
    if str(event.get("source_category", "")).lower() in {"official_news", "local_reviewed_record"}:
        score += 1

    if score >= 10:
        return "urgent", score
    if score >= 7:
        return "high", score
    if score >= 4:
        return "medium", score
    return "low", score


def build_country_watchlist(country_rows: list[dict[str, Any]], triage_queue: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Combine aggregate country attention with urgent signal counts."""

    urgent_counts: Counter[str] = Counter()
    high_counts: Counter[str] = Counter()
    for event in triage_queue:
        country = str(event.get("country", ""))
        if not country:
            continue
        if event.get("priority") == "urgent":
            urgent_counts[country] += 1
        if event.get("priority") in {"urgent", "high"}:
            high_counts[country] += 1

    rows = []
    for row in country_rows:
        country = str(row.get("country", ""))
        if country in {"", "Africa", "Regional Africa", "Not specified"}:
            continue
        attention_score = int(row.get("attention_score", 0) or 0)
        priority = "urgent" if urgent_counts[country] else "high" if high_counts[country] else "monitor"
        rows.append(
            {
                "priority": priority,
                "country": country,
                "attention_score": attention_score,
                "events": int(row.get("events", 0) or 0),
                "risk_events": int(row.get("risk_events", 0) or 0),
                "urgent_signals": urgent_counts[country],
                "high_priority_signals": high_counts[country],
                "open_actions": int(row.get("open_actions", 0) or 0),
                "recommended_action": "open country workspace" if priority in {"urgent", "high"} else "monitor",
            }
        )
    priority_rank = {"urgent": 0, "high": 1, "monitor": 2}
    return sorted(rows, key=lambda row: (priority_rank[row["priority"]], -row["attention_score"]))[:20]


def build_situation_brief(
    *,
    snapshot: dict[str, Any],
    source_health: list[dict[str, Any]],
    triage_queue: list[dict[str, Any]],
    watchlist: list[dict[str, Any]],
    top_risks: list[dict[str, Any]],
    top_outcomes: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Produce a compact monitoring brief."""

    automated = sum(1 for row in source_health if row.get("automation_state") == "automated_feed")
    connector_needed = sum(1 for row in source_health if row.get("automation_state") == "connector_needed")
    urgent_signals = sum(1 for row in triage_queue if row.get("priority") == "urgent")
    top_country = watchlist[0]["country"] if watchlist else "No country"
    top_risk = top_risks[0]["value"] if top_risks else "No dominant risk"
    top_outcome = top_outcomes[0]["value"] if top_outcomes else "No dominant outcome"
    return [
        {
            "section": "Signal Load",
            "brief": (
                f"{snapshot.get('events', 0)} events across {snapshot.get('countries', 0)} countries; "
                f"{snapshot.get('risk_events', 0)} risk-flagged and {urgent_signals} urgent."
            ),
        },
        {
            "section": "Country Watch",
            "brief": f"{top_country} is the top watchlist country based on attention score and urgent/high signals.",
        },
        {
            "section": "Dominant Risk",
            "brief": f"{top_risk} is the leading risk flag in the current monitoring set.",
        },
        {
            "section": "Dominant Outcome",
            "brief": f"{top_outcome} is the most common outcome category.",
        },
        {
            "section": "Source Operations",
            "brief": (
                f"{automated} automated feed source(s), {connector_needed} registered source(s) still need connectors/manual review."
            ),
        },
    ]


def build_agent_next_actions(
    source_health: list[dict[str, Any]],
    triage_queue: list[dict[str, Any]],
    watchlist: list[dict[str, Any]],
    actions: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Return operational next actions for the analyst."""

    next_actions = []
    if any(row.get("automation_state") == "connector_needed" for row in source_health):
        next_actions.append(
            {
                "priority": "high",
                "action": "convert registered web sources into source-specific connectors or approved RSS/API feeds",
                "reason": "Registered manual sources do not create live events automatically.",
            }
        )
    if any(row.get("priority") == "urgent" for row in triage_queue):
        next_actions.append(
            {
                "priority": "urgent",
                "action": "review urgent monitoring signals and decide whether to promote them to bulletin/profile/finance records",
                "reason": "Urgent signals combine relevance, risk, tone, outcome, and source authority.",
            }
        )
    if watchlist:
        next_actions.append(
            {
                "priority": "high",
                "action": f"open the {watchlist[0]['country']} country workspace",
                "reason": "It is the top watchlist country in the current monitoring run.",
            }
        )
    if actions:
        next_actions.append(
            {
                "priority": "medium",
                "action": "clear or assign open action items",
                "reason": f"{sum(1 for action in actions if action.get('status') in {'open', 'in_progress'})} open or in-progress actions remain.",
            }
        )
    return next_actions


def build_country_event_rows(events: list[dict[str, Any]], action_counts: Counter) -> list[dict[str, Any]]:
    """Create country rows for maps and ranking tables."""

    by_country: dict[str, dict[str, Any]] = {}
    for event in events:
        country = event.get("country") or "Not specified"
        item = by_country.setdefault(
            country,
            {
                "country": country,
                "events": 0,
                "risk_events": 0,
                "high_relevance_events": 0,
                "mixed_or_negative_events": 0,
                "open_actions": int(action_counts.get(country, 0)),
                "attention_score": 0,
            },
        )
        item["events"] += 1
        if event.get("risk_flags"):
            item["risk_events"] += 1
        if event.get("relevance") == "high":
            item["high_relevance_events"] += 1
        if str(event.get("sentiment_tone", "")).lower() in {"mixed", "negative"}:
            item["mixed_or_negative_events"] += 1
    for row in by_country.values():
        row["attention_score"] = (
            row["events"]
            + 2 * row["risk_events"]
            + 2 * row["high_relevance_events"]
            + row["mixed_or_negative_events"]
            + row["open_actions"]
        )
    return sorted(by_country.values(), key=lambda row: row["attention_score"], reverse=True)


def build_timeline_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate events by month and tone."""

    counts: Counter[tuple[str, str]] = Counter()
    for event in events:
        period = period_label(str(event.get("published_at") or event.get("collected_at") or ""))
        tone = str(event.get("sentiment_tone") or "unknown")
        counts[(period, tone)] += 1
    rows = [
        {"period": period, "sentiment_tone": tone, "events": count}
        for (period, tone), count in counts.items()
        if period
    ]
    return sorted(rows, key=lambda row: (row["period"], row["sentiment_tone"]))


def build_insight_cards(
    country_rows: list[dict[str, Any]],
    top_risks: list[dict[str, Any]],
    top_outcomes: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Create concise, auditable analyst insights from aggregate signals."""

    cards: list[dict[str, str]] = []
    if country_rows:
        top_country = country_rows[0]
        cards.append(
            {
                "insight": f"{top_country['country']} needs the most attention in the current monitoring set.",
                "why_it_matters": (
                    f"{top_country['events']} events, {top_country['risk_events']} risk-flagged events, "
                    f"and {top_country['open_actions']} open actions."
                ),
                "suggested_action": "Open the country workspace and decide whether to update a profile, bulletin, or finance record.",
            }
        )
    if top_risks:
        risk = top_risks[0]
        cards.append(
            {
                "insight": f"{risk['value']} is the leading risk flag.",
                "why_it_matters": f"It appears in {risk['count']} monitoring events.",
                "suggested_action": "Review whether this risk changes the country brief, partner profile, or review queue priority.",
            }
        )
    if top_outcomes:
        outcome = top_outcomes[0]
        cards.append(
            {
                "insight": f"{outcome['value']} is the dominant observed outcome.",
                "why_it_matters": f"It appears in {outcome['count']} events.",
                "suggested_action": "Use the event table to separate finance commitments, implementation risks, and monitoring-only signals.",
            }
        )
    stale = [event for event in events if not event.get("published_at")]
    if stale:
        cards.append(
            {
                "insight": "Some monitoring events need date cleanup.",
                "why_it_matters": f"{len(stale)} events lack a clear publication date.",
                "suggested_action": "Prioritize source-date review before using these events in trend charts.",
            }
        )
    return cards


def fetch_text(url: str, timeout_seconds: int = 30) -> str:
    """Fetch a text source using a small, explicit user agent."""

    request = Request(url, headers={"User-Agent": "DevFinIntelWorkbench/0.1"})
    with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 - user-approved registry.
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def parse_feed_entries(text: str) -> list[dict[str, str]]:
    """Parse RSS or Atom XML entries with only the Python standard library."""

    root = ET.fromstring(text)
    entries: list[dict[str, str]] = []
    nodes = [node for node in root.iter() if local_name(node.tag) in {"item", "entry"}]
    for node in nodes:
        title = child_text(node, {"title"})
        summary = child_text(node, {"description", "summary", "content", "encoded"})
        url = child_text(node, {"link"})
        if not url:
            link_node = first_child(node, {"link"})
            url = link_node.attrib.get("href", "") if link_node is not None else ""
        published = child_text(node, {"pubDate", "published", "updated", "date"})
        entries.append(
            {
                "title": strip_markup(title),
                "summary": strip_markup(summary),
                "url": url,
                "published_at": published,
            }
        )
    return entries


def child_text(node: ET.Element, names: set[str]) -> str:
    """Return text from the first child with a local tag name in names."""

    child = first_child(node, names)
    if child is None:
        return ""
    return "".join(child.itertext()).strip()


def first_child(node: ET.Element, names: set[str]) -> ET.Element | None:
    """Find a child by local tag name."""

    for child in node:
        if local_name(child.tag) in names:
            return child
    return None


def local_name(tag: str) -> str:
    """Strip XML namespaces from a tag."""

    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def strip_markup(text: str) -> str:
    """Remove lightweight HTML markup from feed summaries."""

    return compact_text(unescape(re.sub(r"<[^>]+>", " ", text)), 1200)


def source_category(source: MonitoringSource) -> str:
    """Classify monitoring-source authority for dashboards."""

    tier = source.credibility_tier.lower()
    publisher = source.publisher.lower()
    if tier == "official" or any(term in publisher for term in ("world bank", "african development bank", "eiti", "iea", "undp")):
        return "official_news"
    if "social" in tier or "social" in source.source_type.lower():
        return "social_signal"
    if "review" in tier:
        return "local_reviewed_record"
    return "independent_news"


def confidence_for_source(source_category_value: str, relevance: str, risks: list[str]) -> float:
    """Assign transparent baseline confidence for triage, not truth."""

    base = {
        "official_news": 0.78,
        "local_reviewed_record": 0.74,
        "independent_news": 0.64,
        "social_signal": 0.45,
    }.get(source_category_value, 0.6)
    if relevance == "high":
        base += 0.05
    if risks:
        base -= 0.03
    return max(0.2, min(base, 0.92))


def infer_outcome(event_type: str, text: str, risks: list[str]) -> str:
    """Normalize event type and risk text into analyst outcome categories."""

    lower = text.lower()
    event_lower = event_type.lower()
    if "delay" in event_lower or "delayed" in lower or "suspended" in lower:
        return "Implementation risk increased"
    if "dispute" in event_lower or "conflict" in event_lower:
        return "Dispute or conflict signal"
    if "approved" in event_lower or "financing" in event_lower:
        return "Financing or investment movement"
    if "launch" in event_lower:
        return "Project launched or advanced"
    if "policy" in event_lower or "reform" in event_lower:
        return "Policy or regulatory movement"
    if risks:
        return "Risk signal"
    return "Monitoring update"


def infer_actors(text: str) -> str:
    """Detect high-level actor categories from monitoring text."""

    lower = text.lower()
    actors = []
    if any(term in lower for term in ("government", "ministry", "minister", "state")):
        actors.append("Government")
    if any(term in lower for term in ("world bank", "afdb", "imf", "ifc", "miga", "donor", "development bank")):
        actors.append("IFI / donor")
    if any(term in lower for term in ("company", "operator", "investor", "private sector")):
        actors.append("Company / investor")
    if any(term in lower for term in ("civil society", "community", "ngo", "workers")):
        actors.append("Civil society / community")
    return "; ".join(unique_preserve_order(actors))


def knowledge_record_text(record: dict[str, Any]) -> str:
    """Build a text body from one stored knowledge record."""

    parts = [
        record.get("title", ""),
        record.get("country", ""),
        record.get("sector", ""),
        record.get("theme", ""),
        record.get("commodity", ""),
        record.get("partner", ""),
        record.get("amount", ""),
        record.get("instrument", ""),
        record.get("event_type", ""),
        record.get("sentiment_tone", ""),
        record.get("risk_flags", ""),
        record.get("recommended_action", ""),
    ]
    fields = record.get("fields")
    if isinstance(fields, dict):
        parts.extend(str(value) for value in fields.values())
    return " ".join(str(part) for part in parts if part)


def unique_events(events: list[MonitoringEvent]) -> list[MonitoringEvent]:
    """Deduplicate events while preserving first occurrence."""

    seen: set[str] = set()
    result: list[MonitoringEvent] = []
    for event in events:
        if event.event_id in seen:
            continue
        seen.add(event.event_id)
        result.append(event)
    return result


def split_labels(value: str) -> list[str]:
    """Split semicolon/comma labels from stored fields."""

    if not value:
        return []
    return unique_preserve_order(label.strip() for label in re.split(r"[;,]", value) if label.strip())


def first_or_default(values: list[str], default: str = "") -> str:
    """Return first non-empty value."""

    for value in values:
        if value:
            return value
    return default


def compact_text(text: str, max_chars: int) -> str:
    """Collapse whitespace and cap text for dashboards."""

    compacted = re.sub(r"\s+", " ", text or "").strip()
    if len(compacted) <= max_chars:
        return compacted
    return compacted[: max_chars - 3].rstrip() + "..."


def top_counts(rows: list[dict[str, Any]], key: str, limit: int = 10) -> list[dict[str, Any]]:
    """Return value counts for one key."""

    counter = Counter(str(row.get(key) or "Not specified") for row in rows if str(row.get(key) or "").strip())
    return [{"value": value, "count": count} for value, count in counter.most_common(limit)]


def top_split_counts(rows: list[dict[str, Any]], key: str, limit: int = 10) -> list[dict[str, Any]]:
    """Return counts for semicolon-separated labels."""

    counter: Counter[str] = Counter()
    for row in rows:
        for value in split_labels(str(row.get(key) or "")):
            counter[value] += 1
    return [{"value": value, "count": count} for value, count in counter.most_common(limit)]


def period_label(value: str) -> str:
    """Return a YYYY-MM-ish period label from a loose date string."""

    if not value:
        return ""
    iso_match = re.search(r"\b(20\d{2})-(\d{2})\b", value)
    if iso_match:
        return f"{iso_match.group(1)}-{iso_match.group(2)}"
    year_match = re.search(r"\b(20\d{2})\b", value)
    if year_match:
        return year_match.group(1)
    return ""
