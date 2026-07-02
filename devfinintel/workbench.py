"""Unified intelligence-workbench helpers.

This module turns separate evidence channels into one analyst workflow:
documents, CSV datasets, monitoring events, and news signals. The goal is not to
hide uncertainty behind a chatbot. The goal is to show what evidence was used,
what changed, what the data suggests, and what still needs review.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from devfinintel.datasets import (
    choose_first_existing,
    compute_numeric_profiles,
    parse_float,
    rank_rows,
    read_csv_rows,
)
from devfinintel.models import EvidenceItem, SourceDocument
from devfinintel.news import meaningful_terms
from devfinintel.utils import stable_id, utc_now_iso


COUNTRY_COLUMN_CANDIDATES = [
    "country",
    "Country",
    "COUNTRY",
    "REF_AREA_LABEL",
    "REF_AREA",
    "economy",
    "Economy",
]
COMMODITY_COLUMN_CANDIDATES = ["commodity", "Commodity", "COMMODITY", "product", "Product"]
YEAR_COLUMN_CANDIDATES = ["year", "Year", "TIME_PERIOD", "date", "Date"]
VALUE_COLUMN_CANDIDATES = ["value", "Value", "OBS_VALUE", "amount", "Amount", "price", "Price"]


def document_evidence_rows(evidence_items: list[EvidenceItem]) -> list[dict[str, Any]]:
    """Flatten retrieved document evidence into rows for UI/export."""

    rows = []
    for item in evidence_items:
        rows.append(
            {
                "source_type": "document",
                "source": item.title,
                "reference": f"p. {item.page_number}",
                "score": round(item.rerank_score, 4),
                "text": item.text[:900],
                "url": item.source_path,
            }
        )
    return rows


def summarize_csv_documents(documents: list[SourceDocument], limit: int = 10) -> list[dict[str, Any]]:
    """Create compact dataset summaries for the unified workbench."""

    summaries: list[dict[str, Any]] = []
    for document in documents:
        if document.source_type != "csv":
            continue
        path = Path(document.source_path)
        if not path.exists():
            continue
        rows = read_csv_rows(path)
        columns = list(rows[0].keys()) if rows else []
        numeric_profiles = compute_numeric_profiles(rows, columns)
        country_column = choose_first_existing(columns, COUNTRY_COLUMN_CANDIDATES)
        commodity_column = choose_first_existing(columns, COMMODITY_COLUMN_CANDIDATES)
        year_column = choose_first_existing(columns, YEAR_COLUMN_CANDIDATES)
        value_column = choose_first_existing(columns, VALUE_COLUMN_CANDIDATES)
        if value_column is None and numeric_profiles:
            value_column = numeric_profiles[0].column
        label_column = country_column or commodity_column or (columns[0] if columns else None)
        top_values = rank_rows(rows, label_column, value_column, reverse=True, limit=5)
        bottom_values = rank_rows(rows, label_column, value_column, reverse=False, limit=5)
        missing_cells = sum(1 for row in rows for column in columns if not row.get(column))
        country_count = len({row.get(country_column, "") for row in rows if country_column and row.get(country_column)})
        commodity_count = len({row.get(commodity_column, "") for row in rows if commodity_column and row.get(commodity_column)})
        years = sorted({row.get(year_column, "") for row in rows if year_column and row.get(year_column)})
        summaries.append(
            {
                "document_id": document.document_id,
                "title": document.title,
                "source_path": document.source_path,
                "rows": len(rows),
                "columns": columns,
                "country_column": country_column or "",
                "commodity_column": commodity_column or "",
                "year_column": year_column or "",
                "value_column": value_column or "",
                "country_count": country_count,
                "commodity_count": commodity_count,
                "years": years,
                "numeric_profiles": [profile.__dict__ for profile in numeric_profiles[:8]],
                "top_values": [{"label": label, "value": value} for label, value in top_values],
                "bottom_values": [{"label": label, "value": value} for label, value in bottom_values],
                "missing_cells": missing_cells,
                "summary": dataset_summary_sentence(
                    document.title,
                    len(rows),
                    len(columns),
                    country_count,
                    commodity_count,
                    years,
                    value_column or "",
                ),
            }
        )
        if len(summaries) >= limit:
            break
    return summaries


def dataset_summary_sentence(
    title: str,
    rows: int,
    columns: int,
    country_count: int,
    commodity_count: int,
    years: list[str],
    value_column: str,
) -> str:
    """Write one plain-English dataset signal."""

    parts = [f"{title} has {rows} rows and {columns} columns"]
    if country_count:
        parts.append(f"{country_count} countries/entities")
    if commodity_count:
        parts.append(f"{commodity_count} commodities/products")
    if years:
        parts.append(f"years {years[0]} to {years[-1]}")
    if value_column:
        parts.append(f"main value column `{value_column}`")
    return "; ".join(parts) + "."


def dataset_rows_for_evidence(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn dataset summaries into evidence rows."""

    rows = []
    for summary in summaries:
        top_values = ", ".join(
            f"{item['label']}: {format_number(item['value'])}" for item in summary.get("top_values", [])[:3]
        )
        text = summary.get("summary", "")
        if top_values:
            text = f"{text} Top values: {top_values}."
        rows.append(
            {
                "source_type": "dataset",
                "source": summary.get("title", "Dataset"),
                "reference": "computed CSV profile",
                "score": 1.0,
                "text": text,
                "url": summary.get("source_path", ""),
            }
        )
    return rows


def news_rows_for_evidence(articles: list[dict[str, Any]], question: str = "", limit: int = 8) -> list[dict[str, Any]]:
    """Turn news articles into ranked evidence rows."""

    ranked = rank_text_rows(
        articles,
        question,
        text_getter=lambda row: f"{row.get('title', '')} {row.get('summary', '')} {row.get('country', '')} "
        f"{row.get('commodity', '')} {row.get('risk_flags', '')}",
    )
    rows = []
    for score, article in ranked[:limit]:
        rows.append(
            {
                "source_type": "news",
                "source": article.get("source", "News"),
                "reference": article.get("published_at", ""),
                "score": round(score, 4),
                "text": f"{article.get('title', '')}. {article.get('summary', '')}",
                "url": article.get("url", ""),
                "country": article.get("country", ""),
                "commodity": article.get("commodity", ""),
                "sentiment_tone": article.get("sentiment_tone", ""),
                "risk_flags": article.get("risk_flags", ""),
            }
        )
    return rows


def monitoring_rows_for_evidence(events: list[dict[str, Any]], question: str = "", limit: int = 8) -> list[dict[str, Any]]:
    """Turn normalized monitoring events into ranked evidence rows."""

    ranked = rank_text_rows(
        events,
        question,
        text_getter=lambda row: f"{row.get('title', '')} {row.get('summary', '')} {row.get('country', '')} "
        f"{row.get('commodity', '')} {row.get('risk_flags', '')} {row.get('recommended_action', '')}",
    )
    rows = []
    for score, event in ranked[:limit]:
        rows.append(
            {
                "source_type": "monitoring_event",
                "source": event.get("source_name", "Monitoring event"),
                "reference": event.get("published_at", ""),
                "score": round(score, 4),
                "text": f"{event.get('title', '')}. {event.get('summary', '')}",
                "url": event.get("url", ""),
                "country": event.get("country", ""),
                "commodity": event.get("commodity", ""),
                "sentiment_tone": event.get("sentiment_tone", ""),
                "risk_flags": event.get("risk_flags", ""),
            }
        )
    return rows


def signal_rows_for_evidence(signals: list[dict[str, Any]], question: str = "", limit: int = 8) -> list[dict[str, Any]]:
    """Turn normalized monitoring signals into ranked evidence rows."""

    ranked = rank_text_rows(
        signals,
        question,
        text_getter=lambda row: f"{row.get('title', '')} {row.get('summary', '')} {row.get('country', '')} "
        f"{row.get('commodity', '')} {row.get('sector', '')} {row.get('risk_flags', '')}",
    )
    rows = []
    for score, signal in ranked[:limit]:
        rows.append(
            {
                "source_type": signal.get("source_type", "signal"),
                "source": signal.get("source_name", "Monitoring signal"),
                "reference": signal.get("date", ""),
                "score": round(score + float(signal.get("relevance_score", 0.0) or 0.0), 4),
                "text": f"{signal.get('title', '')}. {signal.get('summary', '')}",
                "url": signal.get("url", ""),
                "country": signal.get("country", ""),
                "commodity": signal.get("commodity", ""),
                "sector": signal.get("sector", ""),
                "sentiment_tone": signal.get("tone", ""),
                "risk_flags": signal.get("risk_flags", ""),
            }
        )
    return rows


def cluster_rows_for_evidence(clusters: list[dict[str, Any]], question: str = "", limit: int = 6) -> list[dict[str, Any]]:
    """Turn event clusters into evidence rows."""

    ranked = rank_text_rows(
        clusters,
        question,
        text_getter=lambda row: f"{row.get('event_title', '')} {row.get('what_changed', '')} "
        f"{row.get('why_it_matters', '')} {row.get('countries', '')} {row.get('commodities', '')}",
    )
    rows = []
    for score, cluster in ranked[:limit]:
        rows.append(
            {
                "source_type": "event_cluster",
                "source": cluster.get("sources", "Monitoring cluster"),
                "reference": cluster.get("latest_update", ""),
                "score": round(score + cluster.get("signal_count", 0) * 0.1, 4),
                "text": f"{cluster.get('event_title', '')}: {cluster.get('what_changed', '')} {cluster.get('why_it_matters', '')}",
                "url": "",
                "country": cluster.get("countries", ""),
                "commodity": cluster.get("commodities", ""),
                "sector": cluster.get("sectors", ""),
                "sentiment_tone": cluster.get("risk_level", ""),
                "risk_flags": cluster.get("risk_flags", ""),
            }
        )
    return rows


def build_workbench_answer(
    *,
    question: str,
    document_evidence: list[EvidenceItem],
    dataset_summaries: list[dict[str, Any]],
    news_articles: list[dict[str, Any]],
    monitoring_events: list[dict[str, Any]],
    normalized_signals: list[dict[str, Any]] | None = None,
    event_clusters: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a unified answer across docs, data, news, and monitoring records."""

    evidence_rows = (
        document_evidence_rows(document_evidence[:6])
        + dataset_rows_for_evidence(dataset_summaries[:4])
        + cluster_rows_for_evidence(event_clusters or [], question=question, limit=5)
        + signal_rows_for_evidence(normalized_signals or [], question=question, limit=6)
        + news_rows_for_evidence(news_articles, question=question, limit=6)
        + monitoring_rows_for_evidence(monitoring_events, question=question, limit=6)
    )
    evidence_rows = [row for row in evidence_rows if row.get("text")]
    source_counts = Counter(row["source_type"] for row in evidence_rows)
    missing = []
    if not document_evidence:
        missing.append("no matching document excerpts")
    if not dataset_summaries:
        missing.append("no CSV dataset summary")
    if not news_articles:
        missing.append("no news articles")
    if not monitoring_events:
        missing.append("no normalized monitoring events")
    if not normalized_signals:
        missing.append("no live/keyless normalized signals")
    if not event_clusters:
        missing.append("no clustered developments")

    lines = [
        f"# Workbench Answer",
        "",
        f"**Question:** {question}",
        "",
        "## Short Answer",
    ]
    if not evidence_rows:
        lines.append(
            "I do not have enough evidence in the selected documents, datasets, or monitoring/news signals to answer this yet."
        )
    else:
        lines.extend(synthesize_answer_points(evidence_rows, max_points=5))

    lines.extend(["", "## Evidence Used"])
    for index, row in enumerate(evidence_rows[:12], start=1):
        lines.append(
            f"- E{index} [{row['source_type']}]: {row['source']} {row.get('reference', '')} - "
            f"{compact(row.get('text', ''), 220)}"
        )

    lines.extend(["", "## Quality Flags"])
    if missing:
        lines.append("- Missing channels: " + "; ".join(missing) + ".")
    if source_counts.get("news", 0) and all(row.get("url", "").startswith("sample://") for row in evidence_rows if row["source_type"] == "news"):
        lines.append("- News signals are sample/offline items, not verified live current articles.")
    if len(evidence_rows) < 3:
        lines.append("- Evidence volume is low. Treat the answer as a first-pass triage note.")
    if not missing and len(evidence_rows) >= 3:
        lines.append("- Evidence volume is sufficient for first-pass analysis, but source excerpts should still be reviewed.")

    return {
        "answer_markdown": "\n".join(lines),
        "evidence_rows": evidence_rows,
        "source_counts": dict(source_counts),
        "quality_flags": missing,
        "generated_at": utc_now_iso(),
    }


def build_intelligence_brief(
    *,
    focus: str,
    country: str,
    commodity: str,
    topic: str,
    document_evidence: list[EvidenceItem],
    dataset_summaries: list[dict[str, Any]],
    news_articles: list[dict[str, Any]],
    monitoring_events: list[dict[str, Any]],
    monitoring_insights: dict[str, Any],
) -> dict[str, Any]:
    """Create an analyst-style intelligence brief and evidence rows."""

    evidence_rows = (
        document_evidence_rows(document_evidence[:6])
        + dataset_rows_for_evidence(dataset_summaries[:4])
        + news_rows_for_evidence(news_articles, question=focus, limit=8)
        + monitoring_rows_for_evidence(monitoring_events, question=focus, limit=8)
    )
    countries = summarize_values(
        [country]
        + [article.get("country", "") for article in news_articles]
        + [event.get("country", "") for event in monitoring_events]
    )
    commodities = summarize_values(
        [commodity]
        + [article.get("commodity", "") for article in news_articles]
        + [event.get("commodity", "") for event in monitoring_events]
    )
    risks = summarize_values(
        [article.get("risk_flags", "") for article in news_articles]
        + [event.get("risk_flags", "") for event in monitoring_events]
    )
    insight_cards = monitoring_insights.get("insight_cards", []) if monitoring_insights else []

    title_focus = " ".join(part for part in [country, commodity, topic or focus] if part).strip()
    title = f"Africa Energy and Commodities Brief: {title_focus or 'Monitoring Update'}"
    lines = [
        f"# {title}",
        "",
        "## Headline",
        headline_sentence(country=country, commodity=commodity, topic=topic, focus=focus),
        "",
        "## Key Developments",
    ]
    lines.extend(synthesize_answer_points(evidence_rows, max_points=5))
    if insight_cards:
        for card in insight_cards[:3]:
            lines.append(f"- {card.get('title', 'Monitoring insight')}: {card.get('message', '')}")

    lines.extend(
        [
            "",
            "## Countries And Commodities Affected",
            f"- Countries or regions: {', '.join(countries[:10]) if countries else 'Not enough evidence.'}",
            f"- Commodities or sectors: {', '.join(commodities[:10]) if commodities else 'Not enough evidence.'}",
            "",
            "## Risks And Opportunities",
        ]
    )
    if risks:
        lines.append(f"- Risk flags to review: {', '.join(risks[:10])}.")
    else:
        lines.append("- No explicit risk flags were detected in the selected signals.")
    positive_count = sum(1 for row in news_articles + monitoring_events if row.get("sentiment_tone") == "positive")
    mixed_negative_count = sum(
        1 for row in news_articles + monitoring_events if row.get("sentiment_tone") in {"mixed", "negative"}
    )
    lines.append(
        f"- Signal tone: {positive_count} positive and {mixed_negative_count} mixed/negative monitoring items."
    )

    lines.extend(["", "## Data Signals"])
    if dataset_summaries:
        for summary in dataset_summaries[:4]:
            lines.append(f"- {summary.get('summary', '')}")
    else:
        lines.append("- No CSV dataset was selected, so the brief cannot compare quantitative trends yet.")

    lines.extend(["", "## News And Monitoring Signals"])
    news_signal_rows = news_rows_for_evidence(news_articles, question=focus, limit=5)
    if news_signal_rows:
        for row in news_signal_rows:
            lines.append(f"- {compact(row['text'], 240)}")
    else:
        lines.append("- No news items were available for this run.")

    lines.extend(
        [
            "",
            "## Recommended Follow-Up",
            "- Review cited excerpts and article links before external use.",
            "- Promote high-relevance monitoring items into finance/resource records when they contain amounts, partners, or project details.",
            "- Compare official report claims against monitoring items where tone or risk flags diverge.",
            "",
            "## Source And Evidence List",
        ]
    )
    for index, row in enumerate(evidence_rows[:18], start=1):
        lines.append(
            f"- E{index} [{row['source_type']}]: {row['source']} {row.get('reference', '')} - "
            f"{compact(row.get('text', ''), 220)}"
        )

    return {
        "title": title,
        "markdown": "\n".join(lines),
        "evidence_rows": evidence_rows,
        "generated_at": utc_now_iso(),
    }


def evidence_rows_to_items(rows: list[dict[str, Any]], *, prefix: str = "workbench") -> list[EvidenceItem]:
    """Represent non-document workbench rows as exportable EvidenceItems."""

    items = []
    for index, row in enumerate(rows, start=1):
        title = str(row.get("source") or row.get("source_type") or "Workbench evidence")
        url = str(row.get("url") or "")
        text = str(row.get("text") or "")
        items.append(
            EvidenceItem(
                chunk_id=stable_id(prefix, row.get("source_type", ""), title, url, str(index)),
                document_id=stable_id(prefix, row.get("source_type", ""), title),
                title=title,
                source_path=url,
                page_number=1,
                text=text,
                bm25_score=0.0,
                dense_score=0.0,
                rerank_score=float(row.get("score") or 1.0),
            )
        )
    return items


def rank_text_rows(
    rows: list[dict[str, Any]],
    question: str,
    *,
    text_getter,
) -> list[tuple[float, dict[str, Any]]]:
    """Rank rows by transparent term overlap."""

    terms = meaningful_terms(question)
    ranked = []
    for row in rows:
        text = text_getter(row).lower()
        hits = sum(1 for term in terms if term in text)
        relevance = 0.3 if str(row.get("relevance", "")).lower() == "high" else 0.0
        score = hits + relevance
        ranked.append((score, row))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked


def synthesize_answer_points(evidence_rows: list[dict[str, Any]], max_points: int) -> list[str]:
    """Turn evidence rows into concise first-pass analytic points."""

    if not evidence_rows:
        return ["- No evidence points were available."]
    points = []
    seen = set()
    for row in evidence_rows:
        text = compact(row.get("text", ""), 220)
        if not text or text in seen:
            continue
        seen.add(text)
        label = row.get("country") or row.get("source_type")
        risk = row.get("risk_flags")
        suffix = f" Risk flags: {risk}." if risk else ""
        points.append(f"- {label}: {text}{suffix}")
        if len(points) >= max_points:
            break
    return points or ["- Evidence was available, but it was too sparse to synthesize safely."]


def headline_sentence(*, country: str, commodity: str, topic: str, focus: str) -> str:
    """Create a brief headline sentence."""

    focus_bits = [part for part in [country, commodity, topic or focus] if part]
    if focus_bits:
        return "Monitoring signals point to " + ", ".join(focus_bits[:3]) + " as the current analytical focus."
    return "Monitoring signals point to Africa-wide energy and commodity developments requiring evidence review."


def summarize_values(values: list[str]) -> list[str]:
    """Split semicolon-delimited values and keep useful labels."""

    output: list[str] = []
    for value in values:
        for item in str(value or "").split(";"):
            item = item.strip()
            if item and item not in {"Not specified", "Not classified", "Regional Africa"}:
                output.append(item)
    return list(dict.fromkeys(output))


def compact(text: str, limit: int) -> str:
    """Trim whitespace and shorten long evidence text."""

    cleaned = " ".join(str(text).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)].rstrip() + "..."


def format_number(value: Any) -> str:
    """Format a numeric value for display."""

    parsed = parse_float(str(value))
    if parsed is None:
        return str(value)
    if abs(parsed) >= 1_000_000:
        return f"{parsed:,.0f}"
    if abs(parsed) >= 100:
        return f"{parsed:,.1f}"
    return f"{parsed:.3g}"
