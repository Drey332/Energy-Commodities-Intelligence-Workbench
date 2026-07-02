"""Coverage intelligence for country-by-topic knowledge operations.

This module answers a question that a serious institutional knowledge platform
must answer before it generates more prose: "Where do we actually have evidence,
where do we only have a regional source, and where are the gaps?"

The matrix is intentionally conservative. A region-wide report can support
screening, but it should not be treated as equivalent to a reviewed
country-specific record.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from devfinintel.sources import AFRICAN_COUNTRIES, SourceRegistryEntry, split_semicolon


@dataclass(frozen=True)
class CoverageTopic:
    """One canonical topic used in the country coverage matrix."""

    topic_id: str
    label: str
    keywords: tuple[str, ...]
    recommended_source: str


COVERAGE_TOPICS: tuple[CoverageTopic, ...] = (
    CoverageTopic(
        topic_id="energy_access",
        label="Energy Access And Power",
        keywords=("energy", "electricity", "power", "access", "hydropower", "solar", "wind", "renewable"),
        recommended_source="IEA, World Bank energy, AfDB energy, national energy plan",
    ),
    CoverageTopic(
        topic_id="oil_gas",
        label="Oil And Gas",
        keywords=("oil", "gas", "lng", "petroleum", "crude"),
        recommended_source="EITI country report, national petroleum ministry, World Bank country report",
    ),
    CoverageTopic(
        topic_id="mining_critical_minerals",
        label="Mining And Critical Minerals",
        keywords=("mining", "mineral", "critical minerals", "cobalt", "copper", "lithium", "nickel", "manganese", "gold", "bauxite"),
        recommended_source="EITI country report, USGS minerals yearbook, World Bank mining/value-chain report",
    ),
    CoverageTopic(
        topic_id="commodity_value_chains",
        label="Commodity Value Chains",
        keywords=("commodity", "commodities", "value chain", "value chains", "export", "processing", "industrialization"),
        recommended_source="World Bank country economic memorandum, AfDB country diagnostic, OECD/IATI project data",
    ),
    CoverageTopic(
        topic_id="development_finance",
        label="Development Finance",
        keywords=("finance", "financing", "investment", "funding", "grant", "loan", "guarantee", "blended", "donor", "ifi"),
        recommended_source="World Bank Projects, AfDB projects, IATI, OECD CRS, donor country strategy",
    ),
    CoverageTopic(
        topic_id="jobs_infrastructure",
        label="Jobs And Infrastructure",
        keywords=("jobs", "employment", "infrastructure", "transport", "digital", "industry", "industrial"),
        recommended_source="World Bank Africa's Pulse, country economic update, AfDB country strategy",
    ),
    CoverageTopic(
        topic_id="governance_transparency",
        label="Governance And Transparency",
        keywords=("governance", "transparency", "revenue", "accountability", "eiti", "regulation", "institution"),
        recommended_source="EITI country page/report, World Bank governance diagnostic, IMF Article IV",
    ),
    CoverageTopic(
        topic_id="climate_nature",
        label="Climate And Nature",
        keywords=("climate", "nature", "biodiversity", "forest", "conservation", "resilience", "adaptation", "carbon"),
        recommended_source="World Bank CCDR, NDC/LTS, BIOFIN or UNDP country material",
    ),
)


def build_coverage_matrix(
    *,
    source_entries: list[SourceRegistryEntry],
    knowledge_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return country-by-topic coverage rows.

    Source rows and knowledge records are evaluated separately. This lets the UI
    show whether a cell is backed by reviewed structured records or only by
    broader source availability.
    """

    matrix: list[dict[str, Any]] = []
    for _, country, subregion in AFRICAN_COUNTRIES:
        for topic in COVERAGE_TOPICS:
            source_counts = count_sources(country, topic, source_entries)
            record_counts = count_records(country, topic, knowledge_records)
            status = coverage_status(source_counts, record_counts)
            matrix.append(
                {
                    "country": country,
                    "subregion": subregion,
                    "topic_id": topic.topic_id,
                    "topic": topic.label,
                    "status": status,
                    "usable_country_records": record_counts["usable_country_records"],
                    "review_country_records": record_counts["review_country_records"],
                    "regional_records": record_counts["regional_records"],
                    "specific_registered_sources": source_counts["specific_registered_sources"],
                    "specific_downloaded_sources": source_counts["specific_downloaded_sources"],
                    "regional_registered_sources": source_counts["regional_registered_sources"],
                    "regional_downloaded_sources": source_counts["regional_downloaded_sources"],
                    "recommended_next_source": next_source_recommendation(status, topic),
                }
            )
    return matrix


def coverage_summary(matrix: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize the country-topic coverage matrix."""

    total = len(matrix)
    usable = sum(1 for row in matrix if row["status"] == "usable_records")
    review = sum(1 for row in matrix if row["status"] == "needs_review")
    specific_source = sum(1 for row in matrix if row["status"] == "specific_source_ready")
    regional_source = sum(1 for row in matrix if row["status"] == "regional_source_ready")
    registered = sum(1 for row in matrix if row["status"].endswith("registered"))
    specific_registered = sum(1 for row in matrix if row["status"] == "specific_source_registered")
    gaps = sum(1 for row in matrix if row["status"] == "gap")
    country_specific_ready = usable + review + specific_source + specific_registered
    country_specific_gaps = total - country_specific_ready
    return {
        "coverage_cells": total,
        "usable_record_cells": usable,
        "review_record_cells": review,
        "specific_source_ready_cells": specific_source,
        "regional_source_ready_cells": regional_source,
        "registered_only_cells": registered,
        "gap_cells": gaps,
        "country_specific_ready_cells": country_specific_ready,
        "country_specific_gap_cells": country_specific_gaps,
        "country_specific_ready_rate": round(country_specific_ready / total, 3) if total else 0.0,
        "usable_record_rate": round(usable / total, 3) if total else 0.0,
        "needs_work_rate": round((review + registered + gaps + regional_source) / total, 3) if total else 0.0,
        "countries": len({row["country"] for row in matrix}),
        "topics": len({row["topic_id"] for row in matrix}),
    }


def source_backlog(matrix: list[dict[str, Any]], limit: int = 50) -> list[dict[str, Any]]:
    """Return prioritized country-topic source gaps."""

    priority = {
        "gap": 0,
        "regional_source_registered": 1,
        "specific_source_registered": 2,
        "regional_source_ready": 3,
        "specific_source_ready": 4,
        "needs_review": 5,
        "usable_records": 6,
    }
    rows = [
        row
        for row in matrix
        if row["status"] in {
            "gap",
            "regional_source_registered",
            "specific_source_registered",
            "regional_source_ready",
            "specific_source_ready",
            "needs_review",
        }
    ]
    rows.sort(key=lambda row: (priority.get(row["status"], 9), row["country"], row["topic"]))
    return rows[:limit]


def count_sources(
    country: str,
    topic: CoverageTopic,
    source_entries: list[SourceRegistryEntry],
) -> dict[str, int]:
    """Count registry sources that cover one country-topic cell."""

    counts = {
        "specific_registered_sources": 0,
        "specific_downloaded_sources": 0,
        "regional_registered_sources": 0,
        "regional_downloaded_sources": 0,
    }
    for entry in source_entries:
        if not source_matches_topic(entry, topic):
            continue
        country_mode = source_country_mode(entry, country)
        if country_mode == "none":
            continue
        is_downloaded = entry.status == "downloaded"
        if country_mode == "specific":
            counts["specific_registered_sources"] += 1
            counts["specific_downloaded_sources"] += 1 if is_downloaded else 0
        elif country_mode == "regional":
            counts["regional_registered_sources"] += 1
            counts["regional_downloaded_sources"] += 1 if is_downloaded else 0
    return counts


def count_records(
    country: str,
    topic: CoverageTopic,
    records: list[dict[str, Any]],
) -> dict[str, int]:
    """Count knowledge records that cover one country-topic cell."""

    counts = {
        "usable_country_records": 0,
        "review_country_records": 0,
        "regional_records": 0,
    }
    for record in records:
        if not record_matches_topic(record, topic):
            continue
        record_country = str(record.get("country", ""))
        status = str(record.get("review_status", ""))
        if record_country == country:
            if status in {"usable", "approved"}:
                counts["usable_country_records"] += 1
            else:
                counts["review_country_records"] += 1
        elif record_country == "Africa":
            counts["regional_records"] += 1
    return counts


def coverage_status(source_counts: dict[str, int], record_counts: dict[str, int]) -> str:
    """Return the safest status label for one country-topic cell."""

    if record_counts["usable_country_records"]:
        return "usable_records"
    if record_counts["review_country_records"]:
        return "needs_review"
    if source_counts["specific_downloaded_sources"]:
        return "specific_source_ready"
    if source_counts["regional_downloaded_sources"] or record_counts["regional_records"]:
        return "regional_source_ready"
    if source_counts["specific_registered_sources"]:
        return "specific_source_registered"
    if source_counts["regional_registered_sources"]:
        return "regional_source_registered"
    return "gap"


def next_source_recommendation(status: str, topic: CoverageTopic) -> str:
    """Recommend the next source action for a cell."""

    if status == "usable_records":
        return "Maintain and periodically refresh reviewed records."
    if status == "needs_review":
        return "Review extracted record, verify citation and numbers, then approve or reject."
    if status == "specific_source_ready":
        return "Generate a focused work product to extract country-specific records."
    if status == "regional_source_ready":
        return f"Add a country-specific source: {topic.recommended_source}."
    if status == "specific_source_registered":
        return "Download and ingest the registered country-specific source."
    if status == "regional_source_registered":
        return "Download regional source, then add country-specific follow-up source."
    return f"Register and ingest a source such as: {topic.recommended_source}."


def source_matches_topic(entry: SourceRegistryEntry, topic: CoverageTopic) -> bool:
    """Return True when a registry row is relevant to a canonical topic."""

    haystack = " ".join([entry.title, entry.publisher, entry.topics, entry.notes]).lower()
    return any(keyword in haystack for keyword in topic.keywords)


def record_matches_topic(record: dict[str, Any], topic: CoverageTopic) -> bool:
    """Return True when a knowledge record is relevant to a canonical topic."""

    fields = record.get("fields") if isinstance(record.get("fields"), dict) else {}
    haystack = " ".join(
        str(value)
        for value in (
            record.get("title", ""),
            record.get("record_type", ""),
            record.get("sector", ""),
            record.get("theme", ""),
            record.get("commodity", ""),
            record.get("instrument", ""),
            fields,
        )
    ).lower()
    return any(keyword in haystack for keyword in topic.keywords)


def source_country_mode(entry: SourceRegistryEntry, country: str) -> str:
    """Return ``specific``, ``regional``, or ``none`` for source country coverage."""

    if entry.countries.strip().upper() == "ALL_AFRICA":
        return "regional"
    countries = {value.lower() for value in split_semicolon(entry.countries)}
    if country.lower() in countries:
        return "specific"
    return "none"
