"""Institutional role-readiness mapping for the workbench.

This module translates internship-style responsibilities into product
capabilities. The goal is not to hard-code the app for one vacancy; it is to
show that the platform supports the underlying knowledge work: source
monitoring, structured evidence, partner intelligence, finance/resource records,
case studies, synthesis, and reviewable outputs.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


ROLE_REQUIREMENTS = [
    {
        "role": "RBA / Strategic Partnerships",
        "requirement": "Donor and IFI mapping",
        "capability": "Stakeholder / Partner Map",
        "evidence_types": {"partner_profile", "finance_resource_record"},
        "metric_key": "partner_records",
        "implemented": True,
        "why_it_matters": "Supports partner intelligence across donors, IFIs, governments, and private actors.",
        "next_upgrade": "Add entity resolution so World Bank, IBRD, IDA, IFC, and MIGA can be linked under one group where appropriate.",
    },
    {
        "role": "RBA / Strategic Partnerships",
        "requirement": "Two-page partnership profiles",
        "capability": "Partner Profile",
        "evidence_types": {"partner_profile"},
        "metric_key": "partner_records",
        "implemented": True,
        "why_it_matters": "Turns source evidence into reusable partner summaries with citations and review status.",
        "next_upgrade": "Add a strict two-page export template with sections for priorities, financial figures, geography, risks, and next steps.",
    },
    {
        "role": "RBA / Strategic Partnerships",
        "requirement": "Financial figure updates",
        "capability": "Finance / Resource Record",
        "evidence_types": {"finance_resource_record"},
        "metric_key": "finance_records",
        "implemented": True,
        "why_it_matters": "Captures commitments, instruments, sectors, countries, commodities, and cited source pages.",
        "next_upgrade": "Add currency normalization, year-specific amount extraction, and deduplication across documents.",
    },
    {
        "role": "RBA / Strategic Partnerships",
        "requirement": "Africa partnership news scans and bulletin inputs",
        "capability": "Monitoring Digest",
        "evidence_types": {"monitoring_digest"},
        "metric_key": "monitoring_events",
        "implemented": True,
        "why_it_matters": "Normalizes monitoring signals into dated events, outcomes, risk flags, tone, and recommended actions.",
        "next_upgrade": "Add source-specific World Bank, AfDB, UNDP, EITI, and approved-news connectors with deduplication.",
    },
    {
        "role": "RBA / Strategic Partnerships",
        "requirement": "Visual data presentation",
        "capability": "Dataset Insight + Monitoring Visuals",
        "evidence_types": {"dataset_insight"},
        "metric_key": "visual_surfaces",
        "implemented": True,
        "why_it_matters": "Provides CSV profiles, maps, trend charts, risk charts, and country attention views.",
        "next_upgrade": "Add user-editable charts and exportable dashboard snapshots for briefings.",
    },
    {
        "role": "BIOFIN / Planet Hub",
        "requirement": "Cross-country biodiversity finance research",
        "capability": "Country Intelligence Workspace",
        "evidence_types": {"case_study_card", "finance_resource_record", "document_brief"},
        "metric_key": "country_records",
        "implemented": True,
        "why_it_matters": "Lets analysts compare country records, coverage gaps, sectors, risks, actions, and source provenance.",
        "next_upgrade": "Add a country-comparison export and controlled BIOFIN finance-solution vocabulary.",
    },
    {
        "role": "BIOFIN / Planet Hub",
        "requirement": "Nature-finance case-study extraction",
        "capability": "Case Study Card",
        "evidence_types": {"case_study_card"},
        "metric_key": "case_records",
        "implemented": True,
        "why_it_matters": "Extracts country, instrument, actors, amounts, lessons, and source citations for review.",
        "next_upgrade": "Add a case-study quality rubric and lessons-learned taxonomy.",
    },
    {
        "role": "BIOFIN / Planet Hub",
        "requirement": "Finance-source database support",
        "capability": "Knowledge Base + Source Registry",
        "evidence_types": {"finance_resource_record", "case_study_card", "partner_profile"},
        "metric_key": "knowledge_records",
        "implemented": True,
        "why_it_matters": "Stores reusable records by country, sector, commodity, partner, instrument, amount, and review status.",
        "next_upgrade": "Add OECD/IATI/BIOFIN controlled vocabularies and source-specific importers.",
    },
    {
        "role": "BIOFIN / Planet Hub",
        "requirement": "Synthesis of lessons learned and knowledge products",
        "capability": "Knowledge Product Draft",
        "evidence_types": {"case_study_card", "document_brief"},
        "metric_key": "knowledge_product_support",
        "implemented": True,
        "why_it_matters": "Drafts source-grounded synthesis from evidence packs instead of relying on model memory.",
        "next_upgrade": "Add reusable publication templates for blog posts, briefing notes, and internal memos.",
    },
    {
        "role": "BIOFIN / Planet Hub",
        "requirement": "News or press-clipping tracking",
        "capability": "Monitoring Insights",
        "evidence_types": {"monitoring_digest"},
        "metric_key": "monitoring_events",
        "implemented": True,
        "why_it_matters": "Turns press/news/report signals into events, outcomes, risk flags, and reviewable actions.",
        "next_upgrade": "Add approved live feeds, cross-source event deduplication, and analyst-reviewed event labels.",
    },
]


def build_role_alignment_report(
    *,
    knowledge_records: list[dict[str, Any]],
    monitoring_events: list[dict[str, Any]],
    action_items: list[dict[str, Any]],
    source_summary: dict[str, Any],
    coverage_summary: dict[str, Any],
) -> dict[str, Any]:
    """Return readiness metrics for RBA/BIOFIN-style knowledge work."""

    metrics = alignment_metrics(
        knowledge_records=knowledge_records,
        monitoring_events=monitoring_events,
        action_items=action_items,
        source_summary=source_summary,
        coverage_summary=coverage_summary,
    )
    rows = []
    for requirement in ROLE_REQUIREMENTS:
        evidence_count = metric_value(requirement["metric_key"], metrics)
        readiness = readiness_label(requirement["implemented"], evidence_count)
        rows.append(
            {
                "role": requirement["role"],
                "requirement": requirement["requirement"],
                "platform_capability": requirement["capability"],
                "readiness": readiness,
                "evidence_count": evidence_count,
                "why_it_matters": requirement["why_it_matters"],
                "next_upgrade": requirement["next_upgrade"],
            }
        )

    role_scores = []
    for role in sorted({row["role"] for row in rows}):
        role_rows = [row for row in rows if row["role"] == role]
        score = round(
            sum(readiness_points(row["readiness"]) for row in role_rows)
            / max(len(role_rows), 1),
            2,
        )
        role_scores.append({"role": role, "readiness_score": score, "requirements": len(role_rows)})

    executive_summary = build_executive_summary(rows, metrics)
    return {
        "summary": {
            "requirements": len(rows),
            "ready": sum(1 for row in rows if row["readiness"] == "ready"),
            "baseline": sum(1 for row in rows if row["readiness"] == "baseline"),
            "needs_data": sum(1 for row in rows if row["readiness"] == "needs_data"),
            "knowledge_records": metrics["knowledge_records"],
            "monitoring_events": metrics["monitoring_events"],
            "open_actions": metrics["open_actions"],
            "countries_with_records": metrics["countries_with_records"],
        },
        "role_scores": role_scores,
        "requirements": rows,
        "metrics": metrics,
        "executive_summary": executive_summary,
        "next_build_priorities": next_build_priorities(rows, metrics),
    }


def alignment_metrics(
    *,
    knowledge_records: list[dict[str, Any]],
    monitoring_events: list[dict[str, Any]],
    action_items: list[dict[str, Any]],
    source_summary: dict[str, Any],
    coverage_summary: dict[str, Any],
) -> dict[str, int]:
    """Compute evidence counts used by the role-alignment report."""

    type_counts = Counter(record.get("record_type", "") for record in knowledge_records)
    countries = {
        record.get("country")
        for record in knowledge_records
        if record.get("country") and record.get("country") != "Not specified"
    }
    visual_surfaces = 0
    if type_counts.get("dataset_insight", 0):
        visual_surfaces += 1
    if monitoring_events:
        visual_surfaces += 3
    if coverage_summary.get("coverage_cells", 0):
        visual_surfaces += 1
    return {
        "knowledge_records": len(knowledge_records),
        "partner_records": type_counts.get("partner_profile", 0),
        "finance_records": type_counts.get("finance_resource_record", 0),
        "case_records": type_counts.get("case_study_card", 0),
        "document_briefs": type_counts.get("document_brief", 0),
        "dataset_insights": type_counts.get("dataset_insight", 0),
        "monitoring_events": len(monitoring_events),
        "open_actions": sum(1 for action in action_items if action.get("status") in {"open", "in_progress"}),
        "countries_with_records": len(countries),
        "registered_sources": int(source_summary.get("sources", 0) or 0),
        "downloaded_sources": int(source_summary.get("downloaded_sources", 0) or 0),
        "coverage_cells": int(coverage_summary.get("coverage_cells", 0) or 0),
        "coverage_gaps": int(coverage_summary.get("country_specific_gap_cells", 0) or 0),
        "visual_surfaces": visual_surfaces,
        "country_records": len(countries),
        "knowledge_product_support": (
            type_counts.get("case_study_card", 0)
            + type_counts.get("document_brief", 0)
            + type_counts.get("monitoring_digest", 0)
        ),
    }


def metric_value(metric_key: str, metrics: dict[str, int]) -> int:
    """Return the numeric metric for one requirement."""

    return int(metrics.get(metric_key, 0) or 0)


def readiness_label(implemented: bool, evidence_count: int) -> str:
    """Classify readiness using both feature presence and local evidence."""

    if implemented and evidence_count >= 5:
        return "ready"
    if implemented and evidence_count > 0:
        return "baseline"
    if implemented:
        return "needs_data"
    return "not_started"


def readiness_points(readiness: str) -> float:
    """Score readiness labels for role summaries."""

    return {
        "ready": 1.0,
        "baseline": 0.65,
        "needs_data": 0.35,
        "not_started": 0.0,
    }.get(readiness, 0.0)


def build_executive_summary(rows: list[dict[str, Any]], metrics: dict[str, int]) -> list[dict[str, str]]:
    """Create plain-language readiness findings."""

    findings = []
    if metrics["monitoring_events"]:
        findings.append(
            {
                "finding": "The platform now has a live-intelligence backbone.",
                "evidence": f"{metrics['monitoring_events']} monitoring events and {metrics['open_actions']} open analyst actions.",
                "implication": "It can support bulletin/news monitoring workflows, pending stronger live connectors and review labels.",
            }
        )
    if metrics["finance_records"]:
        findings.append(
            {
                "finding": "Finance/resource tracking is operational.",
                "evidence": f"{metrics['finance_records']} finance/resource records.",
                "implication": "This supports RBA financial-figure updates and BIOFIN finance-source database work.",
            }
        )
    if metrics["coverage_gaps"]:
        findings.append(
            {
                "finding": "Coverage gaps are visible instead of hidden.",
                "evidence": f"{metrics['coverage_gaps']} country-topic cells still lack country-specific downloaded sources.",
                "implication": "The next data work should be targeted source acquisition, not random document upload.",
            }
        )
    weak_rows = [row for row in rows if row["readiness"] in {"needs_data", "not_started"}]
    if weak_rows:
        findings.append(
            {
                "finding": "Some capabilities need better seed data or templates.",
                "evidence": f"{len(weak_rows)} requirements are not yet data-ready.",
                "implication": "Prioritize connectors, controlled vocabularies, and export templates before adding unrelated features.",
            }
        )
    return findings


def next_build_priorities(rows: list[dict[str, Any]], metrics: dict[str, int]) -> list[dict[str, str]]:
    """Return a small prioritized engineering backlog."""

    priorities: list[dict[str, str]] = []
    if metrics["monitoring_events"] and metrics["registered_sources"] < 8:
        priorities.append(
            {
                "priority": "1",
                "workstream": "Live source connectors",
                "action": "Add approved institutional RSS/API connectors for World Bank, AfDB, UNDP, EITI, OECD/IATI, and BIOFIN resources.",
                "reason": "The monitoring model exists; source breadth is now the constraint.",
            }
        )
    if metrics["finance_records"]:
        priorities.append(
            {
                "priority": "2",
                "workstream": "Finance normalization",
                "action": "Normalize amounts, currencies, years, instruments, counterparties, and project IDs across records.",
                "reason": "RBA and BIOFIN both depend on reliable financial figures.",
            }
        )
    priorities.append(
        {
            "priority": "3",
            "workstream": "Export templates",
            "action": "Create role-shaped templates: 2-page partner profile, bulletin item, BIOFIN case card, finance-source CSV, and meeting note.",
            "reason": "Internship value is judged by finished work products, not only by database rows.",
        }
    )
    priorities.append(
        {
            "priority": "4",
            "workstream": "Evaluation and governance",
            "action": "Add a small gold set, event-label review, source credibility tracking, and citation/number regression metrics.",
            "reason": "Institutional deployment needs measurable trust and review loops.",
        }
    )
    return priorities
