"""CSV dataset profiling for uploaded finance and indicator files.

RAG is useful for reports and narrative documents, but a CSV file should not be
handled only as loose text rows. A policy or partnerships user often needs basic
dataset facts first: what the rows represent, which columns exist, what years are
covered, whether there are missing values, and what the highest/lowest values
are. This module provides that transparent first-pass profile.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from devfinintel.models import (
    EvidenceItem,
    ExtractionRecord,
    GeneratedOutput,
    SourceDocument,
    VerificationFinding,
)
from devfinintel.schemas import validate_records
from devfinintel.utils import stable_id


@dataclass(frozen=True)
class NumericProfile:
    """Simple descriptive statistics for one numeric CSV column."""

    column: str
    count: int
    minimum: float
    maximum: float
    average: float
    missing_count: int = 0
    standard_deviation: float = 0.0


@dataclass(frozen=True)
class ColumnProfile:
    """Readable profile for one CSV column."""

    column: str
    inferred_type: str
    non_empty_count: int
    missing_count: int
    unique_count: int
    example_values: list[str]


class DatasetProfiler:
    """Create evidence-grounded summaries for selected CSV documents."""

    def profile(self, query: str, documents: list[SourceDocument]) -> GeneratedOutput:
        csv_documents = [document for document in documents if document.source_type == "csv"]
        if not csv_documents:
            return GeneratedOutput(
                task_type="dataset_profile",
                title=f"Dataset Profile: {query}",
                body_markdown=(
                    f"# Dataset Profile: {query}\n\n"
                    "No selected CSV documents were found. Select a CSV file or switch to a document Q&A task."
                ),
                evidence_items=[],
                records=[],
                verification_findings=[
                    VerificationFinding(
                        level="error",
                        message="Dataset profile requested, but no selected evidence source is a CSV file.",
                    )
                ],
                metrics={
                    "citation_coverage": 0.0,
                    "support_overlap": 0.0,
                    "unsupported_number_count": 0.0,
                    "evidence_items": 0.0,
                    "structured_records": 0.0,
                },
            )

        all_sections: list[str] = [f"# Dataset Profile: {query}", ""]
        evidence_items: list[EvidenceItem] = []
        records: list[ExtractionRecord] = []

        for document in csv_documents:
            rows = read_csv_rows(Path(document.source_path))
            profile_text, fields = profile_rows(document, rows)
            all_sections.append(profile_text)
            evidence_items.append(
                EvidenceItem(
                    chunk_id=stable_id("dataset-profile", document.document_id),
                    document_id=document.document_id,
                    title=document.title,
                    source_path=document.source_path,
                    page_number=1,
                    text=profile_text,
                    bm25_score=0.0,
                    dense_score=0.0,
                    rerank_score=1.0,
                )
            )
            records.append(
                ExtractionRecord(
                    record_id=stable_id("dataset-record", document.document_id, query),
                    record_type="dataset_profile",
                    title=f"Dataset profile for {document.title}",
                    fields=fields,
                    evidence_chunk_ids=[stable_id("dataset-profile", document.document_id)],
                    confidence=0.95,
                    review_status="usable",
                )
            )

        records = validate_records(records)
        body = "\n\n".join(all_sections)
        return GeneratedOutput(
            task_type="dataset_profile",
            title=f"Dataset Profile: {query}",
            body_markdown=body,
            evidence_items=evidence_items,
            records=records,
            verification_findings=[
                VerificationFinding(
                    level="pass",
                    message="Dataset profile was computed directly from selected CSV file(s).",
                )
            ],
            metrics={
                "citation_coverage": 1.0,
                "support_overlap": 1.0,
                "unsupported_number_count": 0.0,
                "evidence_items": float(len(evidence_items)),
                "structured_records": float(len(records)),
            },
        )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file as dictionaries while keeping values as plain text."""

    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle))


def profile_rows(document: SourceDocument, rows: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
    """Return Markdown plus structured fields for one CSV document."""

    columns = list(rows[0].keys()) if rows else []
    column_profiles = profile_columns(rows, columns)
    numeric_profiles = compute_numeric_profiles(rows, columns)
    missing_counts = {
        column: sum(1 for row in rows if not row.get(column))
        for column in columns
        if any(not row.get(column) for row in rows)
    }
    likely_label_columns = [column for column in columns if column.endswith("_LABEL")]
    years = sorted({row.get("TIME_PERIOD", "") for row in rows if row.get("TIME_PERIOD")})
    indicators = sorted({row.get("INDICATOR_LABEL", "") for row in rows if row.get("INDICATOR_LABEL")})
    databases = sorted({row.get("DATABASE_ID_LABEL", "") for row in rows if row.get("DATABASE_ID_LABEL")})
    country_column = choose_first_existing(columns, ["REF_AREA_LABEL", "country", "Country", "COUNTRY"])
    value_column = choose_first_existing(columns, ["OBS_VALUE", "value", "Value", "amount", "Amount"])
    top_values = rank_rows(rows, country_column, value_column, reverse=True)
    bottom_values = rank_rows(rows, country_column, value_column, reverse=False)
    value_summary = summarize_value_column(rows, value_column)
    outliers = detect_outliers(rows, country_column, value_column)
    correlations = compute_correlations(rows, numeric_profiles)
    suggested_questions = suggest_questions(
        document=document,
        rows=rows,
        columns=columns,
        numeric_profiles=numeric_profiles,
        country_column=country_column,
        value_column=value_column,
        indicators=indicators,
    )

    lines = [
        f"## {document.title}",
        "",
        f"- Source file: `{document.source_path}` ({document.title}, p. 1)",
        f"- Rows: {len(rows)} ({document.title}, p. 1)",
        f"- Columns: {len(columns)} ({document.title}, p. 1)",
        f"- Years covered: {', '.join(years) if years else 'Not detected'} ({document.title}, p. 1)",
        f"- Indicator labels: {', '.join(indicators[:5]) if indicators else 'Not detected'} ({document.title}, p. 1)",
        f"- Database label: {', '.join(databases[:3]) if databases else 'Not detected'} ({document.title}, p. 1)",
        "",
        "### Key Insights",
    ]
    lines.extend(
        build_key_insights(
            document=document,
            rows=rows,
            country_column=country_column,
            value_column=value_column,
            value_summary=value_summary,
            indicators=indicators,
            top_values=top_values,
            bottom_values=bottom_values,
        )
    )
    lines.extend(
        [
            "",
            "### Suggested Questions",
        ]
    )
    lines.extend(f"- {question}" for question in suggested_questions)
    lines.extend(
        [
            "",
            "### Column Types",
        ]
    )
    if column_profiles:
        for profile in column_profiles:
            examples = ", ".join(profile.example_values[:3]) if profile.example_values else "No examples"
            lines.append(
                f"- `{profile.column}`: {profile.inferred_type}; "
                f"{profile.non_empty_count} non-empty, {profile.missing_count} missing, "
                f"{profile.unique_count} unique; examples: {examples}"
            )
    else:
        lines.append("- No columns detected.")
    lines.extend(
        [
            "",
            "### Numeric Columns",
        ]
    )

    if numeric_profiles:
        for profile in numeric_profiles:
            lines.append(
                f"- `{profile.column}`: count {profile.count}, min {format_number(profile.minimum)}, "
                f"mean {format_number(profile.average)}, max {format_number(profile.maximum)}, "
                f"standard deviation {format_number(profile.standard_deviation)} ({document.title}, p. 1)"
            )
    else:
        lines.append("- No numeric columns detected.")

    if top_values:
        lines.extend(["", "### Highest Values"])
        lines.extend(f"- {label}: {format_number(value)} ({document.title}, p. 1)" for label, value in top_values)

    if bottom_values:
        lines.extend(["", "### Lowest Values"])
        lines.extend(f"- {label}: {format_number(value)} ({document.title}, p. 1)" for label, value in bottom_values)

    if outliers:
        lines.extend(["", "### Potential Outliers"])
        lines.extend(f"- {label}: {format_number(value)} ({document.title}, p. 1)" for label, value in outliers[:8])

    if correlations:
        lines.extend(["", "### Strongest Numeric Relationships"])
        for pair in correlations[:5]:
            lines.append(
                f"- `{pair['left']}` and `{pair['right']}`: correlation {format_number(pair['correlation'])} "
                f"({document.title}, p. 1)"
            )

    lines.extend(
        [
            "",
            "### Data Quality Notes",
            f"- Label columns detected: {', '.join(likely_label_columns[:12]) if likely_label_columns else 'None detected.'}",
            f"- Columns with missing values: {format_missing_counts(missing_counts)}",
            "- Treat this as a first-pass dataset profile. Confirm source definitions before using the indicator analytically.",
        ]
    )

    fields = {
        "source_path": document.source_path,
        "rows": len(rows),
        "columns": columns,
        "column_profiles": [profile.__dict__ for profile in column_profiles],
        "years": years,
        "indicators": indicators,
        "databases": databases,
        "numeric_profiles": [profile.__dict__ for profile in numeric_profiles],
        "highest_values": [{"label": label, "value": value} for label, value in top_values],
        "lowest_values": [{"label": label, "value": value} for label, value in bottom_values],
        "potential_outliers": [{"label": label, "value": value} for label, value in outliers],
        "correlations": correlations,
        "suggested_questions": suggested_questions,
        "primary_value_column": value_column,
        "primary_label_column": country_column,
        "value_summary": value_summary,
        "missing_counts": missing_counts,
    }
    return "\n".join(lines), fields


def profile_columns(rows: list[dict[str, str]], columns: list[str]) -> list[ColumnProfile]:
    """Infer basic column types and data-quality facts."""

    profiles: list[ColumnProfile] = []
    total_rows = len(rows)
    for column in columns:
        values = [str(row.get(column, "")).strip() for row in rows]
        non_empty = [value for value in values if value]
        numeric_values = [parse_float(value) for value in non_empty]
        numeric_count = sum(1 for value in numeric_values if value is not None)
        unique_values = sorted(set(non_empty))
        inferred_type = infer_column_type(column, non_empty, numeric_count)
        profiles.append(
            ColumnProfile(
                column=column,
                inferred_type=inferred_type,
                non_empty_count=len(non_empty),
                missing_count=total_rows - len(non_empty),
                unique_count=len(unique_values),
                example_values=unique_values[:5],
            )
        )
    return profiles


def infer_column_type(column: str, non_empty: list[str], numeric_count: int) -> str:
    """Infer a human-readable column type from values and naming conventions."""

    lower = column.lower()
    if not non_empty:
        return "empty"
    if "date" in lower or "time_period" in lower or lower in {"year"}:
        return "time"
    if lower.endswith("_label") or "country" in lower or "area" in lower:
        return "label/category"
    if numeric_count >= max(2, math.ceil(len(non_empty) * 0.8)):
        return "numeric"
    if len(set(non_empty)) <= max(20, len(non_empty) * 0.2):
        return "category"
    return "text"


def compute_numeric_profiles(rows: list[dict[str, str]], columns: list[str]) -> list[NumericProfile]:
    """Compute simple statistics for columns that are mostly numeric."""

    profiles: list[NumericProfile] = []
    for column in columns:
        values = [parse_float(row.get(column, "")) for row in rows]
        numeric_values = [value for value in values if value is not None and math.isfinite(value)]
        minimum_numeric_values = 1 if len(rows) == 1 else max(2, math.ceil(len(rows) * 0.6))
        if len(numeric_values) >= minimum_numeric_values:
            profiles.append(
                NumericProfile(
                    column=column,
                    count=len(numeric_values),
                    minimum=min(numeric_values),
                    maximum=max(numeric_values),
                    average=mean(numeric_values),
                    missing_count=len(rows) - len(numeric_values),
                    standard_deviation=pstdev(numeric_values) if len(numeric_values) > 1 else 0.0,
                )
            )
    return profiles


def detect_outliers(
    rows: list[dict[str, str]],
    label_column: str | None,
    value_column: str | None,
) -> list[tuple[str, float]]:
    """Detect possible outliers using a simple z-score threshold."""

    if not label_column or not value_column:
        return []
    values = []
    for row in rows:
        value = parse_float(row.get(value_column, ""))
        label = row.get(label_column, "")
        if value is not None and label:
            values.append((label, value))
    if len(values) < 5:
        return []
    numeric_values = [value for _, value in values]
    average = mean(numeric_values)
    deviation = pstdev(numeric_values)
    if deviation == 0:
        return []
    outliers = [(label, value) for label, value in values if abs((value - average) / deviation) >= 2.0]
    outliers.sort(key=lambda item: abs((item[1] - average) / deviation), reverse=True)
    return outliers


def compute_correlations(
    rows: list[dict[str, str]],
    numeric_profiles: list[NumericProfile],
) -> list[dict[str, float | str]]:
    """Compute Pearson correlations for numeric columns with enough variation."""

    numeric_columns = [
        profile.column
        for profile in numeric_profiles
        if profile.count >= 5 and not math.isclose(profile.standard_deviation, 0.0)
    ]
    correlations: list[dict[str, float | str]] = []
    for i, left in enumerate(numeric_columns):
        for right in numeric_columns[i + 1 :]:
            paired = []
            for row in rows:
                left_value = parse_float(row.get(left, ""))
                right_value = parse_float(row.get(right, ""))
                if left_value is not None and right_value is not None:
                    paired.append((left_value, right_value))
            if len(paired) < 5:
                continue
            correlation = pearson_correlation([x for x, _ in paired], [y for _, y in paired])
            if abs(correlation) >= 0.6:
                correlations.append({"left": left, "right": right, "correlation": round(correlation, 4)})
    correlations.sort(key=lambda item: abs(float(item["correlation"])), reverse=True)
    return correlations


def pearson_correlation(left: list[float], right: list[float]) -> float:
    """Compute a small dependency-free Pearson correlation."""

    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_denominator = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_denominator = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    denominator = left_denominator * right_denominator
    return 0.0 if denominator == 0 else numerator / denominator


def suggest_questions(
    *,
    document: SourceDocument,
    rows: list[dict[str, str]],
    columns: list[str],
    numeric_profiles: list[NumericProfile],
    country_column: str | None,
    value_column: str | None,
    indicators: list[str],
) -> list[str]:
    """Generate deterministic follow-up questions for the UI/LLM layer."""

    questions = [
        f"What are the most important insights in {document.title}?",
        "Are there missing values or data-quality issues I should review?",
    ]
    if value_column and country_column:
        questions.append(f"Which {country_column} entries have the highest and lowest {value_column}?")
        questions.append(f"Are there outliers in {value_column}, and what might they imply?")
    if len(numeric_profiles) >= 2:
        questions.append("Which numeric columns are most strongly related?")
    if indicators:
        questions.append(f"What does the indicator '{indicators[0]}' measure, based only on this file?")
    if len(columns) > 15:
        questions.append("Which columns are metadata, labels, and actual analytical values?")
    if len(rows) > 1000:
        questions.append("Can you summarize this large dataset without listing every row?")
    return questions[:7]


def build_key_insights(
    *,
    document: SourceDocument,
    rows: list[dict[str, str]],
    country_column: str | None,
    value_column: str | None,
    value_summary: dict[str, float | int | None],
    indicators: list[str],
    top_values: list[tuple[str, float]],
    bottom_values: list[tuple[str, float]],
) -> list[str]:
    """Create a short plain-English readout of what the dataset says."""

    insights = [
        f"- The file contains {len(rows)} observation rows. ({document.title}, p. 1)"
    ]
    if indicators:
        insights.append(
            f"- The main indicator appears to be: {', '.join(indicators[:3])}. ({document.title}, p. 1)"
        )
    if country_column:
        unique_labels = {row.get(country_column, "") for row in rows if row.get(country_column)}
        insights.append(
            f"- `{country_column}` identifies {len(unique_labels)} distinct places or entities. ({document.title}, p. 1)"
        )
    if value_column and value_summary.get("count"):
        insights.append(
            f"- `{value_column}` is the main numeric value column: min {format_number(value_summary['minimum'])}, "
            f"mean {format_number(value_summary['average'])}, max {format_number(value_summary['maximum'])}. "
            f"({document.title}, p. 1)"
        )
    if top_values:
        insights.append(
            f"- Highest values: {format_ranked_values(top_values[:3])}. ({document.title}, p. 1)"
        )
    if bottom_values:
        insights.append(
            f"- Lowest values: {format_ranked_values(bottom_values[:3])}. ({document.title}, p. 1)"
        )
    insights.append(
        "- This profile describes the dataset contents; it does not by itself explain causality or validate the indicator methodology."
    )
    return insights


def summarize_value_column(rows: list[dict[str, str]], value_column: str | None) -> dict[str, float | int | None]:
    """Summarize the primary numeric value column."""

    if not value_column:
        return {"count": 0, "minimum": None, "maximum": None, "average": None}
    values = [
        value
        for row in rows
        for value in [parse_float(row.get(value_column, ""))]
        if value is not None and math.isfinite(value)
    ]
    if not values:
        return {"count": 0, "minimum": None, "maximum": None, "average": None}
    return {
        "count": len(values),
        "minimum": min(values),
        "maximum": max(values),
        "average": mean(values),
    }


def rank_rows(
    rows: list[dict[str, str]],
    label_column: str | None,
    value_column: str | None,
    reverse: bool,
    limit: int = 5,
) -> list[tuple[str, float]]:
    """Rank rows by a numeric value column and return readable labels."""

    if not label_column or not value_column:
        return []
    scored = []
    for row in rows:
        value = parse_float(row.get(value_column, ""))
        label = row.get(label_column, "")
        if value is not None and label:
            scored.append((label, value))
    scored.sort(key=lambda item: item[1], reverse=reverse)
    return scored[:limit]


def choose_first_existing(columns: list[str], candidates: list[str]) -> str | None:
    """Return the first candidate column that exists in the CSV."""

    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def parse_float(value: str) -> float | None:
    """Parse a number if possible."""

    try:
        return float(str(value).replace(",", "").strip())
    except ValueError:
        return None


def format_number(value: float) -> str:
    """Format numbers compactly for reports."""

    if value is None:
        return "not available"
    if abs(value) >= 1000:
        return f"{value:,.2f}"
    return f"{value:.6g}"


def format_ranked_values(values: list[tuple[str, float]]) -> str:
    """Format ranked label-value pairs for the insight summary."""

    return ", ".join(f"{label} ({format_number(value)})" for label, value in values)


def format_missing_counts(missing_counts: dict[str, int]) -> str:
    """Format missing-value counts for a short data-quality note."""

    if not missing_counts:
        return "None detected."
    shown = list(missing_counts.items())[:10]
    suffix = f" and {len(missing_counts) - 10} more" if len(missing_counts) > 10 else ""
    return ", ".join(f"{column}={count}" for column, count in shown) + suffix
