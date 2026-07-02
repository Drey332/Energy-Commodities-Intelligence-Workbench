"""Schema validation for extracted records.

The project separates facts from writing. This module makes that separation
visible by checking that each extracted record contains the fields expected for
its work product. It is a lightweight local equivalent of JSON-schema governed
extraction: no record is silently treated as complete when key fields are absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from devfinintel.models import ExtractionRecord


SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class RecordSchema:
    """Human-readable schema for one type of structured record."""

    name: str
    required_fields: tuple[str, ...]
    recommended_fields: tuple[str, ...] = ()


SCHEMAS: dict[str, RecordSchema] = {
    "dataset_profile": RecordSchema(
        name="dataset_profile",
        required_fields=("source_path", "rows", "columns", "column_profiles"),
        recommended_fields=(
            "numeric_profiles",
            "highest_values",
            "lowest_values",
            "missing_counts",
            "suggested_questions",
        ),
    ),
    "donor_profile_evidence": RecordSchema(
        name="donor_profile_evidence",
        required_fields=("partner_or_query", "citation"),
        recommended_fields=("financial_figures", "years", "countries_or_regions", "themes"),
    ),
    "biofin_case_evidence": RecordSchema(
        name="biofin_case_evidence",
        required_fields=("citation",),
        recommended_fields=(
            "country",
            "finance_solution",
            "instrument_type",
            "funding_amount",
            "lessons_learned",
        ),
    ),
    "bulletin_item": RecordSchema(
        name="bulletin_item",
        required_fields=("headline_or_query", "citation"),
        recommended_fields=("countries_or_regions", "years", "themes", "summary"),
    ),
    "qa_evidence": RecordSchema(
        name="qa_evidence",
        required_fields=("query", "citation"),
        recommended_fields=("key_sentences",),
    ),
}


def validate_records(records: list[ExtractionRecord]) -> list[ExtractionRecord]:
    """Attach schema metadata and validation warnings to records.

    The function returns new dataclass instances because ``ExtractionRecord`` is
    immutable. The original values are preserved; schema notes are added under
    underscore-prefixed field names so export users can see the audit metadata.
    """

    return [validate_record(record) for record in records]


def validate_record(record: ExtractionRecord) -> ExtractionRecord:
    """Validate one extraction record and return an annotated copy."""

    schema = SCHEMAS.get(record.record_type)
    errors: list[str] = []
    warnings: list[str] = []
    if schema is None:
        schema_name = "unknown"
        errors.append(f"No schema is registered for record type '{record.record_type}'.")
    else:
        schema_name = schema.name
        for field in schema.required_fields:
            if is_missing(record.fields.get(field)):
                errors.append(f"Required field '{field}' is missing or empty.")
        for field in schema.recommended_fields:
            if field not in record.fields:
                warnings.append(f"Recommended field '{field}' is not present.")

    if not record.evidence_chunk_ids:
        errors.append("Record has no evidence chunk ID.")

    fields = dict(record.fields)
    fields["_schema_name"] = schema_name
    fields["_schema_version"] = SCHEMA_VERSION
    fields["_schema_valid"] = not errors
    fields["_schema_errors"] = errors
    fields["_schema_warnings"] = warnings

    review_status = record.review_status
    if errors:
        review_status = "review"

    return ExtractionRecord(
        record_id=record.record_id,
        record_type=record.record_type,
        title=record.title,
        fields=fields,
        evidence_chunk_ids=record.evidence_chunk_ids,
        confidence=record.confidence,
        review_status=review_status,
    )


def schema_summary(records: list[ExtractionRecord]) -> dict[str, Any]:
    """Return compact schema metrics for dashboards and export manifests."""

    total = len(records)
    invalid = 0
    review = 0
    for record in records:
        if record.review_status == "review":
            review += 1
        if record.fields.get("_schema_valid") is False:
            invalid += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "records_checked": total,
        "invalid_records": invalid,
        "review_records": review,
        "schema_valid_rate": round((total - invalid) / total, 3) if total else 1.0,
    }


def is_missing(value: Any) -> bool:
    """Return True for empty values that should not satisfy a schema field."""

    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False
