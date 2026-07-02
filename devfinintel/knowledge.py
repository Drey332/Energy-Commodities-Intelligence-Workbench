"""Knowledge-record layer for the workbench.

Evidence chunks answer "where did this claim come from?" Extraction records
answer "what structured facts did this run pull out?" Knowledge records answer a
different operational question: "what reusable country, partner, finance,
resource, news, or case-study item should an intern review and maintain?"

That separation is what moves the project beyond initial document triage. A
reviewed knowledge record can be filtered by country, sector, commodity,
partner, and status across many runs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from devfinintel.extraction import (
    detect_countries,
    detect_finance_instruments,
    detect_sector,
    detect_themes,
    unique_preserve_order,
)
from devfinintel.models import EvidenceItem, ExtractionRecord, GeneratedOutput
from devfinintel.utils import stable_id, utc_now_iso


GENERIC_RECORD_TYPES = {
    "dataset_profile": "dataset_insight",
    "qa_evidence": "document_brief",
    "donor_profile_evidence": "partner_profile",
    "biofin_case_evidence": "case_study_card",
    "bulletin_item": "monitoring_digest",
}


RESOURCE_KEYWORDS = {
    "oil": "Oil",
    "gas": "Gas",
    "lng": "Liquefied natural gas",
    "petroleum": "Petroleum",
    "crude": "Crude oil",
    "cobalt": "Cobalt",
    "copper": "Copper",
    "lithium": "Lithium",
    "nickel": "Nickel",
    "manganese": "Manganese",
    "bauxite": "Bauxite",
    "iron ore": "Iron ore",
    "gold": "Gold",
    "diamond": "Diamonds",
    "mineral": "Minerals",
    "critical minerals": "Critical minerals",
    "hydropower": "Hydropower",
    "solar": "Solar",
    "wind": "Wind",
    "geothermal": "Geothermal",
    "commodity": "Commodities",
}


SECTOR_KEYWORDS = {
    "energy": "Energy",
    "electricity": "Energy",
    "power": "Power",
    "oil": "Oil and gas",
    "gas": "Oil and gas",
    "mining": "Mining",
    "mineral": "Mining",
    "commodity": "Commodities",
    "infrastructure": "Infrastructure",
    "transport": "Transport",
    "jobs": "Jobs and livelihoods",
    "employment": "Jobs and livelihoods",
    "private sector": "Private sector development",
    "biodiversity": "Nature and biodiversity",
    "climate": "Climate",
    "water": "Water",
    "agriculture": "Agriculture",
}


PARTNER_KEYWORDS = {
    "world bank": "World Bank",
    "international finance corporation": "IFC",
    "ifc": "IFC",
    "miga": "MIGA",
    "african development bank": "African Development Bank",
    "afdb": "African Development Bank",
    "international energy agency": "International Energy Agency",
    "iea": "International Energy Agency",
    "eiti": "Extractive Industries Transparency Initiative",
    "undp": "UNDP",
    "united nations development programme": "UNDP",
    "imf": "International Monetary Fund",
    "oecd": "OECD",
}

EVENT_TYPE_KEYWORDS = {
    "announced": "Announcement",
    "launched": "Project launch",
    "launch": "Project launch",
    "signed": "Agreement signed",
    "approved": "Financing approved",
    "investment": "Investment",
    "financing": "Financing",
    "funding": "Financing",
    "loan": "Financing",
    "grant": "Financing",
    "guarantee": "Financing",
    "policy": "Policy reform",
    "reform": "Policy reform",
    "regulation": "Policy reform",
    "delay": "Delay",
    "delayed": "Delay",
    "suspended": "Delay",
    "dispute": "Dispute",
    "conflict": "Conflict",
    "corruption": "Governance concern",
    "procurement": "Procurement issue",
}


RISK_FLAG_KEYWORDS = {
    "debt": "Debt",
    "sovereign debt": "Debt",
    "environment": "Environmental",
    "environmental": "Environmental",
    "biodiversity": "Environmental",
    "forest": "Environmental",
    "climate": "Climate",
    "drought": "Climate",
    "flood": "Climate",
    "emissions": "Climate",
    "social": "Social license",
    "community": "Social license",
    "resettlement": "Social license",
    "conflict": "Conflict",
    "violence": "Conflict",
    "security": "Conflict",
    "corruption": "Corruption",
    "bribery": "Corruption",
    "governance": "Governance",
    "transparency": "Governance",
    "delay": "Implementation delay",
    "delayed": "Implementation delay",
}


POSITIVE_TONE_TERMS = {
    "approved",
    "launched",
    "expanded",
    "mobilized",
    "improved",
    "growth",
    "opportunity",
    "investment",
    "financing",
    "jobs",
    "access",
}


NEGATIVE_TONE_TERMS = {
    "risk",
    "delay",
    "delayed",
    "conflict",
    "corruption",
    "debt",
    "dispute",
    "decline",
    "shortage",
    "crisis",
    "vulnerability",
    "emissions",
}


MONEY_RE = re.compile(
    r"\b(?P<currency>US\$|\$|USD|EUR|€|GBP|£)\s?(?P<amount>\d[\d,]*(?:\.\d+)?)\s?(?P<scale>million|billion|m|bn)?",
    flags=re.IGNORECASE,
)
YEAR_RE = re.compile(r"\b20\d{2}\b")


@dataclass(frozen=True)
class KnowledgeRecord:
    """A reusable, reviewable fact or work item.

    The top-level fields are intentionally plain and spreadsheet-like because a
    non-technical reviewer should be able to filter them without opening nested
    JSON. The ``fields`` dictionary keeps the original extracted details for
    traceability.
    """

    record_id: str
    record_type: str
    title: str
    country: str
    region: str
    sector: str
    theme: str
    commodity: str
    partner: str
    amount: str
    currency: str
    instrument: str
    event_date: str
    relevance: str
    actors: str
    event_type: str
    sentiment_tone: str
    risk_flags: str
    recommended_action: str
    source_document_id: str
    source_title: str
    source_page: int | None
    source_path: str
    evidence_chunk_ids: list[str]
    fields: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    review_status: str = "review"
    created_at: str = ""
    updated_at: str = ""


def knowledge_records_from_output(output: GeneratedOutput) -> list[KnowledgeRecord]:
    """Create durable knowledge records from one generated output."""

    evidence_by_chunk = {
        item.chunk_id: item
        for item in output.evidence_items
    }
    records: list[KnowledgeRecord] = []
    for extraction_record in output.records:
        item = first_matching_evidence(extraction_record, evidence_by_chunk)
        records.extend(knowledge_records_from_extraction(extraction_record, item))
    return records


def knowledge_records_from_extraction(
    record: ExtractionRecord,
    evidence_item: EvidenceItem | None,
) -> list[KnowledgeRecord]:
    """Convert one extraction record into one or more reusable knowledge records."""

    base_type = GENERIC_RECORD_TYPES.get(record.record_type, "review_note")
    countries = countries_from_record(record, evidence_item)
    if not countries:
        countries = ["Africa" if contains_africa(record, evidence_item) else "Not specified"]

    base_record = build_knowledge_record(
        record=record,
        evidence_item=evidence_item,
        record_type=base_type,
        country=countries[0],
    )
    knowledge_records = [base_record]

    for country in countries[1:]:
        knowledge_records.append(
            build_knowledge_record(
                record=record,
                evidence_item=evidence_item,
                record_type=base_type,
                country=country,
            )
        )

    if should_add_finance_resource_record(record, evidence_item):
        for country in countries:
            knowledge_records.append(
                build_knowledge_record(
                    record=record,
                    evidence_item=evidence_item,
                    record_type="finance_resource_record",
                    country=country,
                )
            )

    return unique_knowledge_records(knowledge_records)


def build_knowledge_record(
    *,
    record: ExtractionRecord,
    evidence_item: EvidenceItem | None,
    record_type: str,
    country: str,
) -> KnowledgeRecord:
    """Build one knowledge row from structured fields plus source metadata."""

    evidence_text = evidence_item.text if evidence_item else ""
    combined_text = " ".join(
        [
            record.title,
            evidence_text,
            " ".join(flatten_values(record.fields)),
        ]
    )
    amount, currency = detect_amount(record.fields, combined_text)
    event_type = first_or_empty(detect_event_types(combined_text))
    risk_flags = detect_risk_flags(combined_text)
    sentiment_tone = detect_sentiment_tone(combined_text)
    recommended_action = recommend_action(record_type, risk_flags, event_type, combined_text)
    actors = detect_actors(record, combined_text)
    now = utc_now_iso()
    source_page = evidence_item.page_number if evidence_item else None
    source_title = evidence_item.title if evidence_item else ""
    source_path = evidence_item.source_path if evidence_item else ""
    source_document_id = evidence_item.document_id if evidence_item else ""
    title_parts = [human_record_type(record_type), country, record.title]
    title = " - ".join(part for part in title_parts if part and part != "Not specified")

    return KnowledgeRecord(
        record_id=stable_id(
            "knowledge",
            record_type,
            record.record_id,
            country,
            source_document_id,
            source_page,
        ),
        record_type=record_type,
        title=title,
        country=country,
        region=region_for_country(country),
        sector=first_or_empty(detect_resource_sectors(combined_text)),
        theme=first_or_empty(detect_record_themes(record, combined_text)),
        commodity=first_or_empty(detect_commodities(combined_text)),
        partner=first_or_empty(detect_partners(record, combined_text)),
        amount=amount,
        currency=currency,
        instrument=first_or_empty(detect_record_instruments(record, combined_text)),
        event_date=first_or_empty(record.fields.get("years") or YEAR_RE.findall(combined_text)),
        relevance=detect_relevance(record_type, combined_text),
        actors="; ".join(actors),
        event_type=event_type,
        sentiment_tone=sentiment_tone,
        risk_flags="; ".join(risk_flags),
        recommended_action=recommended_action,
        source_document_id=source_document_id,
        source_title=source_title,
        source_page=source_page,
        source_path=source_path,
        evidence_chunk_ids=record.evidence_chunk_ids,
        fields={
            **record.fields,
            "_source_extraction_record_id": record.record_id,
            "_source_extraction_record_type": record.record_type,
            "relevance": detect_relevance(record_type, combined_text),
            "actors": actors,
            "event_type": event_type,
            "sentiment_tone": sentiment_tone,
            "risk_flags": risk_flags,
            "recommended_action": recommended_action,
        },
        confidence=record.confidence,
        review_status="review" if record.review_status == "review" or record.confidence < 0.65 else "usable",
        created_at=now,
        updated_at=now,
    )


def enrich_stored_knowledge_record(record: dict[str, Any]) -> dict[str, str]:
    """Derive monitoring-intelligence fields for older stored records.

    This uses the same transparent lexical rules as new record creation. Keeping
    the enrichment deterministic makes the review queue auditable: a reviewer
    can see exactly which source fields led to relevance, tone, risk, and action
    labels instead of treating them as hidden model confidence.
    """

    fields = record.get("fields") or {}
    text = " ".join(
        [
            str(record.get("title", "")),
            str(record.get("country", "")),
            str(record.get("region", "")),
            str(record.get("sector", "")),
            str(record.get("theme", "")),
            str(record.get("commodity", "")),
            str(record.get("partner", "")),
            str(record.get("amount", "")),
            str(record.get("instrument", "")),
            " ".join(flatten_values(fields)),
        ]
    )
    event_type = first_or_empty(detect_event_types(text))
    risk_flags = detect_risk_flags(text)
    record_type = str(record.get("record_type", ""))
    extraction_record = ExtractionRecord(
        record_id=str(fields.get("_source_extraction_record_id", record.get("record_id", ""))),
        record_type=str(fields.get("_source_extraction_record_type", record_type)),
        title=str(record.get("title", "")),
        fields=fields,
        evidence_chunk_ids=record.get("evidence_chunk_ids") or [],
        confidence=float(record.get("confidence") or 0.0),
        review_status=str(record.get("review_status", "review")),
    )
    return {
        "relevance": detect_relevance(record_type, text),
        "actors": "; ".join(detect_actors(extraction_record, text)),
        "event_type": event_type,
        "sentiment_tone": detect_sentiment_tone(text),
        "risk_flags": "; ".join(risk_flags),
        "recommended_action": recommend_action(record_type, risk_flags, event_type, text),
    }


def should_add_finance_resource_record(record: ExtractionRecord, evidence_item: EvidenceItem | None) -> bool:
    """Return True when an extraction also deserves a finance/resource record."""

    text = " ".join([evidence_item.text if evidence_item else "", " ".join(flatten_values(record.fields))])
    has_money = bool(record.fields.get("financial_figures") or record.fields.get("funding_amount") or MONEY_RE.search(text))
    has_resource = bool(detect_commodities(text) or "energy" in text.lower() or "mineral" in text.lower())
    return has_money or has_resource


def first_matching_evidence(
    record: ExtractionRecord,
    evidence_by_chunk: dict[str, EvidenceItem],
) -> EvidenceItem | None:
    """Find the evidence chunk that supports a structured record."""

    for chunk_id in record.evidence_chunk_ids:
        if chunk_id in evidence_by_chunk:
            return evidence_by_chunk[chunk_id]
    return None


def countries_from_record(record: ExtractionRecord, evidence_item: EvidenceItem | None) -> list[str]:
    """Extract country names from known fields and supporting evidence text."""

    candidate_values: list[str] = []
    for field_name in ("country", "countries_or_regions", "countries", "region"):
        value = record.fields.get(field_name)
        if isinstance(value, list):
            candidate_values.extend(str(item) for item in value)
        elif value:
            candidate_values.append(str(value))

    text = " ".join(candidate_values)
    if evidence_item:
        text += " " + evidence_item.text
    countries = detect_countries(text)
    if "Africa" in candidate_values and not countries:
        countries.append("Africa")
    return unique_preserve_order(countries)


def contains_africa(record: ExtractionRecord, evidence_item: EvidenceItem | None) -> bool:
    """Return True when record or evidence is clearly Africa-wide."""

    text = " ".join(flatten_values(record.fields)).lower()
    if evidence_item:
        text += " " + evidence_item.text.lower()
    return "africa" in text or "sub-saharan" in text


def detect_amount(fields: dict[str, Any], text: str) -> tuple[str, str]:
    """Find a money amount and currency for finance/resource review."""

    for field_name in ("financial_figures", "funding_amount", "amount"):
        value = fields.get(field_name)
        values = value if isinstance(value, list) else [value] if value else []
        for item in values:
            match = MONEY_RE.search(str(item))
            if match:
                return normalize_money(match.group(0)), normalize_currency(match.group("currency"))
    match = MONEY_RE.search(text)
    if match:
        return normalize_money(match.group(0)), normalize_currency(match.group("currency"))
    return "", ""


def detect_record_themes(record: ExtractionRecord, text: str) -> list[str]:
    """Detect a broad development theme."""

    values = record.fields.get("themes")
    explicit = values if isinstance(values, list) else [values] if values else []
    return unique_preserve_order([str(value) for value in explicit if value] + detect_themes(text))


def detect_record_instruments(record: ExtractionRecord, text: str) -> list[str]:
    """Detect finance instruments such as loans, grants, or guarantees."""

    values: list[str] = []
    for field_name in ("instrument", "instrument_type", "finance_solution"):
        value = record.fields.get(field_name)
        if isinstance(value, list):
            values.extend(str(item) for item in value if item)
        elif value:
            values.append(str(value))
    values.extend(detect_finance_instruments(text))
    return unique_preserve_order(values)


def detect_resource_sectors(text: str) -> list[str]:
    """Detect broad sectors relevant to development knowledge work."""

    lower = text.lower()
    sectors = [label for keyword, label in SECTOR_KEYWORDS.items() if keyword in lower]
    sectors.extend(detect_sector(text))
    return unique_preserve_order(sectors)


def detect_commodities(text: str) -> list[str]:
    """Detect resource and commodity terms."""

    lower = text.lower()
    return unique_preserve_order(label for keyword, label in RESOURCE_KEYWORDS.items() if keyword in lower)


def detect_partners(record: ExtractionRecord, text: str) -> list[str]:
    """Detect likely partner institutions from fields and text."""

    values: list[str] = []
    for field_name in ("partner", "partner_or_query", "source", "actor", "actors"):
        value = record.fields.get(field_name)
        if isinstance(value, list):
            values.extend(str(item) for item in value if item)
        elif value:
            values.append(str(value))

    lower = text.lower()
    values.extend(label for keyword, label in PARTNER_KEYWORDS.items() if keyword in lower)
    return unique_preserve_order(values)


def detect_actors(record: ExtractionRecord, text: str) -> list[str]:
    """Detect broad actors for monitoring intelligence."""

    actor_terms = {
        "government": "Government",
        "ministry": "Government",
        "regulator": "Government",
        "world bank": "IFI",
        "african development bank": "IFI",
        "afdb": "IFI",
        "imf": "IFI",
        "ifc": "IFI",
        "donor": "Donor",
        "company": "Company",
        "private sector": "Private sector",
        "investor": "Investor",
        "civil society": "Civil society",
        "community": "Community",
        "eiti": "Transparency initiative",
        "undp": "UN agency",
        "united nations": "UN agency",
    }
    lower = text.lower()
    actors = [label for keyword, label in actor_terms.items() if keyword in lower]
    actors.extend(detect_partners(record, text))
    return unique_preserve_order(actors)


def detect_event_types(text: str) -> list[str]:
    """Detect monitoring event types such as investment, reform, or delay."""

    lower = text.lower()
    return unique_preserve_order(label for keyword, label in EVENT_TYPE_KEYWORDS.items() if keyword in lower)


def detect_risk_flags(text: str) -> list[str]:
    """Detect risk flags for monitoring and review."""

    lower = text.lower()
    return unique_preserve_order(label for keyword, label in RISK_FLAG_KEYWORDS.items() if keyword in lower)


def detect_sentiment_tone(text: str) -> str:
    """Classify broad tone as a support signal, not as a truth score."""

    lower = text.lower()
    positive = sum(1 for term in POSITIVE_TONE_TERMS if term in lower)
    negative = sum(1 for term in NEGATIVE_TONE_TERMS if term in lower)
    if positive and negative:
        return "mixed"
    if positive:
        return "positive"
    if negative:
        return "negative"
    return "neutral"


def detect_relevance(record_type: str, text: str) -> str:
    """Score whether a record is useful for the knowledge base."""

    lower = text.lower()
    strong_terms = (
        "energy",
        "oil",
        "gas",
        "mining",
        "mineral",
        "finance",
        "investment",
        "infrastructure",
        "jobs",
        "governance",
        "climate",
        "biodiversity",
    )
    hits = sum(1 for term in strong_terms if term in lower)
    if record_type in {"monitoring_digest", "finance_resource_record", "partner_profile"} and hits >= 2:
        return "high"
    if hits >= 1:
        return "medium"
    return "low"


def recommend_action(
    record_type: str,
    risk_flags: list[str],
    event_type: str,
    text: str,
) -> str:
    """Recommend an operational follow-up action."""

    lower = text.lower()
    if risk_flags:
        return "add to bulletin and review risk flags"
    if record_type == "finance_resource_record" or any(term in lower for term in ("usd", "us$", "investment", "financing")):
        return "create finance record"
    if record_type == "partner_profile" or any(term in lower for term in ("partner", "donor", "world bank", "afdb", "ifi")):
        return "update partner profile"
    if record_type == "monitoring_digest" or event_type:
        return "add to monitoring digest"
    return "monitor only"


def normalize_money(value: str) -> str:
    """Clean a money string for spreadsheet display."""

    return re.sub(r"\s+", " ", value).strip()


def normalize_currency(value: str | None) -> str:
    """Map currency symbols and abbreviations into simple labels."""

    if not value:
        return ""
    normalized = value.upper()
    if normalized in {"US$", "$", "USD"}:
        return "USD"
    if normalized in {"€", "EUR"}:
        return "EUR"
    if normalized in {"£", "GBP"}:
        return "GBP"
    return normalized


def region_for_country(country: str) -> str:
    """Return a light region label for dashboard grouping."""

    mapping = {
        "Algeria": "North Africa",
        "Egypt": "North Africa",
        "Libya": "North Africa",
        "Morocco": "North Africa",
        "Tunisia": "North Africa",
        "Angola": "Central/Southern Africa",
        "Cameroon": "Central Africa",
        "Central African Republic": "Central Africa",
        "Chad": "Central Africa",
        "Democratic Republic of the Congo": "Central Africa",
        "Republic of the Congo": "Central Africa",
        "Congo": "Central Africa",
        "Equatorial Guinea": "Central Africa",
        "Gabon": "Central Africa",
        "Sao Tome and Principe": "Central Africa",
        "Benin": "West Africa",
        "Burkina Faso": "West Africa",
        "Cabo Verde": "West Africa",
        "Cape Verde": "West Africa",
        "Cote d'Ivoire": "West Africa",
        "Gambia": "West Africa",
        "Ghana": "West Africa",
        "Guinea": "West Africa",
        "Guinea-Bissau": "West Africa",
        "Liberia": "West Africa",
        "Mali": "West Africa",
        "Mauritania": "West Africa",
        "Niger": "West Africa",
        "Nigeria": "West Africa",
        "Senegal": "West Africa",
        "Sierra Leone": "West Africa",
        "Togo": "West Africa",
        "Burundi": "East Africa",
        "Comoros": "East Africa",
        "Djibouti": "East Africa",
        "Eritrea": "East Africa",
        "Ethiopia": "East Africa",
        "Kenya": "East Africa",
        "Madagascar": "East Africa",
        "Mauritius": "East Africa",
        "Rwanda": "East Africa",
        "Seychelles": "East Africa",
        "Somalia": "East Africa",
        "South Sudan": "East Africa",
        "Sudan": "East Africa",
        "Tanzania": "East Africa",
        "Uganda": "East Africa",
        "Botswana": "Southern Africa",
        "Eswatini": "Southern Africa",
        "Lesotho": "Southern Africa",
        "Malawi": "Southern Africa",
        "Mozambique": "Southern Africa",
        "Namibia": "Southern Africa",
        "South Africa": "Southern Africa",
        "Zambia": "Southern Africa",
        "Zimbabwe": "Southern Africa",
        "Africa": "Africa",
    }
    return mapping.get(country, "")


def flatten_values(value: Any) -> list[str]:
    """Flatten nested field values into strings for deterministic detection."""

    values: list[str] = []
    if isinstance(value, dict):
        for nested in value.values():
            values.extend(flatten_values(nested))
    elif isinstance(value, list):
        for nested in value:
            values.extend(flatten_values(nested))
    elif value is not None:
        values.append(str(value))
    return values


def first_or_empty(values: Any) -> str:
    """Return the first value from a list-like object, or an empty string."""

    if isinstance(values, list) and values:
        return str(values[0])
    if isinstance(values, str):
        return values
    return ""


def human_record_type(record_type: str) -> str:
    """Turn a system record type into a readable phrase."""

    return record_type.replace("_", " ").title()


def unique_knowledge_records(records: list[KnowledgeRecord]) -> list[KnowledgeRecord]:
    """Deduplicate records created from the same extraction event."""

    seen: set[str] = set()
    result: list[KnowledgeRecord] = []
    for record in records:
        if record.record_id not in seen:
            seen.add(record.record_id)
            result.append(record)
    return result


def coverage_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize knowledge-base coverage for dashboard metrics."""

    countries = {record.get("country", "") for record in records if record.get("country")}
    sectors = {record.get("sector", "") for record in records if record.get("sector")}
    commodities = {record.get("commodity", "") for record in records if record.get("commodity")}
    review = [record for record in records if record.get("review_status") == "review"]
    return {
        "knowledge_records": len(records),
        "countries": len(countries),
        "sectors": len(sectors),
        "commodities": len(commodities),
        "review_queue": len(review),
    }
