"""Structured extraction and evidence-bound drafting.

This module keeps the most important rule of the project: facts are extracted
before prose is written. A funding amount, country, instrument type, or donor
name should appear as a structured record with evidence before it is used in a
profile, bulletin, or case-study narrative.
"""

from __future__ import annotations

import re
from collections import Counter

from devfinintel.models import EvidenceItem, ExtractionRecord
from devfinintel.schemas import validate_records
from devfinintel.utils import sentence_split, stable_id


AFRICAN_COUNTRIES = {
    "Algeria",
    "Angola",
    "Benin",
    "Botswana",
    "Burkina Faso",
    "Burundi",
    "Cameroon",
    "Cabo Verde",
    "Cape Verde",
    "Central African Republic",
    "Chad",
    "Comoros",
    "Congo",
    "Cote d'Ivoire",
    "Democratic Republic of the Congo",
    "Djibouti",
    "Egypt",
    "Equatorial Guinea",
    "Eritrea",
    "Eswatini",
    "Ethiopia",
    "Gabon",
    "Gambia",
    "Ghana",
    "Guinea",
    "Guinea-Bissau",
    "Kenya",
    "Lesotho",
    "Liberia",
    "Libya",
    "Madagascar",
    "Malawi",
    "Mali",
    "Mauritania",
    "Mauritius",
    "Morocco",
    "Mozambique",
    "Namibia",
    "Niger",
    "Nigeria",
    "Rwanda",
    "Sao Tome and Principe",
    "Senegal",
    "Seychelles",
    "Sierra Leone",
    "Somalia",
    "South Africa",
    "South Sudan",
    "Sudan",
    "Tanzania",
    "Togo",
    "Tunisia",
    "Uganda",
    "Zambia",
    "Zimbabwe",
}

FINANCE_INSTRUMENTS = {
    "green bond": "Green bond",
    "blue bond": "Blue bond",
    "sustainability bond": "Sustainability bond",
    "debt-for-nature": "Debt-for-nature swap",
    "debt for nature": "Debt-for-nature swap",
    "payment for ecosystem": "Payments for ecosystem services",
    "pes": "Payments for ecosystem services",
    "biodiversity offset": "Biodiversity offset",
    "subsidy reform": "Subsidy reform",
    "trust fund": "Conservation trust fund",
    "blended finance": "Blended finance",
    "guarantee": "Guarantee",
    "grant": "Grant",
    "loan": "Loan",
    "insurance": "Insurance",
    "tax": "Tax or fee",
    "fee": "Tax or fee",
}

THEME_KEYWORDS = {
    "private sector": "Private sector financing",
    "diaspora": "Diaspora engagement",
    "civil society": "Civil society engagement",
    "cso": "Civil society engagement",
    "climate": "Climate finance",
    "energy": "Energy access",
    "governance": "Governance",
    "poverty": "Poverty reduction",
    "biodiversity": "Biodiversity finance",
    "nature": "Nature-positive development",
    "conservation": "Conservation finance",
}


class StructuredExtractor:
    """Extract task-specific records from retrieved evidence."""

    MONEY_RE = re.compile(
        r"\b(?:US\$|\$|USD|EUR|€|GBP|£)\s?\d[\d,]*(?:\.\d+)?\s?(?:million|billion|m|bn)?"
        r"|\b\d[\d,]*(?:\.\d+)?\s?(?:million|billion|m|bn)\s?(?:USD|EUR|dollars|euros)?",
        flags=re.IGNORECASE,
    )
    YEAR_RE = re.compile(r"\b20\d{2}\b")

    def extract(self, task_type: str, query: str, evidence: list[EvidenceItem]) -> list[ExtractionRecord]:
        if task_type == "donor_profile":
            return validate_records(self._extract_donor_records(query, evidence))
        if task_type == "biofin_case":
            return validate_records(self._extract_biofin_records(query, evidence))
        if task_type == "bulletin":
            return validate_records(self._extract_bulletin_records(query, evidence))
        return validate_records(self._extract_qa_records(query, evidence))

    def _extract_donor_records(
        self, query: str, evidence: list[EvidenceItem]
    ) -> list[ExtractionRecord]:
        records: list[ExtractionRecord] = []
        for item in evidence:
            money_values = unique_preserve_order(self.MONEY_RE.findall(item.text))
            years = unique_preserve_order(self.YEAR_RE.findall(item.text))
            countries = detect_countries(item.text)
            themes = detect_themes(item.text)
            confidence = confidence_from_fields(
                has_money=bool(money_values),
                has_year=bool(years),
                has_place=bool(countries),
                has_theme=bool(themes),
                retrieval_score=item.rerank_score,
            )
            records.append(
                ExtractionRecord(
                    record_id=stable_id("donor", query, item.chunk_id),
                    record_type="donor_profile_evidence",
                    title=f"Donor evidence from {item.title}, p. {item.page_number}",
                    fields={
                        "partner_or_query": query,
                        "financial_figures": money_values,
                        "years": years,
                        "countries_or_regions": countries or ["Africa"] if "africa" in item.text.lower() else countries,
                        "themes": themes,
                        "citation": item.citation,
                    },
                    evidence_chunk_ids=[item.chunk_id],
                    confidence=confidence,
                    review_status="review" if confidence < 0.55 else "usable",
                )
            )
        return records

    def _extract_biofin_records(
        self, query: str, evidence: list[EvidenceItem]
    ) -> list[ExtractionRecord]:
        records: list[ExtractionRecord] = []
        for item in evidence:
            instruments = detect_finance_instruments(item.text)
            countries = detect_countries(item.text)
            money_values = unique_preserve_order(self.MONEY_RE.findall(item.text))
            lessons = first_relevant_sentences(
                item.text,
                keywords=("lesson", "learned", "enabled", "mobilized", "conservation", "biodiversity"),
                limit=2,
            )
            confidence = confidence_from_fields(
                has_money=bool(money_values),
                has_year=bool(self.YEAR_RE.findall(item.text)),
                has_place=bool(countries),
                has_theme=bool(instruments or lessons),
                retrieval_score=item.rerank_score,
            )
            records.append(
                ExtractionRecord(
                    record_id=stable_id("biofin", query, item.chunk_id),
                    record_type="biofin_case_evidence",
                    title=f"BIOFIN case evidence from {item.title}, p. {item.page_number}",
                    fields={
                        "country": countries,
                        "finance_solution": instruments,
                        "instrument_type": instruments,
                        "public_or_private_finance": classify_public_private(item.text),
                        "sector": detect_sector(item.text),
                        "funding_amount": money_values,
                        "lessons_learned": lessons,
                        "citation": item.citation,
                    },
                    evidence_chunk_ids=[item.chunk_id],
                    confidence=confidence,
                    review_status="review" if confidence < 0.55 else "usable",
                )
            )
        return records

    def _extract_bulletin_records(
        self, query: str, evidence: list[EvidenceItem]
    ) -> list[ExtractionRecord]:
        records: list[ExtractionRecord] = []
        for item in evidence:
            years = unique_preserve_order(self.YEAR_RE.findall(item.text))
            countries = detect_countries(item.text)
            themes = detect_themes(item.text)
            summary = first_relevant_sentences(
                item.text,
                keywords=("announced", "launched", "signed", "partnership", "financing", "support"),
                limit=2,
            )
            confidence = confidence_from_fields(
                has_money=bool(self.MONEY_RE.findall(item.text)),
                has_year=bool(years),
                has_place=bool(countries),
                has_theme=bool(themes),
                retrieval_score=item.rerank_score,
            )
            records.append(
                ExtractionRecord(
                    record_id=stable_id("bulletin", query, item.chunk_id),
                    record_type="bulletin_item",
                    title=f"Bulletin item from {item.title}, p. {item.page_number}",
                    fields={
                        "headline_or_query": query,
                        "countries_or_regions": countries or ["Africa"] if "africa" in item.text.lower() else countries,
                        "years": years,
                        "themes": themes,
                        "summary": summary,
                        "citation": item.citation,
                    },
                    evidence_chunk_ids=[item.chunk_id],
                    confidence=confidence,
                    review_status="review" if confidence < 0.55 else "usable",
                )
            )
        return records

    def _extract_qa_records(self, query: str, evidence: list[EvidenceItem]) -> list[ExtractionRecord]:
        records: list[ExtractionRecord] = []
        for item in evidence:
            records.append(
                ExtractionRecord(
                    record_id=stable_id("qa", query, item.chunk_id),
                    record_type="qa_evidence",
                    title=f"Evidence from {item.title}, p. {item.page_number}",
                    fields={
                        "query": query,
                        "key_sentences": first_relevant_sentences(item.text, tuple(query.split()), limit=2),
                        "citation": item.citation,
                    },
                    evidence_chunk_ids=[item.chunk_id],
                    confidence=min(0.95, max(0.25, item.rerank_score)),
                    review_status="usable",
                )
            )
        return records


class EvidenceBoundDrafter:
    """Draft outputs only from retrieved evidence and structured records.

    This class is the safe baseline for "LLM drafting." It behaves like a very
    conservative drafting assistant: it does not invent missing details, and it
    includes citations beside claims. A future LLM adapter can replace this class
    as long as it follows the same evidence-bound contract.
    """

    def draft(
        self,
        task_type: str,
        query: str,
        evidence: list[EvidenceItem],
        records: list[ExtractionRecord],
    ) -> tuple[str, str]:
        if task_type == "donor_profile":
            return self._draft_donor_profile(query, evidence, records)
        if task_type == "biofin_case":
            return self._draft_biofin_case(query, evidence, records)
        if task_type == "bulletin":
            return self._draft_bulletin(query, evidence, records)
        return self._draft_qa(query, evidence, records)

    def _draft_donor_profile(
        self, query: str, evidence: list[EvidenceItem], records: list[ExtractionRecord]
    ) -> tuple[str, str]:
        title = f"Partnership Profile Evidence Brief: {query}"
        money = collect_field_values(records, "financial_figures")
        years = collect_field_values(records, "years")
        countries = collect_field_values(records, "countries_or_regions")
        themes = collect_field_values(records, "themes")

        lines = [
            f"# {title}",
            "",
            "## Evidence-Based Snapshot",
            f"- Profile focus: {query}.",
            f"- Retrieved evidence items reviewed: {len(evidence)}.",
            f"- Financial figures found: {format_list(money) if money else 'No explicit financial figure found in retrieved evidence.'}",
            f"- Years found: {format_list(years) if years else 'No explicit year found in retrieved evidence.'}",
            f"- Countries or regions found: {format_list(countries) if countries else 'No specific country or region found in retrieved evidence.'}",
            f"- Strategic themes found: {format_list(themes) if themes else 'No strategic theme confidently detected.'}",
            "",
            "## Profile Notes",
        ]
        lines.extend(cited_bullets(evidence, max_items=5))
        lines.extend(
            [
                "",
                "## Review Flags",
                "- Confirm all financial figures against the cited source pages before external use.",
                "- Treat this as a first-pass evidence brief, not an official UNDP position.",
            ]
        )
        return title, "\n".join(lines)

    def _draft_biofin_case(
        self, query: str, evidence: list[EvidenceItem], records: list[ExtractionRecord]
    ) -> tuple[str, str]:
        title = f"BIOFIN Nature-Finance Case Evidence Brief: {query}"
        countries = collect_field_values(records, "country")
        instruments = collect_field_values(records, "finance_solution")
        funding = collect_field_values(records, "funding_amount")
        lessons = collect_field_values(records, "lessons_learned")

        lines = [
            f"# {title}",
            "",
            "## Case-Study Fields",
            f"- Countries found: {format_list(countries) if countries else 'No country confidently detected.'}",
            f"- Finance solutions or instruments found: {format_list(instruments) if instruments else 'No instrument confidently detected.'}",
            f"- Funding amounts found: {format_list(funding) if funding else 'No explicit funding amount found in retrieved evidence.'}",
            f"- Lessons learned candidates: {format_list(lessons, limit=5) if lessons else 'No lesson sentence confidently detected.'}",
            "",
            "## Evidence Notes",
        ]
        lines.extend(cited_bullets(evidence, max_items=6))
        lines.extend(
            [
                "",
                "## Review Flags",
                "- Confirm whether each instrument matches BIOFIN's official finance-solution taxonomy.",
                "- Use human review before adding any record to a formal finance-source database.",
            ]
        )
        return title, "\n".join(lines)

    def _draft_bulletin(
        self, query: str, evidence: list[EvidenceItem], records: list[ExtractionRecord]
    ) -> tuple[str, str]:
        title = f"Partnership Bulletin Evidence Brief: {query}"
        themes = collect_field_values(records, "themes")
        countries = collect_field_values(records, "countries_or_regions")
        summaries = collect_field_values(records, "summary")

        lines = [
            f"# {title}",
            "",
            "## Bulletin Inputs",
            f"- Countries or regions found: {format_list(countries) if countries else 'No specific country or region found.'}",
            f"- Themes found: {format_list(themes) if themes else 'No theme confidently detected.'}",
            f"- Candidate bulletin lines: {format_list(summaries, limit=5) if summaries else 'No concise bulletin sentence confidently detected.'}",
            "",
            "## Source Notes",
        ]
        lines.extend(cited_bullets(evidence, max_items=6))
        lines.extend(
            [
                "",
                "## Review Flags",
                "- Confirm dates and partner names before including any item in an Executive Weekly or Monthly Bulletin.",
            ]
        )
        return title, "\n".join(lines)

    def _draft_qa(
        self, query: str, evidence: list[EvidenceItem], records: list[ExtractionRecord]
    ) -> tuple[str, str]:
        if is_overview_query_text(query):
            return self._draft_overview_qa(query, evidence)
        title = f"Evidence-Grounded Answer: {query}"
        lines = [f"# {title}", "", "## Answer From Retrieved Evidence"]
        lines.extend(cited_bullets(evidence, max_items=6))
        lines.extend(["", "## Review Flags", "- Answer is limited to retrieved local evidence."])
        return title, "\n".join(lines)

    def _draft_overview_qa(self, query: str, evidence: list[EvidenceItem]) -> tuple[str, str]:
        """Draft a more useful first-pass overview for broad document questions."""

        title = f"Evidence-Grounded Document Overview: {query}"
        overview_points = overview_bullets(evidence)
        lines = [
            f"# {title}",
            "",
            "## What The Uploaded Document Appears To Be",
        ]
        lines.extend(overview_points[:3])
        lines.extend(
            [
                "",
                "## Main Evidence Signals",
            ]
        )
        lines.extend(overview_points[3:8] or cited_bullets(evidence, max_items=5))
        lines.extend(
            [
                "",
                "## Review Flags",
                "- This is a first-pass overview from retrieved local evidence, not a full expert review of the entire document.",
                "- Use the cited pages to verify details before relying on the summary externally.",
            ]
        )
        return title, "\n".join(lines)


def detect_countries(text: str) -> list[str]:
    """Find African country names mentioned in evidence."""

    found = []
    lower_text = text.lower()
    for country in sorted(AFRICAN_COUNTRIES, key=len, reverse=True):
        if country.lower() in lower_text:
            found.append(country)
    return unique_preserve_order(found)


def detect_finance_instruments(text: str) -> list[str]:
    """Detect likely finance instruments from clear keyword matches."""

    lower_text = text.lower()
    return unique_preserve_order(
        label for keyword, label in FINANCE_INSTRUMENTS.items() if keyword in lower_text
    )


def detect_themes(text: str) -> list[str]:
    """Detect broad partnership or development themes."""

    lower_text = text.lower()
    return unique_preserve_order(label for keyword, label in THEME_KEYWORDS.items() if keyword in lower_text)


def classify_public_private(text: str) -> str:
    """Classify finance source at a high level when evidence contains clear terms."""

    lower_text = text.lower()
    has_private = any(term in lower_text for term in ("private sector", "company", "corporate", "investor"))
    has_public = any(term in lower_text for term in ("government", "public", "ministry", "donor", "oda"))
    if has_private and has_public:
        return "Blended or public-private"
    if has_private:
        return "Private"
    if has_public:
        return "Public"
    return "Not clear from retrieved evidence"


def detect_sector(text: str) -> list[str]:
    """Detect sectors relevant to BIOFIN and partnership work."""

    sector_terms = {
        "forest": "Forests",
        "marine": "Marine and coastal",
        "water": "Water",
        "agriculture": "Agriculture",
        "energy": "Energy",
        "tourism": "Tourism",
        "protected area": "Protected areas",
        "wildlife": "Wildlife",
        "climate": "Climate",
    }
    lower_text = text.lower()
    return unique_preserve_order(label for keyword, label in sector_terms.items() if keyword in lower_text)


def first_relevant_sentences(text: str, keywords: tuple[str, ...], limit: int = 2) -> list[str]:
    """Return short sentences that contain task-relevant keywords."""

    normalized_keywords = [keyword.lower() for keyword in keywords if len(keyword) > 2]
    selected: list[str] = []
    for sentence in sentence_split(text):
        if not is_useful_evidence_sentence(sentence):
            continue
        lower_sentence = sentence.lower()
        if any(keyword in lower_sentence for keyword in normalized_keywords):
            selected.append(sentence[:320])
        if len(selected) >= limit:
            break
    if selected:
        return selected
    fallback = [sentence[:320] for sentence in sentence_split(text) if is_useful_evidence_sentence(sentence)]
    return fallback[:limit]


def is_useful_evidence_sentence(sentence: str) -> bool:
    """Filter out headings and local sample disclaimers from evidence summaries."""

    stripped = sentence.strip()
    lower = stripped.lower()
    if not stripped or stripped.startswith("#"):
        return False
    if "synthetic demonstration data" in lower or "not an official" in lower:
        return False
    if len(stripped) < 35:
        return False
    return True


def confidence_from_fields(
    *,
    has_money: bool,
    has_year: bool,
    has_place: bool,
    has_theme: bool,
    retrieval_score: float,
) -> float:
    """Create an interpretable confidence score for review triage."""

    score = 0.2 + min(retrieval_score, 0.5)
    score += 0.1 if has_money else 0.0
    score += 0.08 if has_year else 0.0
    score += 0.08 if has_place else 0.0
    score += 0.08 if has_theme else 0.0
    return round(min(score, 0.95), 3)


def unique_preserve_order(values) -> list[str]:
    """Return unique non-empty strings while keeping the original order."""

    seen = set()
    result = []
    for value in values:
        if not value:
            continue
        clean = str(value).strip()
        key = clean.lower()
        if key not in seen:
            seen.add(key)
            result.append(clean)
    return result


def collect_field_values(records: list[ExtractionRecord], field_name: str) -> list[str]:
    """Flatten values from a structured field across records."""

    values: list[str] = []
    for record in records:
        value = record.fields.get(field_name)
        if isinstance(value, list):
            values.extend(str(item) for item in value if item)
        elif value:
            values.append(str(value))
    return unique_preserve_order(values)


def format_list(values: list[str], limit: int = 12) -> str:
    """Format a short list for human-readable Markdown."""

    if not values:
        return ""
    shown = values[:limit]
    suffix = f" and {len(values) - limit} more" if len(values) > limit else ""
    return ", ".join(shown) + suffix


def cited_bullets(evidence: list[EvidenceItem], max_items: int) -> list[str]:
    """Create citation-bearing bullets from retrieved evidence."""

    bullets: list[str] = []
    for item in evidence[:max_items]:
        sentences = first_relevant_sentences(item.text, keywords=tuple(item.text.split()[:8]), limit=1)
        sentence = sentences[0] if sentences else item.text[:320]
        bullets.append(f"- {sentence} ({item.citation})")
    if not bullets:
        bullets.append("- No evidence was retrieved for this request.")
    return bullets


def overview_bullets(evidence: list[EvidenceItem]) -> list[str]:
    """Create readable bullets for broad "what is this file" questions."""

    selected: list[str] = []
    priority_keywords = (
        "rapport",
        "report",
        "synthèse",
        "summary",
        "aperçu",
        "overview",
        "recommandation",
        "recommendation",
        "conclusion",
        "résultat",
        "finding",
        "policy",
        "politique",
        "objectif",
        "purpose",
        "service",
        "ecosystem",
        "écosystème",
        "carbon",
        "carbone",
        "forest",
        "forêt",
        "bassin du congo",
    )
    for item in evidence:
        for sentence in sentence_split(clean_pdf_text(item.text)):
            if not is_useful_evidence_sentence(sentence):
                continue
            lower = sentence.lower()
            if any(keyword in lower for keyword in priority_keywords):
                selected.append(f"- {sentence[:420]} ({item.citation})")
                break
        if len(selected) >= 8:
            break

    if selected:
        return selected
    return cited_bullets(evidence, max_items=6)


def clean_pdf_text(text: str) -> str:
    """Reduce common PDF extraction spacing artifacts for display."""

    compact = re.sub(r"\s+", " ", text).strip()
    compact = re.sub(r"\b([A-ZÀ-Ý])\s+(?=[A-ZÀ-Ý]\b)", r"\1", compact)
    compact = compact.replace(" A o û t ", " Août ")
    return compact


def is_overview_query_text(query: str) -> bool:
    """Return True for broad requests asking what a file is about."""

    normalized = query.lower().strip()
    overview_phrases = (
        "what does the uploaded file tell me",
        "what does this file tell me",
        "what is this file about",
        "what is this document about",
        "summarize this",
        "summarise this",
        "summarize the document",
        "summarise the document",
        "tell me about the file",
        "tell me about this document",
    )
    return any(phrase in normalized for phrase in overview_phrases)


def most_common(values: list[str], limit: int = 5) -> list[str]:
    """Return the most common values for dashboard summaries."""

    return [value for value, _ in Counter(values).most_common(limit)]
