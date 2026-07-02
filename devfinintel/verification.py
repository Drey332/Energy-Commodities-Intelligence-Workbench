"""Verification checks for generated outputs.

These checks are not a replacement for expert review. They are guardrails that
make common failure modes visible: missing citations, unsupported numbers, and
claims that do not share enough language with the retrieved evidence.
"""

from __future__ import annotations

import re

from devfinintel.models import EvidenceItem, EvidencePack, GeneratedOutput, VerificationFinding
from devfinintel.utils import normalize_text, tokenize


class OutputVerifier:
    """Run transparent verification checks on a generated output."""

    NUMBER_RE = re.compile(
        r"\b(?:US\$|\$|USD|EUR|€|GBP|£)\s?\d[\d,]*(?:\.\d+)?\s?(?:million|billion|m|bn)?"
        r"|\b20\d{2}\b"
        r"|\b\d[\d,]*(?:\.\d+)?\s?(?:million|billion|m|bn)\b",
        flags=re.IGNORECASE,
    )

    def verify(
        self,
        task_type: str,
        title: str,
        body_markdown: str,
        evidence_items: list[EvidenceItem],
        records,
        evidence_pack: EvidencePack | None = None,
    ) -> GeneratedOutput:
        evidence_text = "\n".join(item.text for item in evidence_items)
        findings: list[VerificationFinding] = []

        citation_coverage = self._citation_coverage(body_markdown)
        if citation_coverage < 0.45:
            findings.append(
                VerificationFinding(
                    level="warning",
                    message=(
                        "Citation coverage is low. Add more page-level citations before using this output externally."
                    ),
                )
            )

        citation_precision = self._citation_precision(body_markdown, evidence_items)
        if citation_precision < 0.9 and evidence_items:
            findings.append(
                VerificationFinding(
                    level="warning",
                    message=(
                        "Some citations in the draft do not match the retrieved evidence pack. "
                        "Check citation titles and page numbers before use."
                    ),
                )
            )

        unsupported_numbers = self._unsupported_numbers(body_markdown, evidence_text)
        for number in unsupported_numbers:
            findings.append(
                VerificationFinding(
                    level="warning",
                    message=f"Number appears in the draft but was not found verbatim in retrieved evidence: {number}",
                )
            )

        support_overlap = self._support_overlap(body_markdown, evidence_text)
        if support_overlap < 0.18 and evidence_items:
            findings.append(
                VerificationFinding(
                    level="warning",
                    message=(
                        "The draft and evidence have low lexical overlap. A reviewer should inspect whether the answer is fully supported."
                    ),
                )
            )

        claim_support_rate = self._claim_support_rate(body_markdown, evidence_items)
        if claim_support_rate < 0.7 and evidence_items:
            findings.append(
                VerificationFinding(
                    level="warning",
                    message=(
                        "Several factual lines have weak direct support from their cited evidence. "
                        "Review line-by-line before external use."
                    ),
                )
            )

        record_traceability = self._record_traceability(records, evidence_items)
        if record_traceability < 1.0 and records:
            findings.append(
                VerificationFinding(
                    level="warning",
                    message="Some structured records reference evidence chunks that are not in the evidence pack.",
                )
            )

        invalid_schema_records = [
            record.record_id
            for record in records
            if record.fields.get("_schema_valid") is False
        ]
        if invalid_schema_records:
            findings.append(
                VerificationFinding(
                    level="warning",
                    message=f"{len(invalid_schema_records)} structured record(s) failed schema validation.",
                )
            )

        if not evidence_items:
            findings.append(
                VerificationFinding(
                    level="error",
                    message="No evidence was retrieved, so the output should not be used as a factual brief.",
                )
            )

        if not findings:
            findings.append(
                VerificationFinding(
                    level="pass",
                    message="Baseline checks passed: citations, numbers, and evidence overlap look acceptable.",
                )
            )

        metrics = {
            "citation_coverage": round(citation_coverage, 3),
            "citation_precision": round(citation_precision, 3),
            "support_overlap": round(support_overlap, 3),
            "claim_support_rate": round(claim_support_rate, 3),
            "record_traceability": round(record_traceability, 3),
            "schema_invalid_records": float(len(invalid_schema_records)),
            "unsupported_number_count": float(len(unsupported_numbers)),
            "evidence_items": float(len(evidence_items)),
            "structured_records": float(len(records)),
        }

        return GeneratedOutput(
            task_type=task_type,
            title=title,
            body_markdown=body_markdown,
            evidence_items=evidence_items,
            records=records,
            verification_findings=findings,
            metrics=metrics,
            evidence_pack=evidence_pack,
        )

    def _citation_coverage(self, markdown: str) -> float:
        """Estimate how many factual lines carry page-level citations."""

        factual_lines = [
            line.strip()
            for line in markdown.splitlines()
            if line.strip()
            and not line.startswith("#")
            and not line.lower().startswith("## review")
            and len(line.strip()) > 25
        ]
        if not factual_lines:
            return 0.0
        cited = [line for line in factual_lines if "p. " in line or "p." in line]
        return len(cited) / len(factual_lines)

    def _citation_precision(self, markdown: str, evidence_items: list[EvidenceItem]) -> float:
        """Check whether draft citation strings exist in the evidence pack."""

        citations = set(re.findall(r"\(([^()]+,\s*p\.?\s*\d+)\)", markdown, flags=re.IGNORECASE))
        if not citations:
            return 0.0 if evidence_items else 1.0
        valid = {normalize_citation(item.citation) for item in evidence_items}
        matched = sum(1 for citation in citations if normalize_citation(citation) in valid)
        return matched / len(citations)

    def _unsupported_numbers(self, markdown: str, evidence_text: str) -> list[str]:
        """Find numbers in the draft that are not present in retrieved evidence."""

        evidence_normalized = normalize_text(evidence_text)
        unsupported = []
        for match in self.NUMBER_RE.findall(markdown):
            if normalize_text(match) not in evidence_normalized:
                unsupported.append(match)
        return sorted(set(unsupported))

    def _support_overlap(self, markdown: str, evidence_text: str) -> float:
        """A simple proxy for faithfulness based on shared content words."""

        draft_terms = {
            term
            for term in tokenize(markdown)
            if len(term) > 4 and term not in {"review", "evidence", "retrieved", "source"}
        }
        evidence_terms = set(tokenize(evidence_text))
        if not draft_terms:
            return 0.0
        return len(draft_terms & evidence_terms) / len(draft_terms)

    def _claim_support_rate(self, markdown: str, evidence_items: list[EvidenceItem]) -> float:
        """Estimate how many factual lines are supported by their cited evidence.

        This is a transparent lexical proxy, not a neural fact checker. It is
        useful because reviewers can see exactly which lines are likely weak.
        """

        factual_lines = [
            line.strip("- ").strip()
            for line in markdown.splitlines()
            if line.strip()
            and not line.startswith("#")
            and not line.lower().startswith("## review")
            and len(line.strip()) > 35
        ]
        if not factual_lines:
            return 1.0

        citation_lookup = {normalize_citation(item.citation): item.text for item in evidence_items}
        supported = 0
        for line in factual_lines:
            cited = re.findall(r"\(([^()]+,\s*p\.?\s*\d+)\)", line, flags=re.IGNORECASE)
            if cited:
                evidence_text = " ".join(
                    citation_lookup.get(normalize_citation(citation), "")
                    for citation in cited
                )
            else:
                evidence_text = " ".join(item.text for item in evidence_items)
            if line_support_overlap(line, evidence_text) >= 0.18:
                supported += 1
        return supported / len(factual_lines)

    def _record_traceability(self, records, evidence_items: list[EvidenceItem]) -> float:
        """Check that structured records point back to retrieved evidence chunks."""

        if not records:
            return 1.0
        evidence_chunk_ids = {item.chunk_id for item in evidence_items}
        traceable = 0
        for record in records:
            if record.evidence_chunk_ids and all(
                chunk_id in evidence_chunk_ids for chunk_id in record.evidence_chunk_ids
            ):
                traceable += 1
        return traceable / len(records)


def normalize_citation(citation: str) -> str:
    """Normalize citation labels such as ``Report, p. 3`` for matching."""

    return re.sub(r"\s+", " ", citation.replace("p.", "p.")).strip().lower()


def line_support_overlap(line: str, evidence_text: str) -> float:
    """Return content-word overlap between one draft line and evidence text."""

    line_terms = {
        term
        for term in tokenize(line)
        if len(term) > 4 and term not in {"review", "evidence", "retrieved", "source"}
    }
    if not line_terms:
        return 1.0
    evidence_terms = set(tokenize(evidence_text))
    return len(line_terms & evidence_terms) / len(line_terms)
