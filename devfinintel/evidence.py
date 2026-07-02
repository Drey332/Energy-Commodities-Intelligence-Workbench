"""Evidence-pack construction and prompt rendering.

The workbench uses retrieval-augmented generation, but it keeps the retrieved
context explicit. An evidence pack is the small, reviewable bundle that a draft,
local LLM answer, or export is allowed to use. This mirrors modern RAG guidance:
retrieve narrowly, label sources, keep diagnostics, and make unsupported answers
abstain instead of guessing.
"""

from __future__ import annotations

import json
from typing import Any

from devfinintel.models import EvidenceItem, EvidencePack, ExtractionRecord
from devfinintel.utils import stable_id, utc_now_iso


class EvidencePackBuilder:
    """Create bounded evidence packs for a single analysis run."""

    def __init__(self, context_budget_chars: int = 12000) -> None:
        self.context_budget_chars = context_budget_chars

    def build(
        self,
        *,
        task_type: str,
        query: str,
        evidence_items: list[EvidenceItem],
        records: list[ExtractionRecord],
        diagnostics: dict[str, Any] | None = None,
    ) -> EvidencePack:
        """Return an auditable pack with stable labels like E1 and R1.

        The context budget is a practical safety guard. Long-context models can
        still miss facts buried in the middle of large prompts, so the app keeps
        the evidence bundle compact and tied to the top-ranked retrieved chunks.
        """

        limited_evidence = fit_evidence_to_budget(evidence_items, self.context_budget_chars)
        citation_map: dict[str, str] = {}
        for index, item in enumerate(limited_evidence, start=1):
            citation_map[f"E{index}"] = item.citation
        for index, record in enumerate(records, start=1):
            citation_map[f"R{index}"] = record.title

        pack_id = stable_id(
            task_type,
            query,
            ",".join(item.chunk_id for item in limited_evidence),
            ",".join(record.record_id for record in records),
            utc_now_iso(),
        )
        return EvidencePack(
            pack_id=pack_id,
            task_type=task_type,
            query=query,
            created_at=utc_now_iso(),
            evidence_items=limited_evidence,
            records=records,
            citation_map=citation_map,
            diagnostics=diagnostics or {},
            context_budget_chars=self.context_budget_chars,
            policy_notes=[
                "Drafting and local LLM answers must use only this evidence pack.",
                "Claims should cite E-labels for source text or R-labels for structured records.",
                "If the evidence pack does not answer a question, the safe behavior is abstention.",
            ],
        )


def fit_evidence_to_budget(
    evidence_items: list[EvidenceItem],
    context_budget_chars: int,
) -> list[EvidenceItem]:
    """Keep evidence within a readable prompt/export budget.

    The current strategy preserves retrieval order. A future learned planner can
    add stronger diversity selection, but this deterministic rule is easier for
    a reviewer to understand today.
    """

    if context_budget_chars <= 0:
        return evidence_items

    selected: list[EvidenceItem] = []
    used = 0
    for item in evidence_items:
        projected = used + len(item.text)
        if selected and projected > context_budget_chars:
            break
        selected.append(item)
        used = projected
    return selected


def render_evidence_pack(pack: EvidencePack, max_evidence_chars: int = 900) -> str:
    """Render an evidence pack as compact text for a local LLM prompt."""

    sections: list[str] = [
        f"Evidence pack: {pack.pack_id}",
        f"Task: {pack.task_type}",
        f"Question: {pack.query}",
    ]
    for index, item in enumerate(pack.evidence_items, start=1):
        sections.append(
            f"[E{index}] Source: {item.citation}\n"
            f"Chunk ID: {item.chunk_id}\n"
            f"Text: {item.text[:max_evidence_chars]}"
        )

    for index, record in enumerate(pack.records, start=1):
        sections.append(
            f"[R{index}] Structured record: {record.title}\n"
            f"Record ID: {record.record_id}\n"
            f"Type: {record.record_type}\n"
            f"Confidence: {record.confidence}\n"
            f"Review status: {record.review_status}\n"
            f"Fields: {compact_json(record.fields)}"
        )
    return "\n\n".join(sections)


def compact_json(value: Any, max_chars: int = 1400) -> str:
    """Serialize nested evidence fields without creating giant prompts."""

    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if len(text) > max_chars:
        return text[: max_chars - 20] + "... [truncated]"
    return text
