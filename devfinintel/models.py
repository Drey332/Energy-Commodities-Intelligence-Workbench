"""Dataclasses used across the document intelligence pipeline.

Dataclasses keep the code simple: each object is just named fields, similar to a
spreadsheet row. That makes the data flow easier to inspect than passing around
unlabeled dictionaries everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SourceDocument:
    """A source file that has been loaded into the evidence database."""

    document_id: str
    title: str
    source_path: str
    source_type: str
    language_hint: str
    parser_backend: str
    loaded_at: str


@dataclass(frozen=True)
class FileManifest:
    """Audit manifest for one uploaded or ingested file.

    A manifest is the evidence-store contract. It tells a reviewer what file was
    loaded, how it was parsed, how large it was, and how many pages/rows/chunks
    were made searchable.
    """

    document_id: str
    title: str
    source_path: str
    source_type: str
    parser_backend: str
    file_sha256: str
    file_size_bytes: int
    language_hint: str
    page_count: int
    row_count: int | None
    column_count: int | None
    chunk_count: int
    loaded_at: str


@dataclass(frozen=True)
class AnalysisSession:
    """A single user-facing analysis run.

    Sessions prevent fragmented architecture. The UI can show one coherent run:
    which files were in scope, which work product was requested, what output was
    produced, and what diagnostics were recorded.
    """

    session_id: str
    title: str
    task_type: str
    query: str
    document_ids: list[str]
    scope_label: str
    created_at: str
    status: str
    output_id: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionItem:
    """One recommended analyst follow-up derived from evidence records.

    This is the bridge from knowledge storage to applied intelligence. It keeps
    recommendations reviewable and status-tracked instead of leaving them as
    prose hidden inside a generated brief.
    """

    action_id: str
    country: str
    action_type: str
    priority: str
    status: str
    title: str
    rationale: str
    source_record_id: str
    source_title: str
    source_page: int | None
    source_path: str
    due_bucket: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class MonitoringSource:
    """A governed source that can feed the monitoring intelligence layer."""

    source_id: str
    name: str
    publisher: str
    url: str
    source_type: str
    scope: str
    topics: str
    countries: str
    credibility_tier: str
    refresh_cadence: str
    status: str
    last_checked_at: str
    notes: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class MonitoringEvent:
    """One dated news, report, or monitoring signal for analyst review."""

    event_id: str
    source_id: str
    source_name: str
    source_url: str
    source_category: str
    title: str
    url: str
    published_at: str
    collected_at: str
    country: str
    region: str
    sector: str
    commodity: str
    actors: str
    event_type: str
    outcome: str
    sentiment_tone: str
    risk_flags: str
    relevance: str
    confidence: float
    summary: str
    recommended_action: str
    source_record_id: str
    status: str
    raw_text: str


@dataclass(frozen=True)
class DocumentPage:
    """Text extracted from one page or one logical section of a source file."""

    document_id: str
    page_number: int
    text: str


@dataclass(frozen=True)
class DocumentChunk:
    """A searchable evidence chunk.

    Chunks are small enough to retrieve precisely, but still carry the page
    number needed for citations.
    """

    chunk_id: str
    document_id: str
    page_number: int
    chunk_index: int
    text: str
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceItem:
    """One retrieved evidence item with transparent scoring details."""

    chunk_id: str
    document_id: str
    title: str
    source_path: str
    page_number: int
    text: str
    bm25_score: float
    dense_score: float
    rerank_score: float

    @property
    def citation(self) -> str:
        """Short citation label used in generated text."""

        return f"{self.title}, p. {self.page_number}"


@dataclass(frozen=True)
class RetrievalDiagnostics:
    """Transparent pass/fail diagnostics for retrieval quality."""

    query: str
    task_type: str
    document_scope_count: int
    chunk_count: int
    returned_count: int
    top_score: float
    average_score: float
    keyword_coverage: float
    passed: bool
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-serializable diagnostics for metrics and audit logs."""

        return {
            "query": self.query,
            "task_type": self.task_type,
            "document_scope_count": self.document_scope_count,
            "chunk_count": self.chunk_count,
            "returned_count": self.returned_count,
            "top_score": self.top_score,
            "average_score": self.average_score,
            "keyword_coverage": self.keyword_coverage,
            "passed": self.passed,
            "warnings": self.warnings,
        }


@dataclass(frozen=True)
class ExtractionRecord:
    """Structured fact extracted from evidence.

    The ``fields`` dictionary stores task-specific values such as funding
    amount, country, instrument type, or partner name. The evidence IDs preserve
    traceability back to retrieved chunks.
    """

    record_id: str
    record_type: str
    title: str
    fields: dict[str, Any]
    evidence_chunk_ids: list[str]
    confidence: float
    review_status: str


@dataclass(frozen=True)
class VerificationFinding:
    """A verification result that can be shown to a human reviewer."""

    level: str
    message: str
    evidence_reference: str = ""


@dataclass(frozen=True)
class EvidencePack:
    """The bounded evidence bundle used for drafting, LLM answers, and export.

    A RAG system is safer when the answer is tied to a specific evidence pack
    instead of the whole document library. This object records the exact chunks,
    structured records, diagnostics, and citation labels available to a run.
    """

    pack_id: str
    task_type: str
    query: str
    created_at: str
    evidence_items: list[EvidenceItem]
    records: list[ExtractionRecord]
    citation_map: dict[str, str]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    context_budget_chars: int = 12000
    policy_notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation for audit exports."""

        return {
            "pack_id": self.pack_id,
            "task_type": self.task_type,
            "query": self.query,
            "created_at": self.created_at,
            "citation_map": self.citation_map,
            "diagnostics": self.diagnostics,
            "context_budget_chars": self.context_budget_chars,
            "policy_notes": self.policy_notes,
            "evidence_items": [
                {
                    "label": f"E{index}",
                    "chunk_id": item.chunk_id,
                    "document_id": item.document_id,
                    "title": item.title,
                    "source_path": item.source_path,
                    "page_number": item.page_number,
                    "citation": item.citation,
                    "bm25_score": item.bm25_score,
                    "dense_score": item.dense_score,
                    "rerank_score": item.rerank_score,
                    "text": item.text,
                }
                for index, item in enumerate(self.evidence_items, start=1)
            ],
            "records": [
                {
                    "label": f"R{index}",
                    "record_id": record.record_id,
                    "record_type": record.record_type,
                    "title": record.title,
                    "fields": record.fields,
                    "evidence_chunk_ids": record.evidence_chunk_ids,
                    "confidence": record.confidence,
                    "review_status": record.review_status,
                }
                for index, record in enumerate(self.records, start=1)
            ],
        }


@dataclass(frozen=True)
class GeneratedOutput:
    """A complete generated answer and the evidence used to produce it."""

    task_type: str
    title: str
    body_markdown: str
    evidence_items: list[EvidenceItem]
    records: list[ExtractionRecord]
    verification_findings: list[VerificationFinding]
    metrics: dict[str, Any]
    evidence_pack: EvidencePack | None = None
