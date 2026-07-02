"""SQLite document store.

SQLite is used as the baseline canonical store because it ships with Python and
can be inspected with many common tools. This is useful in a UN setting: a
reviewer can audit records, citations, and generated outputs without operating a
database server.
"""

from __future__ import annotations

import json
import sqlite3
import csv
from pathlib import Path
from typing import Any

from devfinintel.models import (
    ActionItem,
    AnalysisSession,
    DocumentChunk,
    DocumentPage,
    ExtractionRecord,
    FileManifest,
    MonitoringEvent,
    MonitoringSource,
    SourceDocument,
)
from devfinintel.knowledge import KnowledgeRecord, enrich_stored_knowledge_record
from devfinintel.utils import file_sha256, stable_id, utc_now_iso


def _safe_int(value: Any) -> int | None:
    """Best-effort integer conversion for local data repairs."""

    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float:
    """Best-effort float conversion for local data repairs."""

    try:
        return float(str(value))
    except (TypeError, ValueError):
        return 0.0


class SQLiteDocumentStore:
    """A small, auditable database for source documents and generated records."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_schema(self) -> None:
        """Create tables if they do not already exist."""

        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    language_hint TEXT NOT NULL,
                    parser_backend TEXT NOT NULL,
                    loaded_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS pages (
                    document_id TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    PRIMARY KEY (document_id, page_number),
                    FOREIGN KEY (document_id) REFERENCES documents(document_id)
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES documents(document_id)
                );

                CREATE TABLE IF NOT EXISTS extraction_records (
                    record_id TEXT PRIMARY KEY,
                    record_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    fields_json TEXT NOT NULL,
                    evidence_chunk_ids_json TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    review_status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS knowledge_records (
                    record_id TEXT PRIMARY KEY,
                    record_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    country TEXT NOT NULL,
                    region TEXT NOT NULL,
                    sector TEXT NOT NULL,
                    theme TEXT NOT NULL,
                    commodity TEXT NOT NULL,
                    partner TEXT NOT NULL,
                    amount TEXT NOT NULL,
                    currency TEXT NOT NULL,
                    instrument TEXT NOT NULL,
                    event_date TEXT NOT NULL,
                    relevance TEXT NOT NULL DEFAULT '',
                    actors TEXT NOT NULL DEFAULT '',
                    event_type TEXT NOT NULL DEFAULT '',
                    sentiment_tone TEXT NOT NULL DEFAULT '',
                    risk_flags TEXT NOT NULL DEFAULT '',
                    recommended_action TEXT NOT NULL DEFAULT '',
                    source_document_id TEXT NOT NULL,
                    source_title TEXT NOT NULL,
                    source_page INTEGER,
                    source_path TEXT NOT NULL,
                    evidence_chunk_ids_json TEXT NOT NULL,
                    fields_json TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    review_status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_knowledge_country
                ON knowledge_records(country);

                CREATE INDEX IF NOT EXISTS idx_knowledge_type_status
                ON knowledge_records(record_type, review_status);

                CREATE TABLE IF NOT EXISTS action_items (
                    action_id TEXT PRIMARY KEY,
                    country TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    status TEXT NOT NULL,
                    title TEXT NOT NULL,
                    rationale TEXT NOT NULL,
                    source_record_id TEXT NOT NULL,
                    source_title TEXT NOT NULL,
                    source_page INTEGER,
                    source_path TEXT NOT NULL,
                    due_bucket TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_action_country_status
                ON action_items(country, status);

                CREATE INDEX IF NOT EXISTS idx_action_priority_status
                ON action_items(priority, status);

                CREATE TABLE IF NOT EXISTS monitoring_sources (
                    source_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    publisher TEXT NOT NULL,
                    url TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    topics TEXT NOT NULL,
                    countries TEXT NOT NULL,
                    credibility_tier TEXT NOT NULL,
                    refresh_cadence TEXT NOT NULL,
                    status TEXT NOT NULL,
                    last_checked_at TEXT NOT NULL,
                    notes TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS monitoring_events (
                    event_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    source_category TEXT NOT NULL,
                    title TEXT NOT NULL,
                    url TEXT NOT NULL,
                    published_at TEXT NOT NULL,
                    collected_at TEXT NOT NULL,
                    country TEXT NOT NULL,
                    region TEXT NOT NULL,
                    sector TEXT NOT NULL,
                    commodity TEXT NOT NULL,
                    actors TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    sentiment_tone TEXT NOT NULL,
                    risk_flags TEXT NOT NULL,
                    relevance TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    summary TEXT NOT NULL,
                    recommended_action TEXT NOT NULL,
                    source_record_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    raw_text TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_monitoring_country
                ON monitoring_events(country);

                CREATE INDEX IF NOT EXISTS idx_monitoring_published
                ON monitoring_events(published_at);

                CREATE INDEX IF NOT EXISTS idx_monitoring_status
                ON monitoring_events(status);

                CREATE TABLE IF NOT EXISTS monitoring_runs (
                    monitoring_run_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    query TEXT NOT NULL,
                    filters_json TEXT NOT NULL,
                    sources_attempted_json TEXT NOT NULL,
                    source_statuses_json TEXT NOT NULL,
                    signal_count INTEGER NOT NULL,
                    cluster_count INTEGER NOT NULL,
                    top_countries_json TEXT NOT NULL,
                    top_sectors_json TEXT NOT NULL,
                    top_commodities_json TEXT NOT NULL,
                    top_risk_flags_json TEXT NOT NULL,
                    fallback_used INTEGER NOT NULL,
                    warnings_json TEXT NOT NULL,
                    errors_json TEXT NOT NULL,
                    brief_markdown TEXT NOT NULL,
                    export_paths_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_monitoring_runs_timestamp
                ON monitoring_runs(timestamp);

                CREATE TABLE IF NOT EXISTS promoted_monitoring_items (
                    promotion_id TEXT PRIMARY KEY,
                    monitoring_run_id TEXT NOT NULL,
                    item_type TEXT NOT NULL,
                    source_item_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    country TEXT NOT NULL,
                    region TEXT NOT NULL,
                    sector TEXT NOT NULL,
                    commodity TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    tone TEXT NOT NULL,
                    risk_flags TEXT NOT NULL,
                    relevance_score REAL NOT NULL,
                    confidence_level TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    evidence_text TEXT NOT NULL,
                    supporting_signal_ids_json TEXT NOT NULL,
                    retrieved_at TEXT NOT NULL,
                    promoted_at TEXT NOT NULL,
                    analyst_note TEXT NOT NULL,
                    status TEXT NOT NULL,
                    review_flag TEXT NOT NULL,
                    review_reasons_json TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_promoted_monitoring_run
                ON promoted_monitoring_items(monitoring_run_id);

                CREATE INDEX IF NOT EXISTS idx_promoted_monitoring_status
                ON promoted_monitoring_items(status);

                CREATE TABLE IF NOT EXISTS analysis_sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    query TEXT NOT NULL,
                    document_ids_json TEXT NOT NULL,
                    scope_label TEXT NOT NULL,
                    status TEXT NOT NULL,
                    output_id TEXT NOT NULL,
                    diagnostics_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS generated_outputs (
                    output_id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    markdown_path TEXT NOT NULL,
                    csv_path TEXT NOT NULL,
                    pdf_path TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    action TEXT NOT NULL,
                    details_json TEXT NOT NULL
                );
                """
            )
            self._ensure_knowledge_columns(connection)
            self._repair_misaligned_knowledge_rows(connection)
            self._backfill_knowledge_monitoring_fields(connection)

    def _ensure_knowledge_columns(self, connection: sqlite3.Connection) -> None:
        """Add monitoring-intelligence columns to older local databases."""

        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(knowledge_records)").fetchall()
        }
        for column_name in (
            "relevance",
            "actors",
            "event_type",
            "sentiment_tone",
            "risk_flags",
            "recommended_action",
        ):
            if column_name not in columns:
                try:
                    connection.execute(
                        f"ALTER TABLE knowledge_records ADD COLUMN {column_name} TEXT NOT NULL DEFAULT ''"
                    )
                except sqlite3.OperationalError as exc:
                    # Streamlit and CLI can start at the same time during local
                    # development. If another process added the column after the
                    # PRAGMA read, this migration is already complete.
                    if "duplicate column name" not in str(exc).lower():
                        raise

    def _repair_misaligned_knowledge_rows(self, connection: sqlite3.Connection) -> None:
        """Repair rows written by positional inserts after a schema migration.

        Older databases received the monitoring columns at the end of the table.
        A positional ``INSERT INTO knowledge_records VALUES (...)`` could then
        shift values into the wrong columns. The repair is intentionally narrow:
        it only touches rows where ``source_document_id`` contains a relevance
        label and the appended ``relevance`` column contains JSON evidence IDs.
        """

        rows = connection.execute(
            """
            SELECT *
            FROM knowledge_records
            WHERE source_document_id IN ('high', 'medium', 'low')
              AND relevance LIKE '[%'
            """
        ).fetchall()
        if not rows:
            return

        repairs = []
        for row in rows:
            record = dict(row)
            source_page = _safe_int(record.get("created_at"))
            confidence = _safe_float(record.get("event_type"))
            repairs.append(
                (
                    record["source_document_id"],
                    record["source_title"],
                    str(record["source_page"] or ""),
                    str(record["source_path"] or ""),
                    str(record["evidence_chunk_ids_json"] or ""),
                    str(record["fields_json"] or ""),
                    str(record["confidence"] or ""),
                    str(record["review_status"] or ""),
                    source_page,
                    str(record["updated_at"] or ""),
                    str(record["relevance"] or "[]"),
                    str(record["actors"] or "{}"),
                    confidence,
                    str(record["sentiment_tone"] or "review"),
                    str(record["risk_flags"] or ""),
                    str(record["recommended_action"] or ""),
                    record["record_id"],
                )
            )

        connection.executemany(
            """
            UPDATE knowledge_records
            SET relevance = ?,
                actors = ?,
                event_type = ?,
                sentiment_tone = ?,
                risk_flags = ?,
                recommended_action = ?,
                source_document_id = ?,
                source_title = ?,
                source_page = ?,
                source_path = ?,
                evidence_chunk_ids_json = ?,
                fields_json = ?,
                confidence = ?,
                review_status = ?,
                created_at = ?,
                updated_at = ?
            WHERE record_id = ?
            """,
            repairs,
        )

    def _backfill_knowledge_monitoring_fields(self, connection: sqlite3.Connection) -> None:
        """Populate monitoring fields for rows created before the feature existed."""

        rows = connection.execute(
            """
            SELECT *
            FROM knowledge_records
            WHERE relevance = ''
               OR actors = ''
               OR event_type = ''
               OR sentiment_tone = ''
               OR recommended_action = ''
            """
        ).fetchall()
        if not rows:
            return

        updates = []
        for row in rows:
            record = dict(row)
            try:
                record["fields"] = json.loads(record["fields_json"])
                record["evidence_chunk_ids"] = json.loads(record["evidence_chunk_ids_json"])
            except (json.JSONDecodeError, TypeError):
                continue
            enriched = enrich_stored_knowledge_record(record)
            updates.append(
                (
                    record.get("relevance") or enriched["relevance"],
                    record.get("actors") or enriched["actors"],
                    record.get("event_type") or enriched["event_type"],
                    record.get("sentiment_tone") or enriched["sentiment_tone"],
                    record.get("risk_flags") or enriched["risk_flags"],
                    record.get("recommended_action") or enriched["recommended_action"],
                    record["record_id"],
                )
            )

        if updates:
            connection.executemany(
                """
                UPDATE knowledge_records
                SET relevance = ?,
                    actors = ?,
                    event_type = ?,
                    sentiment_tone = ?,
                    risk_flags = ?,
                    recommended_action = ?
                WHERE record_id = ?
                """,
                updates,
            )

    def save_document(
        self,
        document: SourceDocument,
        pages: list[DocumentPage],
        chunks: list[DocumentChunk],
    ) -> None:
        """Save one parsed document and replace older records with the same ID."""

        with self._connect() as connection:
            connection.execute("DELETE FROM chunks WHERE document_id = ?", (document.document_id,))
            connection.execute("DELETE FROM pages WHERE document_id = ?", (document.document_id,))
            connection.execute("DELETE FROM documents WHERE document_id = ?", (document.document_id,))
            connection.execute(
                """
                INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document.document_id,
                    document.title,
                    document.source_path,
                    document.source_type,
                    document.language_hint,
                    document.parser_backend,
                    document.loaded_at,
                ),
            )
            connection.executemany(
                "INSERT INTO pages VALUES (?, ?, ?)",
                [(page.document_id, page.page_number, page.text) for page in pages],
            )
            connection.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.page_number,
                        chunk.chunk_index,
                        chunk.text,
                        chunk.token_count,
                        json.dumps(chunk.metadata, ensure_ascii=False),
                    )
                    for chunk in chunks
                ],
            )
        self.log_event(
            action="document_ingested",
            details={
                "document_id": document.document_id,
                "title": document.title,
                "pages": len(pages),
                "chunks": len(chunks),
                "parser_backend": document.parser_backend,
            },
        )

    def list_documents(self) -> list[dict[str, Any]]:
        """Return document rows for dashboards and audit views."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT d.*, COUNT(DISTINCT p.page_number) AS pages, COUNT(DISTINCT c.chunk_id) AS chunks
                FROM documents d
                LEFT JOIN pages p ON p.document_id = d.document_id
                LEFT JOIN chunks c ON c.document_id = d.document_id
                GROUP BY d.document_id
                ORDER BY d.loaded_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def list_file_manifests(self, document_ids: list[str] | None = None) -> list[FileManifest]:
        """Return audit manifests for source files.

        The row and column counts are computed from the source CSV when possible.
        That keeps the manifest human-checkable and avoids trusting generated
        prose for basic file facts.
        """

        document_rows = self.list_documents()
        if document_ids is not None:
            allowed = set(document_ids)
            document_rows = [row for row in document_rows if row["document_id"] in allowed]

        manifests: list[FileManifest] = []
        for row in document_rows:
            source_path = Path(row["source_path"])
            row_count: int | None = None
            column_count: int | None = None
            if row["source_type"] == "csv" and source_path.exists():
                row_count, column_count = csv_shape(source_path)

            manifests.append(
                FileManifest(
                    document_id=row["document_id"],
                    title=row["title"],
                    source_path=row["source_path"],
                    source_type=row["source_type"],
                    parser_backend=row["parser_backend"],
                    file_sha256=file_sha256(source_path) if source_path.exists() else "",
                    file_size_bytes=source_path.stat().st_size if source_path.exists() else 0,
                    language_hint=row["language_hint"],
                    page_count=int(row["pages"]),
                    row_count=row_count,
                    column_count=column_count,
                    chunk_count=int(row["chunks"]),
                    loaded_at=row["loaded_at"],
                )
            )
        return manifests

    def get_all_chunks(self) -> list[DocumentChunk]:
        """Load all chunks for indexing."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT chunk_id, document_id, page_number, chunk_index, text,
                       token_count, metadata_json
                FROM chunks
                ORDER BY document_id, page_number, chunk_index
                """
            ).fetchall()
        return [
            DocumentChunk(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                page_number=row["page_number"],
                chunk_index=row["chunk_index"],
                text=row["text"],
                token_count=row["token_count"],
                metadata=json.loads(row["metadata_json"]),
            )
            for row in rows
        ]

    def get_document_lookup(self) -> dict[str, SourceDocument]:
        """Return document metadata keyed by document ID."""

        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM documents").fetchall()
        return {
            row["document_id"]: SourceDocument(
                document_id=row["document_id"],
                title=row["title"],
                source_path=row["source_path"],
                source_type=row["source_type"],
                language_hint=row["language_hint"],
                parser_backend=row["parser_backend"],
                loaded_at=row["loaded_at"],
            )
            for row in rows
        }

    def save_extraction_records(self, records: list[ExtractionRecord]) -> None:
        """Persist structured records extracted from evidence."""

        with self._connect() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO extraction_records
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        record.record_id,
                        record.record_type,
                        record.title,
                        json.dumps(record.fields, ensure_ascii=False),
                        json.dumps(record.evidence_chunk_ids, ensure_ascii=False),
                        record.confidence,
                        record.review_status,
                        utc_now_iso(),
                    )
                    for record in records
                ],
            )
        self.log_event(
            action="extraction_records_saved",
            details={"records": len(records), "record_types": sorted({r.record_type for r in records})},
        )

    def list_extraction_records(self) -> list[dict[str, Any]]:
        """Return extracted structured records for dashboard tables."""

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM extraction_records ORDER BY created_at DESC"
            ).fetchall()
        records: list[dict[str, Any]] = []
        for row in rows:
            record = dict(row)
            record["fields"] = json.loads(row["fields_json"])
            record["evidence_chunk_ids"] = json.loads(row["evidence_chunk_ids_json"])
            records.append(record)
        return records

    def save_knowledge_records(self, records: list[KnowledgeRecord]) -> None:
        """Persist reusable knowledge records for cross-run review.

        Extraction records are tied to one run. Knowledge records are the
        operational database that an intern can maintain: country, sector,
        commodity, partner, amount, source citation, confidence, and review
        status.
        """

        if not records:
            return
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO knowledge_records
                (
                    record_id,
                    record_type,
                    title,
                    country,
                    region,
                    sector,
                    theme,
                    commodity,
                    partner,
                    amount,
                    currency,
                    instrument,
                    event_date,
                    relevance,
                    actors,
                    event_type,
                    sentiment_tone,
                    risk_flags,
                    recommended_action,
                    source_document_id,
                    source_title,
                    source_page,
                    source_path,
                    evidence_chunk_ids_json,
                    fields_json,
                    confidence,
                    review_status,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        record.record_id,
                        record.record_type,
                        record.title,
                        record.country,
                        record.region,
                        record.sector,
                        record.theme,
                        record.commodity,
                        record.partner,
                        record.amount,
                        record.currency,
                        record.instrument,
                        record.event_date,
                        record.relevance,
                        record.actors,
                        record.event_type,
                        record.sentiment_tone,
                        record.risk_flags,
                        record.recommended_action,
                        record.source_document_id,
                        record.source_title,
                        record.source_page,
                        record.source_path,
                        json.dumps(record.evidence_chunk_ids, ensure_ascii=False),
                        json.dumps(record.fields, ensure_ascii=False),
                        record.confidence,
                        record.review_status,
                        record.created_at or utc_now_iso(),
                        utc_now_iso(),
                    )
                    for record in records
                ],
            )
        self.log_event(
            action="knowledge_records_saved",
            details={"records": len(records), "record_types": sorted({r.record_type for r in records})},
        )

    def list_knowledge_records(
        self,
        *,
        record_type: str | None = None,
        review_status: str | None = None,
        country: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Return reusable knowledge records for dashboards and review queues."""

        clauses: list[str] = []
        params: list[Any] = []
        if record_type:
            clauses.append("record_type = ?")
            params.append(record_type)
        if review_status:
            clauses.append("review_status = ?")
            params.append(review_status)
        if country:
            clauses.append("country = ?")
            params.append(country)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM knowledge_records
                {where}
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()

        records: list[dict[str, Any]] = []
        for row in rows:
            record = dict(row)
            record["evidence_chunk_ids"] = json.loads(row["evidence_chunk_ids_json"])
            record["fields"] = json.loads(row["fields_json"])
            records.append(record)
        return records

    def update_knowledge_record_status(self, record_id: str, review_status: str) -> None:
        """Update the review status for one knowledge record."""

        with self._connect() as connection:
            connection.execute(
                """
                UPDATE knowledge_records
                SET review_status = ?, updated_at = ?
                WHERE record_id = ?
                """,
                (review_status, utc_now_iso(), record_id),
            )
        self.log_event(
            action="knowledge_record_status_updated",
            details={"record_id": record_id, "review_status": review_status},
        )

    def sync_action_items(self, actions: list[ActionItem]) -> None:
        """Upsert derived action items while preserving human status changes."""

        if not actions:
            return
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO action_items
                (
                    action_id,
                    country,
                    action_type,
                    priority,
                    status,
                    title,
                    rationale,
                    source_record_id,
                    source_title,
                    source_page,
                    source_path,
                    due_bucket,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(action_id) DO UPDATE SET
                    country = excluded.country,
                    action_type = excluded.action_type,
                    priority = excluded.priority,
                    title = excluded.title,
                    rationale = excluded.rationale,
                    source_record_id = excluded.source_record_id,
                    source_title = excluded.source_title,
                    source_page = excluded.source_page,
                    source_path = excluded.source_path,
                    due_bucket = excluded.due_bucket,
                    updated_at = excluded.updated_at
                """,
                [
                    (
                        action.action_id,
                        action.country,
                        action.action_type,
                        action.priority,
                        action.status,
                        action.title,
                        action.rationale,
                        action.source_record_id,
                        action.source_title,
                        action.source_page,
                        action.source_path,
                        action.due_bucket,
                        action.created_at,
                        utc_now_iso(),
                    )
                    for action in actions
                ],
            )
        self.log_event(
            action="action_items_synced",
            details={"actions": len(actions), "countries": sorted({item.country for item in actions})[:50]},
        )

    def list_action_items(
        self,
        *,
        country: str | None = None,
        status: str | None = None,
        priority: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Return recommended analyst follow-up actions."""

        clauses: list[str] = []
        params: list[Any] = []
        if country:
            clauses.append("country = ?")
            params.append(country)
        if status:
            clauses.append("status = ?")
            params.append(status)
        if priority:
            clauses.append("priority = ?")
            params.append(priority)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT *
                FROM action_items
                {where}
                ORDER BY
                    CASE status
                        WHEN 'open' THEN 0
                        WHEN 'in_progress' THEN 1
                        WHEN 'blocked' THEN 2
                        WHEN 'done' THEN 3
                        WHEN 'dismissed' THEN 4
                        ELSE 9
                    END,
                    CASE priority
                        WHEN 'urgent' THEN 0
                        WHEN 'high' THEN 1
                        WHEN 'medium' THEN 2
                        WHEN 'low' THEN 3
                        ELSE 9
                    END,
                    updated_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def update_action_item_status(self, action_id: str, status: str) -> None:
        """Update the status of one analyst action."""

        with self._connect() as connection:
            connection.execute(
                """
                UPDATE action_items
                SET status = ?, updated_at = ?
                WHERE action_id = ?
                """,
                (status, utc_now_iso(), action_id),
            )
        self.log_event(
            action="action_item_status_updated",
            details={"action_id": action_id, "status": status},
        )

    def sync_monitoring_sources(self, sources: list[MonitoringSource]) -> None:
        """Upsert governed monitoring sources."""

        if not sources:
            return
        now = utc_now_iso()
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO monitoring_sources
                (
                    source_id,
                    name,
                    publisher,
                    url,
                    source_type,
                    scope,
                    topics,
                    countries,
                    credibility_tier,
                    refresh_cadence,
                    status,
                    last_checked_at,
                    notes,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    name = excluded.name,
                    publisher = excluded.publisher,
                    url = excluded.url,
                    source_type = excluded.source_type,
                    scope = excluded.scope,
                    topics = excluded.topics,
                    countries = excluded.countries,
                    credibility_tier = excluded.credibility_tier,
                    refresh_cadence = excluded.refresh_cadence,
                    status = excluded.status,
                    last_checked_at = excluded.last_checked_at,
                    notes = excluded.notes,
                    updated_at = excluded.updated_at
                """,
                [
                    (
                        source.source_id,
                        source.name,
                        source.publisher,
                        source.url,
                        source.source_type,
                        source.scope,
                        source.topics,
                        source.countries,
                        source.credibility_tier,
                        source.refresh_cadence,
                        source.status,
                        source.last_checked_at,
                        source.notes,
                        source.created_at or now,
                        now,
                    )
                    for source in sources
                ],
            )
        self.log_event(
            action="monitoring_sources_synced",
            details={"sources": len(sources), "source_ids": [source.source_id for source in sources]},
        )

    def list_monitoring_sources(self, limit: int = 500) -> list[dict[str, Any]]:
        """Return governed monitoring sources."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM monitoring_sources
                ORDER BY publisher, name
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_monitoring_events(self, events: list[MonitoringEvent]) -> None:
        """Upsert monitoring events while preserving human review status."""

        if not events:
            return
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO monitoring_events
                (
                    event_id,
                    source_id,
                    source_name,
                    source_url,
                    source_category,
                    title,
                    url,
                    published_at,
                    collected_at,
                    country,
                    region,
                    sector,
                    commodity,
                    actors,
                    event_type,
                    outcome,
                    sentiment_tone,
                    risk_flags,
                    relevance,
                    confidence,
                    summary,
                    recommended_action,
                    source_record_id,
                    status,
                    raw_text
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    source_id = excluded.source_id,
                    source_name = excluded.source_name,
                    source_url = excluded.source_url,
                    source_category = excluded.source_category,
                    title = excluded.title,
                    url = excluded.url,
                    published_at = excluded.published_at,
                    collected_at = excluded.collected_at,
                    country = excluded.country,
                    region = excluded.region,
                    sector = excluded.sector,
                    commodity = excluded.commodity,
                    actors = excluded.actors,
                    event_type = excluded.event_type,
                    outcome = excluded.outcome,
                    sentiment_tone = excluded.sentiment_tone,
                    risk_flags = excluded.risk_flags,
                    relevance = excluded.relevance,
                    confidence = excluded.confidence,
                    summary = excluded.summary,
                    recommended_action = excluded.recommended_action,
                    source_record_id = excluded.source_record_id,
                    raw_text = excluded.raw_text
                """,
                [
                    (
                        event.event_id,
                        event.source_id,
                        event.source_name,
                        event.source_url,
                        event.source_category,
                        event.title,
                        event.url,
                        event.published_at,
                        event.collected_at,
                        event.country,
                        event.region,
                        event.sector,
                        event.commodity,
                        event.actors,
                        event.event_type,
                        event.outcome,
                        event.sentiment_tone,
                        event.risk_flags,
                        event.relevance,
                        event.confidence,
                        event.summary,
                        event.recommended_action,
                        event.source_record_id,
                        event.status,
                        event.raw_text,
                    )
                    for event in events
                ],
            )
        self.log_event(
            action="monitoring_events_saved",
            details={"events": len(events), "countries": sorted({event.country for event in events})[:50]},
        )

    def list_monitoring_events(
        self,
        *,
        country: str | None = None,
        sector: str | None = None,
        status: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Return normalized monitoring events for dashboards."""

        clauses: list[str] = []
        params: list[Any] = []
        if country:
            clauses.append("country = ?")
            params.append(country)
        if sector:
            clauses.append("sector = ?")
            params.append(sector)
        if status:
            clauses.append("status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT *
                FROM monitoring_events
                {where}
                ORDER BY
                    CASE relevance
                        WHEN 'high' THEN 0
                        WHEN 'medium' THEN 1
                        WHEN 'low' THEN 2
                        ELSE 9
                    END,
                    published_at DESC,
                    collected_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def update_monitoring_event_status(self, event_id: str, status: str) -> None:
        """Update review status for a monitoring event."""

        with self._connect() as connection:
            connection.execute(
                """
                UPDATE monitoring_events
                SET status = ?
                WHERE event_id = ?
                """,
                (status, event_id),
            )
        self.log_event(
            action="monitoring_event_status_updated",
            details={"event_id": event_id, "status": status},
        )

    def save_monitoring_run(self, run: dict[str, Any]) -> str:
        """Persist one live monitoring run summary without storing secrets."""

        run_id = str(run.get("monitoring_run_id") or stable_id("monitoring-run", run.get("run_timestamp", ""), run.get("query", "")))
        timestamp = str(run.get("run_timestamp") or utc_now_iso())
        source_statuses = run.get("source_statuses", [])
        sources_attempted = [
            {
                "source_name": row.get("source_name", row.get("source", "")),
                "source_type": row.get("source_type", ""),
                "source_status": row.get("source_status", row.get("status", "")),
                "records": row.get("records", 0),
                "secret_visible": "no",
            }
            for row in source_statuses
        ]
        signal_summary = run.get("signal_summary", {}) if isinstance(run.get("signal_summary"), dict) else {}
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO monitoring_runs
                (
                    monitoring_run_id,
                    timestamp,
                    query,
                    filters_json,
                    sources_attempted_json,
                    source_statuses_json,
                    signal_count,
                    cluster_count,
                    top_countries_json,
                    top_sectors_json,
                    top_commodities_json,
                    top_risk_flags_json,
                    fallback_used,
                    warnings_json,
                    errors_json,
                    brief_markdown,
                    export_paths_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(monitoring_run_id) DO UPDATE SET
                    timestamp = excluded.timestamp,
                    query = excluded.query,
                    filters_json = excluded.filters_json,
                    sources_attempted_json = excluded.sources_attempted_json,
                    source_statuses_json = excluded.source_statuses_json,
                    signal_count = excluded.signal_count,
                    cluster_count = excluded.cluster_count,
                    top_countries_json = excluded.top_countries_json,
                    top_sectors_json = excluded.top_sectors_json,
                    top_commodities_json = excluded.top_commodities_json,
                    top_risk_flags_json = excluded.top_risk_flags_json,
                    fallback_used = excluded.fallback_used,
                    warnings_json = excluded.warnings_json,
                    errors_json = excluded.errors_json,
                    brief_markdown = excluded.brief_markdown,
                    export_paths_json = excluded.export_paths_json
                """,
                (
                    run_id,
                    timestamp,
                    str(run.get("query", "")),
                    json.dumps(run.get("filters", {}), ensure_ascii=False),
                    json.dumps(sources_attempted, ensure_ascii=False),
                    json.dumps(source_statuses, ensure_ascii=False),
                    int(run.get("normalized_signal_count", 0) or 0),
                    int(run.get("event_cluster_count", 0) or 0),
                    json.dumps(signal_summary.get("top_countries", []), ensure_ascii=False),
                    json.dumps(signal_summary.get("top_sectors", []), ensure_ascii=False),
                    json.dumps(signal_summary.get("top_commodities", []), ensure_ascii=False),
                    json.dumps(signal_summary.get("top_risk_flags", []), ensure_ascii=False),
                    1 if run.get("fallback_used") else 0,
                    json.dumps(run.get("warnings", []), ensure_ascii=False),
                    json.dumps(run.get("errors", []), ensure_ascii=False),
                    str(run.get("brief_markdown", "")),
                    json.dumps(run.get("export_paths", {}), ensure_ascii=False),
                    utc_now_iso(),
                ),
            )
        self.log_event(
            action="monitoring_run_saved",
            details={
                "monitoring_run_id": run_id,
                "signals": int(run.get("normalized_signal_count", 0) or 0),
                "clusters": int(run.get("event_cluster_count", 0) or 0),
                "fallback_used": bool(run.get("fallback_used")),
            },
        )
        return run_id

    def list_monitoring_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent live monitoring run metadata."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM monitoring_runs
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            for key in (
                "filters_json",
                "sources_attempted_json",
                "source_statuses_json",
                "top_countries_json",
                "top_sectors_json",
                "top_commodities_json",
                "top_risk_flags_json",
                "warnings_json",
                "errors_json",
                "export_paths_json",
            ):
                item[key.removesuffix("_json")] = json.loads(item.pop(key) or "[]")
            item["fallback_used"] = bool(item["fallback_used"])
            result.append(item)
        return result

    def save_promoted_monitoring_item(self, item: dict[str, Any]) -> str:
        """Persist one promoted signal or cluster for review and reuse."""

        promotion_id = str(item.get("promotion_id") or stable_id("promotion", item.get("monitoring_run_id", ""), item.get("item_type", ""), item.get("source_item_id", "")))
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO promoted_monitoring_items
                (
                    promotion_id,
                    monitoring_run_id,
                    item_type,
                    source_item_id,
                    event_id,
                    title,
                    summary,
                    country,
                    region,
                    sector,
                    commodity,
                    event_type,
                    tone,
                    risk_flags,
                    relevance_score,
                    confidence_level,
                    source_name,
                    source_type,
                    source_url,
                    evidence_text,
                    supporting_signal_ids_json,
                    retrieved_at,
                    promoted_at,
                    analyst_note,
                    status,
                    review_flag,
                    review_reasons_json,
                    payload_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(promotion_id) DO UPDATE SET
                    event_id = excluded.event_id,
                    title = excluded.title,
                    summary = excluded.summary,
                    country = excluded.country,
                    region = excluded.region,
                    sector = excluded.sector,
                    commodity = excluded.commodity,
                    event_type = excluded.event_type,
                    tone = excluded.tone,
                    risk_flags = excluded.risk_flags,
                    relevance_score = excluded.relevance_score,
                    confidence_level = excluded.confidence_level,
                    source_name = excluded.source_name,
                    source_type = excluded.source_type,
                    source_url = excluded.source_url,
                    evidence_text = excluded.evidence_text,
                    supporting_signal_ids_json = excluded.supporting_signal_ids_json,
                    analyst_note = excluded.analyst_note,
                    status = excluded.status,
                    review_flag = excluded.review_flag,
                    review_reasons_json = excluded.review_reasons_json,
                    payload_json = excluded.payload_json
                """,
                (
                    promotion_id,
                    str(item.get("monitoring_run_id", "")),
                    str(item.get("item_type", "")),
                    str(item.get("source_item_id", "")),
                    str(item.get("event_id", "")),
                    str(item.get("title", "")),
                    str(item.get("summary", "")),
                    str(item.get("country", "")),
                    str(item.get("region", "")),
                    str(item.get("sector", "")),
                    str(item.get("commodity", "")),
                    str(item.get("event_type", "")),
                    str(item.get("tone", "")),
                    str(item.get("risk_flags", "")),
                    float(item.get("relevance_score", 0.0) or 0.0),
                    str(item.get("confidence_level", "")),
                    str(item.get("source_name", "")),
                    str(item.get("source_type", "")),
                    str(item.get("source_url", "")),
                    str(item.get("evidence_text", "")),
                    json.dumps(item.get("supporting_signal_ids", []), ensure_ascii=False),
                    str(item.get("retrieved_at", "")),
                    str(item.get("promoted_at", utc_now_iso())),
                    str(item.get("analyst_note", "")),
                    str(item.get("status", "new")),
                    str(item.get("review_flag", "")),
                    json.dumps(item.get("review_reasons", []), ensure_ascii=False),
                    json.dumps(item.get("payload", {}), ensure_ascii=False),
                ),
            )
        self.log_event(
            action="monitoring_item_promoted",
            details={
                "promotion_id": promotion_id,
                "monitoring_run_id": item.get("monitoring_run_id", ""),
                "item_type": item.get("item_type", ""),
                "source_item_id": item.get("source_item_id", ""),
                "review_flag": item.get("review_flag", ""),
            },
        )
        return promotion_id

    def list_promoted_monitoring_items(
        self,
        *,
        status: str | None = None,
        review_flag: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Return promoted live monitoring items."""

        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if review_flag:
            clauses.append("review_flag = ?")
            params.append(review_flag)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT *
                FROM promoted_monitoring_items
                {where}
                ORDER BY promoted_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["supporting_signal_ids"] = json.loads(item.pop("supporting_signal_ids_json") or "[]")
            item["review_reasons"] = json.loads(item.pop("review_reasons_json") or "[]")
            item["payload"] = json.loads(item.pop("payload_json") or "{}")
            result.append(item)
        return result

    def update_promoted_monitoring_item(
        self,
        promotion_id: str,
        *,
        status: str | None = None,
        analyst_note: str | None = None,
    ) -> None:
        """Update review status or analyst note for a promoted item."""

        updates = []
        params: list[Any] = []
        if status is not None:
            updates.append("status = ?")
            params.append(status)
        if analyst_note is not None:
            updates.append("analyst_note = ?")
            params.append(analyst_note)
        if not updates:
            return
        params.append(promotion_id)
        with self._connect() as connection:
            connection.execute(
                f"""
                UPDATE promoted_monitoring_items
                SET {', '.join(updates)}
                WHERE promotion_id = ?
                """,
                params,
            )
        self.log_event(
            action="promoted_monitoring_item_updated",
            details={"promotion_id": promotion_id, "status": status, "analyst_note_updated": analyst_note is not None},
        )

    def save_generated_output(
        self,
        task_type: str,
        title: str,
        markdown_path: str,
        csv_path: str,
        pdf_path: str,
        metrics: dict[str, Any],
    ) -> str:
        """Record exported files so the UI can show an audit trail."""

        output_id = stable_id(task_type, title, markdown_path, utc_now_iso())
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO generated_outputs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    output_id,
                    task_type,
                    title,
                    markdown_path,
                    csv_path,
                    pdf_path,
                    json.dumps(metrics, ensure_ascii=False),
                    utc_now_iso(),
                ),
            )
        self.log_event(
            action="generated_output_saved",
            details={
                "output_id": output_id,
                "task_type": task_type,
                "title": title,
                "markdown_path": markdown_path,
                "csv_path": csv_path,
                "pdf_path": pdf_path,
            },
        )
        return output_id

    def save_analysis_session(self, session: AnalysisSession) -> None:
        """Persist one coherent analysis run."""

        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO analysis_sessions
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.session_id,
                    session.title,
                    session.task_type,
                    session.query,
                    json.dumps(session.document_ids, ensure_ascii=False),
                    session.scope_label,
                    session.status,
                    session.output_id,
                    json.dumps(session.diagnostics, ensure_ascii=False, sort_keys=True),
                    session.created_at,
                ),
            )
        self.log_event(
            action="analysis_session_saved",
            details={
                "session_id": session.session_id,
                "task_type": session.task_type,
                "document_ids": session.document_ids,
                "status": session.status,
            },
        )

    def update_analysis_session_output(self, session_id: str, output_id: str) -> None:
        """Attach an exported output ID to an existing analysis session."""

        with self._connect() as connection:
            connection.execute(
                "UPDATE analysis_sessions SET output_id = ? WHERE session_id = ?",
                (output_id, session_id),
            )

    def list_analysis_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent analysis sessions for audit and UI display."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM analysis_sessions
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        sessions: list[dict[str, Any]] = []
        for row in rows:
            session = dict(row)
            session["document_ids"] = json.loads(row["document_ids_json"])
            session["diagnostics"] = json.loads(row["diagnostics_json"])
            sessions.append(session)
        return sessions

    def list_generated_outputs(self) -> list[dict[str, Any]]:
        """Return previous output files for audit and reuse."""

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM generated_outputs ORDER BY created_at DESC"
            ).fetchall()
        outputs: list[dict[str, Any]] = []
        for row in rows:
            output = dict(row)
            output["metrics"] = json.loads(row["metrics_json"])
            outputs.append(output)
        return outputs

    def log_event(self, action: str, details: dict[str, Any], actor: str = "local-user") -> None:
        """Write a human-readable audit event.

        The audit log is intentionally simple. It answers practical questions:
        What was ingested? What was generated? Which parser was used? How many
        records were extracted?
        """

        timestamp = utc_now_iso()
        event_id = stable_id(timestamp, actor, action, json.dumps(details, sort_keys=True))
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO audit_events VALUES (?, ?, ?, ?, ?)",
                (
                    event_id,
                    timestamp,
                    actor,
                    action,
                    json.dumps(details, ensure_ascii=False, sort_keys=True),
                ),
            )

    def list_audit_events(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return recent audit events."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM audit_events
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        events: list[dict[str, Any]] = []
        for row in rows:
            event = dict(row)
            event["details"] = json.loads(row["details_json"])
            events.append(event)
        return events


def csv_shape(path: Path) -> tuple[int, int]:
    """Return ``(row_count, column_count)`` for a CSV file."""

    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        rows = sum(1 for _ in reader)
    return rows, len(columns)
