"""End-to-end orchestration for the workbench.

The pipeline is the single place where the full architecture comes together:

input files -> parsing -> storage -> indexing -> retrieval -> extraction ->
drafting -> verification -> export.

Both the command line and Streamlit dashboard call this class, which keeps the
behavior consistent and easier to audit.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from devfinintel.alignment import build_role_alignment_report
from devfinintel.chunking import chunk_pages
from devfinintel.config import DEFAULT_CONFIG, ProjectConfig
from devfinintel.connectors.eia import fetch_eia_signals
from devfinintel.connectors.gdelt import fetch_gdelt_signals
from devfinintel.connectors.gnews import fetch_gnews_signals
from devfinintel.connectors.guardian import fetch_guardian_signals
from devfinintel.connectors.newsapi import fetch_newsapi_signals
from devfinintel.connectors.reliefweb import fetch_reliefweb_signals
from devfinintel.connectors.worldbank_docs import fetch_worldbank_document_signals
from devfinintel.connectors.worldbank_indicators import fetch_worldbank_indicator_signals
from devfinintel.datasets import DatasetProfiler
from devfinintel.env import get_monitoring_settings, load_project_env, source_key_status
from devfinintel.evidence import EvidencePackBuilder
from devfinintel.events import cluster_signals, clusters_to_evidence_rows, summarize_monitoring_cycle
from devfinintel.exporting import ExportPaths, OutputExporter
from devfinintel.extraction import EvidenceBoundDrafter, StructuredExtractor
from devfinintel.indexing import HybridSearchIndex
from devfinintel.intelligence import build_action_items, build_country_intelligence
from devfinintel.coverage import (
    build_coverage_matrix,
    coverage_summary as matrix_coverage_summary,
    source_backlog,
)
from devfinintel.knowledge import coverage_summary as knowledge_record_summary
from devfinintel.knowledge import knowledge_records_from_output
from devfinintel.llm import DEFAULT_OLLAMA_MODEL, LocalLLMAnswer, OllamaGroundedLLM
from devfinintel.models import AnalysisSession, GeneratedOutput, MonitoringEvent, RetrievalDiagnostics, VerificationFinding
from devfinintel.monitoring import (
    build_monitoring_agent_run,
    build_monitoring_insights,
    collect_monitoring_sources,
    infer_outcome,
    initialize_monitoring_sources,
    load_monitoring_sources,
    monitoring_events_from_knowledge_records,
    save_monitoring_sources,
)
from devfinintel.news import SAMPLE_ARTICLES, fetch_news
from devfinintel.parsing import DocumentParser
from devfinintel.planner import TASK_ALIASES, RetrievalPlanner, RetrievalTask
from devfinintel.schemas import schema_summary
from devfinintel.sources import (
    download_registered_sources,
    downloaded_paths,
    initialize_source_registry,
    load_source_registry,
    registry_summary,
    copy_downloads_to_input,
)
from devfinintel.store import SQLiteDocumentStore
from devfinintel.utils import stable_id, utc_now_iso
from devfinintel.signals import deduplicate_signals, normalize_connector_result, normalize_signal, signal_summary, split_labels
from devfinintel.workbench import (
    build_intelligence_brief,
    build_workbench_answer,
    evidence_rows_to_items,
    summarize_csv_documents,
)


class DocumentIntelligencePipeline:
    """High-level API for ingestion, generation, verification, and export."""

    def __init__(self, config: ProjectConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        load_project_env(self.config.root_dir)
        self.config.ensure_directories()
        self.parser = DocumentParser()
        self.store = SQLiteDocumentStore(self.config.database_path)
        self.planner = RetrievalPlanner()
        self.extractor = StructuredExtractor()
        self.drafter = EvidenceBoundDrafter()
        self.dataset_profiler = DatasetProfiler()
        self.evidence_pack_builder = EvidencePackBuilder()
        self.exporter = OutputExporter(self.config.output_dir)

    def ingest_paths(self, paths: list[Path], language_hint: str = "unknown") -> list[dict]:
        """Parse and store source files.

        Returns a short summary per file so the UI can show what happened.
        """

        summaries = []
        for path in paths:
            parsed = self.parser.parse_path(path, language_hint=language_hint)
            chunks = chunk_pages(
                parsed.pages,
                chunk_size_chars=self.config.chunk_size_chars,
                overlap_chars=self.config.chunk_overlap_chars,
            )
            self.store.save_document(parsed.document, parsed.pages, chunks)
            summaries.append(
                {
                    "title": parsed.document.title,
                    "document_id": parsed.document.document_id,
                    "source_path": parsed.document.source_path,
                    "parser_backend": parsed.document.parser_backend,
                    "pages": len(parsed.pages),
                    "chunks": len(chunks),
                }
            )
        return summaries

    def build_index(self, document_ids: list[str] | None = None) -> HybridSearchIndex:
        """Build an in-memory search index from the current SQLite store."""

        chunks = self.store.get_all_chunks()
        documents = self.store.get_document_lookup()
        if document_ids is not None:
            allowed = set(document_ids)
            chunks = [chunk for chunk in chunks if chunk.document_id in allowed]
            documents = {
                document_id: document
                for document_id, document in documents.items()
                if document_id in allowed
            }
        return HybridSearchIndex(
            chunks=chunks,
            document_lookup=documents,
            embedding_dimensions=self.config.embedding_dimensions,
        )

    def generate(
        self,
        task_type: str,
        query: str,
        top_k: int = 8,
        document_ids: list[str] | None = None,
    ) -> GeneratedOutput:
        """Generate an evidence-grounded output for one task."""

        task_type = TASK_ALIASES.get(task_type, task_type)
        if task_type == "dataset_profile":
            return self.generate_dataset_profile(query=query, document_ids=document_ids)

        resolved_document_ids = self._resolve_document_ids(document_ids)
        index = self.build_index(document_ids=document_ids)
        task = RetrievalTask(task_type=task_type, query=query, top_k=top_k)
        expanded_query = self.planner.build_query(task)
        evidence, diagnostics = index.search_with_diagnostics(
            query=expanded_query,
            task_type=task_type,
            top_k=top_k,
        )
        session_id = self._create_session_id(task_type, query, resolved_document_ids)
        if not diagnostics.passed:
            output = self._abstention_output(task_type, query, diagnostics)
            output.metrics["analysis_session_id"] = session_id
            self._save_session_for_output(
                session_id=session_id,
                task_type=task_type,
                query=query,
                document_ids=resolved_document_ids,
                scope_label=self._scope_label(document_ids),
                status="needs_evidence",
                output=output,
                diagnostics=diagnostics.as_dict(),
            )
            return output

        records = self.extractor.extract(task_type=task_type, query=query, evidence=evidence)
        evidence_pack = self.evidence_pack_builder.build(
            task_type=task_type,
            query=query,
            evidence_items=evidence,
            records=records,
            diagnostics={
                **diagnostics.as_dict(),
                "expanded_query": expanded_query,
                "embedding_backend": index.embedding_backend,
            },
        )
        title, body_markdown = self.drafter.draft(
            task_type=task_type,
            query=query,
            evidence=evidence,
            records=records,
        )
        from devfinintel.verification import OutputVerifier

        output = OutputVerifier().verify(
            task_type=task_type,
            title=title,
            body_markdown=body_markdown,
            evidence_items=evidence,
            records=records,
            evidence_pack=evidence_pack,
        )
        self.store.save_extraction_records(records)
        self.store.save_knowledge_records(knowledge_records_from_output(output))
        self.refresh_action_items()
        self.refresh_monitoring_events(from_sources=False, from_knowledge=True)
        output.metrics.update(schema_summary(records))
        output.metrics["retrieval_passed"] = 1.0 if diagnostics.passed else 0.0
        output.metrics["retrieval_top_score"] = diagnostics.top_score
        output.metrics["retrieval_keyword_coverage"] = diagnostics.keyword_coverage
        output.metrics["evidence_pack_id"] = evidence_pack.pack_id
        output.metrics["analysis_session_id"] = session_id
        self.store.log_event(
            action="output_generated",
            details={
                "task_type": task_type,
                "query": query,
                "top_k": top_k,
                "document_ids": document_ids if document_ids is not None else "all",
                "evidence_items": len(evidence),
                "embedding_backend": index.embedding_backend,
                "metrics": output.metrics,
                "retrieval_diagnostics": diagnostics.as_dict(),
            },
        )
        self._save_session_for_output(
            session_id=session_id,
            task_type=task_type,
            query=query,
            document_ids=resolved_document_ids,
            scope_label=self._scope_label(document_ids),
            status="generated",
            output=output,
            diagnostics=diagnostics.as_dict(),
        )
        return output

    def generate_dataset_profile(
        self,
        query: str,
        document_ids: list[str] | None = None,
    ) -> GeneratedOutput:
        """Generate a computed profile for selected CSV datasets."""

        documents = self.store.get_document_lookup()
        resolved_document_ids = self._resolve_document_ids(document_ids)
        if document_ids is not None:
            selected_documents = [
                document for document_id, document in documents.items() if document_id in set(document_ids)
            ]
        else:
            selected_documents = list(documents.values())

        output = self.dataset_profiler.profile(query=query, documents=selected_documents)
        evidence_pack = self.evidence_pack_builder.build(
            task_type="dataset_profile",
            query=query,
            evidence_items=output.evidence_items,
            records=output.records,
            diagnostics={"dataset_profile": True, "csv_documents": len(selected_documents)},
        )
        output = replace(output, evidence_pack=evidence_pack)
        session_id = self._create_session_id("dataset_profile", query, resolved_document_ids)
        output.metrics["analysis_session_id"] = session_id
        output.metrics["evidence_pack_id"] = evidence_pack.pack_id
        output.metrics.update(schema_summary(output.records))
        output.metrics["retrieval_passed"] = 1.0
        self.store.save_extraction_records(output.records)
        self.store.save_knowledge_records(knowledge_records_from_output(output))
        self.refresh_action_items()
        self.refresh_monitoring_events(from_sources=False, from_knowledge=True)
        self.store.log_event(
            action="dataset_profile_generated",
            details={
                "query": query,
                "document_ids": document_ids if document_ids is not None else "all",
                "evidence_items": len(output.evidence_items),
                "metrics": output.metrics,
            },
        )
        self._save_session_for_output(
            session_id=session_id,
            task_type="dataset_profile",
            query=query,
            document_ids=resolved_document_ids,
            scope_label=self._scope_label(document_ids),
            status="generated",
            output=output,
            diagnostics={"dataset_profile": True, "csv_documents": len(selected_documents)},
        )
        return output

    def export_output(self, output: GeneratedOutput) -> ExportPaths:
        """Write Markdown, CSV, and PDF outputs and record them in the database."""

        evidence_document_ids = sorted({item.document_id for item in output.evidence_items})
        paths = self.exporter.export(
            output,
            file_manifests=self.list_file_manifests(document_ids=evidence_document_ids)
            if evidence_document_ids
            else [],
        )
        output.metrics["export_manifest_path"] = str(paths.manifest_json_path)
        output.metrics["export_package_path"] = str(paths.package_path)
        output_id = self.store.save_generated_output(
            task_type=output.task_type,
            title=output.title,
            markdown_path=str(paths.markdown_path),
            csv_path=str(paths.csv_path),
            pdf_path=str(paths.pdf_path),
            metrics=output.metrics,
        )
        session_id = output.metrics.get("analysis_session_id")
        if isinstance(session_id, str):
            self.store.update_analysis_session_output(session_id, output_id)
        return paths

    def answer_with_local_llm(
        self,
        question: str,
        output: GeneratedOutput,
        model: str = DEFAULT_OLLAMA_MODEL,
    ) -> LocalLLMAnswer:
        """Ask a local LLM to explain the already-retrieved evidence.

        The LLM is intentionally downstream of retrieval and Python analysis. It
        receives only the evidence pack that the app has already shown or
        computed, which is safer than giving it unrestricted access to the whole
        document library.
        """

        answer = OllamaGroundedLLM(model=model).answer_from_output(question, output)
        self.store.log_event(
            action="local_llm_answer_generated",
            details={
                "model": answer.model,
                "question": question,
                "evidence_count": answer.evidence_count,
                "citation_count": answer.citation_count,
                "warning": answer.warning,
            },
        )
        return answer

    def list_file_manifests(self, document_ids: list[str] | None = None) -> list[dict]:
        """Return file manifests as dictionaries for UI and audit exports."""

        return [manifest.__dict__ for manifest in self.store.list_file_manifests(document_ids=document_ids)]

    def initialize_sources(self, overwrite: bool = False) -> tuple[Path, Path]:
        """Create official source registry files.

        The registry is a controlled intake list. It lets the user add many
        official PDFs later without losing source URLs, topics, country coverage,
        or download checksums.
        """

        paths = initialize_source_registry(
            registry_path=self.config.source_registry_path,
            country_coverage_path=self.config.country_coverage_path,
            overwrite=overwrite,
        )
        self.store.log_event(
            action="source_registry_initialized",
            details={
                "source_registry_path": str(paths[0]),
                "country_coverage_path": str(paths[1]),
                "overwrite": overwrite,
            },
        )
        return paths

    def list_source_registry(self) -> list[dict]:
        """Return source registry rows for the dashboard."""

        entries = load_source_registry(self.config.source_registry_path)
        return [entry.as_row() for entry in entries]

    def source_registry_summary(self) -> dict:
        """Return source coverage counts for the dashboard."""

        return registry_summary(load_source_registry(self.config.source_registry_path))

    def initialize_monitoring(self, overwrite: bool = False) -> Path:
        """Create and sync the governed monitoring-source registry."""

        path = initialize_monitoring_sources(
            self.config.monitoring_source_registry_path,
            overwrite=overwrite,
        )
        sources = load_monitoring_sources(path)
        self.store.sync_monitoring_sources(sources)
        self.store.log_event(
            action="monitoring_sources_initialized",
            details={"monitoring_source_registry_path": str(path), "sources": len(sources), "overwrite": overwrite},
        )
        return path

    def list_monitoring_sources(self) -> list[dict]:
        """Return monitoring-source registry rows."""

        if not self.config.monitoring_source_registry_path.exists():
            self.initialize_monitoring()
        sources = load_monitoring_sources(self.config.monitoring_source_registry_path)
        self.store.sync_monitoring_sources(sources)
        return self.store.list_monitoring_sources(limit=500)

    def refresh_monitoring_events(
        self,
        *,
        from_sources: bool = False,
        from_knowledge: bool = True,
        limit: int = 500,
    ) -> dict:
        """Refresh normalized monitoring events from governed inputs."""

        if not self.config.monitoring_source_registry_path.exists():
            self.initialize_monitoring()
        sources = load_monitoring_sources(self.config.monitoring_source_registry_path)
        self.store.sync_monitoring_sources(sources)
        events = []
        source_results: list[dict] = []
        updated_sources = sources
        if from_knowledge:
            events.extend(
                monitoring_events_from_knowledge_records(
                    self.store.list_knowledge_records(limit=10000),
                    limit=limit,
                )
            )
        if from_sources:
            fetched_events, source_results, updated_sources = collect_monitoring_sources(
                sources,
                limit=max(limit - len(events), 0) if events else limit,
            )
            events.extend(fetched_events)
            save_monitoring_sources(self.config.monitoring_source_registry_path, updated_sources)
            self.store.sync_monitoring_sources(updated_sources)
        self.store.save_monitoring_events(events[:limit])
        self.store.log_event(
            action="monitoring_events_refreshed",
            details={
                "from_sources": from_sources,
                "from_knowledge": from_knowledge,
                "events": len(events[:limit]),
                "source_results": source_results,
            },
        )
        return {
            "events_saved": len(events[:limit]),
            "source_results": source_results,
            "sources": [source.__dict__ for source in updated_sources],
        }

    def list_monitoring_events(
        self,
        *,
        country: str | None = None,
        sector: str | None = None,
        status: str | None = None,
        limit: int = 1000,
    ) -> list[dict]:
        """Return normalized monitoring events."""

        events = self.store.list_monitoring_events(
            country=country,
            sector=sector,
            status=status,
            limit=limit,
        )
        if not events and self.store.list_knowledge_records(limit=1):
            self.refresh_monitoring_events(from_sources=False, from_knowledge=True, limit=limit)
            events = self.store.list_monitoring_events(
                country=country,
                sector=sector,
                status=status,
                limit=limit,
            )
        return events

    def update_monitoring_event_status(self, event_id: str, status: str) -> None:
        """Update the review status for a monitoring event."""

        self.store.update_monitoring_event_status(event_id, status)

    def monitoring_insights(self, limit: int = 1000) -> dict:
        """Return monitoring metrics, trend rows, and insight cards."""

        events = self.list_monitoring_events(limit=limit)
        actions = self.list_action_items(limit=1000)
        return build_monitoring_insights(events, actions=actions)

    def run_monitoring_agent(
        self,
        *,
        from_sources: bool = False,
        from_knowledge: bool = True,
        limit: int = 1000,
        monitoring_result: dict | None = None,
        include_current_live_results: bool = False,
    ) -> dict:
        """Run an auditable monitoring supervisor pass."""

        refresh = self.refresh_monitoring_events(
            from_sources=from_sources,
            from_knowledge=from_knowledge,
            limit=limit,
        )
        stored_events = self.list_monitoring_events(limit=limit)
        current_events = (
            self._current_session_events_from_monitoring_result(monitoring_result)
            if include_current_live_results
            else []
        )
        events = stored_events + current_events
        actions = self.list_action_items(limit=1000)
        sources = self.list_monitoring_sources()
        insights = build_monitoring_insights(events, actions=actions)
        agent_run = build_monitoring_agent_run(
            events=events,
            actions=actions,
            sources=sources,
            insights=insights,
        )
        agent_run["refresh"] = refresh
        agent_run["mode"] = (
            "stored/promoted events plus current live monitoring results"
            if include_current_live_results
            else "stored/promoted events only"
        )
        agent_run["evidence_pool"] = {
            "stored_events_count": len(stored_events),
            "promoted_live_items_count": len(self.list_promoted_monitoring_items(limit=10000)),
            "current_session_signals_count": len((monitoring_result or {}).get("normalized_signals", [])) if include_current_live_results else 0,
            "current_session_clusters_count": len((monitoring_result or {}).get("event_clusters", [])) if include_current_live_results else 0,
            "current_session_events_used": len(current_events),
            "total_events_used": len(events),
        }
        agent_run["evidence_rows"] = [
            {
                "event_id": event.get("event_id", ""),
                "title": event.get("title", ""),
                "country": event.get("country", ""),
                "commodity": event.get("commodity", ""),
                "risk_flags": event.get("risk_flags", ""),
                "source_name": event.get("source_name", ""),
                "source_url": event.get("source_url", event.get("url", "")),
                "evidence_origin": event.get("evidence_origin", "stored_monitoring_event"),
                "review_reasons": "; ".join(event.get("review_reasons", [])) if isinstance(event.get("review_reasons"), list) else "",
            }
            for event in events[:100]
        ]
        self.store.log_event(
            action="monitoring_agent_run",
            details={
                "from_sources": from_sources,
                "from_knowledge": from_knowledge,
                "include_current_live_results": include_current_live_results,
                "evidence_pool": agent_run["evidence_pool"],
                "events": len(events),
                "urgent_signals": sum(1 for row in agent_run["triage_queue"] if row["priority"] == "urgent"),
                "watchlist_countries": [row["country"] for row in agent_run["watchlist"][:10]],
            },
        )
        return agent_run

    def source_configuration_status(self) -> list[dict]:
        """Return key-safe source configuration rows for the UI."""

        rows = [
            {"source": "GDELT", "source_type": "news", "status": "keyless public", "secret_visible": "no"},
            {"source": "ReliefWeb", "source_type": "risk_context", "status": "keyless public", "secret_visible": "no"},
            {"source": "World Bank Indicators", "source_type": "dataset_indicator", "status": "keyless public", "secret_visible": "no"},
            {"source": "World Bank Documents", "source_type": "institutional_report", "status": "keyless public", "secret_visible": "no"},
        ]
        rows.extend(source_key_status())
        rows.append({"source": "Fallback sample data", "source_type": "fallback_sample", "status": "available", "secret_visible": "no"})
        return rows

    def run_monitoring_cycle(
        self,
        *,
        query: str = "",
        country: str = "",
        commodity: str = "",
        topic: str = "",
        limit: int | None = None,
        include_optional_key_sources: bool = True,
        include_indicators: bool = True,
        use_live_connectors: bool = True,
    ) -> dict:
        """Run the near-real-time source connector and clustering cycle."""

        settings = get_monitoring_settings()
        effective_limit = limit or settings.max_articles
        source_results: list[dict] = []
        connector_query = query or "Africa energy commodities oil gas mining power investment climate risk"

        if use_live_connectors:
            source_results.extend(
                [
                    fetch_gdelt_signals(
                        query=connector_query,
                        country=country,
                        commodity=commodity,
                        topic=topic,
                        lookback_days=settings.lookback_days,
                        limit=effective_limit,
                        timeout_seconds=12,
                    ),
                    fetch_reliefweb_signals(
                        query="Africa drought flood conflict displacement food security infrastructure energy mining climate",
                        country=country,
                        topic=topic,
                        lookback_days=max(settings.lookback_days, 30),
                        limit=max(10, effective_limit // 2),
                        timeout_seconds=12,
                    ),
                    fetch_worldbank_document_signals(
                        query=connector_query,
                        country=country,
                        commodity=commodity,
                        topic=topic,
                        limit=max(10, effective_limit // 2),
                        timeout_seconds=12,
                    ),
                ]
            )
            if include_indicators:
                source_results.append(
                    fetch_worldbank_indicator_signals(
                        countries=[country] if country else [],
                        limit=200,
                        timeout_seconds=12,
                    )
                )
        else:
            source_results.extend(
                [
                    {
                        "source_name": "GDELT",
                        "source_type": "news",
                        "source_status": "skipped",
                        "records": [],
                        "warnings": ["Live connector execution disabled for this run."],
                        "errors": [],
                        "metadata": {},
                    },
                    {
                        "source_name": "ReliefWeb",
                        "source_type": "risk_context",
                        "source_status": "skipped",
                        "records": [],
                        "warnings": ["Live connector execution disabled for this run."],
                        "errors": [],
                        "metadata": {},
                    },
                    {
                        "source_name": "World Bank Documents",
                        "source_type": "institutional_report",
                        "source_status": "skipped",
                        "records": [],
                        "warnings": ["Live connector execution disabled for this run."],
                        "errors": [],
                        "metadata": {},
                    },
                ]
            )

        if include_optional_key_sources:
            source_results.extend(
                [
                    fetch_newsapi_signals(query=connector_query, lookback_days=settings.lookback_days, limit=max(10, effective_limit // 2)),
                    fetch_gnews_signals(query=connector_query, limit=max(10, effective_limit // 2)),
                    fetch_eia_signals(query=connector_query, limit=max(10, effective_limit // 2)),
                    fetch_guardian_signals(query=connector_query, limit=max(10, effective_limit // 2)),
                ]
            )

        signals = []
        for result in source_results:
            signals.extend(normalize_connector_result(result, default_region=settings.default_region))

        fallback_used = False
        if not signals and settings.use_sample_data:
            fallback_used = True
            source_results.append(
                {
                    "source_name": "Fallback sample data",
                    "source_type": "fallback_sample",
                    "source_status": "used",
                    "records": SAMPLE_ARTICLES,
                    "warnings": ["No live/keyed connector returned signals; sample monitoring data used."],
                    "errors": [],
                    "metadata": {"secret_visible": "no"},
                }
            )
            for article in SAMPLE_ARTICLES[:effective_limit]:
                signals.append(
                    normalize_signal(
                        {
                            "raw_source_id": article["url"],
                            "title": article["title"],
                            "date": article["published_at"],
                            "url": article["url"],
                            "source": article["source"],
                            "summary": article["summary"],
                            "evidence_text": f"{article['title']}. {article['summary']}",
                        },
                        source_type="fallback_sample",
                        source_name="Fallback sample data",
                        source_status="fallback",
                        default_region=settings.default_region,
                    )
                )

        signals = deduplicate_signals(signals)
        clusters = cluster_signals(signals, date_window_days=settings.lookback_days)
        evidence_table = clusters_to_evidence_rows(clusters, signals)
        source_statuses = self._source_status_rows(source_results, fallback_used=fallback_used)
        warnings = [warning for result in source_results for warning in result.get("warnings", [])]
        errors = [error for result in source_results for error in result.get("errors", [])]
        run_timestamp = utc_now_iso()
        run_id = stable_id(
            "monitoring-run",
            run_timestamp,
            connector_query,
            country,
            commodity,
            topic,
            len(signals),
            len(clusters),
        )
        result = {
            "monitoring_run_id": run_id,
            "run_timestamp": run_timestamp,
            "query": connector_query,
            "filters": {"country": country, "commodity": commodity, "topic": topic},
            "settings": {
                "default_region": settings.default_region,
                "lookback_days": settings.lookback_days,
                "max_articles": settings.max_articles,
                "use_sample_data": settings.use_sample_data,
            },
            "source_statuses": source_statuses,
            "source_results": source_results,
            "signals_fetched_by_source": {
                row["source_name"]: len(row.get("records", []))
                for row in source_results
            },
            "normalized_signal_count": len(signals),
            "event_cluster_count": len(clusters),
            "normalized_signals": signals,
            "event_clusters": clusters,
            "top_developments": summarize_monitoring_cycle(
                source_results=source_results,
                signals=signals,
                clusters=clusters,
                fallback_used=fallback_used,
            ),
            "warnings": warnings,
            "errors": errors,
            "fallback_used": fallback_used,
            "signal_summary": signal_summary(signals),
            "evidence_table": evidence_table,
        }
        result["review_queue"] = self.build_monitoring_review_queue(result)
        result["brief_markdown"] = self._monitoring_cycle_brief_markdown(result)
        self.store.save_monitoring_run(result)
        self.store.log_event(
            action="monitoring_cycle_run",
            details={
                "monitoring_run_id": run_id,
                "query": connector_query,
                "filters": result["filters"],
                "sources": {row["source_name"]: row["source_status"] for row in source_statuses},
                "normalized_signals": len(signals),
                "event_clusters": len(clusters),
                "fallback_used": fallback_used,
                "warnings": len(warnings),
                "errors": len(errors),
            },
        )
        return result

    def list_monitoring_runs(self, limit: int = 50) -> list[dict]:
        """Return persisted monitoring-cycle run history."""

        if not hasattr(self.store, "list_monitoring_runs"):
            return []
        return self.store.list_monitoring_runs(limit=limit)

    def list_promoted_monitoring_items(self, limit: int = 500) -> list[dict]:
        """Return persisted promoted signals and clusters."""

        if not hasattr(self.store, "list_promoted_monitoring_items"):
            return []
        return self.store.list_promoted_monitoring_items(limit=limit)

    def promote_monitoring_signal(
        self,
        monitoring_result: dict,
        signal_id: str,
        *,
        analyst_note: str = "",
        status: str = "new",
    ) -> dict:
        """Promote one live signal into persistent monitoring storage."""

        signals = {signal.get("signal_id"): signal for signal in monitoring_result.get("normalized_signals", [])}
        signal = signals.get(signal_id)
        if not signal:
            raise ValueError(f"Signal not found in monitoring run: {signal_id}")
        duplicate_keys = self._duplicate_signal_keys(monitoring_result.get("normalized_signals", []))
        promotion = self._promotion_from_signal(
            signal,
            monitoring_run_id=monitoring_result.get("monitoring_run_id", ""),
            analyst_note=analyst_note,
            status=status,
            duplicate_keys=duplicate_keys,
        )
        event = self._monitoring_event_from_promotion(promotion)
        self.store.save_monitoring_events([event])
        promotion["event_id"] = event.event_id
        promotion["promotion_id"] = self.store.save_promoted_monitoring_item(promotion)
        return promotion

    def promote_monitoring_cluster(
        self,
        monitoring_result: dict,
        event_id: str,
        *,
        analyst_note: str = "",
        status: str = "new",
    ) -> dict:
        """Promote one live event cluster into persistent monitoring storage."""

        clusters = {cluster.get("event_id"): cluster for cluster in monitoring_result.get("event_clusters", [])}
        cluster = clusters.get(event_id)
        if not cluster:
            raise ValueError(f"Cluster not found in monitoring run: {event_id}")
        signals = {signal.get("signal_id"): signal for signal in monitoring_result.get("normalized_signals", [])}
        supporting = [
            signals[signal_id]
            for signal_id in cluster.get("supporting_signal_ids", [])
            if signal_id in signals
        ]
        duplicate_keys = self._duplicate_signal_keys(monitoring_result.get("normalized_signals", []))
        promotion = self._promotion_from_cluster(
            cluster,
            supporting_signals=supporting,
            monitoring_run_id=monitoring_result.get("monitoring_run_id", ""),
            analyst_note=analyst_note,
            status=status,
            duplicate_keys=duplicate_keys,
        )
        event = self._monitoring_event_from_promotion(promotion)
        self.store.save_monitoring_events([event])
        promotion["event_id"] = event.event_id
        promotion["promotion_id"] = self.store.save_promoted_monitoring_item(promotion)
        return promotion

    def promote_high_risk_clusters(self, monitoring_result: dict, *, analyst_note: str = "") -> list[dict]:
        """Promote all high-risk clusters from the current monitoring run."""

        promoted = []
        for cluster in monitoring_result.get("event_clusters", []):
            if str(cluster.get("risk_level", "")).lower() == "high":
                promoted.append(
                    self.promote_monitoring_cluster(
                        monitoring_result,
                        cluster["event_id"],
                        analyst_note=analyst_note,
                    )
                )
        return promoted

    def update_promoted_monitoring_item(
        self,
        promotion_id: str,
        *,
        status: str | None = None,
        analyst_note: str | None = None,
    ) -> None:
        """Update promoted-item review status or analyst note."""

        self.store.update_promoted_monitoring_item(
            promotion_id,
            status=status,
            analyst_note=analyst_note,
        )

    def build_monitoring_review_queue(self, monitoring_result: dict | None = None, *, limit: int = 200) -> list[dict]:
        """Build a review queue from current live and stored promoted items."""

        rows: list[dict[str, Any]] = []
        monitoring_result = monitoring_result or {}
        signals = monitoring_result.get("normalized_signals", [])
        clusters = monitoring_result.get("event_clusters", [])
        promoted_items = self.list_promoted_monitoring_items(limit=limit)
        promoted_source_ids = {item.get("source_item_id", "") for item in promoted_items}
        duplicate_keys = self._duplicate_signal_keys(signals)

        for signal in signals:
            reasons = self._review_reasons_for_signal(signal, duplicate_keys)
            if not reasons:
                continue
            rows.append(
                {
                    "queue_id": f"signal:{signal.get('signal_id', '')}",
                    "item_type": "signal",
                    "source_item_id": signal.get("signal_id", ""),
                    "title": signal.get("title", ""),
                    "country": signal.get("country", ""),
                    "commodity": signal.get("commodity", ""),
                    "sector": signal.get("sector", ""),
                    "risk_flags": signal.get("risk_flags", ""),
                    "confidence_level": self._confidence_from_score(signal.get("relevance_score", 0.0)),
                    "review_flag": "needs_review",
                    "review_reasons": reasons,
                    "status": "promoted" if signal.get("signal_id", "") in promoted_source_ids else "live_session",
                    "evidence_origin": "current_session_live_signal",
                    "promoted": signal.get("signal_id", "") in promoted_source_ids,
                }
            )

        for cluster in clusters:
            reasons = self._review_reasons_for_cluster(cluster, duplicate_keys)
            if not reasons:
                continue
            rows.append(
                {
                    "queue_id": f"cluster:{cluster.get('event_id', '')}",
                    "item_type": "cluster",
                    "source_item_id": cluster.get("event_id", ""),
                    "title": cluster.get("event_title", ""),
                    "country": cluster.get("countries", ""),
                    "commodity": cluster.get("commodities", ""),
                    "sector": cluster.get("sectors", ""),
                    "risk_flags": cluster.get("risk_flags", ""),
                    "confidence_level": cluster.get("confidence_level", ""),
                    "review_flag": "needs_review",
                    "review_reasons": reasons,
                    "status": "promoted" if cluster.get("event_id", "") in promoted_source_ids else "live_session",
                    "evidence_origin": "current_session_event_cluster",
                    "promoted": cluster.get("event_id", "") in promoted_source_ids,
                }
            )

        for item in promoted_items:
            if item.get("status") in {"reviewed", "archived"}:
                continue
            rows.append(
                {
                    "queue_id": f"promotion:{item.get('promotion_id', '')}",
                    "item_type": item.get("item_type", ""),
                    "source_item_id": item.get("source_item_id", ""),
                    "promotion_id": item.get("promotion_id", ""),
                    "title": item.get("title", ""),
                    "country": item.get("country", ""),
                    "commodity": item.get("commodity", ""),
                    "sector": item.get("sector", ""),
                    "risk_flags": item.get("risk_flags", ""),
                    "confidence_level": item.get("confidence_level", ""),
                    "review_flag": item.get("review_flag", ""),
                    "review_reasons": item.get("review_reasons", []),
                    "status": item.get("status", ""),
                    "evidence_origin": "promoted_monitoring_item",
                    "promoted": True,
                }
            )
        return rows[:limit]

    def _promotion_from_signal(
        self,
        signal: dict[str, Any],
        *,
        monitoring_run_id: str,
        analyst_note: str,
        status: str,
        duplicate_keys: set[tuple[str, str, str]],
    ) -> dict[str, Any]:
        """Build a persistable promotion row from one normalized signal."""

        signal_id = str(signal.get("signal_id", ""))
        reasons = self._review_reasons_for_signal(signal, duplicate_keys)
        confidence_level = self._confidence_from_score(signal.get("relevance_score", 0.0))
        promotion_id = stable_id("monitoring-promotion", monitoring_run_id, "signal", signal_id)
        return {
            "promotion_id": promotion_id,
            "monitoring_run_id": monitoring_run_id,
            "item_type": "signal",
            "source_item_id": signal_id,
            "event_id": stable_id("promoted-monitoring-event", promotion_id),
            "title": signal.get("title", ""),
            "summary": signal.get("summary", ""),
            "country": signal.get("country", ""),
            "region": signal.get("region", self._region_from_country(signal.get("country", ""))),
            "sector": signal.get("sector", ""),
            "commodity": signal.get("commodity", ""),
            "event_type": signal.get("event_type", "monitoring signal"),
            "tone": signal.get("tone", ""),
            "risk_flags": signal.get("risk_flags", ""),
            "relevance_score": float(signal.get("relevance_score", 0.0) or 0.0),
            "confidence_level": confidence_level,
            "source_name": signal.get("source_name", ""),
            "source_type": signal.get("source_type", ""),
            "source_url": signal.get("url", ""),
            "evidence_text": signal.get("evidence_text", ""),
            "supporting_signal_ids": [signal_id] if signal_id else [],
            "retrieved_at": signal.get("retrieved_at", ""),
            "promoted_at": utc_now_iso(),
            "analyst_note": analyst_note,
            "status": status,
            "review_flag": "needs_review" if reasons else "",
            "review_reasons": reasons,
            "payload": signal,
        }

    def _promotion_from_cluster(
        self,
        cluster: dict[str, Any],
        *,
        supporting_signals: list[dict[str, Any]],
        monitoring_run_id: str,
        analyst_note: str,
        status: str,
        duplicate_keys: set[tuple[str, str, str]],
    ) -> dict[str, Any]:
        """Build a persistable promotion row from one event cluster."""

        event_id = str(cluster.get("event_id", ""))
        reasons = self._review_reasons_for_cluster(cluster, duplicate_keys)
        first_signal = supporting_signals[0] if supporting_signals else {}
        relevance = (
            sum(float(signal.get("relevance_score", 0.0) or 0.0) for signal in supporting_signals)
            / max(len(supporting_signals), 1)
        )
        promotion_id = stable_id("monitoring-promotion", monitoring_run_id, "cluster", event_id)
        evidence_text = " ".join(
            part
            for part in [
                cluster.get("what_changed", ""),
                cluster.get("why_it_matters", ""),
                cluster.get("evidence_summary", ""),
            ]
            if part
        )
        return {
            "promotion_id": promotion_id,
            "monitoring_run_id": monitoring_run_id,
            "item_type": "cluster",
            "source_item_id": event_id,
            "event_id": stable_id("promoted-monitoring-event", promotion_id),
            "title": cluster.get("event_title", ""),
            "summary": cluster.get("what_changed", ""),
            "country": cluster.get("countries", ""),
            "region": self._region_from_country(cluster.get("countries", "")),
            "sector": cluster.get("sectors", ""),
            "commodity": cluster.get("commodities", ""),
            "event_type": cluster.get("event_type", "monitoring signal"),
            "tone": self._cluster_tone(supporting_signals),
            "risk_flags": cluster.get("risk_flags", ""),
            "relevance_score": round(relevance, 3),
            "confidence_level": cluster.get("confidence_level", ""),
            "source_name": cluster.get("sources", "") or first_signal.get("source_name", ""),
            "source_type": "event_cluster",
            "source_url": first_signal.get("url", ""),
            "evidence_text": evidence_text,
            "supporting_signal_ids": cluster.get("supporting_signal_ids", []),
            "retrieved_at": cluster.get("generated_at", ""),
            "promoted_at": utc_now_iso(),
            "analyst_note": analyst_note,
            "status": status,
            "review_flag": "needs_review" if reasons else "",
            "review_reasons": reasons,
            "payload": {"cluster": cluster, "supporting_signals": supporting_signals},
        }

    def _monitoring_event_from_promotion(self, promotion: dict[str, Any]) -> MonitoringEvent:
        """Convert a promoted signal or cluster into the canonical event model."""

        text = " ".join(
            [
                str(promotion.get("title", "")),
                str(promotion.get("summary", "")),
                str(promotion.get("evidence_text", "")),
            ]
        )
        risks = split_labels(str(promotion.get("risk_flags", "")))
        return MonitoringEvent(
            event_id=str(promotion.get("event_id", "")),
            source_id="promoted-live-monitoring",
            source_name=str(promotion.get("source_name", "")),
            source_url=str(promotion.get("source_url", "")),
            source_category=f"promoted_live_{promotion.get('item_type', 'item')}",
            title=str(promotion.get("title", "")),
            url=str(promotion.get("source_url", "")),
            published_at=self._published_at_from_promotion(promotion),
            collected_at=str(promotion.get("promoted_at", utc_now_iso())),
            country=str(promotion.get("country", "")),
            region=str(promotion.get("region", "")),
            sector=str(promotion.get("sector", "")),
            commodity=str(promotion.get("commodity", "")),
            actors="",
            event_type=str(promotion.get("event_type", "")),
            outcome=infer_outcome(str(promotion.get("event_type", "")), text, risks),
            sentiment_tone=str(promotion.get("tone", "")),
            risk_flags=str(promotion.get("risk_flags", "")),
            relevance=self._relevance_label(float(promotion.get("relevance_score", 0.0) or 0.0), risks),
            confidence=self._confidence_value(
                str(promotion.get("confidence_level", "")),
                float(promotion.get("relevance_score", 0.0) or 0.0),
            ),
            summary=str(promotion.get("summary", "")),
            recommended_action=self._recommended_action_for_promotion(promotion),
            source_record_id=str(promotion.get("promotion_id", "")),
            status=str(promotion.get("status", "new")),
            raw_text=text,
        )

    def _current_session_events_from_monitoring_result(self, monitoring_result: dict | None) -> list[dict[str, Any]]:
        """Convert current live session results into transient event dictionaries."""

        if not monitoring_result:
            return []
        events = []
        duplicate_keys = self._duplicate_signal_keys(monitoring_result.get("normalized_signals", []))
        for signal in monitoring_result.get("normalized_signals", []):
            promotion = self._promotion_from_signal(
                signal,
                monitoring_run_id=monitoring_result.get("monitoring_run_id", ""),
                analyst_note="",
                status="live_session",
                duplicate_keys=duplicate_keys,
            )
            event = self._monitoring_event_from_promotion(promotion)
            row = vars(event)
            row["evidence_origin"] = "current_session_live_signal"
            row["supporting_signal_ids"] = promotion.get("supporting_signal_ids", [])
            row["review_reasons"] = promotion.get("review_reasons", [])
            events.append(row)
        for cluster in monitoring_result.get("event_clusters", []):
            signals = {
                signal.get("signal_id"): signal
                for signal in monitoring_result.get("normalized_signals", [])
            }
            supporting = [
                signals[signal_id]
                for signal_id in cluster.get("supporting_signal_ids", [])
                if signal_id in signals
            ]
            promotion = self._promotion_from_cluster(
                cluster,
                supporting_signals=supporting,
                monitoring_run_id=monitoring_result.get("monitoring_run_id", ""),
                analyst_note="",
                status="live_session",
                duplicate_keys=duplicate_keys,
            )
            event = self._monitoring_event_from_promotion(promotion)
            row = vars(event)
            row["evidence_origin"] = "current_session_event_cluster"
            row["supporting_signal_ids"] = promotion.get("supporting_signal_ids", [])
            row["review_reasons"] = promotion.get("review_reasons", [])
            events.append(row)
        return events

    def _review_reasons_for_signal(
        self,
        signal: dict[str, Any],
        duplicate_keys: set[tuple[str, str, str]],
    ) -> list[str]:
        """Return transparent reasons why a signal needs analyst review."""

        reasons = []
        risk_flags = split_labels(str(signal.get("risk_flags", "")))
        evidence_text = str(signal.get("evidence_text", ""))
        if risk_flags:
            reasons.append("risk flags present")
        if float(signal.get("relevance_score", 0.0) or 0.0) < 0.35:
            reasons.append("low relevance/confidence score")
        if not signal.get("url"):
            reasons.append("missing source URL")
        if len(evidence_text.strip()) < 60:
            reasons.append("weak or short evidence text")
        if self._signal_duplicate_key(signal) in duplicate_keys:
            reasons.append("possible duplicate country/commodity/event signal")
        if signal.get("event_type") in {"monitoring signal", "", None}:
            reasons.append("uncertain event type")
        if signal.get("tone") not in {"positive", "neutral", "negative", "mixed"}:
            reasons.append("uncertain tone classification")
        return reasons

    def _review_reasons_for_cluster(
        self,
        cluster: dict[str, Any],
        duplicate_keys: set[tuple[str, str, str]],
    ) -> list[str]:
        """Return transparent reasons why a cluster needs analyst review."""

        reasons = []
        if str(cluster.get("risk_level", "")).lower() == "high":
            reasons.append("high risk cluster")
        if str(cluster.get("confidence_level", "")).lower() == "low":
            reasons.append("low confidence cluster")
        if int(cluster.get("source_count", 0) or 0) == 0:
            reasons.append("missing source coverage")
        if len(str(cluster.get("evidence_summary", "")).strip()) < 80:
            reasons.append("weak evidence summary")
        if int(cluster.get("signal_count", 0) or 0) > 1 and int(cluster.get("source_count", 0) or 0) > 1:
            reasons.append("multiple sources mention similar country/commodity/event")
        if cluster.get("risk_flags"):
            reasons.append("risk flags present")
        cluster_key = (
            self._first_label(cluster.get("countries", "")),
            self._first_label(cluster.get("commodities", "")),
            str(cluster.get("event_type", "")),
        )
        if cluster_key in duplicate_keys:
            reasons.append("possible duplicate source-signal cluster")
        return reasons

    def _duplicate_signal_keys(self, signals: list[dict[str, Any]]) -> set[tuple[str, str, str]]:
        """Return repeated country/commodity/event keys from normalized signals."""

        counts = Counter(self._signal_duplicate_key(signal) for signal in signals)
        return {key for key, count in counts.items() if count > 1 and any(key)}

    def _signal_duplicate_key(self, signal: dict[str, Any]) -> tuple[str, str, str]:
        """Return a rough dedupe key for monitoring-signal review."""

        return (
            self._first_label(signal.get("country", "")),
            self._first_label(signal.get("commodity", "")),
            str(signal.get("event_type", "")),
        )

    def _first_label(self, value: Any) -> str:
        """Return the first semicolon/comma-separated value."""

        labels = split_labels(str(value or ""))
        return labels[0] if labels else ""

    def _region_from_country(self, value: Any) -> str:
        """Return a coarse region for the first country label."""

        from devfinintel.knowledge import region_for_country

        country = self._first_label(value)
        return region_for_country(country) if country else "Africa"

    def _cluster_tone(self, supporting_signals: list[dict[str, Any]]) -> str:
        """Summarize tone for a cluster."""

        tones = [str(signal.get("tone", "")) for signal in supporting_signals]
        if "negative" in tones or ("mixed" in tones and len(tones) > 1):
            return "negative"
        if "mixed" in tones:
            return "mixed"
        if "positive" in tones and "neutral" not in tones:
            return "positive"
        return "neutral"

    def _confidence_from_score(self, score: Any) -> str:
        """Convert a relevance score to a simple confidence label."""

        value = float(score or 0.0)
        if value >= 0.7:
            return "high"
        if value >= 0.35:
            return "medium"
        return "low"

    def _confidence_value(self, confidence_level: str, relevance_score: float) -> float:
        """Convert confidence label and relevance score to stored event confidence."""

        if confidence_level == "high":
            return max(0.75, min(0.95, relevance_score))
        if confidence_level == "medium":
            return max(0.5, min(0.74, relevance_score))
        return max(0.25, min(0.49, relevance_score or 0.35))

    def _relevance_label(self, relevance_score: float, risks: list[str]) -> str:
        """Convert numeric relevance and risk flags to event relevance."""

        if relevance_score >= 0.7 or risks:
            return "high"
        if relevance_score >= 0.35:
            return "medium"
        return "low"

    def _recommended_action_for_promotion(self, promotion: dict[str, Any]) -> str:
        """Return a conservative next action for a promoted item."""

        if promotion.get("review_flag"):
            return "review risk flags and evidence before external use"
        if promotion.get("item_type") == "cluster":
            return "add to monitoring digest or country watchlist"
        return "monitor and cite source if used in brief"

    def _published_at_from_promotion(self, promotion: dict[str, Any]) -> str:
        """Find the best source date for a promoted item."""

        payload = promotion.get("payload", {})
        if isinstance(payload, dict):
            if payload.get("date"):
                return str(payload.get("date"))
            cluster = payload.get("cluster")
            if isinstance(cluster, dict) and cluster.get("latest_update"):
                return str(cluster.get("latest_update"))
        return str(promotion.get("retrieved_at") or promotion.get("promoted_at") or "")

    def _source_status_rows(self, source_results: list[dict], *, fallback_used: bool) -> list[dict]:
        """Flatten connector result status without secrets."""

        rows = []
        seen = set()
        for result in source_results:
            name = result.get("source_name", "Unknown source")
            if name in seen and name != "Fallback sample data":
                continue
            seen.add(name)
            rows.append(
                {
                    "source_name": name,
                    "source_type": result.get("source_type", ""),
                    "source_status": result.get("source_status", ""),
                    "records": len(result.get("records", [])),
                    "warnings": "; ".join(result.get("warnings", [])[:2]),
                    "errors": "; ".join(result.get("errors", [])[:2]),
                    "secret_visible": "no",
                }
            )
        configured_names = {row["source_name"] for row in rows}
        for row in self.source_configuration_status():
            source = row["source"]
            if source not in configured_names and source not in {"Fallback sample data"}:
                rows.append(
                    {
                        "source_name": source,
                        "source_type": row.get("source_type", row.get("env_var", "")),
                        "source_status": row.get("status", ""),
                        "records": 0,
                        "warnings": "",
                        "errors": "",
                        "secret_visible": "no",
                    }
                )
        if "Fallback sample data" not in {row["source_name"] for row in rows}:
            rows.append(
                {
                    "source_name": "Fallback sample data",
                    "source_type": "fallback_sample",
                    "source_status": "used" if fallback_used else "not used",
                    "records": 0,
                    "warnings": "",
                    "errors": "",
                    "secret_visible": "no",
                }
            )
        return rows

    def _monitoring_cycle_brief_markdown(self, result: dict) -> str:
        """Build a Markdown brief from clustered developments."""

        lines = [
            "# Africa Energy & Commodities Monitoring Brief",
            "",
            f"- Run timestamp: {result['run_timestamp']}",
            f"- Query: {result['query']}",
            f"- Signals: {result['normalized_signal_count']}",
            f"- Event clusters: {result['event_cluster_count']}",
            f"- Fallback sample data used: {'yes' if result['fallback_used'] else 'no'}",
            "",
            "## Executive Summary",
        ]
        if result["top_developments"]:
            for row in result["top_developments"][:5]:
                lines.append(f"- {row['development']}: {row['what_changed']}")
        else:
            lines.append("- No clustered developments were found in this run.")
        lines.extend(["", "## Top Developments"])
        for cluster in result["event_clusters"][:8]:
            lines.extend(
                [
                    f"### {cluster['event_title']}",
                    f"- What changed: {cluster['what_changed']}",
                    f"- Why it matters: {cluster['why_it_matters']}",
                    f"- Countries: {cluster['countries']}",
                    f"- Commodities/sectors: {cluster['commodities']} / {cluster['sectors']}",
                    f"- Risk level: {cluster['risk_level']}",
                    f"- Confidence: {cluster['confidence_level']}",
                ]
            )
        lines.extend(["", "## Country / Commodity Watchlist"])
        watchlist = {}
        for cluster in result["event_clusters"]:
            key = cluster["countries"] or "Regional Africa"
            watchlist.setdefault(key, 0)
            watchlist[key] += int(cluster["signal_count"])
        if watchlist:
            for country, count in sorted(watchlist.items(), key=lambda item: item[1], reverse=True)[:10]:
                lines.append(f"- {country}: {count} supporting signal(s).")
        else:
            lines.append("- No country watchlist could be produced.")
        lines.extend(["", "## Risk And Opportunity Notes"])
        risks = sorted({cluster["risk_flags"] for cluster in result["event_clusters"] if cluster.get("risk_flags")})
        if risks:
            for risk in risks[:8]:
                lines.append(f"- Review risk flags: {risk}.")
        else:
            lines.append("- No explicit risk flags were detected in the clustered signals.")
        lines.extend(["", "## Source / Evidence Table"])
        for row in result["evidence_table"][:20]:
            lines.append(
                f"- {row['source_name']} | {row['country']} | {row['title']} | {row['url']}"
            )
        lines.extend(["", "## Low-Confidence Or Missing-Data Flags"])
        low_confidence = [cluster for cluster in result["event_clusters"] if cluster["confidence_level"] == "low"]
        if low_confidence:
            lines.append(f"- {len(low_confidence)} cluster(s) have low confidence and need review.")
        if result["warnings"]:
            lines.append(f"- Connector warnings: {'; '.join(result['warnings'][:5])}.")
        if result["errors"]:
            lines.append(f"- Connector errors: {'; '.join(result['errors'][:5])}.")
        if not result["warnings"] and not result["errors"] and not low_confidence:
            lines.append("- No major source-quality flags were produced by the baseline checks.")
        lines.extend(
            [
                "",
                "## Suggested Follow-Up Questions",
                "- What changed this week in Nigeria's oil and fuel market?",
                "- Which countries show energy access or grid-reliability concerns?",
                "- Which critical-minerals signals should become finance/resource records?",
                "- Which clusters need a country-specific official source before external use?",
            ]
        )
        return "\n".join(lines)

    def fetch_news_signals(
        self,
        *,
        query: str,
        country: str = "",
        commodity: str = "",
        topic: str = "",
        limit: int = 20,
    ) -> dict:
        """Return live or sample news signals for the unified workbench."""

        bundle = fetch_news(
            query=query,
            country=country,
            commodity=commodity,
            topic=topic,
            limit=limit,
        )
        self.store.log_event(
            action="news_signals_fetched",
            details={
                "provider": bundle["provider"],
                "status": bundle["status"],
                "query": bundle["query"],
                "articles": len(bundle["articles"]),
                "warning": bundle.get("warning", ""),
            },
        )
        return bundle

    def document_excerpts(
        self,
        *,
        document_ids: list[str] | None = None,
        keyword: str = "",
        limit: int = 25,
    ) -> list[dict]:
        """Return readable document chunks for preview and keyword inspection."""

        allowed = set(document_ids or [])
        chunks = self.store.get_all_chunks()
        documents = self.store.get_document_lookup()
        if document_ids is not None:
            chunks = [chunk for chunk in chunks if chunk.document_id in allowed]
        keyword_lower = keyword.strip().lower()
        if keyword_lower:
            chunks = [chunk for chunk in chunks if keyword_lower in chunk.text.lower()]
        rows = []
        for chunk in chunks[:limit]:
            document = documents.get(chunk.document_id)
            if not document:
                continue
            rows.append(
                {
                    "document_id": chunk.document_id,
                    "title": document.title,
                    "page": chunk.page_number,
                    "chunk": chunk.chunk_index,
                    "excerpt": " ".join(chunk.text.split())[:1200],
                    "source_path": document.source_path,
                }
            )
        return rows

    def dataset_summaries(self, document_ids: list[str] | None = None, limit: int = 10) -> list[dict]:
        """Return compact summaries for selected CSV documents."""

        documents = list(self.store.get_document_lookup().values())
        if document_ids is not None:
            allowed = set(document_ids)
            documents = [document for document in documents if document.document_id in allowed]
        return summarize_csv_documents(documents, limit=limit)

    def answer_workbench_question(
        self,
        *,
        question: str,
        document_ids: list[str] | None = None,
        news_articles: list[dict] | None = None,
        monitoring_result: dict | None = None,
        top_k: int = 8,
        monitoring_limit: int = 100,
    ) -> dict:
        """Answer one question across documents, datasets, news, and monitoring events."""

        document_evidence = self._workbench_document_evidence(
            query=question,
            document_ids=document_ids,
            top_k=top_k,
        )
        answer = build_workbench_answer(
            question=question,
            document_evidence=document_evidence,
            dataset_summaries=self.dataset_summaries(document_ids=document_ids, limit=5),
            news_articles=news_articles or [],
            monitoring_events=self.list_monitoring_events(limit=monitoring_limit),
            normalized_signals=(monitoring_result or {}).get("normalized_signals", []),
            event_clusters=(monitoring_result or {}).get("event_clusters", []),
        )
        self.store.log_event(
            action="workbench_question_answered",
            details={
                "question": question,
                "document_ids": document_ids if document_ids is not None else "all",
                "evidence_rows": len(answer["evidence_rows"]),
                "source_counts": answer["source_counts"],
                "quality_flags": answer["quality_flags"],
            },
        )
        return answer

    def generate_workbench_brief(
        self,
        *,
        focus: str,
        country: str = "",
        commodity: str = "",
        topic: str = "",
        document_ids: list[str] | None = None,
        news_articles: list[dict] | None = None,
        top_k: int = 8,
    ) -> GeneratedOutput:
        """Generate an exportable intelligence brief from all workbench channels."""

        document_evidence = self._workbench_document_evidence(
            query=" ".join(part for part in [focus, country, commodity, topic] if part),
            document_ids=document_ids,
            top_k=top_k,
        )
        monitoring_events = self.list_monitoring_events(limit=150)
        brief = build_intelligence_brief(
            focus=focus,
            country=country,
            commodity=commodity,
            topic=topic,
            document_evidence=document_evidence,
            dataset_summaries=self.dataset_summaries(document_ids=document_ids, limit=5),
            news_articles=news_articles or [],
            monitoring_events=monitoring_events,
            monitoring_insights=self.monitoring_insights(limit=1000),
        )
        evidence_items = evidence_rows_to_items(brief["evidence_rows"], prefix="workbench-brief")
        evidence_pack = self.evidence_pack_builder.build(
            task_type="monitoring_digest",
            query=focus,
            evidence_items=evidence_items,
            records=[],
            diagnostics={
                "workbench_brief": True,
                "document_ids": document_ids if document_ids is not None else "all",
                "news_articles": len(news_articles or []),
                "monitoring_events": len(monitoring_events),
            },
        )
        output = GeneratedOutput(
            task_type="monitoring_digest",
            title=brief["title"],
            body_markdown=brief["markdown"],
            evidence_items=evidence_items,
            records=[],
            verification_findings=[
                VerificationFinding(
                    level="pass" if evidence_items else "review",
                    message=(
                        "Workbench brief generated from visible evidence rows."
                        if evidence_items
                        else "Workbench brief generated with limited evidence; add documents, data, or news."
                    ),
                )
            ],
            metrics={
                "citation_coverage": 1.0 if evidence_items else 0.0,
                "support_overlap": 1.0 if evidence_items else 0.0,
                "unsupported_number_count": 0.0,
                "evidence_items": float(len(evidence_items)),
                "structured_records": 0.0,
                "workbench_brief": 1.0,
                "evidence_pack_id": evidence_pack.pack_id,
            },
            evidence_pack=evidence_pack,
        )
        self.store.log_event(
            action="workbench_brief_generated",
            details={
                "focus": focus,
                "country": country,
                "commodity": commodity,
                "topic": topic,
                "evidence_items": len(evidence_items),
            },
        )
        return output

    def _workbench_document_evidence(
        self,
        *,
        query: str,
        document_ids: list[str] | None,
        top_k: int,
    ):
        """Retrieve document evidence without applying the stricter generation gate."""

        if document_ids == []:
            return []
        index = self.build_index(document_ids=document_ids)
        return index.search(query=query or "Africa energy commodities", task_type="qa", top_k=top_k)

    def role_alignment_report(self) -> dict:
        """Return internship-alignment readiness for RBA/BIOFIN-style work."""

        if not self.list_action_items(limit=1) and self.store.list_knowledge_records(limit=1):
            self.refresh_action_items()
        if not self.list_monitoring_events(limit=1) and self.store.list_knowledge_records(limit=1):
            self.refresh_monitoring_events(from_sources=False, from_knowledge=True, limit=500)
        return build_role_alignment_report(
            knowledge_records=self.store.list_knowledge_records(limit=10000),
            monitoring_events=self.list_monitoring_events(limit=1000),
            action_items=self.list_action_items(limit=1000),
            source_summary=self.source_registry_summary(),
            coverage_summary=self.source_coverage_summary(),
        )

    def download_sources(
        self,
        *,
        limit: int | None = None,
        ingest: bool = True,
        language_hint: str = "English",
    ) -> list[dict]:
        """Download registered official sources and optionally ingest them."""

        if not self.config.source_registry_path.exists():
            self.initialize_sources()
        results = download_registered_sources(
            self.config.source_registry_path,
            self.config.source_download_dir,
            limit=limit,
        )
        downloaded = downloaded_paths(results)
        ingest_summaries: list[dict] = []
        if ingest and downloaded:
            ingest_paths = copy_downloads_to_input(downloaded, self.config.input_dir)
            ingest_summaries = self.ingest_paths(ingest_paths, language_hint=language_hint)

        self.store.log_event(
            action="sources_downloaded",
            details={
                "requested_limit": limit,
                "downloaded": len(downloaded),
                "ingested": len(ingest_summaries),
                "results": [result.__dict__ for result in results],
            },
        )
        summaries = []
        for result in results:
            row = result.__dict__.copy()
            row["ingested"] = any(
                summary.get("source_path", "").endswith(Path(result.local_path).name)
                for summary in ingest_summaries
            )
            summaries.append(row)
        return summaries

    def list_knowledge_records(
        self,
        *,
        record_type: str | None = None,
        review_status: str | None = None,
        country: str | None = None,
        limit: int = 1000,
    ) -> list[dict]:
        """Return reusable knowledge records."""

        return self.store.list_knowledge_records(
            record_type=record_type,
            review_status=review_status,
            country=country,
            limit=limit,
        )

    def update_knowledge_record_status(self, record_id: str, review_status: str) -> None:
        """Mark a knowledge record as reviewed, usable, or rejected."""

        self.store.update_knowledge_record_status(record_id, review_status)

    def knowledge_coverage_summary(self) -> dict:
        """Return cross-run country/theme/sector coverage metrics."""

        return knowledge_record_summary(self.store.list_knowledge_records(limit=10000))

    def source_coverage_matrix(self) -> list[dict]:
        """Return the country-by-topic evidence coverage matrix."""

        return build_coverage_matrix(
            source_entries=load_source_registry(self.config.source_registry_path),
            knowledge_records=self.store.list_knowledge_records(limit=10000),
        )

    def source_coverage_summary(self) -> dict:
        """Return summary metrics for the country-by-topic coverage matrix."""

        return matrix_coverage_summary(self.source_coverage_matrix())

    def source_backlog(self, limit: int = 50) -> list[dict]:
        """Return prioritized source and review gaps."""

        return source_backlog(self.source_coverage_matrix(), limit=limit)

    def refresh_action_items(self, limit: int = 500) -> list[dict]:
        """Refresh the analyst action queue from records and coverage gaps."""

        actions = build_action_items(
            knowledge_records=self.store.list_knowledge_records(limit=10000),
            source_backlog_rows=self.source_backlog(limit=100),
            limit=limit,
        )
        self.store.sync_action_items(actions)
        return self.store.list_action_items(limit=limit)

    def list_action_items(
        self,
        *,
        country: str | None = None,
        status: str | None = None,
        priority: str | None = None,
        limit: int = 500,
    ) -> list[dict]:
        """Return analyst follow-up actions."""

        return self.store.list_action_items(
            country=country,
            status=status,
            priority=priority,
            limit=limit,
        )

    def update_action_item_status(self, action_id: str, status: str) -> None:
        """Update an analyst action status."""

        self.store.update_action_item_status(action_id, status)

    def country_intelligence(self, country: str) -> dict:
        """Return country snapshot, actions, stakeholders, and signal comparison."""

        actions = self.list_action_items(country=country, limit=500)
        if not actions:
            self.refresh_action_items()
            actions = self.list_action_items(country=country, limit=500)
        return build_country_intelligence(
            country=country,
            knowledge_records=self.store.list_knowledge_records(limit=10000),
            coverage_matrix=self.source_coverage_matrix(),
            action_items=actions,
        )

    def _resolve_document_ids(self, document_ids: list[str] | None) -> list[str]:
        """Return concrete document IDs for a scoped or full-library run."""

        if document_ids is not None:
            return list(document_ids)
        return [document.document_id for document in self.store.get_document_lookup().values()]

    def _scope_label(self, document_ids: list[str] | None) -> str:
        """Human-readable scope label for sessions."""

        if document_ids is None:
            return "full_evidence_library"
        if len(document_ids) == 1:
            return "single_document"
        return "selected_documents"

    def _create_session_id(self, task_type: str, query: str, document_ids: list[str]) -> str:
        """Create a stable-ish run ID with a timestamp component."""

        return stable_id(task_type, query, ",".join(document_ids), utc_now_iso())

    def _save_session_for_output(
        self,
        *,
        session_id: str,
        task_type: str,
        query: str,
        document_ids: list[str],
        scope_label: str,
        status: str,
        output: GeneratedOutput,
        diagnostics: dict,
    ) -> None:
        """Persist one coherent analysis session."""

        self.store.save_analysis_session(
            AnalysisSession(
                session_id=session_id,
                title=output.title,
                task_type=task_type,
                query=query,
                document_ids=document_ids,
                scope_label=scope_label,
                created_at=utc_now_iso(),
                status=status,
                diagnostics=diagnostics,
            )
        )

    def _abstention_output(
        self,
        task_type: str,
        query: str,
        diagnostics: RetrievalDiagnostics,
    ) -> GeneratedOutput:
        """Return a safe output when retrieval is too weak."""

        warnings = "\n".join(f"- {warning}" for warning in diagnostics.warnings)
        evidence_pack = self.evidence_pack_builder.build(
            task_type=task_type,
            query=query,
            evidence_items=[],
            records=[],
            diagnostics=diagnostics.as_dict(),
        )
        body = (
            f"# Not Enough Evidence: {query}\n\n"
            "The system did not find enough relevant evidence in the selected files to answer safely.\n\n"
            "## Retrieval Diagnostics\n"
            f"- Documents in scope: {diagnostics.document_scope_count}\n"
            f"- Chunks searched: {diagnostics.chunk_count}\n"
            f"- Evidence items returned: {diagnostics.returned_count}\n"
            f"- Top retrieval score: {diagnostics.top_score}\n"
            f"- Query keyword coverage: {diagnostics.keyword_coverage}\n\n"
            "## Warnings\n"
            f"{warnings if warnings else '- Retrieval did not pass the safety gate.'}\n\n"
            "Try selecting a different document, using more specific terms, or ingesting a source that directly contains the answer."
        )
        return GeneratedOutput(
            task_type=task_type,
            title=f"Not Enough Evidence: {query}",
            body_markdown=body,
            evidence_items=[],
            records=[],
            verification_findings=[
                VerificationFinding(
                    level="error",
                    message="Retrieval did not pass the evidence-quality gate.",
                )
            ],
            metrics={
                "citation_coverage": 0.0,
                "support_overlap": 0.0,
                "unsupported_number_count": 0.0,
                "evidence_items": 0.0,
                "structured_records": 0.0,
                "retrieval_passed": 0.0,
                "retrieval_top_score": diagnostics.top_score,
                "retrieval_keyword_coverage": diagnostics.keyword_coverage,
                "evidence_pack_id": evidence_pack.pack_id,
            },
            evidence_pack=evidence_pack,
        )
