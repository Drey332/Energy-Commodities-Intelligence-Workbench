"""End-to-end tests for the document intelligence pipeline."""

from __future__ import annotations

import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from devfinintel.config import ProjectConfig
from devfinintel.llm import build_evidence_pack
from devfinintel.models import EvidenceItem, ExtractionRecord
from devfinintel.knowledge import KnowledgeRecord
from devfinintel.pipeline import DocumentIntelligencePipeline
from devfinintel.sources import SourceRegistryEntry, save_source_registry
from devfinintel.store import SQLiteDocumentStore


class PipelineTest(unittest.TestCase):
    def test_ingest_generate_and_export(self) -> None:
        """The core pipeline should ingest evidence, generate a brief, and export files."""

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = ProjectConfig(
                root_dir=root,
                data_dir=root / "data",
                input_dir=root / "data" / "input",
                news_dir=root / "data" / "news",
                finance_dir=root / "data" / "finance",
                sources_dir=root / "data" / "sources",
                source_download_dir=root / "data" / "sources" / "downloads",
                source_registry_path=root / "data" / "sources" / "source_registry.csv",
                monitoring_source_registry_path=root / "data" / "sources" / "monitoring_sources.csv",
                country_coverage_path=root / "data" / "sources" / "africa_country_coverage.csv",
                storage_dir=root / "storage",
                output_dir=root / "outputs",
                database_path=root / "storage" / "workbench.sqlite",
            )
            config.ensure_directories()
            source = config.input_dir / "sample.md"
            source.write_text(
                """
                In 2025, an IFI announced US$250 million for Kenya and Ghana.
                The partnership focused on biodiversity finance, private sector
                financing, green bond preparation, and conservation trust fund design.
                Lessons learned included aligning public budgets with local
                conservation priorities before launching blended finance.
                """,
                encoding="utf-8",
            )

            pipeline = DocumentIntelligencePipeline(config=config)
            summaries = pipeline.ingest_paths([source], language_hint="English")
            self.assertEqual(len(summaries), 1)
            self.assertGreater(summaries[0]["chunks"], 0)

            output = pipeline.generate(
                task_type="biofin_case",
                query="green bond biodiversity finance Kenya Ghana",
                top_k=3,
            )
            self.assertGreater(len(output.evidence_items), 0)
            self.assertGreater(len(output.records), 0)
            self.assertIn("BIOFIN", output.title)
            self.assertIn("p. 1", output.body_markdown)
            self.assertTrue(pipeline.list_knowledge_records())

            paths = pipeline.export_output(output)
            self.assertTrue(paths.markdown_path.exists())
            self.assertTrue(paths.csv_path.exists())
            self.assertTrue(paths.pdf_path.exists())
            self.assertTrue(paths.evidence_json_path.exists())
            self.assertTrue(paths.manifest_json_path.exists())
            self.assertTrue(paths.package_path.exists())
            self.assertIn("evidence_pack_id", output.metrics)

    def test_scoped_generation_and_dataset_profile(self) -> None:
        """Generation can be limited to selected documents and can profile CSVs."""

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = ProjectConfig(
                root_dir=root,
                data_dir=root / "data",
                input_dir=root / "data" / "input",
                news_dir=root / "data" / "news",
                finance_dir=root / "data" / "finance",
                sources_dir=root / "data" / "sources",
                source_download_dir=root / "data" / "sources" / "downloads",
                source_registry_path=root / "data" / "sources" / "source_registry.csv",
                monitoring_source_registry_path=root / "data" / "sources" / "monitoring_sources.csv",
                country_coverage_path=root / "data" / "sources" / "africa_country_coverage.csv",
                storage_dir=root / "storage",
                output_dir=root / "outputs",
                database_path=root / "storage" / "workbench.sqlite",
            )
            config.ensure_directories()
            narrative = config.input_dir / "sample.md"
            narrative.write_text(
                "In 2025, a sample donor announced US$250 million for Ghana biodiversity finance.",
                encoding="utf-8",
            )
            csv_file = config.input_dir / "indicator.csv"
            csv_file.write_text(
                "country,value,TIME_PERIOD,INDICATOR_LABEL\n"
                "Kenya,0.20,2023,AI regulation\n"
                "Ghana,0.10,2023,AI regulation\n"
                "Ethiopia,0.05,2023,AI regulation\n",
                encoding="utf-8",
            )

            pipeline = DocumentIntelligencePipeline(config=config)
            narrative_summary = pipeline.ingest_paths([narrative], language_hint="English")[0]
            csv_summary = pipeline.ingest_paths([csv_file], language_hint="English")[0]

            scoped_output = pipeline.generate(
                task_type="qa",
                query="what does the indicator file say",
                top_k=3,
                document_ids=[csv_summary["document_id"]],
            )
            self.assertTrue(scoped_output.evidence_items)
            self.assertTrue(
                all(item.document_id == csv_summary["document_id"] for item in scoped_output.evidence_items)
            )
            self.assertNotIn(narrative_summary["document_id"], [item.document_id for item in scoped_output.evidence_items])

            profile = pipeline.generate(
                task_type="dataset_profile",
                query="what does the uploaded file tell me",
                document_ids=[csv_summary["document_id"]],
            )
            self.assertIn("Rows: 3", profile.body_markdown)
            self.assertIn("Highest Values", profile.body_markdown)
            self.assertIn("Suggested Questions", profile.body_markdown)
            self.assertIn("column_profiles", profile.records[0].fields)
            self.assertTrue(profile.evidence_pack)
            self.assertEqual(profile.records[0].fields["_schema_valid"], True)

            manifests = pipeline.list_file_manifests(document_ids=[csv_summary["document_id"]])
            self.assertEqual(len(manifests), 1)
            self.assertEqual(manifests[0]["row_count"], 3)
            self.assertEqual(manifests[0]["column_count"], 4)
            self.assertTrue(pipeline.knowledge_coverage_summary()["knowledge_records"])

            sessions = pipeline.store.list_analysis_sessions()
            self.assertTrue(sessions)

            abstention = pipeline.generate(
                task_type="qa",
                query="what does this say about climate finance?",
                document_ids=[],
            )
            self.assertIn("Not Enough Evidence", abstention.title)
            self.assertEqual(abstention.metrics["retrieval_passed"], 0.0)

    def test_source_registry_download_and_ingest(self) -> None:
        """A registry source can be downloaded, snapshotted, and ingested."""

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = ProjectConfig(
                root_dir=root,
                data_dir=root / "data",
                input_dir=root / "data" / "input",
                news_dir=root / "data" / "news",
                finance_dir=root / "data" / "finance",
                sources_dir=root / "data" / "sources",
                source_download_dir=root / "data" / "sources" / "downloads",
                source_registry_path=root / "data" / "sources" / "source_registry.csv",
                monitoring_source_registry_path=root / "data" / "sources" / "monitoring_sources.csv",
                country_coverage_path=root / "data" / "sources" / "africa_country_coverage.csv",
                storage_dir=root / "storage",
                output_dir=root / "outputs",
                database_path=root / "storage" / "workbench.sqlite",
            )
            config.ensure_directories()
            pipeline = DocumentIntelligencePipeline(config=config)

            registry_path, coverage_path = pipeline.initialize_sources()
            self.assertTrue(registry_path.exists())
            self.assertTrue(coverage_path.exists())

            source_page = root / "official_source.html"
            source_page.write_text(
                "<html><body><h1>Africa energy source</h1>"
                "<p>Kenya and Ghana energy infrastructure investment and mining value chains.</p>"
                "</body></html>",
                encoding="utf-8",
            )
            save_source_registry(
                config.source_registry_path,
                [
                    SourceRegistryEntry(
                        source_id="local-official-source",
                        title="Local Official Source",
                        publisher="Test Publisher",
                        year="2026",
                        url=source_page.as_uri(),
                        source_type="web_page",
                        topics="energy; mining; infrastructure",
                        countries="Kenya; Ghana",
                        regions="Africa",
                        license_note="Test fixture.",
                    )
                ],
            )

            results = pipeline.download_sources(limit=1, ingest=True)
            self.assertEqual(results[0]["status"], "downloaded")
            self.assertTrue(Path(results[0]["local_path"]).exists())
            self.assertTrue(pipeline.store.list_documents())

            matrix = pipeline.source_coverage_matrix()
            self.assertTrue(matrix)
            summary = pipeline.source_coverage_summary()
            self.assertEqual(summary["countries"], 54)
            self.assertGreater(summary["topics"], 0)
            kenya_energy = [
                row
                for row in matrix
                if row["country"] == "Kenya" and row["topic_id"] == "energy_access"
            ][0]
            self.assertIn(kenya_energy["status"], {"specific_source_ready", "usable_records"})
            self.assertTrue(pipeline.source_backlog(limit=5))

    def test_monitoring_intelligence_fields_are_populated(self) -> None:
        """Monitoring digests should create reviewable intelligence fields."""

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = ProjectConfig(
                root_dir=root,
                data_dir=root / "data",
                input_dir=root / "data" / "input",
                news_dir=root / "data" / "news",
                finance_dir=root / "data" / "finance",
                sources_dir=root / "data" / "sources",
                source_download_dir=root / "data" / "sources" / "downloads",
                source_registry_path=root / "data" / "sources" / "source_registry.csv",
                monitoring_source_registry_path=root / "data" / "sources" / "monitoring_sources.csv",
                country_coverage_path=root / "data" / "sources" / "africa_country_coverage.csv",
                storage_dir=root / "storage",
                output_dir=root / "outputs",
                database_path=root / "storage" / "workbench.sqlite",
            )
            config.ensure_directories()
            source = config.input_dir / "monitoring.md"
            source.write_text(
                """
                In 2026, the World Bank approved US$1 billion financing for
                Kenya energy infrastructure and power-sector investment. Civil
                society warned about debt, governance, climate, and community
                social-license risks after a project delay.
                """,
                encoding="utf-8",
            )

            pipeline = DocumentIntelligencePipeline(config=config)
            pipeline.ingest_paths([source], language_hint="English")
            pipeline.generate(
                task_type="monitoring_digest",
                query="Kenya energy infrastructure financing project delay governance risk",
                top_k=3,
            )
            records = pipeline.list_knowledge_records(record_type="monitoring_digest", limit=5)
            self.assertTrue(records)
            record = records[0]
            self.assertEqual(record["relevance"], "high")
            self.assertIn(record["sentiment_tone"], {"mixed", "negative"})
            self.assertTrue(record["actors"])
            self.assertTrue(record["event_type"])
            self.assertIn("Governance", record["risk_flags"])
            self.assertEqual(record["recommended_action"], "add to bulletin and review risk flags")
            actions = pipeline.refresh_action_items(limit=50)
            self.assertTrue(actions)
            kenya_actions = pipeline.list_action_items(country="Kenya", limit=10)
            self.assertTrue(kenya_actions)
            self.assertIn(kenya_actions[0]["priority"], {"urgent", "high", "medium"})
            self.assertEqual(kenya_actions[0]["status"], "open")
            intelligence = pipeline.country_intelligence("Kenya")
            self.assertEqual(intelligence["snapshot"]["country"], "Kenya")
            self.assertGreaterEqual(intelligence["snapshot"]["open_actions"], 1)
            self.assertTrue(intelligence["stakeholders"])
            self.assertTrue(intelligence["comparisons"])

            monitor_path = pipeline.initialize_monitoring()
            self.assertTrue(monitor_path.exists())
            refresh = pipeline.refresh_monitoring_events(from_sources=False, from_knowledge=True, limit=50)
            self.assertGreaterEqual(refresh["events_saved"], 1)
            events = pipeline.list_monitoring_events(country="Kenya", limit=10)
            self.assertTrue(events)
            self.assertEqual(events[0]["country"], "Kenya")
            self.assertTrue(events[0]["outcome"])
            self.assertIn(events[0]["sentiment_tone"], {"mixed", "negative", "positive", "neutral"})
            insights = pipeline.monitoring_insights(limit=50)
            self.assertGreaterEqual(insights["snapshot"]["events"], 1)
            self.assertTrue(insights["country_rows"])
            self.assertTrue(insights["insight_cards"])
            agent_run = pipeline.run_monitoring_agent(from_sources=False, from_knowledge=True, limit=50)
            self.assertTrue(agent_run["briefing"])
            self.assertTrue(agent_run["source_health"])
            self.assertTrue(agent_run["triage_queue"])
            self.assertTrue(agent_run["next_actions"])
            alignment = pipeline.role_alignment_report()
            self.assertGreaterEqual(alignment["summary"]["requirements"], 10)
            self.assertTrue(alignment["role_scores"])
            self.assertTrue(alignment["requirements"])
            self.assertTrue(alignment["next_build_priorities"])

    def test_knowledge_store_migration_uses_named_columns(self) -> None:
        """Saving records after an old schema migration must not shift columns."""

        with TemporaryDirectory() as tmp:
            database_path = Path(tmp) / "workbench.sqlite"
            with sqlite3.connect(database_path) as connection:
                connection.executescript(
                    """
                    CREATE TABLE knowledge_records (
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
                    """
                )

            store = SQLiteDocumentStore(database_path)
            store.save_knowledge_records(
                [
                    KnowledgeRecord(
                        record_id="record-1",
                        record_type="monitoring_digest",
                        title="Kenya energy financing",
                        country="Kenya",
                        region="East Africa",
                        sector="Energy",
                        theme="Energy access",
                        commodity="",
                        partner="World Bank",
                        amount="US$1 billion",
                        currency="USD",
                        instrument="Loan",
                        event_date="2026",
                        relevance="high",
                        actors="IFI; Government",
                        event_type="Financing approved",
                        sentiment_tone="mixed",
                        risk_flags="Debt; Governance",
                        recommended_action="add to bulletin and review risk flags",
                        source_document_id="doc-1",
                        source_title="Source title",
                        source_page=7,
                        source_path="/tmp/source.pdf",
                        evidence_chunk_ids=["chunk-1"],
                        fields={"summary": "World Bank approved financing with debt risk."},
                        confidence=0.91,
                        review_status="usable",
                        created_at="2026-07-01T00:00:00+00:00",
                    )
                ]
            )
            row = store.list_knowledge_records(limit=1)[0]
            self.assertEqual(row["source_document_id"], "doc-1")
            self.assertEqual(row["source_title"], "Source title")
            self.assertEqual(row["source_page"], 7)
            self.assertEqual(row["source_path"], "/tmp/source.pdf")
            self.assertEqual(row["relevance"], "high")
            self.assertEqual(row["event_type"], "Financing approved")
            self.assertEqual(row["evidence_chunk_ids"], ["chunk-1"])
            self.assertEqual(row["fields"]["summary"], "World Bank approved financing with debt risk.")

    def test_llm_evidence_pack_labels_sources(self) -> None:
        """The local LLM prompt receives labelled evidence instead of raw files."""

        evidence = [
            EvidenceItem(
                chunk_id="chunk-1",
                document_id="doc-1",
                title="indicator",
                source_path="/tmp/indicator.csv",
                page_number=1,
                text="Kenya has value 0.20 in 2023.",
                bm25_score=1.0,
                dense_score=0.5,
                rerank_score=0.9,
            )
        ]
        records = [
            ExtractionRecord(
                record_id="record-1",
                record_type="dataset_profile",
                title="Dataset profile",
                fields={"highest_values": [{"label": "Kenya", "value": 0.20}]},
                evidence_chunk_ids=["chunk-1"],
                confidence=0.95,
                review_status="usable",
            )
        ]
        pack = build_evidence_pack(evidence, records)
        self.assertIn("[E1]", pack)
        self.assertIn("[R1]", pack)
        self.assertIn("Kenya", pack)


if __name__ == "__main__":
    unittest.main()
