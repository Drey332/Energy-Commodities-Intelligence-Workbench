"""Tests for connector-backed monitoring architecture."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from devfinintel.config import ProjectConfig
from devfinintel.connectors.gdelt import fetch_gdelt_signals
from devfinintel.connectors.reliefweb import fetch_reliefweb_signals
from devfinintel.env import load_project_env, source_key_status
from devfinintel.events import cluster_signals
from devfinintel.pipeline import DocumentIntelligencePipeline
from devfinintel.signals import normalize_signal


class MonitoringCycleTest(unittest.TestCase):
    def make_pipeline(self, root: Path) -> DocumentIntelligencePipeline:
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
        return DocumentIntelligencePipeline(config=config)

    def test_env_loading_and_key_status_do_not_expose_values(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_path = root / ".env"
            fake_key = "dummytestvalue"
            env_path.write_text(
                f"{'NEWSAPI_API_KEY'}={fake_key}\n"
                "DEVFIN_NEWS_LOOKBACK_DAYS=14\n",
                encoding="utf-8",
            )
            old_value = os.environ.get("NEWSAPI_API_KEY")
            try:
                os.environ.pop("NEWSAPI_API_KEY", None)
                load_project_env(root)
                self.assertEqual(os.environ.get("NEWSAPI_API_KEY"), fake_key)
                statuses = source_key_status()
                rendered = str(statuses)
                self.assertIn("configured", rendered)
                self.assertNotIn(fake_key, rendered)
            finally:
                if old_value is None:
                    os.environ.pop("NEWSAPI_API_KEY", None)
                else:
                    os.environ["NEWSAPI_API_KEY"] = old_value

    def test_keyless_connectors_fail_gracefully(self) -> None:
        def failing_fetcher(url: str, **kwargs):
            raise RuntimeError("offline")

        gdelt = fetch_gdelt_signals(query="Africa energy", fetcher=failing_fetcher)
        reliefweb = fetch_reliefweb_signals(query="Africa conflict energy", fetcher=failing_fetcher)
        self.assertEqual(gdelt["source_status"], "failed")
        self.assertEqual(reliefweb["source_status"], "failed")
        self.assertTrue(gdelt["errors"])
        self.assertTrue(reliefweb["errors"])

    def test_signal_normalization_classifies_domain_fields(self) -> None:
        signal = normalize_signal(
            {
                "title": "Nigeria fuel prices and refinery regulation raise inflation risk",
                "date": "2026-07-01",
                "url": "https://example.org/nigeria-fuel",
                "summary": "Oil market pressure, regulation, and fuel imports are affecting energy security.",
                "evidence_text": "Nigeria fuel prices, oil refinery regulation, inflation, and imports.",
            },
            source_type="news",
            source_name="Test Source",
            source_status="live/keyless",
        )
        self.assertEqual(signal["country"], "Nigeria")
        self.assertIn("Oil", signal["commodity"])
        self.assertIn(signal["event_type"], {"regulation", "market pressure"})
        self.assertGreater(signal["relevance_score"], 0.5)

    def test_event_clustering_groups_related_signals(self) -> None:
        signals = [
            normalize_signal(
                {
                    "title": "Nigeria fuel prices rise after refinery import changes",
                    "date": "2026-07-01",
                    "url": "https://example.org/1",
                    "summary": "Fuel prices and oil imports are under pressure.",
                },
                source_type="news",
                source_name="Source A",
                source_status="live/keyless",
            ),
            normalize_signal(
                {
                    "title": "Nigeria refinery regulation affects fuel imports",
                    "date": "2026-07-02",
                    "url": "https://example.org/2",
                    "summary": "Regulation and refinery operations affect refined fuel supply.",
                },
                source_type="news",
                source_name="Source B",
                source_status="live/keyless",
            ),
        ]
        clusters = cluster_signals(signals)
        self.assertTrue(clusters)
        self.assertGreaterEqual(clusters[0]["signal_count"], 1)
        self.assertIn("Nigeria", clusters[0]["countries"])

    def test_monitoring_cycle_output_shape_with_sample_fallback(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline = self.make_pipeline(root)
            result = pipeline.run_monitoring_cycle(
                query="Africa energy commodities",
                use_live_connectors=False,
                include_optional_key_sources=False,
                include_indicators=False,
                limit=5,
            )
            self.assertTrue(result["fallback_used"])
            self.assertGreater(result["normalized_signal_count"], 0)
            self.assertGreater(result["event_cluster_count"], 0)
            self.assertTrue(result["evidence_table"])
            self.assertIn("Executive Summary", result["brief_markdown"])

    def test_monitoring_run_metadata_is_persisted(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = self.make_pipeline(Path(tmp))
            result = pipeline.run_monitoring_cycle(
                query="Africa energy commodities",
                use_live_connectors=False,
                include_optional_key_sources=False,
                include_indicators=False,
                limit=5,
            )
            runs = pipeline.list_monitoring_runs()
            self.assertTrue(runs)
            self.assertEqual(runs[0]["monitoring_run_id"], result["monitoring_run_id"])
            self.assertGreaterEqual(runs[0]["signal_count"], 1)

    def test_promote_signal_persists_event_and_promotion(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = self.make_pipeline(Path(tmp))
            result = pipeline.run_monitoring_cycle(
                query="Africa energy commodities",
                use_live_connectors=False,
                include_optional_key_sources=False,
                include_indicators=False,
                limit=5,
            )
            signal_id = result["normalized_signals"][0]["signal_id"]
            promoted = pipeline.promote_monitoring_signal(result, signal_id, analyst_note="track this")
            self.assertEqual(promoted["item_type"], "signal")
            self.assertEqual(promoted["monitoring_run_id"], result["monitoring_run_id"])
            self.assertTrue(pipeline.list_promoted_monitoring_items())
            events = pipeline.list_monitoring_events(limit=20)
            self.assertTrue(any(event["event_id"] == promoted["event_id"] for event in events))

    def test_promote_cluster_persists_event_and_supporting_ids(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = self.make_pipeline(Path(tmp))
            result = pipeline.run_monitoring_cycle(
                query="Africa energy commodities",
                use_live_connectors=False,
                include_optional_key_sources=False,
                include_indicators=False,
                limit=5,
            )
            cluster = result["event_clusters"][0]
            promoted = pipeline.promote_monitoring_cluster(result, cluster["event_id"])
            self.assertEqual(promoted["item_type"], "cluster")
            self.assertTrue(promoted["supporting_signal_ids"])
            stored = pipeline.list_promoted_monitoring_items()
            self.assertEqual(stored[0]["source_item_id"], cluster["event_id"])

    def test_supervisor_can_include_or_exclude_current_session_results(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = self.make_pipeline(Path(tmp))
            result = pipeline.run_monitoring_cycle(
                query="Africa energy commodities",
                use_live_connectors=False,
                include_optional_key_sources=False,
                include_indicators=False,
                limit=5,
            )
            stored_only = pipeline.run_monitoring_agent(
                from_sources=False,
                from_knowledge=False,
                monitoring_result=result,
                include_current_live_results=False,
            )
            with_current = pipeline.run_monitoring_agent(
                from_sources=False,
                from_knowledge=False,
                monitoring_result=result,
                include_current_live_results=True,
            )
            self.assertEqual(stored_only["evidence_pool"]["current_session_events_used"], 0)
            self.assertGreater(with_current["evidence_pool"]["current_session_events_used"], 0)
            self.assertGreater(
                with_current["evidence_pool"]["total_events_used"],
                stored_only["evidence_pool"]["total_events_used"],
            )

    def test_review_queue_flags_current_session_items(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = self.make_pipeline(Path(tmp))
            result = pipeline.run_monitoring_cycle(
                query="Africa energy commodities",
                use_live_connectors=False,
                include_optional_key_sources=False,
                include_indicators=False,
                limit=5,
            )
            queue = pipeline.build_monitoring_review_queue(result)
            self.assertTrue(queue)
            self.assertTrue(any(row["review_reasons"] for row in queue))


if __name__ == "__main__":
    unittest.main()
