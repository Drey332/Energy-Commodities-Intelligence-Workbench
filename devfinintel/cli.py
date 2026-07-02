"""Command-line entrypoint for the workbench.

Examples:

    python -m devfinintel.cli ingest data/input
    python -m devfinintel.cli generate --task donor_profile --query "AfDB climate finance Africa"

The CLI is useful for reproducibility: a reviewer can rerun the same command and
compare the generated output files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from devfinintel.pipeline import DocumentIntelligencePipeline
from devfinintel.planner import TASK_LABELS


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Development Finance Intelligence Workbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Parse files and store searchable evidence.")
    ingest_parser.add_argument("paths", nargs="+", help="Files or folders to ingest.")
    ingest_parser.add_argument("--language", default="unknown", help="Optional language hint, e.g. English/French.")

    generate_parser = subparsers.add_parser("generate", help="Generate an evidence-grounded work product.")
    generate_parser.add_argument(
        "--task",
        choices=sorted(TASK_LABELS),
        default="qa",
        help="Type of work product to generate.",
    )
    generate_parser.add_argument("--query", required=True, help="Plain-language request or profile topic.")
    generate_parser.add_argument("--top-k", type=int, default=8, help="Number of evidence chunks to retrieve.")
    generate_parser.add_argument(
        "--document-id",
        action="append",
        default=None,
        help="Optional document ID to limit retrieval. Can be passed more than once.",
    )

    subparsers.add_parser("documents", help="List ingested documents.")
    subparsers.add_parser("audit", help="Show recent audit events.")

    sources_parser = subparsers.add_parser("sources", help="Manage the official source registry.")
    sources_parser.add_argument(
        "--init",
        action="store_true",
        help="Create the seed source registry and Africa country coverage roster.",
    )
    sources_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing source registry files when used with --init.",
    )

    download_parser = subparsers.add_parser("download-sources", help="Download and ingest registered sources.")
    download_parser.add_argument("--limit", type=int, default=3, help="Maximum registry rows to download.")
    download_parser.add_argument(
        "--no-ingest",
        action="store_true",
        help="Download files but do not copy them into data/input and ingest them.",
    )

    knowledge_parser = subparsers.add_parser("knowledge", help="List reusable knowledge records.")
    knowledge_parser.add_argument("--record-type", default=None, help="Optional knowledge record type filter.")
    knowledge_parser.add_argument("--country", default=None, help="Optional country filter.")
    knowledge_parser.add_argument("--review-status", default=None, help="Optional review status filter.")
    knowledge_parser.add_argument("--limit", type=int, default=100, help="Maximum records to print.")

    coverage_parser = subparsers.add_parser("coverage", help="Show country-topic coverage and source gaps.")
    coverage_parser.add_argument("--country", default=None, help="Optional country filter.")
    coverage_parser.add_argument("--topic", default=None, help="Optional topic ID or label filter.")
    coverage_parser.add_argument("--backlog", action="store_true", help="Show prioritized gaps instead of the full matrix.")
    coverage_parser.add_argument("--limit", type=int, default=50, help="Maximum rows to print.")

    actions_parser = subparsers.add_parser("actions", help="Show recommended analyst follow-up actions.")
    actions_parser.add_argument("--country", default=None, help="Optional country filter.")
    actions_parser.add_argument("--status", default=None, help="Optional action status filter.")
    actions_parser.add_argument("--priority", default=None, help="Optional priority filter.")
    actions_parser.add_argument("--refresh", action="store_true", help="Refresh derived actions before listing.")
    actions_parser.add_argument("--limit", type=int, default=50, help="Maximum rows to print.")

    country_parser = subparsers.add_parser("country", help="Show a country intelligence workspace summary.")
    country_parser.add_argument("--country", required=True, help="Country name to inspect.")
    country_parser.add_argument("--refresh-actions", action="store_true", help="Refresh action items first.")
    country_parser.add_argument("--limit", type=int, default=12, help="Maximum rows per section.")

    monitor_parser = subparsers.add_parser("monitor", help="Refresh and inspect monitoring intelligence events.")
    monitor_parser.add_argument("--init", action="store_true", help="Create the governed monitoring-source registry.")
    monitor_parser.add_argument("--overwrite", action="store_true", help="Overwrite the monitoring-source registry.")
    monitor_parser.add_argument("--refresh", action="store_true", help="Refresh monitoring events.")
    monitor_parser.add_argument("--agent", action="store_true", help="Run the monitoring supervisor and print a situation brief.")
    monitor_parser.add_argument(
        "--live-sources",
        action="store_true",
        help="Fetch configured RSS/Atom sources as well as local reviewed knowledge records.",
    )
    monitor_parser.add_argument("--country", default=None, help="Optional country filter.")
    monitor_parser.add_argument("--sector", default=None, help="Optional sector filter.")
    monitor_parser.add_argument("--status", default=None, help="Optional event status filter.")
    monitor_parser.add_argument("--limit", type=int, default=50, help="Maximum rows to print.")

    alignment_parser = subparsers.add_parser("alignment", help="Show RBA/BIOFIN role-readiness alignment.")
    alignment_parser.add_argument("--limit", type=int, default=50, help="Maximum requirement rows to print.")

    args = parser.parse_args()
    pipeline = DocumentIntelligencePipeline()

    if args.command == "ingest":
        files = collect_files([Path(path) for path in args.paths])
        summaries = pipeline.ingest_paths(files, language_hint=args.language)
        for summary in summaries:
            print(
                f"Ingested {summary['title']}: {summary['pages']} pages, "
                f"{summary['chunks']} chunks, parser={summary['parser_backend']}"
            )
    elif args.command == "generate":
        output = pipeline.generate(
            task_type=args.task,
            query=args.query,
            top_k=args.top_k,
            document_ids=args.document_id,
        )
        paths = pipeline.export_output(output)
        print(output.body_markdown)
        print()
        print(f"Markdown: {paths.markdown_path}")
        print(f"CSV: {paths.csv_path}")
        print(f"PDF: {paths.pdf_path}")
        print(f"Evidence JSON: {paths.evidence_json_path}")
        print(f"Run manifest: {paths.manifest_json_path}")
        print(f"Review package: {paths.package_path}")
    elif args.command == "documents":
        for document in pipeline.store.list_documents():
            print(
                f"{document['document_id']} | {document['title']} | pages={document['pages']} | chunks={document['chunks']} | "
                f"parser={document['parser_backend']}"
            )
    elif args.command == "sources":
        if args.init:
            registry_path, coverage_path = pipeline.initialize_sources(overwrite=args.overwrite)
            print(f"Source registry: {registry_path}")
            print(f"Africa country coverage: {coverage_path}")
        rows = pipeline.list_source_registry()
        if not rows:
            print("No source registry yet. Run: python -m devfinintel.cli sources --init")
        for row in rows:
            print(
                f"{row['source_id']} | {row['publisher']} | {row['year']} | "
                f"{row['status']} | {row['title']}"
            )
    elif args.command == "download-sources":
        results = pipeline.download_sources(limit=args.limit, ingest=not args.no_ingest)
        for result in results:
            print(
                f"{result['status']} | {result['source_id']} | "
                f"{result['local_path']} | {result.get('message', '')}"
            )
    elif args.command == "knowledge":
        records = pipeline.list_knowledge_records(
            record_type=args.record_type,
            country=args.country,
            review_status=args.review_status,
            limit=args.limit,
        )
        for record in records:
            print(
                f"{record['record_id']} | {record['record_type']} | {record['country']} | "
                f"{record['sector']} | {record['review_status']} | {record['title']}"
            )
    elif args.command == "coverage":
        rows = pipeline.source_backlog(limit=args.limit) if args.backlog else pipeline.source_coverage_matrix()
        if args.country:
            rows = [row for row in rows if row["country"].lower() == args.country.lower()]
        if args.topic:
            topic = args.topic.lower()
            rows = [
                row
                for row in rows
                if topic in row["topic_id"].lower() or topic in row["topic"].lower()
            ]
        summary = pipeline.source_coverage_summary()
        print(
            "Coverage: "
            f"{summary['usable_record_cells']} usable record cells, "
            f"{summary['regional_source_ready_cells']} regional-source-ready cells, "
            f"{summary['country_specific_gap_cells']} country-specific gaps across "
            f"{summary['coverage_cells']} country-topic cells"
        )
        for row in rows[: args.limit]:
            print(
                f"{row['country']} | {row['topic']} | {row['status']} | "
                f"usable_records={row['usable_country_records']} | "
                f"specific_sources={row['specific_downloaded_sources']} | "
                f"regional_sources={row['regional_downloaded_sources']} | "
                f"next={row['recommended_next_source']}"
            )
    elif args.command == "actions":
        if args.refresh:
            pipeline.refresh_action_items(limit=max(args.limit, 100))
        rows = pipeline.list_action_items(
            country=args.country,
            status=args.status,
            priority=args.priority,
            limit=args.limit,
        )
        for row in rows:
            print(
                f"{row['priority']} | {row['status']} | {row['country']} | "
                f"{row['action_type']} | {row['title']}"
            )
    elif args.command == "country":
        if args.refresh_actions:
            pipeline.refresh_action_items()
        payload = pipeline.country_intelligence(args.country)
        snapshot = payload["snapshot"]
        print(
            f"{snapshot['country']}: {snapshot['records']} records, "
            f"{snapshot['open_actions']} open actions, "
            f"{snapshot['high_priority_actions']} high-priority actions, "
            f"{snapshot['stakeholders']} stakeholders, {snapshot['comparisons']} comparisons"
        )
        print("Top risks:", snapshot["risk_flags"])
        print("Top actions:", snapshot["recommended_actions"])
        print("\nActions")
        for row in payload["actions"][: args.limit]:
            print(f"{row['priority']} | {row['status']} | {row['action_type']} | {row['title']}")
        print("\nStakeholders")
        for row in payload["stakeholders"][: args.limit]:
            print(f"{row['stakeholder']} | {row['role']} | records={row['records']} | risks={row['risk_flags']}")
        print("\nOfficial vs monitoring")
        for row in payload["comparisons"][: args.limit]:
            print(f"{row['relation']} | {row['topic']} | {row['risk_flags']} | {row['recommended_action']}")
    elif args.command == "monitor":
        if args.init:
            path = pipeline.initialize_monitoring(overwrite=args.overwrite)
            print(f"Monitoring source registry: {path}")
        if args.agent:
            agent_run = pipeline.run_monitoring_agent(
                from_sources=args.live_sources,
                from_knowledge=True,
                limit=max(args.limit, 100),
            )
            print(f"Monitoring agent run: {agent_run['generated_at']} | mode={agent_run['mode']}")
            for row in agent_run["briefing"]:
                print(f"{row['section']}: {row['brief']}")
            print("\nWatchlist")
            for row in agent_run["watchlist"][: args.limit]:
                print(f"{row['priority']} | {row['country']} | score={row['attention_score']} | action={row['recommended_action']}")
            print("\nSignal triage")
            for row in agent_run["triage_queue"][: args.limit]:
                print(f"{row['priority']} | {row['country']} | {row['outcome']} | {row['title']}")
            return
        if args.refresh:
            result = pipeline.refresh_monitoring_events(
                from_sources=args.live_sources,
                from_knowledge=True,
                limit=max(args.limit, 100),
            )
            print(f"Monitoring events saved: {result['events_saved']}")
            for source_result in result["source_results"]:
                print(f"{source_result['status']} | {source_result['source_id']} | {source_result['message']}")
        insights = pipeline.monitoring_insights(limit=max(args.limit, 100))
        snapshot = insights["snapshot"]
        print(
            "Monitoring: "
            f"{snapshot['events']} events, {snapshot['countries']} countries, "
            f"{snapshot['risk_events']} risk events, "
            f"{snapshot['high_relevance_events']} high-relevance events"
        )
        for card in insights["insight_cards"][:5]:
            print(f"Insight: {card['insight']} | Action: {card['suggested_action']}")
        rows = pipeline.list_monitoring_events(
            country=args.country,
            sector=args.sector,
            status=args.status,
            limit=args.limit,
        )
        for row in rows:
            print(
                f"{row['published_at']} | {row['country']} | {row['sector']} | "
                f"{row['outcome']} | {row['sentiment_tone']} | {row['title']}"
            )
    elif args.command == "alignment":
        report = pipeline.role_alignment_report()
        summary = report["summary"]
        print(
            "Role alignment: "
            f"{summary['ready']} ready, {summary['baseline']} baseline, "
            f"{summary['needs_data']} need data across {summary['requirements']} requirements"
        )
        for finding in report["executive_summary"]:
            print(f"Finding: {finding['finding']} | {finding['implication']}")
        print("\nRole scores")
        for score in report["role_scores"]:
            print(f"{score['role']} | {score['readiness_score']:.2f} | requirements={score['requirements']}")
        print("\nRequirements")
        for row in report["requirements"][: args.limit]:
            print(
                f"{row['readiness']} | {row['role']} | {row['requirement']} | "
                f"{row['platform_capability']} | evidence={row['evidence_count']}"
            )
    elif args.command == "audit":
        for event in pipeline.store.list_audit_events(limit=50):
            print(f"{event['timestamp']} | {event['action']} | {event['details']}")


def collect_files(paths: list[Path]) -> list[Path]:
    """Collect supported files from files or directories."""

    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.suffix.lower() in SUPPORTED_EXTENSIONS:
                    files.append(child)
        elif path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    if not files:
        raise SystemExit("No supported files found. Use PDF, TXT, MD, or CSV.")
    return files


if __name__ == "__main__":
    main()
