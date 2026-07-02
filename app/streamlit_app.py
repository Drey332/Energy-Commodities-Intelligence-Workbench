"""Streamlit dashboard for the Development Finance Intelligence Workbench.

The UI calls the same pipeline as the command line. That is deliberate: whether
someone uses buttons or a terminal command, the evidence path is identical and
therefore auditable.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

import streamlit as st

from devfinintel.env import source_key_status
from devfinintel.sources import AFRICAN_COUNTRIES

if TYPE_CHECKING:
    from devfinintel.pipeline import DocumentIntelligencePipeline


COUNTRY_ISO3_BY_NAME = {country.lower(): iso3 for iso3, country, _ in AFRICAN_COUNTRIES}
COUNTRY_ISO3_BY_NAME.update(
    {
        "côte d'ivoire": "CIV",
        "cote d’ivoire": "CIV",
        "drc": "COD",
        "congo, dem. rep.": "COD",
        "congo, rep.": "COG",
        "republic of congo": "COG",
    }
)


try:
    import pandas as pd
except Exception:  # pragma: no cover - Streamlit runtime convenience.
    pd = None

try:
    import plotly.express as px
except Exception:  # pragma: no cover - Streamlit runtime convenience.
    px = None


st.set_page_config(
    page_title="Development Finance Intelligence Workbench",
    layout="wide",
)


def get_pipeline() -> "DocumentIntelligencePipeline":
    """Create a pipeline object for the Streamlit run.

    This is intentionally not cached. During active development Streamlit can
    keep imported modules alive after code reloads, which can hide new pipeline
    methods from the UI unless the module is refreshed.
    """

    import devfinintel.store as store_module
    import devfinintel.pipeline as pipeline_module

    importlib.reload(store_module)
    pipeline_module = importlib.reload(pipeline_module)
    return pipeline_module.DocumentIntelligencePipeline()


def source_status_rows_for_ui(pipeline_obj: "DocumentIntelligencePipeline") -> list[dict]:
    """Return source-status rows even if Streamlit has a stale pipeline object."""

    if hasattr(pipeline_obj, "source_configuration_status"):
        return pipeline_obj.source_configuration_status()
    rows = [
        {"source": "GDELT", "source_type": "news", "status": "keyless public", "secret_visible": "no"},
        {"source": "ReliefWeb", "source_type": "risk_context", "status": "keyless public", "secret_visible": "no"},
        {"source": "World Bank Indicators", "source_type": "dataset_indicator", "status": "keyless public", "secret_visible": "no"},
        {"source": "World Bank Documents", "source_type": "institutional_report", "status": "keyless public", "secret_visible": "no"},
    ]
    rows.extend(source_key_status())
    rows.append({"source": "Fallback sample data", "source_type": "fallback_sample", "status": "available", "secret_visible": "no"})
    return rows


def iso3_for_country(country: Any) -> str:
    """Return ISO-3 for a country name used in dashboard maps."""

    return COUNTRY_ISO3_BY_NAME.get(str(country).strip().lower(), "")


def download_button(column, label: str, path: Path, mime: str, key: str) -> None:
    """Render a download button for one exported file."""

    if not path.exists():
        column.caption(f"{label} unavailable")
        return
    column.download_button(
        label,
        data=path.read_bytes(),
        file_name=path.name,
        mime=mime,
        width="stretch",
        key=key,
    )


def flatten_records(records) -> list[dict]:
    """Convert ExtractionRecord objects into rows for Streamlit tables."""

    rows = []
    for record in records:
        row = {
            "record_id": record.record_id,
            "record_type": record.record_type,
            "title": record.title,
            "confidence": record.confidence,
            "review_status": record.review_status,
            "evidence_chunk_ids": "; ".join(record.evidence_chunk_ids),
        }
        for key, value in record.fields.items():
            row[key] = display_value(value)
        rows.append(row)
    return rows


def flatten_stored_records(records: list[dict]) -> list[dict]:
    """Convert stored database rows into dashboard rows."""

    rows = []
    for record in records:
        row = {
            "record_id": record["record_id"],
            "record_type": record["record_type"],
            "title": record["title"],
            "confidence": record["confidence"],
            "review_status": record["review_status"],
            "created_at": record["created_at"],
            "evidence_chunk_ids": "; ".join(record["evidence_chunk_ids"]),
        }
        for key, value in record["fields"].items():
            row[key] = display_value(value)
        rows.append(row)
    return rows


def flatten_knowledge_records(records: list[dict]) -> list[dict]:
    """Convert operational knowledge records into dashboard rows."""

    keys = [
        "record_id",
        "record_type",
        "title",
        "country",
        "region",
        "sector",
        "theme",
        "commodity",
        "partner",
        "amount",
        "currency",
        "instrument",
        "event_date",
        "relevance",
        "actors",
        "event_type",
        "sentiment_tone",
        "risk_flags",
        "recommended_action",
        "source_title",
        "source_page",
        "confidence",
        "review_status",
        "updated_at",
    ]
    rows = []
    for record in records:
        row = {key: display_value(record.get(key)) for key in keys}
        row["evidence_chunk_ids"] = display_value(record.get("evidence_chunk_ids"))
        row["source_path"] = record.get("source_path", "")
        rows.append(row)
    return rows


def flatten_action_items(actions: list[dict]) -> list[dict]:
    """Convert action items into dashboard rows."""

    keys = [
        "action_id",
        "country",
        "action_type",
        "priority",
        "status",
        "title",
        "rationale",
        "source_record_id",
        "source_title",
        "source_page",
        "due_bucket",
        "updated_at",
    ]
    return [{key: display_value(action.get(key)) for key in keys} for action in actions]


def flatten_monitoring_events(events: list[dict]) -> list[dict]:
    """Convert monitoring events into compact dashboard rows."""

    keys = [
        "event_id",
        "published_at",
        "country",
        "sector",
        "commodity",
        "event_type",
        "outcome",
        "sentiment_tone",
        "risk_flags",
        "relevance",
        "recommended_action",
        "source_name",
        "source_category",
        "status",
        "confidence",
        "title",
        "summary",
        "url",
    ]
    return [{key: display_value(event.get(key)) for key in keys} for event in events]


def country_map_rows(coverage_matrix: list[dict], actions: list[dict]) -> list[dict]:
    """Build one row per country for the Africa intelligence map."""

    action_counts: dict[str, int] = {}
    urgent_counts: dict[str, int] = {}
    for action in actions:
        country = action.get("country", "")
        if not country:
            continue
        if action.get("status") in {"open", "in_progress"}:
            action_counts[country] = action_counts.get(country, 0) + 1
            if action.get("priority") in {"urgent", "high"}:
                urgent_counts[country] = urgent_counts.get(country, 0) + 1

    by_country: dict[str, dict] = {}
    for row in coverage_matrix:
        country = row.get("country", "")
        if not country:
            continue
        item = by_country.setdefault(
            country,
            {
                "country": country,
                "usable_cells": 0,
                "country_specific_gaps": 0,
                "regional_only_cells": 0,
                "open_actions": 0,
                "high_priority_actions": 0,
                "attention_score": 0,
            },
        )
        if row.get("status") == "usable_records":
            item["usable_cells"] += 1
        if int(row.get("specific_downloaded_sources", 0)) == 0:
            item["country_specific_gaps"] += 1
        if row.get("status") == "regional_source_ready":
            item["regional_only_cells"] += 1

    for country, item in by_country.items():
        item["open_actions"] = action_counts.get(country, 0)
        item["high_priority_actions"] = urgent_counts.get(country, 0)
        item["attention_score"] = (
            item["high_priority_actions"] * 3
            + item["open_actions"]
            + item["country_specific_gaps"]
        )
    return list(by_country.values())


def flatten_audit_events(events: list[dict]) -> list[dict]:
    """Convert audit events into display-safe table rows."""

    rows = []
    for event in events:
        row = dict(event)
        row["details"] = display_value(row.get("details"))
        rows.append(row)
    return rows


def flatten_sessions(sessions: list[dict]) -> list[dict]:
    """Convert analysis sessions into table rows."""

    rows = []
    for session in sessions:
        row = dict(session)
        row["document_ids"] = display_value(row.get("document_ids"))
        row["diagnostics"] = display_value(row.get("diagnostics"))
        rows.append(row)
    return rows


def display_value(value: Any) -> str:
    """Render nested values safely in Streamlit tables.

    Dataset profiles contain nested structures such as lists of dictionaries for
    highest values, lowest values, and numeric summaries. A table cell can only
    display text-like values, so this function converts those structures into
    readable JSON instead of crashing.
    """

    if isinstance(value, list):
        if all(not isinstance(item, (dict, list)) for item in value):
            return "; ".join(str(item) for item in value)
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return "" if value is None else str(value)


def ingest_uploaded_files(uploaded_files, language_hint: str) -> list[str]:
    """Copy uploaded files into the project and ingest them.

    This gives the app a PartyRock-like flow: a user can upload a file and press
    Analyze without needing to understand the separate ingestion step.
    """

    saved_paths = []
    for uploaded in uploaded_files or []:
        destination = pipeline.config.input_dir / Path(uploaded.name).name
        destination.write_bytes(uploaded.getbuffer())
        saved_paths.append(destination)
    if not saved_paths:
        return []
    summaries = pipeline.ingest_paths(saved_paths, language_hint=language_hint)
    document_ids = [summary["document_id"] for summary in summaries]
    st.session_state["latest_ingested_document_ids"] = document_ids
    st.session_state["latest_ingest_summaries"] = summaries
    return document_ids


def render_dataset_visuals(output) -> None:
    """Render Python/Plotly charts for dataset profile outputs."""

    if output.task_type != "dataset_profile" or pd is None:
        return

    dataset_records = [record for record in output.records if record.record_type == "dataset_profile"]
    if not dataset_records:
        return

    st.subheader("Dataset Visuals")
    for record in dataset_records:
        fields = record.fields
        highest = fields.get("highest_values", [])
        lowest = fields.get("lowest_values", [])
        numeric_profiles = fields.get("numeric_profiles", [])

        st.markdown(f"**{record.title}**")
        chart_cols = st.columns(2)
        if highest:
            highest_df = pd.DataFrame(highest)
            with chart_cols[0]:
                if px is not None:
                    fig = px.bar(
                        highest_df,
                        x="value",
                        y="label",
                        orientation="h",
                        title="Highest values",
                    )
                    fig.update_layout(yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.bar_chart(highest_df.set_index("label")["value"])
        if lowest:
            lowest_df = pd.DataFrame(lowest)
            with chart_cols[1]:
                if px is not None:
                    fig = px.bar(
                        lowest_df,
                        x="value",
                        y="label",
                        orientation="h",
                        title="Lowest values",
                    )
                    fig.update_layout(yaxis={"categoryorder": "total descending"})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.bar_chart(lowest_df.set_index("label")["value"])

        if numeric_profiles:
            st.dataframe(pd.DataFrame(numeric_profiles), width="stretch", hide_index=True)


def render_suggested_questions(output) -> None:
    """Show deterministic follow-up questions produced by the analysis layer."""

    questions: list[str] = []
    for record in output.records:
        values = record.fields.get("suggested_questions")
        if isinstance(values, list):
            questions.extend(str(value) for value in values)
    if not questions:
        return
    st.subheader("Suggested Questions")
    for question in questions[:7]:
        st.markdown(f"- {question}")


def selected_documents_are_csv(document_ids: list[str] | None, document_rows: list[dict]) -> bool:
    """Return True when all selected evidence sources are CSV files."""

    if not document_ids:
        return False
    source_types = {
        row["document_id"]: row["source_type"]
        for row in document_rows
    }
    return all(source_types.get(document_id) == "csv" for document_id in document_ids)


def selected_documents_include_csv(document_ids: list[str] | None, document_rows: list[dict]) -> bool:
    """Return True when the selected scope includes at least one CSV file."""

    source_types = {
        row["document_id"]: row["source_type"]
        for row in document_rows
    }
    if document_ids is None:
        return any(row["source_type"] == "csv" for row in document_rows)
    return any(source_types.get(document_id) == "csv" for document_id in document_ids)


def resolve_effective_task_type(
    task_type: str,
    selected_document_ids: list[str] | None,
    document_rows: list[dict],
) -> tuple[str, str]:
    """Choose a safe task when the selected file type does not match the UI mode."""

    if task_type == "dataset_profile" and not selected_documents_include_csv(selected_document_ids, document_rows):
        return (
            "qa",
            "The selected upload is not a CSV, so the app used evidence-grounded document Q&A instead of Dataset / CSV profile.",
        )
    return task_type, ""


def render_export_downloads(paths) -> None:
    """Render exports while tolerating stale Streamlit session objects.

    Streamlit can keep old dataclass instances in session state after code
    reloads. ``getattr`` prevents an old three-file ExportPaths object from
    crashing the whole app while the user is testing.
    """

    download_cols = st.columns(3)
    output_id = getattr(paths, "output_id", "") or Path(str(paths.markdown_path)).stem
    download_button(download_cols[0], "Markdown", paths.markdown_path, "text/markdown", f"export_{output_id}_markdown")
    download_button(download_cols[1], "CSV", paths.csv_path, "text/csv", f"export_{output_id}_csv")
    download_button(download_cols[2], "PDF", paths.pdf_path, "application/pdf", f"export_{output_id}_pdf")

    audit_paths = [
        ("Evidence JSON", getattr(paths, "evidence_json_path", None), "application/json"),
        ("Run Manifest", getattr(paths, "manifest_json_path", None), "application/json"),
        ("Review Package", getattr(paths, "package_path", None), "application/zip"),
    ]
    if all(path is not None for _, path, _ in audit_paths):
        audit_download_cols = st.columns(3)
        for column, (label, path, mime) in zip(audit_download_cols, audit_paths):
            download_button(column, label, path, mime, f"export_{output_id}_{Path(str(path)).suffix}_{label}")
    else:
        st.caption("Audit exports will appear after the next analysis run.")


def render_metric_explanations() -> None:
    """Explain the dashboard metrics in plain language.

    These numbers are audit heuristics. They help a reviewer understand how
    strongly the draft is tied to retrieved evidence, but they are not model
    confidence scores and they do not prove that an answer is true.
    """

    with st.expander("How these numbers are calculated"):
        st.markdown(
            """
- **Citation coverage**: factual-looking Markdown lines with a page citation divided by all factual-looking lines.
- **Support overlap**: content-word overlap between the generated draft and the retrieved evidence text.
- **Evidence items**: number of retrieved evidence chunks included in the evidence pack.
- **Records**: number of structured records extracted from those evidence chunks.
- **Retrieval gate**: `Pass` when evidence exists and either top score is at least `0.08` or keyword coverage is at least `15%`.
- **Top score**: highest hybrid retrieval score among selected chunks: normalized BM25 keyword score + dense similarity score + transparent task boost.
- **Keyword coverage**: meaningful query terms found in the retrieved evidence divided by meaningful query terms in the expanded query.

These are transparent review signals, not truth percentages. Use the citations and evidence pack for final review.
            """.strip()
        )


def find_column(columns: list[str], candidates: list[str]) -> str | None:
    """Find a likely analytical column by exact or loose name match."""

    lower_lookup = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lower_lookup:
            return lower_lookup[candidate.lower()]
    for column in columns:
        lower = column.lower()
        if any(candidate.lower() in lower for candidate in candidates):
            return column
    return None


def load_csv_dataframe(document_row: dict) -> Any:
    """Load a selected CSV document into pandas if available."""

    if pd is None:
        return None
    path = Path(document_row["source_path"])
    if not path.exists():
        return None
    return pd.read_csv(path)


def dataframe_filters(df) -> tuple[Any, dict[str, str]]:
    """Render standard country/commodity/year filters when columns exist."""

    country_col = find_column(list(df.columns), ["country", "ref_area_label", "economy", "area"])
    commodity_col = find_column(list(df.columns), ["commodity", "product", "resource", "fuel", "mineral"])
    year_col = find_column(list(df.columns), ["year", "time_period", "date"])
    filters: dict[str, str] = {}
    filter_cols = st.columns(3)
    filtered = df.copy()
    with filter_cols[0]:
        if country_col:
            countries = sorted(str(value) for value in df[country_col].dropna().astype(str).unique())
            selected = st.selectbox("Country/entity filter", ["All"] + countries[:250])
            if selected != "All":
                filtered = filtered[filtered[country_col].astype(str) == selected]
                filters[country_col] = selected
        else:
            st.caption("No country column detected.")
    with filter_cols[1]:
        if commodity_col:
            commodities = sorted(str(value) for value in df[commodity_col].dropna().astype(str).unique())
            selected = st.selectbox("Commodity/product filter", ["All"] + commodities[:250])
            if selected != "All":
                filtered = filtered[filtered[commodity_col].astype(str) == selected]
                filters[commodity_col] = selected
        else:
            st.caption("No commodity column detected.")
    with filter_cols[2]:
        if year_col:
            years = sorted(str(value) for value in df[year_col].dropna().astype(str).unique())
            selected = st.selectbox("Year/date filter", ["All"] + years[-150:])
            if selected != "All":
                filtered = filtered[filtered[year_col].astype(str) == selected]
                filters[year_col] = selected
        else:
            st.caption("No year/date column detected.")
    return filtered, filters


def render_data_charts(df, key_prefix: str) -> None:
    """Render useful automatic charts from a filtered dataframe."""

    if pd is None or px is None or df is None or df.empty:
        return
    columns = list(df.columns)
    country_col = find_column(columns, ["country", "ref_area_label", "economy", "area"])
    commodity_col = find_column(columns, ["commodity", "product", "resource", "fuel", "mineral"])
    year_col = find_column(columns, ["year", "time_period", "date"])
    numeric_columns = [
        column
        for column in list(df.select_dtypes(include="number").columns)
        if column not in {year_col, country_col, commodity_col}
    ]
    chart_cols = st.columns(2)
    if year_col and numeric_columns:
        value_col = st.selectbox("Time-series value", numeric_columns, key=f"{key_prefix}_line_value")
        line_df = df.copy()
        line_df[year_col] = line_df[year_col].astype(str)
        if country_col:
            group_cols = [year_col, country_col]
            line_df = line_df.groupby(group_cols, dropna=False)[value_col].sum().reset_index()
            fig = px.line(line_df, x=year_col, y=value_col, color=country_col, markers=True, title="Trend by country/entity")
        else:
            line_df = line_df.groupby(year_col, dropna=False)[value_col].sum().reset_index()
            fig = px.line(line_df, x=year_col, y=value_col, markers=True, title="Trend")
        with chart_cols[0]:
            st.plotly_chart(fig, use_container_width=True)
    if numeric_columns:
        value_col = st.selectbox("Ranking value", numeric_columns, key=f"{key_prefix}_rank_value")
        label_col = country_col or commodity_col or columns[0]
        ranked = df[[label_col, value_col]].dropna().copy()
        if not ranked.empty:
            ranked = ranked.groupby(label_col, dropna=False)[value_col].sum().reset_index()
            ranked = ranked.sort_values(value_col, ascending=False).head(15)
            fig = px.bar(ranked, x=value_col, y=label_col, orientation="h", title="Top values")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            with chart_cols[1]:
                st.plotly_chart(fig, use_container_width=True)

    quality_cols = st.columns(2)
    missing = df.isna().sum().sort_values(ascending=False).head(20).reset_index()
    missing.columns = ["column", "missing"]
    with quality_cols[0]:
        if not missing.empty:
            fig = px.bar(missing, x="missing", y="column", orientation="h", title="Missing values by column")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
    with quality_cols[1]:
        if len(numeric_columns) >= 2:
            corr = df[numeric_columns].corr(numeric_only=True)
            fig = px.imshow(corr, text_auto=True, aspect="auto", title="Numeric correlation heatmap")
            st.plotly_chart(fig, use_container_width=True)


def dataframe_csv_bytes(df) -> bytes:
    """Return CSV bytes for download."""

    return df.to_csv(index=False).encode("utf-8")


def render_evidence_rows(rows: list[dict], key: str) -> None:
    """Show evidence rows and provide a CSV export."""

    if not rows:
        st.info("No evidence rows available.")
        return
    if pd is not None:
        evidence_df = pd.DataFrame(rows)
        st.dataframe(evidence_df, width="stretch", hide_index=True)
        st.download_button(
            "Download evidence/source list",
            evidence_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{key}_evidence.csv",
            mime="text/csv",
            width="stretch",
            key=f"{key}_evidence_download",
        )
    else:
        st.json(rows, expanded=False)


def download_json_button(label: str, payload: dict, file_name: str, key: str) -> None:
    """Download a JSON payload without exposing secrets."""

    st.download_button(
        label,
        json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=file_name,
        mime="application/json",
        width="stretch",
        key=key,
    )


def supervisor_brief_markdown(agent_run: dict) -> str:
    """Format a monitoring supervisor run as Markdown."""

    lines = [
        "# Monitoring Supervisor Situation Brief",
        "",
        f"- Generated at: {agent_run.get('generated_at', '')}",
        f"- Mode: {agent_run.get('mode', '')}",
        "",
        "## Evidence Pool",
    ]
    for key, value in agent_run.get("evidence_pool", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Briefing"])
    for row in agent_run.get("briefing", []):
        lines.append(f"- {row.get('section', '')}: {row.get('brief', '')}")
    lines.extend(["", "## Watchlist"])
    for row in agent_run.get("watchlist", [])[:20]:
        lines.append(f"- {row.get('priority', '')}: {row.get('country', '')} ({row.get('recommended_action', '')})")
    lines.extend(["", "## Urgent / High Signals"])
    for row in agent_run.get("triage_queue", [])[:20]:
        lines.append(f"- {row.get('priority', '')}: {row.get('country', '')} | {row.get('title', '')}")
    lines.extend(["", "## Next Actions"])
    for row in agent_run.get("next_actions", []):
        lines.append(f"- {row.get('priority', '')}: {row.get('action', '')} -- {row.get('reason', '')}")
    return "\n".join(lines)


pipeline = get_pipeline()

st.title("Africa Energy & Commodities Intelligence Workbench")
st.caption(
    "One analyst workspace for official reports, uploaded datasets, monitoring signals, news, "
    "evidence-backed questions, and short decision-ready briefs."
)

documents = pipeline.store.list_documents()
csv_documents = [row for row in documents if row["source_type"] == "csv"]
document_label_by_id = {
    row["document_id"]: f"{row['title']} ({row['source_type']}, {row['pages']} pages)"
    for row in documents
}
session_latest_document_ids = [
    document_id
    for document_id in st.session_state.get("latest_ingested_document_ids", [])
    if document_id in document_label_by_id
]
latest_document_ids = session_latest_document_ids or ([documents[0]["document_id"]] if documents else [])

with st.sidebar:
    st.header("Workbench Controls")
    uploaded_files = st.file_uploader(
        "Upload reports or datasets",
        type=["pdf", "txt", "md", "csv"],
        accept_multiple_files=True,
    )
    language_hint = st.selectbox("Language hint", ["unknown", "English", "French", "English/French"])
    if st.button("Ingest uploads", type="primary", width="stretch"):
        document_ids = ingest_uploaded_files(uploaded_files, language_hint=language_hint)
        if document_ids:
            st.success(f"Ingested {len(document_ids)} file(s).")
            st.rerun()
        else:
            st.warning("No files selected.")

    st.divider()
    focus_query = st.text_input(
        "Monitoring focus",
        value="Africa energy commodities development finance risks",
    )
    country_filter = st.text_input("Country focus", value="")
    commodity_filter = st.text_input("Commodity focus", value="")
    topic_filter = st.selectbox(
        "Topic focus",
        [
            "",
            "energy access",
            "oil and gas",
            "mining",
            "critical minerals",
            "electricity",
            "renewables",
            "infrastructure",
            "commodity prices",
            "governance",
            "climate risk",
            "development finance",
        ],
    )
    top_k = st.slider("Document evidence items", min_value=3, max_value=15, value=8)
    news_limit = st.slider("News items", min_value=5, max_value=40, value=15)

    scope = st.selectbox(
        "Document scope",
        ["latest", "selected", "all"],
        index=0 if latest_document_ids else 2,
        format_func=lambda value: {
            "latest": "Current/latest upload",
            "selected": "Selected library files",
            "all": "Full evidence library",
        }[value],
    )
    if scope == "latest":
        selected_document_ids: list[str] | None = latest_document_ids
        if latest_document_ids:
            st.caption("Using: " + ", ".join(document_label_by_id[doc_id] for doc_id in latest_document_ids))
        else:
            selected_document_ids = None
            st.caption("No current upload found, using full library.")
    elif scope == "selected":
        selected_document_ids = st.multiselect(
            "Library files",
            options=list(document_label_by_id),
            default=latest_document_ids or list(document_label_by_id)[:3],
            format_func=lambda value: document_label_by_id[value],
        )
    else:
        selected_document_ids = None

    st.divider()
    if st.button("Refresh news signals", width="stretch"):
        with st.spinner("Fetching live or fallback news signals..."):
            st.session_state["news_bundle"] = pipeline.fetch_news_signals(
                query=focus_query,
                country=country_filter,
                commodity=commodity_filter,
                topic=topic_filter,
                limit=int(news_limit),
            )
    if st.button("Run monitoring cycle", type="primary", width="stretch"):
        with st.spinner("Fetching keyless sources, normalizing signals, and clustering developments..."):
            if hasattr(pipeline, "run_monitoring_cycle"):
                st.session_state["monitoring_cycle_result"] = pipeline.run_monitoring_cycle(
                    query=focus_query,
                    country=country_filter,
                    commodity=commodity_filter,
                    topic=topic_filter,
                    limit=int(news_limit),
                    include_optional_key_sources=True,
                    include_indicators=True,
                    use_live_connectors=True,
                )
            else:
                st.error("The monitoring-cycle backend is not loaded yet. Restart Streamlit and try again.")
    include_live_for_supervisor = st.checkbox(
        "Include current live monitoring results",
        value=True,
        help="When checked, the supervisor uses the current monitoring-cycle signals and clusters in addition to stored/promoted events.",
    )
    if st.button("Run monitoring supervisor", width="stretch"):
        with st.spinner("Refreshing local events and building monitoring run..."):
            st.session_state["monitoring_agent"] = pipeline.run_monitoring_agent(
                from_sources=False,
                from_knowledge=True,
                limit=1000,
                monitoring_result=st.session_state.get("monitoring_cycle_result"),
                include_current_live_results=include_live_for_supervisor,
            )

if "news_bundle" not in st.session_state:
    st.session_state["news_bundle"] = pipeline.fetch_news_signals(
        query=focus_query,
        country=country_filter,
        commodity=commodity_filter,
        topic=topic_filter,
        limit=int(news_limit),
    )

news_bundle = st.session_state["news_bundle"]
news_articles = news_bundle.get("articles", [])
monitoring_cycle = st.session_state.get("monitoring_cycle_result")
monitoring_insights = pipeline.monitoring_insights(limit=1000)
monitoring_snapshot = monitoring_insights["snapshot"]
knowledge_summary = pipeline.knowledge_coverage_summary()

metric_cols = st.columns(6)
metric_cols[0].metric("Documents", len(documents))
metric_cols[1].metric("Datasets", len(csv_documents))
metric_cols[2].metric("News signals", len(news_articles))
metric_cols[3].metric(
    "Cycle signals",
    int((monitoring_cycle or {}).get("normalized_signal_count", monitoring_snapshot["events"])),
)
metric_cols[4].metric("Countries", int(max(monitoring_snapshot["countries"], knowledge_summary["countries"])))
metric_cols[5].metric("Event clusters", int((monitoring_cycle or {}).get("event_cluster_count", 0)))

panel_tabs = st.tabs(["Overview", "Documents", "Data", "News", "Ask & Brief", "Review & Sources"])

with panel_tabs[0]:
    st.subheader("What Is Happening?")
    status_rows = (monitoring_cycle or {}).get("source_statuses") or source_status_rows_for_ui(pipeline)
    st.markdown("**Source status**")
    st.dataframe(status_rows, width="stretch", hide_index=True)
    if monitoring_cycle:
        st.markdown("**Current monitoring run**")
        st.caption(
            f"Run ID: {monitoring_cycle.get('monitoring_run_id', '')} | "
            f"Timestamp: {monitoring_cycle.get('run_timestamp', '')} | "
            f"Query: {monitoring_cycle.get('query', '')}"
        )
        cycle_cols = st.columns(4)
        cycle_cols[0].metric("Normalized signals", int(monitoring_cycle["normalized_signal_count"]))
        cycle_cols[1].metric("Event clusters", int(monitoring_cycle["event_cluster_count"]))
        cycle_cols[2].metric("Fallback used", "yes" if monitoring_cycle["fallback_used"] else "no")
        cycle_cols[3].metric("Warnings", len(monitoring_cycle.get("warnings", [])))
        if monitoring_cycle.get("top_developments"):
            st.markdown("**Top clustered developments**")
            st.dataframe(monitoring_cycle["top_developments"], width="stretch", hide_index=True)
    overview_cols = st.columns([0.48, 0.52], gap="large")
    with overview_cols[0]:
        st.markdown("**What changed?**")
        if news_bundle.get("warning"):
            st.warning(news_bundle["warning"])
        if news_bundle.get("what_changed"):
            st.dataframe(news_bundle["what_changed"], width="stretch", hide_index=True)
        if monitoring_insights.get("insight_cards"):
            st.markdown("**Analyst insight cards**")
            st.dataframe(monitoring_insights["insight_cards"], width="stretch", hide_index=True)
    with overview_cols[1]:
        agent_run = st.session_state.get("monitoring_agent")
        if agent_run:
            st.markdown("**Monitoring supervisor briefing**")
            st.caption(f"Last run: {agent_run['generated_at']} | {agent_run['mode']}")
            if agent_run.get("evidence_pool"):
                st.markdown("**Supervisor evidence pool**")
                st.dataframe([agent_run["evidence_pool"]], width="stretch", hide_index=True)
            st.dataframe(agent_run["briefing"], width="stretch", hide_index=True)
            st.markdown("**Next actions**")
            st.dataframe(agent_run["next_actions"], width="stretch", hide_index=True)
            st.download_button(
                "Download supervisor brief Markdown",
                supervisor_brief_markdown(agent_run).encode("utf-8"),
                file_name="monitoring_supervisor_brief.md",
                mime="text/markdown",
                width="stretch",
                key="monitoring_supervisor_brief_markdown",
            )
        else:
            st.info("Run the monitoring supervisor from the sidebar to generate a situation brief and action queue.")

    if pd is not None and px is not None and monitoring_insights["country_rows"]:
        country_event_df = pd.DataFrame(
            [
                row
                for row in monitoring_insights["country_rows"]
                if row.get("country") not in {"Regional Africa", "Not specified"}
            ]
        )
        if not country_event_df.empty:
            country_event_df["iso3"] = country_event_df["country"].map(iso3_for_country)
            country_event_df = country_event_df[country_event_df["iso3"] != ""]
        if not country_event_df.empty:
            fig = px.choropleth(
                country_event_df,
                locations="iso3",
                locationmode="ISO-3",
                color="attention_score",
                hover_name="country",
                hover_data={
                    "events": True,
                    "risk_events": True,
                    "high_relevance_events": True,
                    "mixed_or_negative_events": True,
                    "open_actions": True,
                },
                color_continuous_scale="YlOrRd",
                title="Africa Monitoring Attention Map",
            )
            fig.update_geos(scope="africa", fitbounds="locations", visible=False)
            fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
            st.plotly_chart(fig, use_container_width=True)

        chart_cols = st.columns(3)
        with chart_cols[0]:
            if monitoring_insights["top_risks"]:
                fig = px.bar(pd.DataFrame(monitoring_insights["top_risks"]), x="count", y="value", orientation="h", title="Risk Flags")
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)
        with chart_cols[1]:
            if monitoring_insights["top_event_types"]:
                fig = px.bar(pd.DataFrame(monitoring_insights["top_event_types"]), x="count", y="value", orientation="h", title="Event Types")
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)
        with chart_cols[2]:
            if monitoring_insights["top_outcomes"]:
                fig = px.bar(pd.DataFrame(monitoring_insights["top_outcomes"]), x="count", y="value", orientation="h", title="Observed Outcomes")
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)
    if monitoring_cycle and monitoring_cycle.get("event_clusters"):
        st.subheader("Clustered Developments")
        for cluster in monitoring_cycle["event_clusters"][:8]:
            with st.expander(f"{cluster['risk_level'].upper()} | {cluster['event_title']} | {cluster['signal_count']} signal(s)"):
                st.write(cluster["what_changed"])
                st.write(cluster["why_it_matters"])
                detail_cols = st.columns(4)
                detail_cols[0].metric("Sources", int(cluster["source_count"]))
                detail_cols[1].metric("Confidence", cluster["confidence_level"])
                detail_cols[2].metric("Latest", cluster["latest_update"] or "undated")
                detail_cols[3].metric("Risk", cluster["risk_level"])
                st.caption(cluster["evidence_summary"])

with panel_tabs[1]:
    st.subheader("Document Intelligence")
    if not documents:
        st.info("Upload or ingest PDFs, reports, text files, or CSVs to inspect document content.")
    else:
        doc_options = [row["document_id"] for row in documents]
        default_doc = selected_document_ids[0] if isinstance(selected_document_ids, list) and selected_document_ids else doc_options[0]
        selected_preview_doc = st.selectbox(
            "Document preview",
            options=doc_options,
            index=doc_options.index(default_doc) if default_doc in doc_options else 0,
            format_func=lambda value: document_label_by_id[value],
        )
        selected_row = next(row for row in documents if row["document_id"] == selected_preview_doc)
        meta_cols = st.columns(5)
        meta_cols[0].metric("Pages", int(selected_row["pages"]))
        meta_cols[1].metric("Chunks", int(selected_row["chunks"]))
        meta_cols[2].metric("Type", selected_row["source_type"])
        meta_cols[3].metric("Parser", selected_row["parser_backend"])
        meta_cols[4].metric("Loaded", selected_row["loaded_at"][:10])
        st.caption(selected_row["source_path"])

        keyword = st.text_input("Search inside document", value="")
        excerpts = pipeline.document_excerpts(
            document_ids=[selected_preview_doc],
            keyword=keyword,
            limit=20,
        )
        if excerpts:
            for excerpt in excerpts[:8]:
                with st.expander(f"{excerpt['title']} | p. {excerpt['page']} | chunk {excerpt['chunk']}"):
                    st.write(excerpt["excerpt"])
            st.dataframe(excerpts, width="stretch", hide_index=True)
        else:
            st.warning("No matching excerpts found.")

        doc_question = st.text_input("Ask this document", value="What are the main energy or commodity risks?")
        if st.button("Answer from document evidence", width="stretch"):
            with st.spinner("Searching document excerpts..."):
                st.session_state["document_answer"] = pipeline.answer_workbench_question(
                    question=doc_question,
                    document_ids=[selected_preview_doc],
                    news_articles=[],
                    top_k=int(top_k),
                )
        if st.session_state.get("document_answer"):
            st.markdown(st.session_state["document_answer"]["answer_markdown"])
            render_evidence_rows(st.session_state["document_answer"]["evidence_rows"], "document_answer")

with panel_tabs[2]:
    st.subheader("Data Explorer")
    if pd is None:
        st.info("Install pandas to use the interactive data explorer.")
    elif not csv_documents:
        st.info("No CSV datasets are ingested yet. Upload a CSV in the sidebar.")
    else:
        csv_id = st.selectbox(
            "Dataset",
            [row["document_id"] for row in csv_documents],
            format_func=lambda value: document_label_by_id[value],
        )
        csv_row = next(row for row in csv_documents if row["document_id"] == csv_id)
        df = load_csv_dataframe(csv_row)
        if df is None:
            st.error("Could not load the selected CSV file.")
        else:
            st.caption(csv_row["source_path"])
            profile_cols = st.columns(4)
            profile_cols[0].metric("Rows", len(df))
            profile_cols[1].metric("Columns", len(df.columns))
            profile_cols[2].metric("Missing cells", int(df.isna().sum().sum()))
            profile_cols[3].metric("Numeric columns", len(df.select_dtypes(include="number").columns))
            filtered_df, active_filters = dataframe_filters(df)
            st.markdown("**Filtered preview**")
            st.dataframe(filtered_df.head(250), width="stretch", hide_index=True)
            st.download_button(
                "Download filtered data CSV",
                dataframe_csv_bytes(filtered_df),
                file_name=f"{Path(csv_row['source_path']).stem}_filtered.csv",
                mime="text/csv",
                width="stretch",
                key=f"data_explorer_filtered_csv_{csv_id}",
            )
            st.markdown("**Column profile**")
            type_rows = [
                {
                    "column": column,
                    "dtype": str(df[column].dtype),
                    "missing": int(df[column].isna().sum()),
                    "unique": int(df[column].nunique(dropna=True)),
                }
                for column in df.columns
            ]
            st.dataframe(type_rows, width="stretch", hide_index=True)
            render_data_charts(filtered_df, "data_explorer")

            data_question = st.text_input("Ask about this dataset", value="What are the most important data signals?")
            if st.button("Answer from dataset and selected evidence", width="stretch"):
                with st.spinner("Combining dataset summary with selected evidence..."):
                    st.session_state["data_answer"] = pipeline.answer_workbench_question(
                        question=data_question,
                        document_ids=[csv_id],
                        news_articles=news_articles,
                        top_k=int(top_k),
                    )
            if st.session_state.get("data_answer"):
                st.markdown(st.session_state["data_answer"]["answer_markdown"])
                render_evidence_rows(st.session_state["data_answer"]["evidence_rows"], "data_answer")

with panel_tabs[3]:
    st.subheader("Live News Monitoring")
    if monitoring_cycle:
        st.markdown("**Monitoring cycle signals**")
        signals = monitoring_cycle.get("normalized_signals", [])
        clusters = monitoring_cycle.get("event_clusters", [])
        promoted_items = pipeline.list_promoted_monitoring_items(limit=1000)
        promoted_source_ids = {item.get("source_item_id", "") for item in promoted_items}
        review_rows = pipeline.build_monitoring_review_queue(monitoring_cycle)
        review_reasons_by_source = {
            row.get("source_item_id", ""): "; ".join(row.get("review_reasons", []))
            for row in review_rows
            if row.get("source_item_id")
        }
        if monitoring_cycle.get("monitoring_run_id"):
            st.caption(
                f"Live/session run: {monitoring_cycle['monitoring_run_id']} | "
                f"{len(promoted_source_ids)} promoted item(s) currently stored."
            )
        if pd is not None and signals:
            signals_df = pd.DataFrame(signals)
            signals_df["promoted"] = signals_df["signal_id"].astype(str).isin(promoted_source_ids)
            signals_df["review_reasons"] = signals_df["signal_id"].astype(str).map(review_reasons_by_source).fillna("")
            signal_filter_cols = st.columns(6)
            with signal_filter_cols[0]:
                signal_country = st.selectbox(
                    "Signal country",
                    ["All"] + sorted({value for value in signals_df["country"].dropna().astype(str) if value}),
                )
            with signal_filter_cols[1]:
                signal_commodity = st.selectbox(
                    "Signal commodity",
                    ["All"] + sorted({value for value in signals_df["commodity"].dropna().astype(str) if value}),
                )
            with signal_filter_cols[2]:
                signal_sector = st.selectbox(
                    "Signal sector",
                    ["All"] + sorted({value for value in signals_df["sector"].dropna().astype(str) if value}),
                )
            with signal_filter_cols[3]:
                signal_event = st.selectbox(
                    "Signal event type",
                    ["All"] + sorted({value for value in signals_df["event_type"].dropna().astype(str) if value}),
                )
            with signal_filter_cols[4]:
                signal_source = st.selectbox(
                    "Signal source",
                    ["All"] + sorted({value for value in signals_df["source_name"].dropna().astype(str) if value}),
                )
            with signal_filter_cols[5]:
                signal_risk = st.selectbox(
                    "Signal risk",
                    ["All"] + sorted({value for value in signals_df["risk_flags"].dropna().astype(str) if value}),
                )
            shown_signals = signals_df.copy()
            for column, selected in [
                ("country", signal_country),
                ("commodity", signal_commodity),
                ("sector", signal_sector),
                ("event_type", signal_event),
                ("source_name", signal_source),
                ("risk_flags", signal_risk),
            ]:
                if selected != "All":
                    shown_signals = shown_signals[shown_signals[column] == selected]
            visible_columns = [
                "date",
                "source_name",
                "source_status",
                "title",
                "country",
                "commodity",
                "sector",
                "event_type",
                "tone",
                "risk_flags",
                "relevance_score",
                "promoted",
                "review_reasons",
                "url",
            ]
            st.dataframe(shown_signals[visible_columns], width="stretch", hide_index=True)
            st.markdown("**Promote signals**")
            signal_options = {
                f"{row['source_name']} | {row['country']} | {row['title'][:90]}": row["signal_id"]
                for _, row in shown_signals.iterrows()
            }
            selected_signal_labels = st.multiselect(
                "Select live signals to promote into stored monitoring events",
                options=list(signal_options),
                key="promote_signal_selection",
            )
            signal_note = st.text_input("Analyst note for selected signal promotion", key="signal_promotion_note")
            if st.button("Promote selected signals", width="stretch", key="promote_selected_signals"):
                promoted = []
                for label in selected_signal_labels:
                    promoted.append(
                        pipeline.promote_monitoring_signal(
                            monitoring_cycle,
                            signal_options[label],
                            analyst_note=signal_note,
                        )
                    )
                st.success(f"Promoted {len(promoted)} signal(s).")
            export_cols = st.columns(3)
            export_cols[0].download_button(
                "Download signals CSV",
                shown_signals.to_csv(index=False).encode("utf-8"),
                file_name="monitoring_signals.csv",
                mime="text/csv",
                width="stretch",
                key="monitoring_signals_table_csv",
            )
            if clusters:
                clusters_df = pd.DataFrame(clusters)
                export_cols[1].download_button(
                    "Download event clusters CSV",
                    clusters_df.to_csv(index=False).encode("utf-8"),
                    file_name="event_clusters.csv",
                    mime="text/csv",
                    width="stretch",
                    key="monitoring_event_clusters_table_csv",
                )
            with export_cols[2]:
                download_json_button("Download monitoring run JSON", monitoring_cycle, "monitoring_cycle.json", "monitoring_run_table_json")
        elif not signals:
            st.info("Run the monitoring cycle from the sidebar to fetch and normalize source signals.")

        if clusters:
            st.markdown("**Event clusters**")
            high_risk_clusters = [cluster for cluster in clusters if str(cluster.get("risk_level", "")).lower() == "high"]
            promote_cols = st.columns([0.55, 0.45])
            cluster_note = promote_cols[0].text_input("Analyst note for cluster promotion", key="cluster_promotion_note")
            if promote_cols[1].button("Promote all high-risk clusters", width="stretch", key="promote_high_risk_clusters"):
                promoted = pipeline.promote_high_risk_clusters(monitoring_cycle, analyst_note=cluster_note)
                st.success(f"Promoted {len(promoted)} high-risk cluster(s).")
            for cluster in clusters[:10]:
                with st.expander(f"{cluster['event_title']} | {cluster['confidence_level']} confidence"):
                    st.write(cluster["what_changed"])
                    st.write(cluster["why_it_matters"])
                    cluster_promoted = cluster.get("event_id", "") in promoted_source_ids
                    cluster_reasons = review_reasons_by_source.get(cluster.get("event_id", ""), "")
                    st.caption(
                        f"Sources: {cluster['sources']} | Risk flags: {cluster['risk_flags'] or 'none detected'} | "
                        f"Promoted: {'yes' if cluster_promoted else 'no'} | Review: {cluster_reasons or 'none'}"
                    )
                    if st.button(
                        "Promote this cluster",
                        key=f"promote_cluster_{cluster['event_id']}",
                        disabled=cluster_promoted,
                    ):
                        pipeline.promote_monitoring_cluster(
                            monitoring_cycle,
                            cluster["event_id"],
                            analyst_note=cluster_note,
                        )
                        st.success("Cluster promoted into stored monitoring events.")
        st.divider()

    st.caption(f"Provider: {news_bundle.get('provider')} | Status: {news_bundle.get('status')} | Query: {news_bundle.get('query')}")
    if news_bundle.get("warning"):
        st.warning(news_bundle["warning"])
    if news_articles:
        if pd is not None:
            news_df = pd.DataFrame(news_articles)
            filter_cols = st.columns(4)
            with filter_cols[0]:
                country_values = sorted({value for value in news_df["country"].dropna().astype(str) if value})
                selected_news_country = st.selectbox("News country", ["All"] + country_values)
            with filter_cols[1]:
                commodity_values = sorted({value for value in news_df["commodity"].dropna().astype(str) if value})
                selected_news_commodity = st.selectbox("News commodity", ["All"] + commodity_values)
            with filter_cols[2]:
                tone_values = sorted({value for value in news_df["sentiment_tone"].dropna().astype(str) if value})
                selected_news_tone = st.selectbox("News tone", ["All"] + tone_values)
            with filter_cols[3]:
                relevance_values = sorted({value for value in news_df["relevance"].dropna().astype(str) if value})
                selected_news_relevance = st.selectbox("News relevance", ["All"] + relevance_values)
            shown_news = news_df.copy()
            if selected_news_country != "All":
                shown_news = shown_news[shown_news["country"] == selected_news_country]
            if selected_news_commodity != "All":
                shown_news = shown_news[shown_news["commodity"] == selected_news_commodity]
            if selected_news_tone != "All":
                shown_news = shown_news[shown_news["sentiment_tone"] == selected_news_tone]
            if selected_news_relevance != "All":
                shown_news = shown_news[shown_news["relevance"] == selected_news_relevance]
            st.dataframe(
                shown_news[
                    [
                        "published_at",
                        "source",
                        "title",
                        "country",
                        "commodity",
                        "topic_tags",
                        "sentiment_tone",
                        "risk_flags",
                        "recommended_action",
                        "url",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )
            st.download_button(
                "Download news signals CSV",
                shown_news.to_csv(index=False).encode("utf-8"),
                file_name="news_signals.csv",
                mime="text/csv",
                width="stretch",
                key="news_signals_csv",
            )
        for article in news_articles[:8]:
            with st.expander(f"{article['source']} | {article['title']}"):
                st.write(article["summary"])
                st.write(f"Country/region: {article['country']}")
                st.write(f"Commodity: {article['commodity']}")
                st.write(f"Risk flags: {article['risk_flags'] or 'none detected'}")
                if article["url"].startswith(("http://", "https://")):
                    st.link_button("Open source", article["url"])
                elif article["url"]:
                    st.caption(f"Source locator: {article['url']}")
    else:
        st.info("No news articles available.")

with panel_tabs[4]:
    st.subheader("Unified Question Answering And Briefs")
    question = st.text_area(
        "Ask across documents, datasets, news, and monitoring signals",
        value="What are the major risks and opportunities in African energy and commodities right now?",
        height=90,
    )
    ask_cols = st.columns([0.5, 0.5])
    with ask_cols[0]:
        if st.button("Answer with evidence trail", type="primary", width="stretch"):
            with st.spinner("Combining document, dataset, news, and monitoring evidence..."):
                st.session_state["workbench_answer"] = pipeline.answer_workbench_question(
                    question=question,
                    document_ids=selected_document_ids,
                    news_articles=news_articles,
                    monitoring_result=monitoring_cycle,
                    top_k=int(top_k),
                )
    with ask_cols[1]:
        if st.button("Generate intelligence brief", width="stretch"):
            with st.spinner("Writing exportable intelligence brief..."):
                output = pipeline.generate_workbench_brief(
                    focus=question,
                    country=country_filter,
                    commodity=commodity_filter,
                    topic=topic_filter,
                    document_ids=selected_document_ids,
                    news_articles=news_articles,
                    top_k=int(top_k),
                )
                paths = pipeline.export_output(output)
                st.session_state["workbench_brief_output"] = output
                st.session_state["workbench_brief_paths"] = paths

    answer = st.session_state.get("workbench_answer")
    if answer:
        st.markdown(answer["answer_markdown"])
        render_evidence_rows(answer["evidence_rows"], "workbench_answer")

    if monitoring_cycle:
        st.subheader("Monitoring-Cycle Intelligence Brief")
        st.markdown(monitoring_cycle["brief_markdown"])
        brief_cols = st.columns(4)
        brief_cols[0].download_button(
            "Download monitoring brief Markdown",
            monitoring_cycle["brief_markdown"].encode("utf-8"),
            file_name="monitoring_intelligence_brief.md",
            mime="text/markdown",
            width="stretch",
            key="monitoring_brief_markdown",
        )
        if pd is not None:
            brief_cols[1].download_button(
                "Download signals CSV",
                pd.DataFrame(monitoring_cycle.get("normalized_signals", [])).to_csv(index=False).encode("utf-8"),
                file_name="monitoring_signals.csv",
                mime="text/csv",
                width="stretch",
                key="monitoring_brief_signals_csv",
            )
            brief_cols[2].download_button(
                "Download clusters CSV",
                pd.DataFrame(monitoring_cycle.get("event_clusters", [])).to_csv(index=False).encode("utf-8"),
                file_name="event_clusters.csv",
                mime="text/csv",
                width="stretch",
                key="monitoring_brief_clusters_csv",
            )
        with brief_cols[3]:
            download_json_button("Download run JSON", monitoring_cycle, "monitoring_cycle.json", "monitoring_brief_run_json")

    brief_output = st.session_state.get("workbench_brief_output")
    if brief_output:
        st.markdown(brief_output.body_markdown)
        render_metric_explanations()
        render_export_downloads(st.session_state.get("workbench_brief_paths"))

with panel_tabs[5]:
    st.subheader("Human Review, Sources, And Audit")
    review_records = pipeline.list_knowledge_records(review_status="review", limit=200)
    if review_records:
        st.markdown("**Knowledge review queue**")
        st.dataframe(flatten_knowledge_records(review_records), width="stretch", hide_index=True)
        review_labels = {
            f"{record['title']} [{record['record_id']}]": record["record_id"]
            for record in review_records
        }
        status_cols = st.columns([0.7, 0.3])
        with status_cols[0]:
            selected_review_label = st.selectbox("Record needing review", list(review_labels))
        with status_cols[1]:
            new_status = st.selectbox("Set status", ["usable", "approved", "rejected", "review"])
        if st.button("Update knowledge review status", width="stretch"):
            pipeline.update_knowledge_record_status(review_labels[selected_review_label], new_status)
            st.success("Review status updated.")
    else:
        st.success("No knowledge records currently marked for review.")

    st.markdown("**Monitoring review queue**")
    review_queue = pipeline.build_monitoring_review_queue(monitoring_cycle)
    if pd is not None and review_queue:
        review_df = pd.DataFrame(review_queue)
        if "review_reasons" in review_df.columns:
            review_df["review_reasons"] = review_df["review_reasons"].apply(display_value)
        st.dataframe(review_df, width="stretch", hide_index=True)
        st.download_button(
            "Download review queue CSV",
            review_df.to_csv(index=False).encode("utf-8"),
            file_name="monitoring_review_queue.csv",
            mime="text/csv",
            width="stretch",
            key="monitoring_review_queue_csv",
        )
        queue_labels = {
            f"{row.get('item_type', '')} | {row.get('status', '')} | {row.get('title', '')[:100]}": row
            for row in review_queue
        }
        review_action_cols = st.columns([0.45, 0.25, 0.3])
        with review_action_cols[0]:
            selected_queue_label = st.selectbox("Review queue item", list(queue_labels), key="selected_monitoring_review_item")
        selected_queue_item = queue_labels[selected_queue_label]
        with review_action_cols[1]:
            queue_status = st.selectbox("Review action", ["reviewed", "archived", "new"], key="monitoring_review_status")
        with review_action_cols[2]:
            queue_note = st.text_input("Analyst note", key="monitoring_review_note")
        action_cols = st.columns(3)
        if action_cols[0].button("Apply review status", width="stretch", key="apply_monitoring_review_status"):
            promotion_id = selected_queue_item.get("promotion_id")
            if promotion_id:
                pipeline.update_promoted_monitoring_item(
                    promotion_id,
                    status=queue_status,
                    analyst_note=queue_note,
                )
                st.success("Promoted item review status updated.")
            else:
                st.warning("This live/session item must be promoted before review status can be persisted.")
        if action_cols[1].button("Promote queue item", width="stretch", key="promote_review_queue_item"):
            if not monitoring_cycle:
                st.warning("Run a monitoring cycle before promoting live queue items.")
            elif selected_queue_item.get("item_type") == "signal":
                pipeline.promote_monitoring_signal(
                    monitoring_cycle,
                    selected_queue_item["source_item_id"],
                    analyst_note=queue_note,
                )
                st.success("Signal promoted.")
            elif selected_queue_item.get("item_type") == "cluster":
                pipeline.promote_monitoring_cluster(
                    monitoring_cycle,
                    selected_queue_item["source_item_id"],
                    analyst_note=queue_note,
                )
                st.success("Cluster promoted.")
            else:
                st.info("This item is already a stored promotion; use review status instead.")
        if action_cols[2].button("Archive promoted item", width="stretch", key="archive_monitoring_review_item"):
            promotion_id = selected_queue_item.get("promotion_id")
            if promotion_id:
                pipeline.update_promoted_monitoring_item(
                    promotion_id,
                    status="archived",
                    analyst_note=queue_note,
                )
                st.success("Promoted item archived.")
            else:
                st.warning("Only promoted items can be archived persistently.")
    else:
        st.info("No monitoring review items currently need attention.")

    st.markdown("**Promoted live monitoring items**")
    promoted_items = pipeline.list_promoted_monitoring_items(limit=1000)
    if pd is not None and promoted_items:
        promoted_df = pd.DataFrame(promoted_items)
        for column in ("supporting_signal_ids", "review_reasons", "payload"):
            if column in promoted_df.columns:
                promoted_df[column] = promoted_df[column].apply(display_value)
        st.dataframe(promoted_df, width="stretch", hide_index=True)
        st.download_button(
            "Download promoted events CSV",
            promoted_df.to_csv(index=False).encode("utf-8"),
            file_name="promoted_monitoring_items.csv",
            mime="text/csv",
            width="stretch",
            key="promoted_monitoring_items_csv",
        )
    else:
        st.info("No live monitoring signals or clusters have been promoted yet.")

    st.markdown("**Monitoring run history**")
    run_history = pipeline.list_monitoring_runs(limit=50)
    if pd is not None and run_history:
        run_df = pd.DataFrame(run_history)
        for column in (
            "filters",
            "sources_attempted",
            "source_statuses",
            "top_countries",
            "top_sectors",
            "top_commodities",
            "top_risk_flags",
            "warnings",
            "errors",
            "export_paths",
        ):
            if column in run_df.columns:
                run_df[column] = run_df[column].apply(display_value)
        st.dataframe(run_df, width="stretch", hide_index=True)
        st.download_button(
            "Download monitoring run history CSV",
            run_df.to_csv(index=False).encode("utf-8"),
            file_name="monitoring_run_history.csv",
            mime="text/csv",
            width="stretch",
            key="monitoring_run_history_csv",
        )
        download_json_button(
            "Download monitoring run history JSON",
            {"monitoring_runs": run_history},
            "monitoring_run_history.json",
            "monitoring_run_history_json",
        )
    else:
        st.info("No monitoring runs have been persisted yet.")

    st.markdown("**Monitoring event explorer**")
    events = pipeline.list_monitoring_events(limit=1000)
    if events:
        st.dataframe(flatten_monitoring_events(events[:500]), width="stretch", hide_index=True)
    else:
        st.info("No monitoring events yet. Generate work products or run the monitoring supervisor.")

    with st.expander("Official and monitoring source administration"):
        source_cols = st.columns(4)
        if source_cols[0].button("Initialize official registry", width="stretch"):
            registry_path, coverage_path = pipeline.initialize_sources()
            st.success(f"Created registry at {registry_path} and country coverage at {coverage_path}.")
        if source_cols[1].button("Initialize monitoring registry", width="stretch"):
            path = pipeline.initialize_monitoring()
            st.success(f"Monitoring registry ready at {path}.")
        download_limit = source_cols[2].number_input("Source download limit", min_value=1, max_value=25, value=3)
        if source_cols[3].button("Download official sources", width="stretch"):
            with st.spinner("Downloading and ingesting registered official sources..."):
                try:
                    results = pipeline.download_sources(limit=int(download_limit), ingest=True)
                    st.session_state["latest_source_download_results"] = results
                    st.success(f"Processed {len(results)} source(s).")
                except Exception as exc:
                    st.error(str(exc))

        if pipeline.config.source_registry_path.exists():
            st.markdown("**Official source registry**")
            st.dataframe(pipeline.list_source_registry(), width="stretch", hide_index=True)
        st.markdown("**Monitoring sources**")
        st.dataframe(pipeline.list_monitoring_sources(), width="stretch", hide_index=True)
        if st.session_state.get("latest_source_download_results"):
            st.markdown("**Latest source download results**")
            st.dataframe(st.session_state["latest_source_download_results"], width="stretch", hide_index=True)

    with st.expander("Audit tables"):
        st.markdown("**File manifests**")
        st.dataframe(pipeline.list_file_manifests(), width="stretch", hide_index=True)
        st.markdown("**Analysis sessions**")
        st.dataframe(flatten_sessions(pipeline.store.list_analysis_sessions(limit=50)), width="stretch", hide_index=True)
        st.markdown("**Generated files**")
        st.dataframe(pipeline.store.list_generated_outputs(), width="stretch", hide_index=True)
        st.markdown("**Audit events**")
        st.dataframe(flatten_audit_events(pipeline.store.list_audit_events(limit=100)), width="stretch", hide_index=True)
