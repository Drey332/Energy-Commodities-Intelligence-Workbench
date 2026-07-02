# Technical Architecture

## Overview

The Africa Energy & Commodities Intelligence Workbench is a local Python and
Streamlit application. It combines source connectors, document parsing, CSV
profiling, normalized monitoring signals, event clustering, evidence-backed Q&A,
review queues, and exports.

The system is intentionally modular. Connectors collect raw rows. Normalizers
turn those rows into a common schema. Retrieval and evidence packs support
Q&A and brief generation. SQLite stores documents, chunks, monitoring events,
analysis sessions, generated outputs, review records, and audit events.

## Source Connectors

Connectors return a common envelope:

```text
source_name
source_type
source_status
records
query
url
warnings
errors
metadata
retrieved_at
```

Implemented connector modules:

- `devfinintel/connectors/gdelt.py`
- `devfinintel/connectors/reliefweb.py`
- `devfinintel/connectors/worldbank_indicators.py`
- `devfinintel/connectors/worldbank_docs.py`
- `devfinintel/connectors/newsapi.py`
- `devfinintel/connectors/gnews.py`
- `devfinintel/connectors/guardian.py`
- `devfinintel/connectors/eia.py`

Keyless connectors can run without secrets. Optional-key connectors return a
safe missing-key result when the relevant environment variable is absent.

## Environment Variable Handling

`devfinintel/env.py` loads `.env` from the project root without overriding
deployment environment variables. It exposes only key presence/status to the UI.
The actual values are never printed.

Supported secret variables:

```bash
NEWSAPI_API_KEY=
GNEWS_API_KEY=
EIA_API_KEY=
GUARDIAN_API_KEY=
```

Supported non-secret settings:

```bash
DEVFIN_DEFAULT_REGION=Africa
DEVFIN_NEWS_LOOKBACK_DAYS=7
DEVFIN_MAX_ARTICLES=50
DEVFIN_USE_SAMPLE_DATA=true
```

## Document And Dataset Intake

`devfinintel/parsing.py` parses local files:

- PDF through Docling when installed, then PyMuPDF, then pypdf.
- TXT and Markdown through plain text reading.
- CSV through Python's CSV reader into row-level evidence text.

`devfinintel/chunking.py` creates page-preserving chunks. Each chunk keeps
document ID and page number so later answers can cite their source.

`devfinintel/datasets.py` computes dataset profiles for CSV files, including
column typing, missingness, numeric profiles, ranking rows, outlier candidates,
correlations, suggested questions, and chart specifications.

## Normalized Signal Schema

`devfinintel/signals.py` maps raw connector records into a shared monitoring
signal with fields such as:

```text
signal_id
source_type
source_name
source_status
title
date
url
country
region
commodity
sector
event_type
summary
tone
risk_flags
relevance_score
evidence_text
retrieved_at
raw_source_id
metadata
```

Classification is deterministic and transparent. It uses term dictionaries and
lightweight detection helpers for country, commodity, sector, event type, tone,
risk flags, and relevance.

## Clustering Logic

`devfinintel/events.py` groups deduplicated signals by:

- country,
- commodity or sector topic,
- event type,
- date bucket.

Each cluster contains:

- event title,
- affected countries,
- commodities,
- sectors,
- event type,
- risk level,
- what changed,
- why it matters,
- supporting signal IDs,
- source count,
- confidence level,
- evidence summary.

The clustering is deterministic and auditable. It is useful for first-pass
monitoring, but it is not a substitute for expert judgment.

## Retrieval, Q&A, And Brief Generation

`devfinintel/indexing.py` implements transparent hybrid retrieval:

- BM25 keyword matching for exact policy terms, countries, donors, commodities,
  and finance words.
- Local hashing-based dense retrieval fallback for softer text similarity.
- Optional `sentence-transformers` backend if `DEVFIN_EMBEDDING_MODEL` is set.
- Transparent reranking with task-specific boosts.
- Retrieval diagnostics and a simple evidence-quality gate.

`devfinintel/evidence.py` builds bounded evidence packs with labels, diagnostics,
citation maps, and context-budget notes.

`devfinintel/workbench.py` combines document rows, dataset summaries, news rows,
and monitoring rows into a unified evidence surface for Q&A and brief generation.

`devfinintel/extraction.py` and `devfinintel/verification.py` support structured
records, citation checks, support overlap, numeric checks, schema checks, and
traceability metrics.

## Monitoring Supervisor

`devfinintel/monitoring.py` maintains a governed monitoring-source registry and
turns reviewed records or supported feeds into monitoring events. It also builds
monitoring insights and a deterministic supervisor run that summarizes:

- source health,
- event counts,
- high-priority risks,
- country watchlist,
- situation brief,
- recommended next actions.

The supervisor can run on stored/promoted events only or include current live
monitoring-cycle results when the analyst selects that option in the UI.

## Local Storage

`devfinintel/store.py` uses SQLite for auditable local storage. It manages:

- documents,
- pages,
- chunks,
- file manifests,
- extraction records,
- knowledge records,
- monitoring sources,
- monitoring events,
- monitoring runs,
- promoted monitoring items,
- action items,
- analysis sessions,
- generated outputs,
- audit events.

SQLite is used because it is inspectable, dependency-light, and appropriate for
a local portfolio workbench.

## Exports

`devfinintel/exporting.py` writes:

- Markdown reports,
- CSV record tables,
- PDF reports,
- evidence JSON,
- run manifest JSON,
- ZIP review packages.

Exports are designed for reviewability. The goal is to make the evidence trail
visible rather than only producing polished prose.

## Fallback Behavior

The app is designed to run even when optional services are unavailable:

- Missing API keys produce key-safe source-status rows.
- Public/keyless connectors are attempted when possible.
- Sample monitoring data can be used when live data is unavailable.
- Retrieval can abstain when selected evidence is weak.
- Optional local LLM support is downstream of evidence retrieval and is not
  required for the core app.

## Tests

The test suite uses Python's built-in `unittest` runner. Current tests cover
pipeline generation, monitoring-cycle metadata, promoted signal/cluster storage,
supervisor inclusion/exclusion of current live results, and review queue
behavior.

Run:

```bash
python3 -m compileall devfinintel app
python3 -m unittest discover -s tests
```

## Limitations

- Public APIs and news feeds can be delayed, incomplete, or rate limited.
- Deterministic classification is explainable but imperfect.
- Event clustering is a first-pass grouping method, not expert verification.
- Sentiment/tone is used as a risk and urgency layer, not as a truth score.
- The retrieval gate and quality metrics are heuristics.
- The platform is designed for public, non-confidential data.
