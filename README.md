# Africa Energy & Commodities Intelligence Workbench

A public-data intelligence workbench for monitoring African energy, commodity,
finance, and risk signals from live news, institutional reports, uploaded
documents, and structured datasets.

This project helps turn scattered information - news articles, reports,
datasets, and institutional documents - into structured monitoring signals,
clustered developments, evidence-backed Q&A, dashboards, review queues, and
analyst-style briefs.

## Table Of Contents

- [Why This Project Matters](#why-this-project-matters)
- [Core Problem](#core-problem)
- [What The Platform Does](#what-the-platform-does)
- [Key Features](#key-features)
- [Example Use Cases](#example-use-cases)
- [Tech Stack](#tech-stack)
- [Data And Source Connectors](#data-and-source-connectors)
- [Architecture](#architecture)
- [Analyst Workflow](#analyst-workflow)
- [Setup](#setup)
- [Environment Variables](#environment-variables)
- [Running Tests](#running-tests)
- [Sample Outputs](#sample-outputs)
- [Why This Demonstrates Data And Knowledge Skills](#why-this-demonstrates-data-and-knowledge-skills)
- [Data Ethics And Security](#data-ethics-and-security)
- [Limitations](#limitations)
- [Roadmap](#roadmap)

## Why This Project Matters

African energy and commodity systems affect development finance, climate
resilience, infrastructure, trade, fiscal stability, jobs, private-sector
investment, and regional risk. Analysts working on these issues often have to
track fast-moving developments across fragmented reports, press releases,
datasets, news sources, and partner updates.

The workbench shows how public data and document intelligence can organize that
information into decision-ready knowledge products. It supports monitoring,
evidence synthesis, knowledge management, dashboarding, human review, and
source-transparent brief generation. The goal is not to replace analysts. The
goal is to make their evidence trail clearer and their first-pass monitoring
work faster.

## Core Problem

Development and policy teams often receive large volumes of PDFs, reports, CSVs,
news items, project updates, and partner materials. The challenge is not only
access to information. The harder task is turning that information into usable
knowledge:

- What country or region is affected?
- Which sector or commodity is involved?
- Is the item about financing, policy reform, investment, risk, or delay?
- Which source supports the claim?
- Is the signal strong enough to include in a brief?
- What should be reviewed, escalated, or monitored next?

This workbench is built around that workflow.

## What The Platform Does

1. Ingests PDFs, text files, Markdown files, CSV datasets, and live/public source
   results.
2. Parses documents into page-preserving evidence chunks.
3. Profiles CSV datasets with rows, columns, numeric ranges, missingness,
   rankings, correlations, and charts.
4. Normalizes raw monitoring content into structured signals.
5. Classifies signals by country, region, sector, commodity, event type, risk
   flags, tone, source, and relevance.
6. Clusters related signals into developments.
7. Supports evidence-backed questions across documents, data summaries, news,
   and monitoring events.
8. Generates exportable intelligence briefs and review packages.
9. Promotes important signals or clusters into persistent local monitoring
   storage.
10. Maintains review queues, source status tables, run history, file manifests,
    and audit records.

## Key Features

- Unified Monitoring Intelligence workbench for African energy, commodities,
  finance, and risk.
- Live/keyless source monitoring through public connectors.
- Optional API-key source monitoring for richer news and data coverage.
- Document inspection, search, chunk retrieval, and evidence-grounded Q&A.
- Dataset profiling with charts and computed statistics.
- Normalized monitoring-signal schema.
- Event clustering for related developments.
- Evidence-backed Q&A and intelligence brief generation.
- Monitoring Supervisor that summarizes current situation, source health, and
  next actions.
- Review queue for weak evidence, low confidence, missing sources, duplicate-like
  items, and high-risk signals.
- Exportable Markdown, CSV, PDF, evidence JSON, manifest JSON, and ZIP review
  packages.
- Source transparency, local SQLite audit storage, and quality flags.

## Example Use Cases

- Track recent developments in Nigeria's oil and fuel sector.
- Monitor copper and cobalt developments in Democratic Republic of the Congo and
  Zambia.
- Follow South Africa electricity, power reliability, and investment risk.
- Compare news signals with structured commodity, finance, or energy datasets.
- Generate a short Africa energy and commodities monitoring brief.
- Review high-risk or low-confidence signals before including them in a brief.
- Support development-sector research, knowledge management, source monitoring,
  and policy brief preparation.

## Tech Stack

The stack is intentionally local and auditable:

- Python 3.10+
- Streamlit for the dashboard
- pandas for tabular exploration
- Plotly for charts
- SQLite for local document, monitoring, audit, and review storage
- pypdf for baseline PDF parsing
- ReportLab for PDF export
- urllib-based API connectors for public and optional-key data sources
- Deterministic BM25 plus local hashing-based dense retrieval fallback
- Optional `sentence-transformers` backend through `DEVFIN_EMBEDDING_MODEL`
- Optional PyMuPDF and Docling parsing paths when installed
- `unittest` test suite
- `.env`-based local configuration for optional API keys

The code does not require a hosted LLM or cloud vector database. Optional local
Ollama support can answer follow-up questions from the retrieved evidence pack
only.

## Data And Source Connectors

The monitoring layer is built around connector envelopes and normalized signal
records. The app can run without any API keys.

Keyless/public sources:

- GDELT
- ReliefWeb
- World Bank Indicators
- World Bank Documents

Optional API-key sources:

- NewsAPI
- GNews
- Guardian
- EIA

The app uses `.env` for optional keys and shows only key status, never key
values. If keys are missing or live sources fail, the platform can still run
through keyless public connectors and clearly marked sample fallback data.

## Architecture

```text
Sources
  |
  |-- keyless connectors
  |-- optional-key connectors
  |-- uploaded PDFs / CSVs / text files
  |-- official source registry
  |
Connectors and parsers
  |
Normalized signals + page-preserving evidence chunks + dataset summaries
  |
Event clustering + retrieval diagnostics + evidence packs
  |
Q&A / briefs / review queue / monitoring supervisor / exports
```

Module map:

```text
app/streamlit_app.py              Streamlit intelligence workbench
devfinintel/connectors/           GDELT, ReliefWeb, World Bank, NewsAPI, GNews, Guardian, EIA
devfinintel/env.py                .env loading and key-safe source status
devfinintel/parsing.py            PDF, TXT, Markdown, and CSV parsing
devfinintel/chunking.py           page-preserving evidence chunks
devfinintel/store.py              SQLite document, monitoring, audit, and review store
devfinintel/indexing.py           BM25 + local dense retrieval + transparent reranking
devfinintel/signals.py            normalized monitoring-signal schema and classifiers
devfinintel/events.py             event clustering and monitoring-cycle summaries
devfinintel/monitoring.py         source registry, events, insights, and supervisor logic
devfinintel/datasets.py           CSV profiling and chart-ready summaries
devfinintel/workbench.py          unified document/data/news/monitoring Q&A and briefs
devfinintel/evidence.py           bounded evidence packs for drafting and export
devfinintel/verification.py       citation, support, schema, and numeric checks
devfinintel/exporting.py          Markdown, CSV, PDF, JSON, manifest, and ZIP exports
devfinintel/pipeline.py           end-to-end orchestration
tests/                            unittest coverage for core workflows
```

## Analyst Workflow

Typical monitoring workflow:

```text
Run Monitoring Cycle
  -> Inspect source status
  -> Review normalized signals
  -> Review event clusters
  -> Promote important signals or clusters
  -> Run Monitoring Supervisor
  -> Ask evidence-backed questions
  -> Generate an intelligence brief
  -> Review quality flags and source trail
  -> Export outputs
```

The review queue is important. It prevents the platform from treating every live
news item as reliable knowledge. Signals can be flagged for low confidence,
missing source URLs, weak evidence, high risk, uncertain classification, or
duplicate-like behavior before they are reused in briefings.

## Setup

Clone the repository after creating it on GitHub:

```bash
git clone https://github.com/Drey332/africa-energy-commodities-intelligence.git
cd africa-energy-commodities-intelligence
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional local upgrades:

```bash
pip install -r requirements-advanced.txt
```

Create a local environment file:

```bash
cp .env.example .env
```

Add optional API keys to `.env` if you have them. The app runs without keys.

Run the dashboard:

```bash
streamlit run app/streamlit_app.py
```

Generated local state is written to:

- `storage/workbench.sqlite`
- `outputs/`
- `data/sources/downloads/`

These paths are ignored by Git.

## Environment Variables

`.env.example` documents the supported local settings:

```bash
NEWSAPI_API_KEY=
GNEWS_API_KEY=
EIA_API_KEY=
GUARDIAN_API_KEY=

DEVFIN_DEFAULT_REGION=Africa
DEVFIN_NEWS_LOOKBACK_DAYS=7
DEVFIN_MAX_ARTICLES=50
DEVFIN_USE_SAMPLE_DATA=true
```

Do not commit `.env`. The repository includes `.env.example` for setup only.

## Running Tests

```bash
PYTHONPYCACHEPREFIX=/private/tmp/devfin_pycache python3 -m compileall devfinintel app
PYTHONPYCACHEPREFIX=/private/tmp/devfin_pycache python3 -m unittest discover -s tests
```

If you are not on a restricted filesystem, the `PYTHONPYCACHEPREFIX` prefix is
optional:

```bash
python3 -m compileall devfinintel app
python3 -m unittest discover -s tests
```

## Sample Outputs

Screenshots are not committed yet. Suggested screenshots to add under
`docs/screenshots/`:

- Monitoring Intelligence overview
- Source status panel
- Normalized signals table
- Event clusters
- Monitoring Supervisor brief
- Dataset explorer
- Document evidence view
- Export buttons and review queue

Exported files include:

- `.md` readable brief or dataset profile
- `.csv` structured records
- `.pdf` shareable report
- `.evidence.json` evidence pack, records, metrics, and findings
- `.manifest.json` source and output manifest
- `.zip` review package

## Why This Demonstrates Data And Knowledge Skills

This project demonstrates development-sector data and knowledge capabilities in
a concrete workflow:

- Development data analysis through CSV profiling, rankings, missingness checks,
  and chart-ready summaries.
- Knowledge management through reusable records, source registries, review
  queues, audit tables, and export packages.
- Document intelligence through page-level parsing, chunking, retrieval, and
  source-grounded Q&A.
- Public-source monitoring through live/keyless connectors, optional API-key
  connectors, source health status, and fallback behavior.
- Evidence synthesis through event clustering, monitoring briefs, source trails,
  and human-review flags.
- Policy-relevant communication through analyst-style summaries, country and
  commodity framing, risk flags, recommended actions, and exportable briefs.
- Responsible AI/data use through local-first processing, key-safe environment
  handling, abstention behavior, and quality metrics.

For recruiters in organizations such as UNDP, the World Bank, UNESCO, or other
development institutions, the project shows the ability to turn messy public
information into transparent, reviewable knowledge products.

## Data Ethics And Security

- `.env` is ignored and should never be committed.
- Optional API keys are loaded locally and never printed in the UI.
- The app is designed for public and non-confidential materials.
- Do not upload confidential, personal, or restricted documents.
- Live signals are treated as monitoring inputs, not verified facts.
- Analyst review remains part of the workflow before signals are reused in
  external products.

## Limitations

- Public APIs can have delays, rate limits, missing fields, or coverage gaps.
- Live results depend on network access and optional API availability.
- Event clustering is deterministic and not equivalent to expert judgment.
- Sentiment/tone is a monitoring and risk signal, not a claim that an article is
  objectively positive or negative.
- Source quality varies across news, institutional reports, and datasets.
- The current retrieval gate and verification metrics are transparent heuristics,
  not calibrated truth scores.
- The platform supports analyst review. It does not replace analysts.

## Roadmap

- Stronger entity resolution for countries, partners, projects, and companies.
- Better deduplication across live news, official reports, and promoted events.
- More polished promoted-event workflows, watchlists, and review analytics.
- Monitoring run comparison across weeks.
- Improved country and commodity watchlists.
- Evaluation harness for Q&A, retrieval, briefs, citation precision, and numeric
  faithfulness.
- Additional institutional connectors for AfDB, IEA, EITI, OECD, IATI, and World
  Bank project data.
- Improved maps, time-series analytics, and side-by-side official-report versus
  live-signal comparison.
- More screenshot examples and public demo data.
