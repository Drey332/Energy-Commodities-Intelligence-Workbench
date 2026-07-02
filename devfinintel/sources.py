"""Official-source registry and downloader.

This module is the entry point for turning the project from an upload-only demo
into a managed knowledge platform. It keeps a small registry of official source
URLs, records which countries and themes each source is meant to cover, and
downloads files into a predictable folder with checksums.

The design is intentionally conservative. A real policy assistant should not
silently scrape the open web and then mix unverified documents into an evidence
store. Instead, every source starts as a registry row that a reviewer can open,
edit, approve, or remove.
"""

from __future__ import annotations

import csv
import html
import json
import re
import shutil
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from devfinintel.utils import file_sha256, slugify, stable_id, utc_now_iso


AFRICAN_COUNTRIES: tuple[tuple[str, str, str], ...] = (
    ("DZA", "Algeria", "North Africa"),
    ("AGO", "Angola", "Central/Southern Africa"),
    ("BEN", "Benin", "West Africa"),
    ("BWA", "Botswana", "Southern Africa"),
    ("BFA", "Burkina Faso", "West Africa"),
    ("BDI", "Burundi", "East Africa"),
    ("CPV", "Cabo Verde", "West Africa"),
    ("CMR", "Cameroon", "Central Africa"),
    ("CAF", "Central African Republic", "Central Africa"),
    ("TCD", "Chad", "Central Africa"),
    ("COM", "Comoros", "East Africa"),
    ("COG", "Republic of the Congo", "Central Africa"),
    ("COD", "Democratic Republic of the Congo", "Central Africa"),
    ("CIV", "Cote d'Ivoire", "West Africa"),
    ("DJI", "Djibouti", "East Africa"),
    ("EGY", "Egypt", "North Africa"),
    ("GNQ", "Equatorial Guinea", "Central Africa"),
    ("ERI", "Eritrea", "East Africa"),
    ("SWZ", "Eswatini", "Southern Africa"),
    ("ETH", "Ethiopia", "East Africa"),
    ("GAB", "Gabon", "Central Africa"),
    ("GMB", "Gambia", "West Africa"),
    ("GHA", "Ghana", "West Africa"),
    ("GIN", "Guinea", "West Africa"),
    ("GNB", "Guinea-Bissau", "West Africa"),
    ("KEN", "Kenya", "East Africa"),
    ("LSO", "Lesotho", "Southern Africa"),
    ("LBR", "Liberia", "West Africa"),
    ("LBY", "Libya", "North Africa"),
    ("MDG", "Madagascar", "East Africa"),
    ("MWI", "Malawi", "Southern Africa"),
    ("MLI", "Mali", "West Africa"),
    ("MRT", "Mauritania", "West Africa"),
    ("MUS", "Mauritius", "East Africa"),
    ("MAR", "Morocco", "North Africa"),
    ("MOZ", "Mozambique", "Southern Africa"),
    ("NAM", "Namibia", "Southern Africa"),
    ("NER", "Niger", "West Africa"),
    ("NGA", "Nigeria", "West Africa"),
    ("RWA", "Rwanda", "East Africa"),
    ("STP", "Sao Tome and Principe", "Central Africa"),
    ("SEN", "Senegal", "West Africa"),
    ("SYC", "Seychelles", "East Africa"),
    ("SLE", "Sierra Leone", "West Africa"),
    ("SOM", "Somalia", "East Africa"),
    ("ZAF", "South Africa", "Southern Africa"),
    ("SSD", "South Sudan", "East Africa"),
    ("SDN", "Sudan", "East Africa"),
    ("TZA", "Tanzania", "East Africa"),
    ("TGO", "Togo", "West Africa"),
    ("TUN", "Tunisia", "North Africa"),
    ("UGA", "Uganda", "East Africa"),
    ("ZMB", "Zambia", "Southern Africa"),
    ("ZWE", "Zimbabwe", "Southern Africa"),
)


REGISTRY_FIELDS = (
    "source_id",
    "title",
    "publisher",
    "year",
    "url",
    "source_type",
    "topics",
    "countries",
    "regions",
    "license_note",
    "status",
    "local_path",
    "retrieved_at",
    "file_sha256",
    "notes",
)


@dataclass(frozen=True)
class SourceRegistryEntry:
    """One auditable source row.

    ``countries`` can contain a semicolon-separated list of country names, or
    ``ALL_AFRICA`` for sources that are meant to support region-wide screening.
    """

    source_id: str
    title: str
    publisher: str
    year: str
    url: str
    source_type: str
    topics: str
    countries: str
    regions: str
    license_note: str
    status: str = "registered"
    local_path: str = ""
    retrieved_at: str = ""
    file_sha256: str = ""
    notes: str = ""

    def as_row(self) -> dict[str, str]:
        """Return the registry entry as a CSV row."""

        return {field: str(getattr(self, field)) for field in REGISTRY_FIELDS}


@dataclass(frozen=True)
class SourceDownloadResult:
    """Short download summary for CLI and dashboard display."""

    source_id: str
    title: str
    status: str
    local_path: str
    file_sha256: str
    message: str = ""


DEFAULT_SEED_SOURCES: tuple[SourceRegistryEntry, ...] = (
    SourceRegistryEntry(
        source_id="iea-africa-energy-outlook-2022",
        title="Africa Energy Outlook 2022",
        publisher="International Energy Agency",
        year="2022",
        url="https://iea.blob.core.windows.net/assets/220b2862-33a6-47bd-81e9-00e586f4d384/AfricaEnergyOutlook2022.pdf",
        source_type="pdf",
        topics="energy access; oil and gas; critical minerals; investment; clean energy transition",
        countries="ALL_AFRICA",
        regions="Africa",
        license_note="IEA page cites CC BY 4.0 for the report page; confirm redistribution terms before external sharing.",
        notes="Flagship Africa energy source for access, oil/gas, minerals, investment, and transition questions.",
    ),
    SourceRegistryEntry(
        source_id="iea-african-hydropower-climate-2020",
        title="Climate Impacts on African Hydropower",
        publisher="International Energy Agency",
        year="2020",
        url="https://iea.blob.core.windows.net/assets/4878b887-dbc3-470a-bf74-df0304d537e1/ClimateimpactsonAfricanhydropower_CORR.pdf",
        source_type="pdf",
        topics="hydropower; climate resilience; energy access; infrastructure",
        countries="ALL_AFRICA",
        regions="Africa",
        license_note="Confirm IEA content-use terms before redistribution.",
        notes="Useful for country-specific hydropower exposure and resilience records.",
    ),
    SourceRegistryEntry(
        source_id="iea-clean-energy-north-africa-2020",
        title="Clean Energy Transitions in North Africa",
        publisher="International Energy Agency",
        year="2020",
        url="https://www.iea.org/reports/clean-energy-transitions-in-north-africa",
        source_type="web_page",
        topics="clean energy transition; oil and gas; renewables; energy efficiency; North Africa",
        countries="Algeria; Egypt; Libya; Morocco; Tunisia",
        regions="North Africa",
        license_note="Registry stores source page; confirm download and redistribution terms before sharing.",
        notes="The page is converted to a local Markdown snapshot when downloaded.",
    ),
    SourceRegistryEntry(
        source_id="worldbank-africas-pulse-2025",
        title="Africa's Pulse October 2025: Pathways to Job Creation in Africa",
        publisher="World Bank",
        year="2025",
        url="https://www.worldbank.org/en/publication/africa-pulse",
        source_type="web_page",
        topics="jobs; energy infrastructure; mineral value chains; macroeconomy; private sector development",
        countries="ALL_AFRICA",
        regions="Sub-Saharan Africa",
        license_note="World Bank page links to official PDF; confirm Open Knowledge Repository terms before redistribution.",
        notes="Seed source for development, jobs, infrastructure, energy, and mineral-value-chain monitoring.",
    ),
    SourceRegistryEntry(
        source_id="eiti-country-reports",
        title="EITI Country Reports",
        publisher="Extractive Industries Transparency Initiative",
        year="2026",
        url="https://eiti.org/eiti-country-reports",
        source_type="web_page",
        topics="extractives governance; oil; gas; mining; revenues; commodity transparency",
        countries="Angola; Burkina Faso; Cameroon; Central African Republic; Chad; Cote d'Ivoire; Democratic Republic of the Congo; Ethiopia; Gabon; Ghana; Guinea; Liberia; Madagascar; Malawi; Mali; Mauritania; Mozambique; Niger; Nigeria; Republic of the Congo; Senegal; Seychelles; Sierra Leone; Sao Tome and Principe; Tanzania; Togo; Uganda; Zambia",
        regions="Africa",
        license_note="EITI content use policy allows republication unless otherwise noted; cite source and links.",
        notes="Use as a registry hub before downloading individual country reports.",
    ),
    SourceRegistryEntry(
        source_id="eiti-countries",
        title="EITI Countries",
        publisher="Extractive Industries Transparency Initiative",
        year="2026",
        url="https://eiti.org/countries",
        source_type="web_page",
        topics="extractives governance; country status; validation outcome; accountability",
        countries="Angola; Burkina Faso; Cameroon; Central African Republic; Chad; Cote d'Ivoire; Democratic Republic of the Congo; Ethiopia; Gabon; Ghana; Guinea; Liberia; Madagascar; Malawi; Mali; Mauritania; Mozambique; Niger; Nigeria; Republic of the Congo; Senegal; Seychelles; Sierra Leone; Sao Tome and Principe; Tanzania; Togo; Uganda; Zambia",
        regions="Africa",
        license_note="EITI content use policy allows republication unless otherwise noted; cite source and links.",
        notes="Country-page hub for extractives governance status and follow-up report discovery.",
    ),
)


def initialize_source_registry(
    registry_path: Path,
    country_coverage_path: Path,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Create the registry CSV and the 54-country Africa coverage roster."""

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not registry_path.exists():
        save_source_registry(registry_path, list(DEFAULT_SEED_SOURCES))

    if overwrite or not country_coverage_path.exists():
        write_country_coverage(country_coverage_path)

    return registry_path, country_coverage_path


def write_country_coverage(path: Path) -> None:
    """Write the Africa country roster used for coverage checks."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("iso3", "country", "subregion"))
        writer.writeheader()
        for iso3, country, subregion in AFRICAN_COUNTRIES:
            writer.writerow({"iso3": iso3, "country": country, "subregion": subregion})


def load_source_registry(path: Path) -> list[SourceRegistryEntry]:
    """Read source rows from a registry CSV."""

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        entries = []
        for row in reader:
            values = {field: row.get(field, "") for field in REGISTRY_FIELDS}
            entries.append(SourceRegistryEntry(**values))
        return entries


def save_source_registry(path: Path, entries: list[SourceRegistryEntry]) -> None:
    """Write source rows to a registry CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        for entry in entries:
            writer.writerow(entry.as_row())


def download_registered_sources(
    registry_path: Path,
    download_dir: Path,
    *,
    limit: int | None = None,
    source_ids: set[str] | None = None,
    ingestible_only: bool = True,
) -> list[SourceDownloadResult]:
    """Download source files listed in the registry and update the registry.

    HTML pages are converted into small Markdown snapshots. This means a source
    page can still be cited and searched locally while the user later decides
    whether to download deeper country-level PDFs.
    """

    entries = load_source_registry(registry_path)
    selected = [
        entry
        for entry in entries
        if (source_ids is None or entry.source_id in source_ids)
        and entry.status != "downloaded"
        and (not ingestible_only or entry.source_type in {"pdf", "web_page", "txt", "md", "csv"})
    ]
    if limit is not None:
        selected = selected[: max(limit, 0)]

    updated_by_id: dict[str, SourceRegistryEntry] = {entry.source_id: entry for entry in entries}
    results: list[SourceDownloadResult] = []
    for entry in selected:
        result, updated_entry = download_source(entry, download_dir)
        updated_by_id[entry.source_id] = updated_entry
        results.append(result)

    save_source_registry(registry_path, [updated_by_id[entry.source_id] for entry in entries])
    return results


def download_source(entry: SourceRegistryEntry, download_dir: Path) -> tuple[SourceDownloadResult, SourceRegistryEntry]:
    """Download one source entry into the local source-download folder."""

    download_dir.mkdir(parents=True, exist_ok=True)
    try:
        content, content_type = fetch_url(entry.url)
        extension = choose_extension(entry, content_type)
        destination = download_dir / f"{entry.source_id}-{slugify(entry.title, 50)}{extension}"
        if extension == ".md":
            text = html_to_markdown_snapshot(content.decode("utf-8", errors="replace"), entry)
            destination.write_text(text, encoding="utf-8")
        else:
            destination.write_bytes(content)

        sidecar = destination.with_suffix(destination.suffix + ".source.json")
        sidecar.write_text(
            json.dumps(
                {
                    "source": entry.as_row(),
                    "downloaded_at": utc_now_iso(),
                    "local_path": str(destination),
                    "file_sha256": file_sha256(destination),
                    "content_type": content_type,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        checksum = file_sha256(destination)
        updated = replace_entry(
            entry,
            status="downloaded",
            local_path=str(destination),
            retrieved_at=utc_now_iso(),
            file_sha256=checksum,
        )
        return (
            SourceDownloadResult(
                source_id=entry.source_id,
                title=entry.title,
                status="downloaded",
                local_path=str(destination),
                file_sha256=checksum,
            ),
            updated,
        )
    except Exception as exc:
        updated = replace_entry(entry, status="error", notes=f"{entry.notes} | download error: {exc}")
        return (
            SourceDownloadResult(
                source_id=entry.source_id,
                title=entry.title,
                status="error",
                local_path=entry.local_path,
                file_sha256=entry.file_sha256,
                message=str(exc),
            ),
            updated,
        )


def fetch_url(url: str) -> tuple[bytes, str]:
    """Fetch a URL with a clear user agent and no hidden browser automation."""

    parsed = urlparse(url)
    if parsed.scheme == "file":
        path = Path(parsed.path)
        return path.read_bytes(), "application/octet-stream"
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "DevelopmentFinanceIntelligenceWorkbench/0.1"
            )
        },
    )
    with urlopen(request, timeout=60) as response:  # noqa: S310 - user-approved source registry.
        content_type = response.headers.get("content-type", "")
        return response.read(), content_type


def choose_extension(entry: SourceRegistryEntry, content_type: str) -> str:
    """Choose a local file extension from registry metadata and HTTP headers."""

    lower_url = entry.url.lower()
    lower_type = content_type.lower()
    if entry.source_type == "pdf" or "application/pdf" in lower_type or lower_url.endswith(".pdf"):
        return ".pdf"
    if entry.source_type == "csv" or "text/csv" in lower_type or lower_url.endswith(".csv"):
        return ".csv"
    if entry.source_type in {"txt", "md"}:
        return f".{entry.source_type}"
    return ".md"


def html_to_markdown_snapshot(raw_html: str, entry: SourceRegistryEntry) -> str:
    """Convert a source web page into a simple searchable Markdown snapshot."""

    parser = VisibleTextParser()
    parser.feed(raw_html)
    visible_lines = []
    for line in parser.lines:
        clean = re.sub(r"\s+", " ", html.unescape(line)).strip()
        if len(clean) >= 25:
            visible_lines.append(clean)
    deduped = unique_lines(visible_lines)
    body = "\n\n".join(deduped[:250])
    return (
        f"# {entry.title}\n\n"
        f"- Publisher: {entry.publisher}\n"
        f"- Year: {entry.year}\n"
        f"- Source URL: {entry.url}\n"
        f"- Topics: {entry.topics}\n"
        f"- Countries: {entry.countries}\n"
        f"- Regions: {entry.regions}\n\n"
        "## Page Snapshot\n\n"
        f"{body}\n"
    )


class VisibleTextParser(HTMLParser):
    """Tiny visible-text extractor for source-page snapshots."""

    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and data.strip():
            self.lines.append(data.strip())


def unique_lines(lines: list[str]) -> list[str]:
    """Remove repeated menu/footer lines while preserving page order."""

    seen: set[str] = set()
    result: list[str] = []
    for line in lines:
        key = line.lower()
        if key not in seen:
            seen.add(key)
            result.append(line)
    return result


def replace_entry(entry: SourceRegistryEntry, **updates: str) -> SourceRegistryEntry:
    """Return a new registry entry with selected fields replaced."""

    values = entry.as_row()
    values.update(updates)
    return SourceRegistryEntry(**{field: values.get(field, "") for field in REGISTRY_FIELDS})


def downloaded_paths(results: list[SourceDownloadResult]) -> list[Path]:
    """Return successfully downloaded files that the parser can ingest."""

    paths: list[Path] = []
    for result in results:
        path = Path(result.local_path)
        if result.status == "downloaded" and path.suffix.lower() in {".pdf", ".txt", ".md", ".csv"}:
            paths.append(path)
    return paths


def copy_downloads_to_input(paths: list[Path], input_dir: Path) -> list[Path]:
    """Copy downloaded source files into ``data/input`` for normal ingestion."""

    input_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for path in paths:
        destination = input_dir / path.name
        if path.resolve() != destination.resolve():
            shutil.copy2(path, destination)
        copied.append(destination)
    return copied


def registry_summary(entries: list[SourceRegistryEntry]) -> dict[str, Any]:
    """Return simple registry coverage statistics for the dashboard."""

    covered_countries: set[str] = set()
    topics: set[str] = set()
    for entry in entries:
        for country in expand_countries(entry.countries):
            covered_countries.add(country)
        for topic in split_semicolon(entry.topics):
            topics.add(topic)
    return {
        "sources": len(entries),
        "downloaded_sources": sum(1 for entry in entries if entry.status == "downloaded"),
        "countries_covered": len(covered_countries),
        "topics": sorted(topics),
    }


def expand_countries(countries: str) -> list[str]:
    """Expand ``ALL_AFRICA`` into the 54-country roster."""

    if countries.strip().upper() == "ALL_AFRICA":
        return [country for _, country, _ in AFRICAN_COUNTRIES]
    return split_semicolon(countries)


def split_semicolon(value: str) -> list[str]:
    """Split a semicolon-separated registry field."""

    return [part.strip() for part in value.split(";") if part.strip()]
