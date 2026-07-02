"""Project configuration and filesystem paths.

The application keeps all local state inside the project folder:

- ``data/input``: source reports, PDFs, news files, and finance datasets.
- ``storage``: the local SQLite database.
- ``outputs``: generated Markdown, CSV, and PDF files.

Keeping state in predictable folders makes the project easier to audit. A UN
reviewer can open the database or outputs folder and see what evidence was used
without knowing any cloud infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    """Small configuration object shared by the pipeline and Streamlit app."""

    root_dir: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = root_dir / "data"
    input_dir: Path = data_dir / "input"
    news_dir: Path = data_dir / "news"
    finance_dir: Path = data_dir / "finance"
    sources_dir: Path = data_dir / "sources"
    source_download_dir: Path = sources_dir / "downloads"
    source_registry_path: Path = sources_dir / "source_registry.csv"
    monitoring_source_registry_path: Path = sources_dir / "monitoring_sources.csv"
    country_coverage_path: Path = sources_dir / "africa_country_coverage.csv"
    storage_dir: Path = root_dir / "storage"
    output_dir: Path = root_dir / "outputs"
    database_path: Path = storage_dir / "workbench.sqlite"

    chunk_size_chars: int = 1600
    chunk_overlap_chars: int = 250
    embedding_dimensions: int = 384

    def ensure_directories(self) -> None:
        """Create local folders used by the app.

        This is intentionally explicit. It lets a non-technical reviewer inspect
        the folders and understand where inputs, database files, and outputs live.
        """

        for path in (
            self.data_dir,
            self.input_dir,
            self.news_dir,
            self.finance_dir,
            self.sources_dir,
            self.source_download_dir,
            self.storage_dir,
            self.output_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


DEFAULT_CONFIG = ProjectConfig()
