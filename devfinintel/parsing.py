"""Document parsing with source metadata and page-level traceability.

The parser's job is deliberately narrow: convert each source file into page-like
text records while preserving where each piece of text came from. Better parsing
can be added later, but downstream modules should never need to guess the source
document or page number.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from devfinintel.models import DocumentPage, SourceDocument
from devfinintel.utils import file_sha256, readable_title_from_path, stable_id, utc_now_iso


@dataclass(frozen=True)
class ParsedDocument:
    """A parsed source document and its page-level text."""

    document: SourceDocument
    pages: list[DocumentPage]


class DocumentParser:
    """Convert local source files into auditable page records."""

    def parse_path(self, path: Path, language_hint: str = "unknown") -> ParsedDocument:
        """Parse one file into a ``ParsedDocument``.

        ``language_hint`` is optional because many UNDP documents are bilingual
        or mixed-language. The hint is stored as metadata; it is not used to hide
        evidence from retrieval.
        """

        path = path.resolve()
        suffix = path.suffix.lower()
        document_id = stable_id(str(path), path.stat().st_size, file_sha256(path))

        if suffix == ".pdf":
            page_texts, backend = self._parse_pdf(path)
            source_type = "pdf"
        elif suffix in {".txt", ".md"}:
            page_texts, backend = self._parse_plain_text(path)
            source_type = suffix.lstrip(".")
        elif suffix == ".csv":
            page_texts, backend = self._parse_csv(path)
            source_type = "csv"
        else:
            page_texts, backend = self._parse_plain_text(path)
            source_type = suffix.lstrip(".") or "unknown"

        document = SourceDocument(
            document_id=document_id,
            title=readable_title_from_path(path),
            source_path=str(path),
            source_type=source_type,
            language_hint=language_hint,
            parser_backend=backend,
            loaded_at=utc_now_iso(),
        )
        pages = [
            DocumentPage(document_id=document_id, page_number=i + 1, text=text.strip())
            for i, text in enumerate(page_texts)
            if text.strip()
        ]
        return ParsedDocument(document=document, pages=pages)

    def _parse_pdf(self, path: Path) -> tuple[list[str], str]:
        """Extract page text from a PDF.

        The preferred backend is Docling when it is installed because it is
        designed for layout-aware conversion of PDFs, tables, and document
        structure. If Docling is unavailable, the parser falls back to PyMuPDF
        and then pypdf. The chosen backend is stored in the manifest so reviewers
        know the evidence quality level.
        """

        try:
            from docling.document_converter import DocumentConverter  # type: ignore

            converter = DocumentConverter()
            result = converter.convert(str(path))
            markdown = result.document.export_to_markdown()
            # Docling returns a structured document. The lightweight baseline
            # stores the Markdown representation as one logical page so later
            # code can still cite and retrieve it. A production upgrade can map
            # Docling elements back to exact page/table/caption IDs.
            return [markdown], "docling-markdown"
        except Exception:
            pass

        try:
            import fitz  # type: ignore

            page_texts: list[str] = []
            with fitz.open(path) as pdf:
                for page in pdf:
                    page_texts.append(page.get_text("text"))
            return page_texts, "pymupdf"
        except Exception:
            pass

        try:
            from pypdf import PdfReader  # type: ignore

            reader = PdfReader(str(path))
            page_texts = [page.extract_text() or "" for page in reader.pages]
            return page_texts, "pypdf"
        except Exception as exc:
            raise RuntimeError(
                f"Could not parse PDF {path}. Install PyMuPDF or pypdf, or use TXT/CSV input."
            ) from exc

    def _parse_plain_text(self, path: Path) -> tuple[list[str], str]:
        """Read a text-like file and split it into page-sized sections."""

        text = path.read_text(encoding="utf-8", errors="replace")
        # Form-feed is commonly used to mark page breaks in extracted text.
        pages = [part.strip() for part in text.split("\f") if part.strip()]
        return pages or [text], "plain-text"

    def _parse_csv(self, path: Path) -> tuple[list[str], str]:
        """Represent CSV rows as readable evidence text.

        CSV files often hold news clippings or finance datasets. Instead of
        hiding them in a dataframe, we convert each row into a simple labelled
        text block that can be searched and cited like a document page.
        """

        rows: list[str] = []
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            for row_number, row in enumerate(reader, start=1):
                values = [f"{key}: {value}" for key, value in row.items() if value]
                rows.append(f"CSV row {row_number}\n" + "\n".join(values))
        return rows, "csv-dict-reader"
