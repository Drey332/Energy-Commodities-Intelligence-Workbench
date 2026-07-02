"""Small utility functions shared across the workbench."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path


def utc_now_iso() -> str:
    """Return an ISO timestamp for audit logs and output filenames."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stable_id(*parts: object, length: int = 16) -> str:
    """Create a repeatable short identifier from text-like values.

    We use stable IDs so that re-ingesting the same document does not create a
    confusing pile of duplicate records. The ID is not secret; it is simply a
    compact label derived from the document path, page number, or chunk text.
    """

    raw = "||".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def file_sha256(path: Path) -> str:
    """Return a SHA-256 checksum for a local file.

    A checksum is a fingerprint of the file contents. It lets the app recognize
    the same uploaded file even if the modification time changes, which avoids
    creating confusing duplicate document IDs during repeated analysis.
    """

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    """Normalize text for search while preserving human-readable output elsewhere."""

    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return normalized.lower()


def tokenize(text: str) -> list[str]:
    """Tokenize English/French policy text for transparent keyword search.

    This is not meant to be a perfect linguistic model. It is a readable,
    auditable tokenizer that works well enough for BM25 search and avoids a black
    box dependency in the baseline application.
    """

    return re.findall(r"[a-zA-ZÀ-ÿ0-9][a-zA-ZÀ-ÿ0-9'-]{1,}", normalize_text(text))


def sentence_split(text: str) -> list[str]:
    """Split text into sentences using a conservative rule-based splitter."""

    parts = re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-Ý0-9])", text.strip())
    return [part.strip() for part in parts if part.strip()]


def slugify(text: str, max_length: int = 70) -> str:
    """Create a safe filename slug from a title or query."""

    text = normalize_text(text)
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return (text[:max_length].strip("-") or "output")


def readable_title_from_path(path: Path) -> str:
    """Turn a file path into a title that looks reasonable in citations."""

    return path.stem.replace("_", " ").replace("-", " ").strip()
