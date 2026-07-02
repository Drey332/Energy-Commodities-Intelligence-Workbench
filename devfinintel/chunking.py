"""Split page text into smaller evidence chunks.

The system cites pages, but retrieval works better on smaller passages. This
module keeps both: each chunk is small enough to search precisely and still
remembers its source page.
"""

from __future__ import annotations

from devfinintel.models import DocumentChunk, DocumentPage
from devfinintel.utils import stable_id, tokenize


def chunk_pages(
    pages: list[DocumentPage],
    chunk_size_chars: int,
    overlap_chars: int,
) -> list[DocumentChunk]:
    """Create overlapping chunks from parsed pages.

    Overlap prevents a paragraph split from losing context at the boundary. For
    a policy reviewer, the practical effect is simple: a generated citation can
    still point to the correct page even when the relevant sentence is near the
    end of one chunk and the beginning of another.
    """

    chunks: list[DocumentChunk] = []
    for page in pages:
        text = " ".join(page.text.split())
        if not text:
            continue

        start = 0
        chunk_index = 0
        while start < len(text):
            end = min(start + chunk_size_chars, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_id = stable_id(page.document_id, page.page_number, chunk_index, chunk_text)
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=page.document_id,
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                        text=chunk_text,
                        token_count=len(tokenize(chunk_text)),
                        metadata={"char_start": start, "char_end": end},
                    )
                )
            if end == len(text):
                break
            start = max(end - overlap_chars, start + 1)
            chunk_index += 1
    return chunks

