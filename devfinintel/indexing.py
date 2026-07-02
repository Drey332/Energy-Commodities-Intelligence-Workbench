"""Hybrid retrieval: BM25 keywords + dense similarity + transparent reranking.

This module implements the search layer that sits between the evidence database
and generated outputs. It intentionally exposes its scoring pieces so a reviewer
can see why a passage was selected.

Baseline design:
- BM25 finds exact policy terms, donor names, country names, and finance words.
- Dense similarity finds softer matches such as related wording in English/French.
- Reranking adds task-specific boosts for evidence that looks useful for a donor
  profile, BIOFIN case card, bulletin, or general Q&A.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Protocol

from devfinintel.models import DocumentChunk, EvidenceItem, RetrievalDiagnostics, SourceDocument
from devfinintel.utils import normalize_text, tokenize


class Embedder(Protocol):
    """Small interface for any embedding backend."""

    backend_name: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one numeric vector per text."""


class HashingCharNgramEmbedder:
    """Deterministic local fallback for multilingual-friendly dense retrieval.

    This is not as strong as BGE-M3 or another trained embedding model. Its value
    is auditability and zero setup: character n-grams catch related English/French
    spellings, accents are normalized, and every vector is produced locally.
    """

    backend_name = "local-hashing-char-ngram"

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        normalized = normalize_text(text)
        compact = re.sub(r"\s+", " ", normalized)
        tokens = tokenize(compact)

        features: list[str] = tokens[:]
        for token in tokens:
            if len(token) >= 4:
                features.extend(token[i : i + 4] for i in range(len(token) - 3))
        for i in range(max(0, len(compact) - 4)):
            features.append(compact[i : i + 5])

        for feature in features:
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class OptionalSentenceTransformerEmbedder:
    """Use a local sentence-transformers model when one is explicitly configured.

    The code does not download a model automatically. In an institutional setting
    that is important: a user should know when a model is added, where it came
    from, and whether documents leave the machine. Set ``DEVFIN_EMBEDDING_MODEL``
    to a locally available model name/path to enable this backend.
    """

    def __init__(self) -> None:
        model_name = os.getenv("DEVFIN_EMBEDDING_MODEL")
        if not model_name:
            raise RuntimeError("DEVFIN_EMBEDDING_MODEL is not set.")
        from sentence_transformers import SentenceTransformer  # type: ignore

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.backend_name = f"sentence-transformers:{model_name}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return [list(map(float, vector)) for vector in vectors]


@dataclass(frozen=True)
class SearchResult:
    """Internal search result before it is converted to an EvidenceItem."""

    chunk: DocumentChunk
    bm25_score: float
    dense_score: float
    final_score: float


class BM25Index:
    """Transparent keyword index for exact search."""

    def __init__(self, chunks: list[DocumentChunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(chunk.text) for chunk in chunks]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.average_doc_length = sum(self.doc_lengths) / max(1, len(self.doc_lengths))
        self.term_frequencies = [Counter(tokens) for tokens in self.doc_tokens]
        self.document_frequencies: dict[str, int] = defaultdict(int)
        for frequencies in self.term_frequencies:
            for term in frequencies:
                self.document_frequencies[term] += 1
        self.total_documents = len(chunks)

    def scores(self, query: str) -> list[float]:
        """Return BM25 scores for all chunks."""

        query_terms = tokenize(query)
        scores: list[float] = []
        for index, frequencies in enumerate(self.term_frequencies):
            score = 0.0
            doc_length = self.doc_lengths[index] or 1
            for term in query_terms:
                if term not in frequencies:
                    continue
                df = self.document_frequencies.get(term, 0)
                idf = math.log(1 + (self.total_documents - df + 0.5) / (df + 0.5))
                tf = frequencies[term]
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * doc_length / max(1.0, self.average_doc_length)
                )
                score += idf * (tf * (self.k1 + 1)) / denominator
            scores.append(score)
        return scores


class TransparentReranker:
    """Task-specific reranker with simple, reviewable rules."""

    TASK_KEYWORDS = {
        "donor_profile": {
            "donor",
            "partner",
            "partnership",
            "financing",
            "commitment",
            "portfolio",
            "africa",
            "strategic",
            "ifi",
            "development bank",
        },
        "biofin_case": {
            "biodiversity",
            "finance solution",
            "conservation",
            "ecosystem",
            "nature",
            "green bond",
            "subsidy",
            "trust fund",
            "payment for ecosystem",
        },
        "bulletin": {
            "news",
            "announced",
            "launched",
            "signed",
            "partnership",
            "weekly",
            "monthly",
            "private sector",
            "diaspora",
            "cso",
        },
        "qa": set(),
    }

    def boost(self, text: str, task_type: str) -> float:
        """Return a small score boost based on task-relevant evidence signals."""

        normalized = normalize_text(text)
        keywords = self.TASK_KEYWORDS.get(task_type, set())
        keyword_hits = sum(1 for keyword in keywords if keyword in normalized)
        has_money = bool(re.search(r"(us\$|\$|eur|€|gbp|£)\s?\d|usd\s?\d|\d+\s?(million|billion)", normalized))
        has_year = bool(re.search(r"\b20\d{2}\b", normalized))
        length_ok = 100 <= len(text) <= 2200

        boost = min(keyword_hits * 0.035, 0.25)
        if has_money and task_type in {"donor_profile", "biofin_case", "bulletin"}:
            boost += 0.08
        if has_year and task_type in {"donor_profile", "bulletin"}:
            boost += 0.04
        if length_ok:
            boost += 0.03
        return boost


class HybridSearchIndex:
    """Search index built from the current SQLite evidence store."""

    def __init__(
        self,
        chunks: list[DocumentChunk],
        document_lookup: dict[str, SourceDocument],
        embedding_dimensions: int = 384,
    ) -> None:
        self.chunks = chunks
        self.document_lookup = document_lookup
        self.bm25 = BM25Index(chunks)
        self.reranker = TransparentReranker()
        self.embedder = self._choose_embedder(embedding_dimensions)
        self.chunk_vectors = self.embedder.embed([chunk.text for chunk in chunks]) if chunks else []

    @property
    def embedding_backend(self) -> str:
        """Name of the dense retrieval backend used for this index."""

        return self.embedder.backend_name

    def _choose_embedder(self, embedding_dimensions: int) -> Embedder:
        try:
            return OptionalSentenceTransformerEmbedder()
        except Exception:
            return HashingCharNgramEmbedder(dimensions=embedding_dimensions)

    def search(self, query: str, task_type: str, top_k: int = 8) -> list[EvidenceItem]:
        """Retrieve the best evidence for a task and query."""

        if not self.chunks:
            return []

        bm25_scores = self.bm25.scores(query)
        query_vector = self.embedder.embed([query])[0]
        dense_scores = [cosine_similarity(query_vector, vector) for vector in self.chunk_vectors]

        normalized_bm25 = min_max_normalize(bm25_scores)
        normalized_dense = min_max_normalize(dense_scores)

        candidates: list[SearchResult] = []
        for index, chunk in enumerate(self.chunks):
            base_score = 0.55 * normalized_bm25[index] + 0.35 * normalized_dense[index]
            final_score = base_score + self.reranker.boost(chunk.text, task_type)
            candidates.append(
                SearchResult(
                    chunk=chunk,
                    bm25_score=bm25_scores[index],
                    dense_score=dense_scores[index],
                    final_score=final_score,
                )
            )

        candidates.sort(key=lambda result: result.final_score, reverse=True)
        evidence_items: list[EvidenceItem] = []
        for result in candidates[:top_k]:
            document = self.document_lookup[result.chunk.document_id]
            evidence_items.append(
                EvidenceItem(
                    chunk_id=result.chunk.chunk_id,
                    document_id=result.chunk.document_id,
                    title=document.title,
                    source_path=document.source_path,
                    page_number=result.chunk.page_number,
                    text=result.chunk.text,
                    bm25_score=round(result.bm25_score, 6),
                    dense_score=round(result.dense_score, 6),
                    rerank_score=round(result.final_score, 6),
                )
            )
        return evidence_items

    def search_with_diagnostics(
        self,
        query: str,
        task_type: str,
        top_k: int = 8,
    ) -> tuple[list[EvidenceItem], RetrievalDiagnostics]:
        """Retrieve evidence and return pass/fail diagnostics."""

        evidence_items = self.search(query=query, task_type=task_type, top_k=top_k)
        diagnostics = self.diagnose(query=query, task_type=task_type, evidence_items=evidence_items)
        return evidence_items, diagnostics

    def diagnose(
        self,
        query: str,
        task_type: str,
        evidence_items: list[EvidenceItem],
    ) -> RetrievalDiagnostics:
        """Score retrieval quality with transparent thresholds.

        The thresholds are intentionally conservative. They are not a benchmark;
        they are a safety gate that catches cases where the evidence pack is too
        thin to support a factual answer.
        """

        query_terms = meaningful_terms(query)
        evidence_terms = set(tokenize(" ".join(item.text for item in evidence_items)))
        keyword_coverage = (
            len(set(query_terms) & evidence_terms) / len(set(query_terms))
            if query_terms
            else 0.0
        )
        scores = [item.rerank_score for item in evidence_items]
        top_score = max(scores) if scores else 0.0
        average_score = sum(scores) / len(scores) if scores else 0.0

        warnings: list[str] = []
        if not evidence_items:
            warnings.append("No evidence chunks were retrieved.")
        if top_score < 0.08:
            warnings.append("Top retrieval score is low.")
        if query_terms and keyword_coverage < 0.15:
            warnings.append("Retrieved evidence covers few meaningful query terms.")

        passed = bool(evidence_items) and (top_score >= 0.08 or keyword_coverage >= 0.15)
        return RetrievalDiagnostics(
            query=query,
            task_type=task_type,
            document_scope_count=len(self.document_lookup),
            chunk_count=len(self.chunks),
            returned_count=len(evidence_items),
            top_score=round(top_score, 6),
            average_score=round(average_score, 6),
            keyword_coverage=round(keyword_coverage, 6),
            passed=passed,
            warnings=warnings,
        )


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Cosine similarity for already-normalized or ordinary vectors."""

    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left)) or 1.0
    right_norm = math.sqrt(sum(b * b for b in right)) or 1.0
    return numerator / (left_norm * right_norm)


def min_max_normalize(values: list[float]) -> list[float]:
    """Normalize scores to 0-1 so BM25 and dense scores can be combined."""

    if not values:
        return []
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def meaningful_terms(text: str) -> list[str]:
    """Return query terms that are useful for rough retrieval coverage checks."""

    stopwords = {
        "about",
        "does",
        "file",
        "from",
        "have",
        "tell",
        "that",
        "the",
        "this",
        "what",
        "with",
        "uploaded",
    }
    return [term for term in tokenize(text) if len(term) > 2 and term not in stopwords]
