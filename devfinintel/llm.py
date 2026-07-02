"""Local LLM support for evidence-grounded conversation.

Sensitive uploads should not be sent to a hosted model by default. This module
therefore talks to a local Ollama server at ``localhost``. The LLM is used only
for the "talking" layer: Python still performs parsing, statistics, retrieval,
and chart preparation.

Important guardrail:
The prompt tells the model to answer only from the evidence pack. This cannot
mathematically guarantee zero hallucinations, but it sharply reduces risk and
makes failures easier to catch because every answer must cite evidence labels
such as ``[E1]``.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

from devfinintel.evidence import EvidencePackBuilder, render_evidence_pack
from devfinintel.models import EvidenceItem, ExtractionRecord, GeneratedOutput


DEFAULT_OLLAMA_MODEL = "qwen3:8b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/api"


@dataclass(frozen=True)
class LocalLLMAnswer:
    """LLM response plus audit metadata."""

    answer_markdown: str
    model: str
    evidence_count: int
    citation_count: int
    warning: str = ""


class OllamaGroundedLLM:
    """Small Ollama client using Python's standard library."""

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        timeout_seconds: int = 120,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def answer_from_output(self, question: str, output: GeneratedOutput) -> LocalLLMAnswer:
        """Answer a follow-up question from one generated output's evidence pack."""

        if output.evidence_pack is not None:
            evidence_pack = render_evidence_pack(output.evidence_pack)
        else:
            evidence_pack = build_evidence_pack(output.evidence_items, output.records)
        if not evidence_pack:
            return LocalLLMAnswer(
                answer_markdown="I do not have retrieved evidence to answer from.",
                model=self.model,
                evidence_count=0,
                citation_count=0,
                warning="No evidence was available for the local LLM.",
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a local evidence-grounded analysis assistant for sensitive policy and dataset files.\n"
                    "Use ONLY the evidence in the user's message.\n"
                    "Do not use outside knowledge, memory, assumptions, or web facts.\n"
                    "If the evidence does not answer the question, say exactly: "
                    "'I do not have enough evidence in the uploaded file to answer that.'\n"
                    "Cite every factual claim with evidence labels like [E1] or [R1].\n"
                    "For numbers, copy the exact value from the evidence.\n"
                    "For charts or statistics, describe only what the Python-computed evidence says.\n"
                    "Do not reveal hidden reasoning. Keep the answer concise and practical."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Evidence pack:\n{evidence_pack}\n\n"
                    "Answer in Markdown. Use bullet points when useful."
                ),
            },
        ]
        content = self._chat(messages)
        citation_count = len(re.findall(r"\[(?:E|R)\d+\]", content))
        warning = ""
        if citation_count == 0:
            warning = "The LLM answer did not include evidence labels. Treat it as needing review."
            content += (
                "\n\n**Review note:** This answer did not include evidence labels, "
                "so it should be checked against the evidence pack before use."
            )
        return LocalLLMAnswer(
            answer_markdown=strip_qwen_thinking(content),
            model=self.model,
            evidence_count=len(output.evidence_items) + len(output.records),
            citation_count=citation_count,
            warning=warning,
        )

    def _chat(self, messages: list[dict[str, str]]) -> str:
        """Call Ollama's local chat API with deterministic settings."""

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0,
                "top_p": 0.1,
                "num_predict": 900,
            },
        }
        request = urllib.request.Request(
            url=f"{self.base_url}/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Could not reach local Ollama. Start Ollama and make sure the model is available locally."
            ) from exc

        message = data.get("message", {})
        return str(message.get("content", "")).strip()


def build_evidence_pack(
    evidence_items: list[EvidenceItem],
    records: list[ExtractionRecord],
    max_evidence_chars: int = 900,
) -> str:
    """Build a compact cited evidence pack for the LLM prompt."""

    pack = EvidencePackBuilder().build(
        task_type="local_llm_follow_up",
        query="follow-up question",
        evidence_items=evidence_items,
        records=records,
    )
    return render_evidence_pack(pack, max_evidence_chars=max_evidence_chars)


def strip_qwen_thinking(text: str) -> str:
    """Remove any accidental Qwen-style thinking block from the final answer."""

    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
