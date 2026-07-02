"""Retrieval planning for different UNDP-style work products.

The same evidence database can serve several tasks. A donor profile needs
different evidence than a BIOFIN case card or a bulletin. The planner expands the
user's plain-language request with task-specific terms so retrieval is more
likely to find the right passages.
"""

from __future__ import annotations

from dataclasses import dataclass

from devfinintel.indexing import HybridSearchIndex
from devfinintel.models import EvidenceItem


TASK_LABELS = {
    "dataset_profile": "Dataset Insight",
    "qa": "Evidence-grounded Q&A",
    "document_brief": "Document Brief",
    "donor_profile": "Partner Profile",
    "partner_profile": "Partner Profile",
    "finance_record": "Finance / Resource Record",
    "biofin_case": "Case Study Card",
    "case_study_card": "Case Study Card",
    "bulletin": "Monitoring Digest",
    "monitoring_digest": "Monitoring Digest",
    "meeting_notes": "Meeting Notes",
    "knowledge_product": "Knowledge Product Draft",
    "stakeholder_map": "Stakeholder / Partner Map",
}


TASK_ALIASES = {
    "document_brief": "qa",
    "partner_profile": "donor_profile",
    "finance_record": "donor_profile",
    "case_study_card": "biofin_case",
    "monitoring_digest": "bulletin",
    "meeting_notes": "qa",
    "knowledge_product": "qa",
    "stakeholder_map": "donor_profile",
}


@dataclass(frozen=True)
class RetrievalTask:
    """User request plus the work-product type."""

    task_type: str
    query: str
    top_k: int = 8


class RetrievalPlanner:
    """Build search queries and evidence packs for specific tasks."""

    OVERVIEW_EXPANSION = (
        "summary overview executive summary purpose objective findings conclusions "
        "recommendations key messages report tells me document about "
        "résumé synthèse aperçu objectif objectifs résultats conclusions "
        "recommandations messages clés rapport document"
    )

    TASK_EXPANSIONS = {
        "donor_profile": (
            "donor IFI DAC non-DAC partnership profile financial figures "
            "commitments strategic priorities Africa collaboration portfolio"
        ),
        "partner_profile": (
            "partner stakeholder institution donor investor IFI collaboration portfolio "
            "strategic priorities financing commitments country engagement"
        ),
        "finance_record": (
            "finance funding amount investment loan grant guarantee oil gas mining "
            "critical minerals commodity revenue infrastructure project country"
        ),
        "biofin_case": (
            "BIOFIN biodiversity finance nature-based solution conservation "
            "finance instrument case study country lessons learned"
        ),
        "case_study_card": (
            "case study lesson learned country intervention finance instrument "
            "implementation result actor development outcome"
        ),
        "bulletin": (
            "Africa partnership news donor IFI private sector financing diaspora "
            "CSO announcement monthly bulletin executive weekly"
        ),
        "monitoring_digest": (
            "news monitoring announcement update risk opportunity partner country "
            "Africa energy oil gas mining commodities financing"
        ),
        "meeting_notes": (
            "meeting minutes agenda participants action items follow up decision "
            "partner stakeholder next steps"
        ),
        "knowledge_product": (
            "knowledge product briefing note guidance lessons learned synthesis "
            "recommendations source evidence country policy"
        ),
        "stakeholder_map": (
            "stakeholder partner donor IFI government ministry private sector civil society "
            "role interest influence country engagement"
        ),
        "dataset_profile": "dataset CSV columns rows values indicators country ranking metadata",
        "document_brief": "",
        "qa": "",
    }

    def build_query(self, task: RetrievalTask) -> str:
        """Return the expanded query used for retrieval."""

        if task.task_type == "qa" and is_overview_query(task.query):
            return f"{task.query.strip()} {self.OVERVIEW_EXPANSION}".strip()
        expansion = self.TASK_EXPANSIONS.get(task.task_type, "")
        return f"{task.query.strip()} {expansion}".strip()

    def retrieve(self, index: HybridSearchIndex, task: RetrievalTask) -> list[EvidenceItem]:
        """Return page-cited evidence for a task."""

        expanded_query = self.build_query(task)
        return index.search(expanded_query, task.task_type, top_k=task.top_k)


def is_overview_query(query: str) -> bool:
    """Return True for broad requests asking what a file is about."""

    normalized = query.lower().strip()
    overview_phrases = (
        "what does the uploaded file tell me",
        "what does this file tell me",
        "what is this file about",
        "what is this document about",
        "summarize this",
        "summarise this",
        "summarize the document",
        "summarise the document",
        "tell me about the file",
        "tell me about this document",
    )
    return any(phrase in normalized for phrase in overview_phrases)
