"""Development Finance Intelligence Workbench.

This package is intentionally written as plain Python modules rather than a
large framework. The target reviewer may be a policy, partnerships, or finance
colleague rather than a software engineer, so each module maps to one visible
step in the evidence pipeline:

1. Parse documents.
2. Store the cleaned evidence.
3. Build a searchable index.
4. Retrieve evidence for a task.
5. Extract structured facts.
6. Draft an output from those facts.
7. Verify and export the result.
"""

from devfinintel.pipeline import DocumentIntelligencePipeline

__all__ = ["DocumentIntelligencePipeline"]

