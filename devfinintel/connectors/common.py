"""Shared connector helpers."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from devfinintel.utils import utc_now_iso


USER_AGENT = "devfinintel-monitoring/1.0"


def fetch_json(url: str, *, timeout_seconds: int = 20, headers: dict[str, str] | None = None) -> dict[str, Any]:
    """Fetch JSON from a public endpoint."""

    request = Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    return json.loads(urlopen(request, timeout=timeout_seconds).read().decode("utf-8"))


def connector_result(
    *,
    source_name: str,
    source_type: str,
    source_status: str,
    records: list[dict[str, Any]] | None = None,
    query: str = "",
    url: str = "",
    warnings: list[str] | None = None,
    errors: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a normalized connector envelope."""

    return {
        "source_name": source_name,
        "source_type": source_type,
        "source_status": source_status,
        "records": records or [],
        "query": query,
        "url": url,
        "warnings": warnings or [],
        "errors": errors or [],
        "metadata": metadata or {},
        "retrieved_at": utc_now_iso(),
    }


def build_url(base_url: str, params: dict[str, Any]) -> str:
    """Build a URL with compact query parameters."""

    clean = {key: value for key, value in params.items() if value not in {None, ""}}
    return f"{base_url}?{urlencode(clean)}"


def missing_key_result(source_name: str, env_var: str) -> dict[str, Any]:
    """Return a key-safe missing-key connector result."""

    return connector_result(
        source_name=source_name,
        source_type="optional_key",
        source_status="missing key",
        warnings=[f"{source_name} not queried because {env_var} is not configured."],
        metadata={"env_var": env_var, "secret_visible": "no"},
    )
