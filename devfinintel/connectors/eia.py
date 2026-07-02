"""Optional-key EIA connector stub.

The EIA API can be valuable for energy data, but the exact series selection
should be deliberate. This first connector reports key status and keeps the
monitoring cycle non-fatal until a vetted series list is added.
"""

from __future__ import annotations

import os

from devfinintel.connectors.common import connector_result, missing_key_result


def fetch_eia_signals(*, query: str = "", limit: int = 25) -> dict:
    """Return EIA connector status without exposing the key."""

    if not os.getenv("EIA_API_KEY"):
        return missing_key_result("EIA", "EIA_API_KEY")
    return connector_result(
        source_name="EIA",
        source_type="dataset_indicator",
        source_status="configured/not queried",
        records=[],
        query=query,
        warnings=["EIA_API_KEY is configured, but no vetted EIA series connector has been enabled yet."],
        metadata={"requested_limit": limit, "secret_visible": "no"},
    )
