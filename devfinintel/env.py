"""Environment loading and key-safe source configuration.

API keys are optional and must never be printed or stored in outputs. This
module loads a local `.env` file when present, without overriding deployment
environment variables, and exposes only boolean/key-status metadata to the UI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


SECRET_NAMES = [
    "NEWSAPI_API_KEY",
    "GNEWS_API_KEY",
    "EIA_API_KEY",
    "GUARDIAN_API_KEY",
]


@dataclass(frozen=True)
class MonitoringSettings:
    """Non-secret runtime settings for a monitoring cycle."""

    default_region: str
    lookback_days: int
    max_articles: int
    use_sample_data: bool


def load_project_env(root_dir: Path) -> None:
    """Load `.env` from the project root if available.

    `python-dotenv` is used when installed. A small fallback parser keeps the
    project dependency-light and avoids failing in minimal environments.
    """

    env_path = root_dir / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(env_path, override=False)
        return
    except Exception:
        pass

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_monitoring_settings() -> MonitoringSettings:
    """Return non-secret settings with safe defaults."""

    return MonitoringSettings(
        default_region=os.getenv("DEVFIN_DEFAULT_REGION", "Africa") or "Africa",
        lookback_days=parse_int(os.getenv("DEVFIN_NEWS_LOOKBACK_DAYS"), 7, minimum=1, maximum=90),
        max_articles=parse_int(os.getenv("DEVFIN_MAX_ARTICLES"), 50, minimum=5, maximum=200),
        use_sample_data=parse_bool(os.getenv("DEVFIN_USE_SAMPLE_DATA"), default=True),
    )


def source_key_status() -> list[dict[str, str]]:
    """Return key presence without exposing secret values."""

    rows = []
    for name in SECRET_NAMES:
        rows.append(
            {
                "source": key_to_source_name(name),
                "env_var": name,
                "status": "configured" if bool(os.getenv(name)) else "missing key",
                "secret_visible": "no",
            }
        )
    return rows


def parse_int(value: str | None, default: int, *, minimum: int, maximum: int) -> int:
    """Parse a bounded integer environment value."""

    try:
        parsed = int(str(value))
    except Exception:
        parsed = default
    return min(max(parsed, minimum), maximum)


def parse_bool(value: str | None, default: bool) -> bool:
    """Parse common boolean environment values."""

    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def key_to_source_name(name: str) -> str:
    """Map an environment variable to a user-facing source name."""

    return {
        "NEWSAPI_API_KEY": "NewsAPI",
        "GNEWS_API_KEY": "GNews",
        "EIA_API_KEY": "EIA",
        "GUARDIAN_API_KEY": "Guardian",
    }.get(name, name)
