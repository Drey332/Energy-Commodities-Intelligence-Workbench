"""Keyless World Bank indicator connector."""

from __future__ import annotations

from typing import Any, Callable

from devfinintel.connectors.common import build_url, connector_result, fetch_json


WORLD_BANK_INDICATOR_API = "https://api.worldbank.org/v2/country/{countries}/indicator/{indicator}"

COUNTRY_ISO3 = {
    "Nigeria": "NGA",
    "Ghana": "GHA",
    "South Africa": "ZAF",
    "Kenya": "KEN",
    "Angola": "AGO",
    "Senegal": "SEN",
    "Mozambique": "MOZ",
    "DRC": "COD",
    "Democratic Republic of the Congo": "COD",
    "Zambia": "ZMB",
    "Tanzania": "TZA",
    "Ethiopia": "ETH",
    "Egypt": "EGY",
    "Morocco": "MAR",
    "Namibia": "NAM",
    "Rwanda": "RWA",
    "Côte d’Ivoire": "CIV",
    "Cote d'Ivoire": "CIV",
    "Cameroon": "CMR",
}

DEFAULT_COUNTRIES = ["NGA", "GHA", "ZAF", "KEN", "AGO", "SEN", "MOZ", "COD", "ZMB", "TZA", "ETH"]

INDICATORS = {
    "access_to_electricity": "EG.ELC.ACCS.ZS",
    "gdp_current_usd": "NY.GDP.MKTP.CD",
    "population": "SP.POP.TOTL",
    "inflation": "FP.CPI.TOTL.ZG",
    "exports_goods_services": "NE.EXP.GNFS.CD",
    "renewable_electricity_output": "EG.ELC.RNEW.ZS",
}


def fetch_worldbank_indicator_signals(
    *,
    countries: list[str] | None = None,
    indicators: list[str] | None = None,
    limit: int = 200,
    timeout_seconds: int = 20,
    fetcher: Callable[..., dict[str, Any] | list[Any]] = fetch_json,
) -> dict[str, Any]:
    """Fetch public World Bank indicator time series and return latest rows."""

    country_codes = normalize_country_codes(countries or [])
    if not country_codes:
        country_codes = DEFAULT_COUNTRIES
    selected_indicators = indicators or ["access_to_electricity", "gdp_current_usd", "population"]
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    urls: list[str] = []
    for indicator_name in selected_indicators:
        indicator_code = INDICATORS.get(indicator_name, indicator_name)
        url = build_url(
            WORLD_BANK_INDICATOR_API.format(
                countries=";".join(country_codes),
                indicator=indicator_code,
            ),
            {"format": "json", "per_page": min(max(limit, 1), 20000), "mrv": 5},
        )
        urls.append(url)
        try:
            payload = fetcher(url, timeout_seconds=timeout_seconds)
            rows = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
            records.extend(normalize_indicator_row(row, indicator_name, indicator_code) for row in rows if row)
        except Exception as exc:
            errors.append(f"{indicator_code}: {exc}")
    return connector_result(
        source_name="World Bank Indicators",
        source_type="dataset_indicator",
        source_status="live/keyless" if records else "failed",
        records=records,
        query="; ".join(selected_indicators),
        url=" | ".join(urls[:3]),
        errors=errors,
        metadata={"countries": country_codes, "indicators": selected_indicators, "returned": len(records)},
    )


def normalize_country_codes(countries: list[str]) -> list[str]:
    """Map country names to World Bank ISO3 codes."""

    codes = []
    for country in countries:
        value = str(country).strip()
        if not value:
            continue
        if len(value) == 3 and value.isalpha():
            codes.append(value.upper())
        elif value in COUNTRY_ISO3:
            codes.append(COUNTRY_ISO3[value])
    return list(dict.fromkeys(codes))


def normalize_indicator_row(row: dict[str, Any], indicator_name: str, indicator_code: str) -> dict[str, Any]:
    """Map a World Bank indicator record to raw connector fields."""

    country = row.get("country") or {}
    country_name = country.get("value", "") if isinstance(country, dict) else ""
    year = str(row.get("date", ""))
    value = row.get("value")
    title = f"{country_name} {indicator_name} {year}"
    return {
        "raw_source_id": f"{country_name}-{indicator_code}-{year}",
        "title": title,
        "date": year,
        "url": "https://data.worldbank.org/",
        "source": "World Bank Data",
        "country": country_name,
        "summary": f"{indicator_name} ({indicator_code}) for {country_name} in {year}: {value}",
        "value": value,
        "indicator": indicator_name,
        "indicator_code": indicator_code,
        "evidence_text": f"{country_name} {indicator_name} {year}: {value}",
        "metadata": {"source": "World Bank Indicators", "year": year, "value": value},
    }
