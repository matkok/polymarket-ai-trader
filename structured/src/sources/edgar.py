"""SEC EDGAR API adapter — filings, XBRL facts for EPS/revenue."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
import structlog

from src.contracts.earnings import EarningsContractSpec
from src.sources.base import FetchResult, SourceAdapter
from src.sources.ticker_resolver import resolve_company, resolve_ticker

logger = structlog.get_logger(__name__)

_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

_DEFAULT_USER_AGENT = "agent-trader-structured research@example.com"


class EDGARAdapter(SourceAdapter):
    """Fetch filing data from SEC EDGAR APIs."""

    def __init__(self, user_agent: str = "") -> None:
        self._user_agent = user_agent or _DEFAULT_USER_AGENT

    @property
    def name(self) -> str:
        return "edgar"

    def _headers(self) -> dict[str, str]:
        return {
            "User-Agent": self._user_agent,
            "Accept": "application/json",
        }

    async def fetch(self, spec: Any) -> FetchResult:
        """Fetch EDGAR data for an earnings contract spec."""
        now = datetime.now(timezone.utc)

        if not isinstance(spec, EarningsContractSpec):
            return FetchResult(
                source_name=self.name, source_key="",
                ts_source=now, raw_json={}, normalized_json={},
                error="invalid_spec_type",
            )

        # Resolve CIK.
        cik = spec.cik
        if not cik:
            if spec.ticker:
                resolved = resolve_ticker(spec.ticker)
                if resolved:
                    cik = resolved[0]
            if not cik and spec.company:
                resolved = resolve_company(spec.company)
                if resolved:
                    cik = resolved[0]

        if not cik:
            return FetchResult(
                source_name=self.name,
                source_key=spec.ticker or spec.company,
                ts_source=now, raw_json={}, normalized_json={},
                error="cik_not_resolved",
            )

        # Choose endpoint based on metric.
        if spec.metric in ("eps", "revenue"):
            return await self._fetch_company_facts(cik, spec, now)
        elif spec.metric.startswith("filing_"):
            return await self._fetch_submissions(cik, spec, now)
        else:
            return await self._fetch_submissions(cik, spec, now)

    async def _fetch_company_facts(
        self, cik: str, spec: EarningsContractSpec, now: datetime
    ) -> FetchResult:
        """Fetch XBRL company facts for EPS/revenue."""
        url = _COMPANY_FACTS_URL.format(cik=cik)
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url, headers=self._headers())
                resp.raise_for_status()
                raw = resp.json()
        except Exception as exc:
            logger.exception("edgar_facts_error", cik=cik)
            return FetchResult(
                source_name=self.name, source_key=cik,
                ts_source=now, raw_json={}, normalized_json={},
                error=f"api_error: {exc}",
            )

        normalized = self._parse_facts(raw, spec)
        if "error" in normalized:
            return FetchResult(
                source_name=self.name, source_key=cik,
                ts_source=now, raw_json=raw, normalized_json=normalized,
                error=normalized["error"],
            )

        quality = 1.0 if normalized.get("has_filed") else 0.3
        return FetchResult(
            source_name=self.name, source_key=cik,
            ts_source=now, raw_json=raw, normalized_json=normalized,
            quality_score=quality,
        )

    async def _fetch_submissions(
        self, cik: str, spec: EarningsContractSpec, now: datetime
    ) -> FetchResult:
        """Fetch recent submissions to check filing status."""
        url = _SUBMISSIONS_URL.format(cik=cik)
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url, headers=self._headers())
                resp.raise_for_status()
                raw = resp.json()
        except Exception as exc:
            logger.exception("edgar_submissions_error", cik=cik)
            return FetchResult(
                source_name=self.name, source_key=cik,
                ts_source=now, raw_json={}, normalized_json={},
                error=f"api_error: {exc}",
            )

        normalized = self._parse_submissions(raw, spec)
        if "error" in normalized:
            return FetchResult(
                source_name=self.name, source_key=cik,
                ts_source=now, raw_json=raw, normalized_json=normalized,
                error=normalized["error"],
            )

        quality = 1.0 if normalized.get("filing_found") else 0.3
        return FetchResult(
            source_name=self.name, source_key=cik,
            ts_source=now, raw_json=raw, normalized_json=normalized,
            quality_score=quality,
        )

    @staticmethod
    def _parse_facts(raw: dict[str, Any], spec: EarningsContractSpec) -> dict[str, Any]:
        """Parse XBRL company facts for EPS or revenue."""
        facts = raw.get("facts", {})
        us_gaap = facts.get("us-gaap", {})

        if spec.metric == "eps":
            concept = us_gaap.get("EarningsPerShareDiluted", {})
        elif spec.metric == "revenue":
            concept = us_gaap.get("Revenues", {}) or us_gaap.get("RevenueFromContractWithCustomerExcludingAssessedTax", {})
        else:
            return {"error": f"unsupported_metric: {spec.metric}"}

        units = concept.get("units", {})
        # EPS is in USD/shares, Revenue in USD.
        values_list = units.get("USD/shares", []) or units.get("USD", [])
        if not values_list:
            return {"error": "no_xbrl_data", "has_filed": False}

        # Sort by end date, newest first.
        sorted_values = sorted(values_list, key=lambda x: x.get("end", ""), reverse=True)

        latest = sorted_values[0] if sorted_values else None
        if latest is None:
            return {"error": "no_data_points", "has_filed": False}

        result: dict[str, Any] = {
            "metric": spec.metric,
            "latest_value": latest.get("val"),
            "latest_end_date": latest.get("end"),
            "latest_filed": latest.get("filed"),
            "fiscal_period": latest.get("fp", ""),
            "has_filed": True,
        }

        # Add historical values.
        history = []
        for v in sorted_values[:12]:
            history.append({
                "value": v.get("val"),
                "end_date": v.get("end"),
                "fiscal_period": v.get("fp", ""),
            })
        result["history"] = history

        return result

    @staticmethod
    def _parse_submissions(
        raw: dict[str, Any], spec: EarningsContractSpec
    ) -> dict[str, Any]:
        """Parse submissions to check if a filing has been made."""
        recent = raw.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])

        target_form = spec.filing_type or "10-K"

        # Find the most recent matching filing.
        for i, form in enumerate(forms):
            if form.upper() == target_form.upper():
                return {
                    "filing_found": True,
                    "filing_type": form,
                    "filing_date": filing_dates[i] if i < len(filing_dates) else None,
                    "company": raw.get("name", ""),
                    "cik": raw.get("cik", ""),
                }

        return {
            "filing_found": False,
            "filing_type": target_form,
            "company": raw.get("name", ""),
            "cik": raw.get("cik", ""),
        }

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    "https://data.sec.gov/submissions/CIK0000320193.json",
                    headers=self._headers(),
                )
                return resp.status_code == 200
        except Exception:
            return False
