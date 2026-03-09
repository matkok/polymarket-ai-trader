"""Ticker → CIK resolver for SEC EDGAR lookups.

Static dict of ~50 common tickers. Fallback to EDGAR full-text search.
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)

# Static mapping of common tickers to CIK numbers.
_TICKER_TO_CIK: dict[str, tuple[str, str]] = {
    # ticker: (CIK zero-padded to 10 digits, company name)
    "AAPL": ("0000320193", "Apple Inc."),
    "MSFT": ("0000789019", "Microsoft Corporation"),
    "AMZN": ("0001018724", "Amazon.com Inc."),
    "GOOGL": ("0001652044", "Alphabet Inc."),
    "GOOG": ("0001652044", "Alphabet Inc."),
    "META": ("0001326801", "Meta Platforms Inc."),
    "TSLA": ("0001318605", "Tesla Inc."),
    "NVDA": ("0001045810", "NVIDIA Corporation"),
    "BRK.A": ("0001067983", "Berkshire Hathaway Inc."),
    "BRK.B": ("0001067983", "Berkshire Hathaway Inc."),
    "JPM": ("0000019617", "JPMorgan Chase & Co."),
    "JNJ": ("0000200406", "Johnson & Johnson"),
    "V": ("0001403161", "Visa Inc."),
    "PG": ("0000080424", "Procter & Gamble Co."),
    "UNH": ("0000731766", "UnitedHealth Group Inc."),
    "HD": ("0000354950", "Home Depot Inc."),
    "MA": ("0001141391", "Mastercard Inc."),
    "DIS": ("0001744489", "Walt Disney Co."),
    "BAC": ("0000070858", "Bank of America Corp."),
    "XOM": ("0000034088", "Exxon Mobil Corporation"),
    "PFE": ("0000078003", "Pfizer Inc."),
    "KO": ("0000021344", "Coca-Cola Co."),
    "PEP": ("0000077476", "PepsiCo Inc."),
    "CSCO": ("0000858877", "Cisco Systems Inc."),
    "NFLX": ("0001065280", "Netflix Inc."),
    "INTC": ("0000050863", "Intel Corporation"),
    "AMD": ("0000002488", "Advanced Micro Devices Inc."),
    "CRM": ("0001108524", "Salesforce Inc."),
    "ORCL": ("0001341439", "Oracle Corporation"),
    "WMT": ("0000104169", "Walmart Inc."),
    "CVX": ("0000093410", "Chevron Corporation"),
    "ABBV": ("0001551152", "AbbVie Inc."),
    "LLY": ("0000059478", "Eli Lilly and Co."),
    "MRK": ("0000310158", "Merck & Co. Inc."),
    "AVGO": ("0001649338", "Broadcom Inc."),
    "COST": ("0000909832", "Costco Wholesale Corp."),
    "TMO": ("0000097745", "Thermo Fisher Scientific Inc."),
    "NKE": ("0000320187", "Nike Inc."),
    "ADBE": ("0000796343", "Adobe Inc."),
    "QCOM": ("0000804328", "Qualcomm Inc."),
    "TXN": ("0000097476", "Texas Instruments Inc."),
    "UPS": ("0001090727", "United Parcel Service Inc."),
    "GS": ("0000886982", "Goldman Sachs Group Inc."),
    "MS": ("0000895421", "Morgan Stanley"),
    "IBM": ("0000051143", "International Business Machines Corp."),
    "GE": ("0000040545", "General Electric Co."),
    "CAT": ("0000018230", "Caterpillar Inc."),
    "BA": ("0000012927", "Boeing Co."),
    "MMM": ("0000066740", "3M Company"),
    "GM": ("0001467858", "General Motors Co."),
}


def resolve_ticker(ticker: str) -> tuple[str, str] | None:
    """Resolve a ticker symbol to (CIK, company_name).

    Returns ``None`` if the ticker is not in the static map.
    """
    return _TICKER_TO_CIK.get(ticker.upper())


def resolve_company(company_name: str) -> tuple[str, str] | None:
    """Resolve a company name to (CIK, ticker).

    Performs a case-insensitive substring match against known companies.
    Returns ``None`` if no match found.
    """
    name_lower = company_name.strip().lower()
    for ticker, (cik, name) in _TICKER_TO_CIK.items():
        if name_lower in name.lower() or name.lower().startswith(name_lower):
            return (cik, ticker)
    return None
