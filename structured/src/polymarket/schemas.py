"""Pydantic models for Polymarket API responses.

Covers the Gamma API (market discovery) and CLOB API (order books, prices).
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field


class GammaMarket(BaseModel):
    """Market from Gamma API."""

    condition_id: str = Field(alias="conditionId", default="")
    question: str = ""
    description: str = ""
    slug: str = ""
    category: str = ""
    end_date_iso: str = Field(alias="endDateIso", default="")
    active: bool = True
    closed: bool = False
    liquidity: float = 0.0
    volume: float = 0.0
    outcome_prices: str = Field(alias="outcomePrices", default="")  # JSON string "[0.55, 0.45]"
    outcomes: str = ""  # JSON string '["Yes","No"]'

    model_config = {"populate_by_name": True}

    def best_bid_ask(self) -> tuple[float | None, float | None]:
        """Parse outcomePrices to get (best_bid_yes, best_ask_yes).

        ``outcomePrices`` is a JSON string like ``'["0.55","0.45"]'``.
        For paper-trading (M1) we simplify: bid = ask = yes_price.
        """
        try:
            prices = json.loads(self.outcome_prices)
            yes_price = float(prices[0])
            return yes_price, yes_price  # simplified for M1
        except (json.JSONDecodeError, IndexError, ValueError):
            return None, None


class GammaEvent(BaseModel):
    """Event from Gamma API, containing multiple markets."""

    id: str = ""
    title: str = ""
    slug: str = ""
    markets: list[GammaMarket] = []


class CLOBOrderBookEntry(BaseModel):
    """Single price level in a CLOB order book."""

    price: str
    size: str


class CLOBOrderBook(BaseModel):
    """Order book snapshot from CLOB API."""

    market: str = ""
    asset_id: str = ""
    bids: list[CLOBOrderBookEntry] = []
    asks: list[CLOBOrderBookEntry] = []

    def best_bid(self) -> float | None:
        """Return the best (highest) bid price, or ``None`` if empty."""
        if self.bids:
            return float(self.bids[0].price)
        return None

    def best_ask(self) -> float | None:
        """Return the best (lowest) ask price, or ``None`` if empty."""
        if self.asks:
            return float(self.asks[0].price)
        return None


class CLOBPrice(BaseModel):
    """Price from CLOB API."""

    token_id: str = ""
    price: float = 0.0
