"""Tests for src.polymarket — schemas, gamma_client, clob_client, ws_watcher."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.polymarket.clob_client import CLOBClient
from src.polymarket.gamma_client import GammaClient
from src.polymarket.schemas import (
    CLOBOrderBook,
    CLOBOrderBookEntry,
    CLOBPrice,
    GammaEvent,
    GammaMarket,
)
from src.polymarket.ws_watcher import WSWatcher


# ---- Schemas ---------------------------------------------------------------


class TestGammaMarket:
    """GammaMarket model validation and helper methods."""

    def test_defaults(self) -> None:
        m = GammaMarket()
        assert m.condition_id == ""
        assert m.question == ""
        assert m.active is True
        assert m.closed is False
        assert m.liquidity == 0.0
        assert m.volume == 0.0
        assert m.outcome_prices == ""
        assert m.outcomes == ""

    def test_from_alias(self) -> None:
        """Model accepts aliased field names from the API."""
        data = {
            "conditionId": "0xabc123",
            "question": "Will it rain?",
            "endDateIso": "2026-03-01T00:00:00Z",
            "outcomePrices": '["0.55","0.45"]',
        }
        m = GammaMarket.model_validate(data)
        assert m.condition_id == "0xabc123"
        assert m.end_date_iso == "2026-03-01T00:00:00Z"
        assert m.outcome_prices == '["0.55","0.45"]'

    def test_populate_by_name(self) -> None:
        """Model also accepts the Python field names directly."""
        m = GammaMarket(condition_id="abc", end_date_iso="2026-01-01")
        assert m.condition_id == "abc"
        assert m.end_date_iso == "2026-01-01"

    def test_best_bid_ask_valid(self) -> None:
        m = GammaMarket(outcome_prices='["0.55","0.45"]')
        bid, ask = m.best_bid_ask()
        assert bid == 0.55
        assert ask == 0.55

    def test_best_bid_ask_empty_string(self) -> None:
        m = GammaMarket(outcome_prices="")
        bid, ask = m.best_bid_ask()
        assert bid is None
        assert ask is None

    def test_best_bid_ask_invalid_json(self) -> None:
        m = GammaMarket(outcome_prices="not json")
        bid, ask = m.best_bid_ask()
        assert bid is None
        assert ask is None

    def test_best_bid_ask_empty_array(self) -> None:
        m = GammaMarket(outcome_prices="[]")
        bid, ask = m.best_bid_ask()
        assert bid is None
        assert ask is None

    def test_best_bid_ask_non_numeric(self) -> None:
        m = GammaMarket(outcome_prices='["abc","def"]')
        bid, ask = m.best_bid_ask()
        assert bid is None
        assert ask is None


class TestGammaEvent:
    """GammaEvent model validation."""

    def test_defaults(self) -> None:
        e = GammaEvent()
        assert e.id == ""
        assert e.title == ""
        assert e.slug == ""
        assert e.markets == []

    def test_with_nested_markets(self) -> None:
        data = {
            "id": "evt-1",
            "title": "Election 2026",
            "markets": [
                {"conditionId": "c1", "question": "Q1"},
                {"conditionId": "c2", "question": "Q2"},
            ],
        }
        e = GammaEvent.model_validate(data)
        assert e.id == "evt-1"
        assert len(e.markets) == 2
        assert e.markets[0].condition_id == "c1"
        assert e.markets[1].question == "Q2"


class TestCLOBOrderBook:
    """CLOBOrderBook model and helper methods."""

    def test_best_bid_empty(self) -> None:
        ob = CLOBOrderBook()
        assert ob.best_bid() is None

    def test_best_ask_empty(self) -> None:
        ob = CLOBOrderBook()
        assert ob.best_ask() is None

    def test_best_bid(self) -> None:
        ob = CLOBOrderBook(
            bids=[
                CLOBOrderBookEntry(price="0.55", size="100"),
                CLOBOrderBookEntry(price="0.54", size="200"),
            ]
        )
        assert ob.best_bid() == 0.55

    def test_best_ask(self) -> None:
        ob = CLOBOrderBook(
            asks=[
                CLOBOrderBookEntry(price="0.56", size="150"),
                CLOBOrderBookEntry(price="0.57", size="50"),
            ]
        )
        assert ob.best_ask() == 0.56


class TestCLOBPrice:
    """CLOBPrice model validation."""

    def test_defaults(self) -> None:
        p = CLOBPrice()
        assert p.token_id == ""
        assert p.price == 0.0

    def test_custom(self) -> None:
        p = CLOBPrice(token_id="tok-1", price=0.65)
        assert p.token_id == "tok-1"
        assert p.price == 0.65


# ---- GammaClient -----------------------------------------------------------


def _make_httpx_response(data: list | dict, status_code: int = 200) -> httpx.Response:
    """Create a fake httpx.Response with JSON body."""
    return httpx.Response(
        status_code=status_code,
        json=data,
        request=httpx.Request("GET", "https://test"),
    )


class TestGammaClient:
    """GammaClient async methods with mocked HTTP."""

    async def test_get_markets(self) -> None:
        payload = [
            {"conditionId": "c1", "question": "Q1", "outcomePrices": '["0.6","0.4"]'},
            {"conditionId": "c2", "question": "Q2"},
        ]
        mock_response = _make_httpx_response(payload)

        with patch("src.polymarket.gamma_client.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            gc = GammaClient(base_url="https://test-gamma")
            markets = await gc.get_markets(limit=10, offset=0)

        assert len(markets) == 2
        assert markets[0].condition_id == "c1"
        assert markets[1].question == "Q2"
        mock_client.get.assert_called_once_with(
            "https://test-gamma/markets",
            params={"limit": 10, "offset": 0, "active": "true", "closed": "false"},
        )

    async def test_get_markets_http_error(self) -> None:
        error_response = httpx.Response(
            status_code=500,
            request=httpx.Request("GET", "https://test"),
        )

        with patch("src.polymarket.gamma_client.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = error_response
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            gc = GammaClient(base_url="https://test-gamma")
            with pytest.raises(httpx.HTTPStatusError):
                await gc.get_markets()

    async def test_get_all_active_markets_pagination(self) -> None:
        """Stops when an empty page is returned."""
        page0 = [{"conditionId": f"c{i}", "question": f"Q{i}"} for i in range(100)]
        page1 = [{"conditionId": "c100", "question": "Q100"}]
        page2: list[dict] = []  # empty => stop

        call_count = 0

        async def fake_get(url: str, params: dict | None = None) -> httpx.Response:
            nonlocal call_count
            pages = [page0, page1, page2]
            resp = _make_httpx_response(pages[call_count])
            call_count += 1
            return resp

        with patch("src.polymarket.gamma_client.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get.side_effect = fake_get
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            gc = GammaClient(base_url="https://test-gamma")
            markets = await gc.get_all_active_markets(max_pages=5)

        assert len(markets) == 101
        assert call_count == 3

    async def test_get_events(self) -> None:
        payload = [
            {
                "id": "e1",
                "title": "Event 1",
                "markets": [{"conditionId": "c1", "question": "Q1"}],
            },
        ]
        mock_response = _make_httpx_response(payload)

        with patch("src.polymarket.gamma_client.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            gc = GammaClient(base_url="https://test-gamma")
            events = await gc.get_events(limit=10)

        assert len(events) == 1
        assert events[0].id == "e1"
        assert len(events[0].markets) == 1

    def test_base_url_trailing_slash_stripped(self) -> None:
        gc = GammaClient(base_url="https://example.com/")
        assert gc.base_url == "https://example.com"

    def test_default_base_url(self) -> None:
        gc = GammaClient()
        assert gc.base_url == "https://gamma-api.polymarket.com"


# ---- CLOBClient ------------------------------------------------------------


class TestCLOBClient:
    """CLOBClient async methods with mocked HTTP."""

    async def test_get_order_book(self) -> None:
        payload = {
            "market": "mkt-1",
            "asset_id": "asset-1",
            "bids": [{"price": "0.55", "size": "100"}],
            "asks": [{"price": "0.56", "size": "200"}],
        }
        mock_response = _make_httpx_response(payload)

        with patch("src.polymarket.clob_client.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            cc = CLOBClient(base_url="https://test-clob")
            ob = await cc.get_order_book("tok-1")

        assert ob.market == "mkt-1"
        assert ob.best_bid() == 0.55
        assert ob.best_ask() == 0.56
        mock_client.get.assert_called_once_with(
            "https://test-clob/book", params={"token_id": "tok-1"}
        )

    async def test_get_order_book_http_error(self) -> None:
        error_response = httpx.Response(
            status_code=404,
            request=httpx.Request("GET", "https://test"),
        )

        with patch("src.polymarket.clob_client.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = error_response
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            cc = CLOBClient(base_url="https://test-clob")
            with pytest.raises(httpx.HTTPStatusError):
                await cc.get_order_book("tok-1")

    async def test_get_price(self) -> None:
        payload = {"price": 0.65}
        mock_response = _make_httpx_response(payload)

        with patch("src.polymarket.clob_client.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            cc = CLOBClient(base_url="https://test-clob")
            price = await cc.get_price("tok-1")

        assert price == 0.65

    async def test_get_price_returns_none_on_error(self) -> None:
        with patch("src.polymarket.clob_client.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.ConnectError("connection refused")
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            cc = CLOBClient(base_url="https://test-clob")
            price = await cc.get_price("tok-1")

        assert price is None

    async def test_get_midpoint(self) -> None:
        payload = {"mid": 0.555}
        mock_response = _make_httpx_response(payload)

        with patch("src.polymarket.clob_client.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            cc = CLOBClient(base_url="https://test-clob")
            mid = await cc.get_midpoint("tok-1")

        assert mid == 0.555

    async def test_get_midpoint_returns_none_on_error(self) -> None:
        with patch("src.polymarket.clob_client.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.TimeoutException("timed out")
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            cc = CLOBClient(base_url="https://test-clob")
            mid = await cc.get_midpoint("tok-1")

        assert mid is None

    def test_base_url_trailing_slash_stripped(self) -> None:
        cc = CLOBClient(base_url="https://example.com/")
        assert cc.base_url == "https://example.com"

    def test_default_base_url(self) -> None:
        cc = CLOBClient()
        assert cc.base_url == "https://clob.polymarket.com"


# ---- WSWatcher --------------------------------------------------------------


class TestWSWatcher:
    """WSWatcher callback registration and lifecycle."""

    def test_default_ws_url(self) -> None:
        watcher = WSWatcher()
        assert watcher.ws_url == "wss://ws-subscriptions-clob.polymarket.com/ws/market"

    def test_custom_ws_url(self) -> None:
        watcher = WSWatcher(ws_url="wss://custom.test/ws")
        assert watcher.ws_url == "wss://custom.test/ws"

    def test_on_price_update_registers_callback(self) -> None:
        watcher = WSWatcher()
        assert len(watcher._callbacks) == 0

        async def my_callback(data: dict) -> None:
            pass

        watcher.on_price_update(my_callback)
        assert len(watcher._callbacks) == 1
        assert watcher._callbacks[0] is my_callback

    def test_stop_sets_running_false(self) -> None:
        watcher = WSWatcher()
        watcher._running = True
        watcher.stop()
        assert watcher._running is False

    async def test_subscribe_dispatches_messages(self) -> None:
        """Verify callbacks receive parsed JSON messages."""
        received: list[dict] = []

        async def capture(data: dict) -> None:
            received.append(data)

        watcher = WSWatcher(ws_url="wss://test")
        watcher.on_price_update(capture)

        messages = [
            json.dumps({"event": "price_change", "price": "0.60"}),
            json.dumps({"event": "price_change", "price": "0.62"}),
        ]

        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        mock_ws.__aiter__ = lambda self: self
        call_count = 0

        async def async_iter_messages(self: AsyncMock) -> None:
            pass

        # Build an async iterator that yields messages then signals stop
        async def message_iterator() -> None:
            nonlocal call_count
            for msg in messages:
                yield msg  # type: ignore[misc]
            watcher.stop()

        mock_ws.__aiter__ = lambda s: message_iterator()

        with patch("src.polymarket.ws_watcher.websockets.connect") as mock_connect:
            mock_connect.return_value.__aenter__ = AsyncMock(return_value=mock_ws)
            mock_connect.return_value.__aexit__ = AsyncMock(return_value=False)

            await watcher.subscribe(asset_ids=["asset-1"])

        assert len(received) == 2
        assert received[0]["price"] == "0.60"
        assert received[1]["price"] == "0.62"
        mock_ws.send.assert_called_once_with(
            json.dumps({"type": "market", "assets_ids": ["asset-1"]})
        )

    async def test_subscribe_handles_invalid_json(self) -> None:
        """Invalid JSON messages are logged but do not crash the watcher."""
        received: list[dict] = []

        async def capture(data: dict) -> None:
            received.append(data)

        watcher = WSWatcher(ws_url="wss://test")
        watcher.on_price_update(capture)

        messages = [
            "not valid json",
            json.dumps({"valid": True}),
        ]

        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()

        async def message_iterator() -> None:
            for msg in messages:
                yield msg  # type: ignore[misc]
            watcher.stop()

        mock_ws.__aiter__ = lambda s: message_iterator()

        with patch("src.polymarket.ws_watcher.websockets.connect") as mock_connect:
            mock_connect.return_value.__aenter__ = AsyncMock(return_value=mock_ws)
            mock_connect.return_value.__aexit__ = AsyncMock(return_value=False)

            await watcher.subscribe(asset_ids=["asset-1"])

        # Only the valid message should have been dispatched
        assert len(received) == 1
        assert received[0]["valid"] is True

    async def test_subscribe_multiple_asset_ids(self) -> None:
        """Each asset ID gets its own subscription message."""
        watcher = WSWatcher(ws_url="wss://test")

        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()

        async def empty_iterator() -> None:
            watcher.stop()
            return
            yield  # type: ignore[misc]

        mock_ws.__aiter__ = lambda s: empty_iterator()

        with patch("src.polymarket.ws_watcher.websockets.connect") as mock_connect:
            mock_connect.return_value.__aenter__ = AsyncMock(return_value=mock_ws)
            mock_connect.return_value.__aexit__ = AsyncMock(return_value=False)

            await watcher.subscribe(asset_ids=["a1", "a2", "a3"])

        assert mock_ws.send.call_count == 3
