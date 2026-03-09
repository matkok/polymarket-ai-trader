"""WebSocket subscriber for CLOB price updates."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Coroutine
from typing import Any

import structlog
import websockets

type PriceCallback = Callable[[dict[str, Any]], Coroutine[Any, Any, None]]


class WSWatcher:
    """WebSocket subscriber for CLOB price updates.

    Connects to the Polymarket WebSocket endpoint, subscribes to the
    requested asset IDs, and dispatches incoming messages to registered
    callbacks.  Automatically reconnects on disconnection.
    """

    WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

    def __init__(self, ws_url: str | None = None) -> None:
        self.ws_url = ws_url or self.WS_URL
        self.logger = structlog.get_logger(__name__)
        self._running = False
        self._callbacks: list[PriceCallback] = []

    def on_price_update(self, callback: PriceCallback) -> None:
        """Register a callback for price updates.

        The callback receives the parsed JSON message dict.
        """
        self._callbacks.append(callback)

    async def subscribe(self, asset_ids: list[str]) -> None:
        """Connect to WebSocket and subscribe to price updates for given asset IDs.

        Runs until :meth:`stop` is called.  Reconnects on disconnection.
        """
        self._running = True
        while self._running:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    for asset_id in asset_ids:
                        sub_msg = json.dumps({
                            "type": "market",
                            "assets_ids": [asset_id],
                        })
                        await ws.send(sub_msg)
                        self.logger.info("ws_subscribed", asset_id=asset_id)

                    async for message in ws:
                        if not self._running:
                            break
                        try:
                            data = json.loads(message)
                            for cb in self._callbacks:
                                await cb(data)
                        except json.JSONDecodeError:
                            self.logger.warning("ws_invalid_json")
            except websockets.ConnectionClosed:
                if self._running:
                    self.logger.warning("ws_disconnected_reconnecting")
                    await asyncio.sleep(5)
            except Exception as exc:
                if self._running:
                    self.logger.error("ws_error", error=str(exc))
                    await asyncio.sleep(10)

    def stop(self) -> None:
        """Signal the subscriber to stop."""
        self._running = False
