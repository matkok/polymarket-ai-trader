"""Compute market microstructure signals from snapshot history."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import structlog

from src.db.repository import Repository
from src.signals.schemas import MicrostructureSignal

logger = structlog.get_logger(__name__)


class MicrostructureComputer:
    """Compute microstructure signals from historical market snapshots."""

    def __init__(self, repo: Repository) -> None:
        self.repo = repo

    async def compute(
        self,
        market_id: str,
        current_snap,
    ) -> MicrostructureSignal:
        """Compute microstructure signals for a market.

        Args:
            market_id: The market to compute signals for.
            current_snap: The current MarketSnapshot.

        Returns:
            MicrostructureSignal with available fields populated.
        """
        now = datetime.now(timezone.utc)
        signal = MicrostructureSignal()

        if current_snap is None or current_snap.mid is None:
            return signal

        current_mid = current_snap.mid

        # Compute odds moves at 1h, 6h, 24h.
        for hours, attr in [(1, "odds_move_1h"), (6, "odds_move_6h"), (24, "odds_move_24h")]:
            target_ts = now - timedelta(hours=hours)
            try:
                old_snap = await self.repo.get_snapshot_at(market_id, target_ts)
                if old_snap is not None and old_snap.mid is not None:
                    setattr(signal, attr, abs(current_mid - old_snap.mid))
            except Exception:
                logger.warning(
                    "microstructure_odds_move_error",
                    market_id=market_id,
                    hours=hours,
                )

        # Compute volume ratio (current volume / 7-day average).
        try:
            since_7d = now - timedelta(days=7)
            snapshots = await self.repo.get_snapshots_since(market_id, since_7d)
            volumes = [s.volume for s in snapshots if s.volume is not None and s.volume > 0]
            if volumes and current_snap.volume is not None and current_snap.volume > 0:
                avg_volume = sum(volumes) / len(volumes)
                if avg_volume > 0:
                    signal.volume_ratio_24h = current_snap.volume / avg_volume
        except Exception:
            logger.warning("microstructure_volume_ratio_error", market_id=market_id)

        # Compute spread.
        if current_snap.best_ask is not None and current_snap.best_bid is not None:
            signal.spread_current = current_snap.best_ask - current_snap.best_bid

        # Compute spread widening (current spread / 7-day average spread).
        if signal.spread_current is not None:
            try:
                since_7d = now - timedelta(days=7)
                snapshots = await self.repo.get_snapshots_since(market_id, since_7d)
                spreads = []
                for s in snapshots:
                    if s.best_ask is not None and s.best_bid is not None:
                        spread = s.best_ask - s.best_bid
                        if spread > 0:
                            spreads.append(spread)
                if spreads:
                    avg_spread = sum(spreads) / len(spreads)
                    if avg_spread > 0:
                        signal.spread_widening = signal.spread_current / avg_spread
            except Exception:
                logger.warning("microstructure_spread_widening_error", market_id=market_id)

        return signal
