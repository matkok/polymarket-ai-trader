"""Core trading engine that orchestrates all operations.

The :class:`TradingEngine` drives market ingestion, candidate scanning,
position lifecycle review, and daily resets.  It is intended to be
invoked by APScheduler jobs configured in ``main.py``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from dataclasses import asdict, fields as dataclass_fields
from datetime import datetime, timedelta, timezone

import structlog

from src.aggregation.aggregator import Aggregator, AggregationResult
from src.app.candidate_selector import CandidateSelector
from src.config.policy import Policy, policy_version_hash
from src.db.repository import Repository
from src.evaluation.scorer import ModelScorer
from src.evidence.embedder import EvidenceEmbedder
from src.evidence.fulltext import FullTextFetcher
from src.evidence.linker import EvidenceLinker
from src.evidence.rss_ingestor import RSSIngestor
from src.evidence.xai_search import XAISearchClient
from src.execution.fills import calculate_realized_pnl, calculate_unrealized_pnl
from src.execution.paper_executor import PaperExecutor
from src.llm.panel import PanelOrchestrator
from src.llm.prompt_builder import PROMPT_VERSION
from src.llm.schemas import ModelTier, PanelAgent
from src.packets.builder import PacketBuilder
from src.polymarket.gamma_client import GammaClient
from src.portfolio.lifecycle import LifecycleAction, PositionLifecycle
from src.portfolio.risk_manager import RiskManager
from src.portfolio.sizing import SizingInput, compute_size
from src.signals.collector import SignalCollector
from src.signals.triage import TriageScorer

logger = structlog.get_logger(__name__)


class TradingEngine:
    """Core trading engine that orchestrates all M1 operations."""

    def __init__(
        self,
        repo: Repository,
        gamma_client: GammaClient,
        policy: Policy,
        rss_ingestor: RSSIngestor | None = None,
        packet_builder: PacketBuilder | None = None,
        panel_orchestrator: PanelOrchestrator | None = None,
        aggregator: Aggregator | None = None,
        xai_search_client: XAISearchClient | None = None,
        embedder: EvidenceEmbedder | None = None,
        fulltext_fetcher: FullTextFetcher | None = None,
        signal_collector: SignalCollector | None = None,
        triage_scorer: TriageScorer | None = None,
        clob_client: object | None = None,
    ) -> None:
        self.repo = repo
        self.gamma = gamma_client
        self.policy = policy
        self.selector = CandidateSelector(policy)
        self.risk_manager = RiskManager(policy)
        self.lifecycle = PositionLifecycle(policy)
        self.executor = PaperExecutor(policy)
        self.policy_hash = policy_version_hash(policy)
        self.daily_realized_pnl = 0.0
        self.rss_ingestor = rss_ingestor
        self.packet_builder = packet_builder
        self.panel = panel_orchestrator
        self.aggregator = aggregator
        self.xai_search_client = xai_search_client
        self.fulltext_fetcher = fulltext_fetcher
        self.scorer = ModelScorer(repo, policy)
        self.evidence_linker = EvidenceLinker(
            max_per_market=policy.max_evidence_items_per_packet,
            embedder=embedder,
            similarity_threshold=policy.evidence_similarity_threshold,
        )
        self.signal_collector = signal_collector
        self.triage_scorer = triage_scorer
        self.clob_client = clob_client
        self._trade_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Fee and order-book helpers
    # ------------------------------------------------------------------

    def _get_fee_rate(self, market_id: str, market: object | None = None) -> float:
        """Return the appropriate fee rate for a market.

        Standard Polymarket markets have 0% fees.  15-minute crypto
        markets still charge fees (``crypto_15min_fee_rate``).
        """
        if market is not None:
            question = getattr(market, "question", "") or ""
            if "15-min" in question.lower() or "15min" in question.lower():
                return self.policy.crypto_15min_fee_rate
        return self.policy.fee_rate

    async def _fetch_order_book(
        self, market_id: str, side: str = "BUY_YES"
    ) -> list[tuple[float, float]] | None:
        """Fetch order book for *market_id* from CLOB client.

        Returns list of (price, size_eur) levels sorted appropriately
        for the trade side, or None on failure/timeout.
        """
        if self.clob_client is None:
            return None
        try:
            book = await asyncio.wait_for(
                self.clob_client.get_order_book(market_id),
                timeout=5.0,
            )
            # Convert CLOBOrderBook entries to (price, size) tuples.
            if side in ("BUY_YES", "BUY_NO"):
                entries = [(float(e.price), float(e.size)) for e in book.asks]
            else:
                entries = [(float(e.price), float(e.size)) for e in book.bids]
            return entries if entries else None
        except Exception:
            logger.debug("order_book_fetch_error", market_id=market_id)
            return None

    # ------------------------------------------------------------------
    # Market ingestion
    # ------------------------------------------------------------------

    async def ingest_markets(self) -> int:
        """Fetch markets from Gamma API and upsert to DB.

        Returns the count of markets ingested.
        """
        logger.info("market_ingestion_start")
        gamma_markets = await self.gamma.get_all_active_markets()

        market_dicts: list[dict] = []
        snapshot_dicts: list[dict] = []
        for gm in gamma_markets:
            if not gm.condition_id:
                continue

            # Parse resolution time from end_date_iso.
            resolution_time = None
            if gm.end_date_iso:
                try:
                    resolution_time = datetime.fromisoformat(
                        gm.end_date_iso.replace("Z", "+00:00")
                    )
                except ValueError:
                    pass

            now = datetime.now(timezone.utc)
            market_dicts.append(
                {
                    "market_id": gm.condition_id,
                    "question": gm.question,
                    "rules_text": gm.description or None,
                    "category": gm.category or None,
                    "resolution_time_utc": resolution_time,
                    "status": "active" if gm.active and not gm.closed else "closed",
                    "updated_ts_utc": now,
                }
            )

            # Prepare snapshot (insert after markets are upserted).
            bid, ask = gm.best_bid_ask()
            mid = (bid + ask) / 2 if bid is not None and ask is not None else None
            snapshot_dicts.append(
                {
                    "market_id": gm.condition_id,
                    "ts_utc": now,
                    "best_bid": bid,
                    "best_ask": ask,
                    "mid": mid,
                    "liquidity": gm.liquidity,
                    "volume": gm.volume,
                }
            )

        # Deduplicate by market_id (Gamma API can return the same market
        # on multiple pages).  Keep the last occurrence for each id.
        seen: dict[str, int] = {}
        for idx, md in enumerate(market_dicts):
            seen[md["market_id"]] = idx
        unique_idxs = set(seen.values())
        market_dicts = [market_dicts[i] for i in sorted(unique_idxs)]
        snapshot_dicts = [snapshot_dicts[i] for i in sorted(unique_idxs)]

        # Upsert markets first (FK parent), then insert snapshots.
        if market_dicts:
            await self.repo.bulk_upsert_markets(market_dicts)

        if snapshot_dicts:
            await self.repo.bulk_add_snapshots(snapshot_dicts)

        logger.info(
            "market_ingestion_done",
            markets=len(market_dicts),
            snapshots=len(snapshot_dicts),
        )
        return len(market_dicts)

    # ------------------------------------------------------------------
    # Evidence ingestion
    # ------------------------------------------------------------------

    async def ingest_evidence(self) -> int:
        """Fetch RSS feeds, store evidence items to DB.

        Returns the count of new evidence items stored.
        """
        if self.rss_ingestor is None:
            logger.warning("evidence_ingestion_skipped", reason="no_rss_ingestor")
            return 0

        logger.info("evidence_ingestion_start")
        articles = await self.rss_ingestor.fetch_all()

        now = datetime.now(timezone.utc)
        items: list[dict] = []
        for article in articles:
            content_hash = self.rss_ingestor._compute_hash(
                article.url, article.extracted_text
            )
            items.append(
                {
                    "ts_utc": now,
                    "source_type": "rss",
                    "url": article.url,
                    "title": article.title,
                    "published_ts_utc": article.published_ts,
                    "extracted_text": article.extracted_text[
                        : self.policy.evidence_excerpt_max_chars
                    ],
                    "content_hash": content_hash,
                }
            )

        # Enrich articles with short text by fetching full content.
        if self.fulltext_fetcher:
            items = await self.fulltext_fetcher.enrich_articles(items)

        inserted = await self.repo.bulk_upsert_evidence(items)
        logger.info(
            "evidence_ingestion_done",
            articles_fetched=len(articles),
            items_stored=inserted,
        )
        return inserted

    # ------------------------------------------------------------------
    # Signal collection
    # ------------------------------------------------------------------

    async def collect_signals(self) -> int:
        """Collect signals for active markets.

        Returns the number of signal snapshots stored.
        """
        if self.signal_collector is None:
            return 0

        logger.info("signal_collection_start")
        markets = await self.repo.get_active_markets()

        snapshots: dict = {}
        for m in markets:
            snap = await self.repo.get_latest_snapshot(m.market_id)
            if snap:
                snapshots[m.market_id] = snap

        bundles = await self.signal_collector.collect_all(markets, snapshots)

        stored = 0
        now = datetime.now(timezone.utc)
        for market_id, bundle in bundles.items():
            try:
                data: dict = {
                    "market_id": market_id,
                    "ts_utc": now,
                }
                if bundle.microstructure:
                    data["odds_move_1h"] = bundle.microstructure.odds_move_1h
                    data["odds_move_6h"] = bundle.microstructure.odds_move_6h
                    data["odds_move_24h"] = bundle.microstructure.odds_move_24h
                    data["volume_ratio_24h"] = bundle.microstructure.volume_ratio_24h
                    data["spread_current"] = bundle.microstructure.spread_current
                    data["spread_widening"] = bundle.microstructure.spread_widening
                if bundle.evidence_freshness:
                    data["evidence_count_6h"] = bundle.evidence_freshness.evidence_count_6h
                    data["evidence_count_24h"] = bundle.evidence_freshness.evidence_count_24h
                    data["credible_evidence_6h"] = bundle.evidence_freshness.credible_evidence_6h
                if bundle.google_trends:
                    data["google_trends_spike"] = bundle.google_trends.spike_score
                if bundle.wikipedia:
                    data["wikipedia_spike"] = bundle.wikipedia.spike_score

                await self.repo.add_signal_snapshot(data)
                stored += 1
            except Exception:
                logger.exception("signal_snapshot_store_error", market_id=market_id)

        logger.info("signal_collection_done", stored=stored)
        return stored

    # ------------------------------------------------------------------
    # Candidate scanning
    # ------------------------------------------------------------------

    async def run_candidate_scan(self) -> int:
        """Select candidates and execute paper trades.

        Returns the count of trades executed.
        """
        async with self._trade_lock:
            return await self._run_candidate_scan_impl()

    async def _run_candidate_scan_impl(self) -> int:
        """Inner implementation of candidate scan (lock held by caller)."""
        logger.info("candidate_scan_start")

        markets = await self.repo.get_active_markets()
        positions = await self.repo.get_open_positions()
        open_market_ids = {p.market_id for p in positions}
        total_unrealized_pnl = sum(
            getattr(p, "unrealized_pnl", 0.0) or 0.0 for p in positions
        )

        # Build snapshots dict.
        snapshots: dict = {}
        for m in markets:
            snap = await self.repo.get_latest_snapshot(m.market_id)
            if snap:
                snapshots[m.market_id] = snap

        candidates = self.selector.select(markets, snapshots, open_market_ids)
        logger.info("candidates_selected", count=len(candidates))

        # Panel cooldown: skip markets recently evaluated by LLM panel.
        if self.policy.panel_cooldown_hours > 0 and candidates:
            cooldown_since = datetime.now(timezone.utc) - timedelta(
                hours=self.policy.panel_cooldown_hours
            )
            try:
                recently_paneled = await self.repo.get_recently_paneled_market_ids(
                    cooldown_since
                )
                before_count = len(candidates)
                candidates = [
                    c for c in candidates
                    if c.market_id not in recently_paneled
                ]
                skipped = before_count - len(candidates)
                if skipped:
                    logger.info(
                        "panel_cooldown_filtered",
                        skipped=skipped,
                        remaining=len(candidates),
                    )
            except Exception:
                logger.exception("panel_cooldown_query_error")

        # Triage: filter candidates using signal-based scoring.
        triage_results: dict = {}
        if self.signal_collector and self.triage_scorer:
            candidate_markets = [
                m for m in markets if m.market_id in {c.market_id for c in candidates}
            ]
            bundles = await self.signal_collector.collect_all(
                candidate_markets, snapshots
            )
            now_ts = datetime.now(timezone.utc)
            for candidate in candidates:
                bundle = bundles.get(candidate.market_id)
                if bundle:
                    triage = self.triage_scorer.score(bundle)
                    triage_results[candidate.market_id] = triage
                    logger.info(
                        "triage_scored",
                        market_id=candidate.market_id,
                        triage_score=round(triage.triage_score, 3),
                        should_panel=triage.should_panel,
                        guardrails=triage.guardrail_flags,
                    )
                    # Store signal snapshot with triage score.
                    try:
                        data: dict = {
                            "market_id": candidate.market_id,
                            "ts_utc": now_ts,
                            "triage_score": triage.triage_score,
                            "triage_reasons": triage.reasons,
                        }
                        if bundle.microstructure:
                            data["odds_move_1h"] = bundle.microstructure.odds_move_1h
                            data["odds_move_6h"] = bundle.microstructure.odds_move_6h
                            data["odds_move_24h"] = bundle.microstructure.odds_move_24h
                            data["volume_ratio_24h"] = bundle.microstructure.volume_ratio_24h
                            data["spread_current"] = bundle.microstructure.spread_current
                            data["spread_widening"] = bundle.microstructure.spread_widening
                        if bundle.evidence_freshness:
                            data["evidence_count_6h"] = bundle.evidence_freshness.evidence_count_6h
                            data["evidence_count_24h"] = bundle.evidence_freshness.evidence_count_24h
                            data["credible_evidence_6h"] = bundle.evidence_freshness.credible_evidence_6h
                        if bundle.google_trends:
                            data["google_trends_spike"] = bundle.google_trends.spike_score
                        if bundle.wikipedia:
                            data["wikipedia_spike"] = bundle.wikipedia.spike_score
                        await self.repo.add_signal_snapshot(data)
                    except Exception:
                        logger.exception(
                            "triage_signal_store_error",
                            market_id=candidate.market_id,
                        )

            # Filter to only triaged candidates.
            candidates = [
                c for c in candidates
                if triage_results.get(c.market_id) is None
                or triage_results[c.market_id].should_panel
            ]
            logger.info("candidates_after_triage", count=len(candidates))

        # Build packets for candidates (M2: stored for audit, not yet used for decisions).
        await self._build_packets_for_candidates(
            candidates, markets, snapshots, positions
        )

        trades_executed = 0
        for candidate in candidates:
            snap = snapshots.get(candidate.market_id)
            if not snap or snap.mid is None:
                continue

            # Skip markets with a recent veto.
            if self.policy.no_add_if_recent_veto_minutes > 0:
                try:
                    recent_agg = await self.repo.get_latest_aggregation(candidate.market_id)
                    if recent_agg and recent_agg.aggregation_json:
                        agg_data = recent_agg.aggregation_json
                        age_minutes = (
                            datetime.now(timezone.utc) - recent_agg.ts_utc
                        ).total_seconds() / 60
                        if (
                            age_minutes < self.policy.no_add_if_recent_veto_minutes
                            and agg_data.get("veto")
                        ):
                            logger.info(
                                "entry_blocked_recent_veto",
                                market_id=candidate.market_id,
                                veto_age_minutes=round(age_minutes),
                            )
                            continue
                except Exception:
                    logger.exception(
                        "recent_veto_check_error",
                        market_id=candidate.market_id,
                    )

            if self.panel and self.aggregator:
                # M3: Run LLM panel.
                triage_bundle = None
                triage = triage_results.get(candidate.market_id)
                if triage and self.signal_collector:
                    # Re-use signal bundle from triage if available.
                    candidate_markets_for_bundle = [
                        m for m in markets if m.market_id == candidate.market_id
                    ]
                    if candidate_markets_for_bundle:
                        try:
                            bundles = await self.signal_collector.collect_all(
                                candidate_markets_for_bundle, snapshots
                            )
                            triage_bundle = bundles.get(candidate.market_id)
                        except Exception:
                            logger.exception(
                                "signal_bundle_fetch_error",
                                market_id=candidate.market_id,
                            )
                agg_result = await self._run_panel_for_candidate(
                    candidate, snap, snapshots,
                    signal_bundle=triage_bundle,
                )
                if agg_result is None or not agg_result.trade_allowed:
                    logger.info(
                        "panel_trade_blocked",
                        market_id=candidate.market_id,
                        reason="panel_veto_or_disagreement"
                        if agg_result
                        else "panel_failed",
                    )
                    continue

                # Reentry cooldown + stability check.
                if self.policy.position_reentry_cooldown_hours > 0:
                    try:
                        last_trade_ts = await self.repo.get_latest_order_ts(
                            candidate.market_id
                        )
                        if last_trade_ts:
                            hours_since_trade = (
                                datetime.now(timezone.utc) - last_trade_ts
                            ).total_seconds() / 3600
                            if hours_since_trade < self.policy.position_reentry_cooldown_hours:
                                prior_agg_row = await self.repo.get_latest_aggregation(
                                    candidate.market_id
                                )
                                if prior_agg_row and prior_agg_row.aggregation_json:
                                    prior = prior_agg_row.aggregation_json
                                    prior_p_market = prior.get("p_market", snap.mid)
                                    prior_side = prior.get("consensus_side")
                                    new_side = (
                                        "BUY_YES"
                                        if agg_result.p_consensus > snap.mid
                                        else "BUY_NO"
                                    )

                                    # Signed edges (positive = in favor of the side).
                                    if new_side == "BUY_YES":
                                        new_signed_edge = agg_result.p_consensus - snap.mid
                                    else:
                                        new_signed_edge = snap.mid - agg_result.p_consensus
                                    if prior_side == "BUY_YES":
                                        prior_signed_edge = (
                                            prior.get("p_consensus", prior_p_market)
                                            - prior_p_market
                                        )
                                    else:
                                        prior_signed_edge = (
                                            prior_p_market
                                            - prior.get("p_consensus", prior_p_market)
                                        )

                                    stability_ok = (
                                        new_side == prior_side
                                        and new_signed_edge > 0
                                        and new_signed_edge > prior_signed_edge
                                        and agg_result.confidence > prior.get("confidence", 0)
                                        and agg_result.disagreement
                                        < prior.get("disagreement", 1)
                                    )
                                    if not stability_ok:
                                        logger.info(
                                            "entry_blocked_reentry_stability",
                                            market_id=candidate.market_id,
                                            hours_since_trade=round(hours_since_trade, 1),
                                            new_side=new_side,
                                            prior_side=prior_side,
                                            new_signed_edge=round(new_signed_edge, 4),
                                            prior_signed_edge=round(prior_signed_edge, 4),
                                        )
                                        continue
                    except Exception:
                        logger.exception(
                            "reentry_cooldown_check_error",
                            market_id=candidate.market_id,
                        )

                sizing_input = SizingInput(
                    p_consensus=agg_result.p_consensus,
                    p_market=snap.mid,
                    confidence=agg_result.confidence,
                    disagreement=agg_result.disagreement,
                    best_bid=snap.best_bid or 0.0,
                    best_ask=snap.best_ask or 0.0,
                )
            else:
                # M1 fallback: synthetic edge.
                p_market = snap.mid
                p_consensus = p_market + (candidate.score - 0.5) * 0.1
                p_consensus = max(0.01, min(0.99, p_consensus))
                sizing_input = SizingInput(
                    p_consensus=p_consensus,
                    p_market=p_market,
                    confidence=1.0,
                    disagreement=0.0,
                    best_bid=snap.best_bid or 0.0,
                    best_ask=snap.best_ask or 0.0,
                )
            sizing_result = compute_size(sizing_input, self.policy)

            if sizing_result.skip_reason:
                logger.info(
                    "trade_skipped",
                    market_id=candidate.market_id,
                    reason=sizing_result.skip_reason,
                )
                continue

            # Apply triage guardrails to sizing.
            triage = triage_results.get(candidate.market_id)
            if triage:
                if "social_only_no_credible_source" in triage.guardrail_flags:
                    logger.info(
                        "trade_blocked_guardrail",
                        market_id=candidate.market_id,
                        guardrail="social_only_no_credible_source",
                    )
                    continue
                guardrail_factor = 1.0
                if "wide_spread" in triage.guardrail_flags:
                    guardrail_factor *= 0.50
                    logger.info(
                        "guardrail_size_reduction",
                        market_id=candidate.market_id,
                        guardrail="wide_spread",
                        factor=0.50,
                    )
                if "spread_widening" in triage.guardrail_flags:
                    guardrail_factor *= 0.75
                    logger.info(
                        "guardrail_size_reduction",
                        market_id=candidate.market_id,
                        guardrail="spread_widening",
                        factor=0.75,
                    )
                if guardrail_factor < 1.0:
                    sizing_result.clamped_size_eur *= guardrail_factor

            # Risk check.
            risk_check = self.risk_manager.check_new_trade(
                size_eur=sizing_result.clamped_size_eur,
                market_id=candidate.market_id,
                current_positions=positions,
                daily_realized_pnl=self.daily_realized_pnl,
                total_unrealized_pnl=total_unrealized_pnl,
            )
            if not risk_check.allowed:
                logger.info(
                    "trade_blocked",
                    market_id=candidate.market_id,
                    violations=risk_check.violations,
                )
                continue

            # Execute paper trade.
            bid = snap.best_bid or snap.mid
            ask = snap.best_ask or snap.mid
            market_obj = next(
                (m for m in markets if m.market_id == candidate.market_id), None
            )
            fee_rate = self._get_fee_rate(candidate.market_id, market_obj)
            ob = await self._fetch_order_book(candidate.market_id, side=sizing_result.side)
            fill = self.executor.execute(
                side=sizing_result.side,
                size_eur=sizing_result.clamped_size_eur,
                best_bid=bid,
                best_ask=ask,
                fee_rate_override=fee_rate,
                order_book=ob,
            )

            # Record decision.
            now = datetime.now(timezone.utc)
            decision_id = await self.repo.add_decision(
                {
                    "market_id": candidate.market_id,
                    "ts_utc": now,
                    "action": "OPEN",
                    "size_eur": fill.size_eur,
                    "reason_json": {
                        "edge": sizing_result.edge,
                        "side": sizing_result.side,
                        "candidate_score": candidate.score,
                    },
                    "policy_version": self.policy_hash,
                }
            )

            # Record order.
            order_id = await self.repo.add_order(
                {
                    "decision_id": decision_id,
                    "market_id": candidate.market_id,
                    "side": fill.side,
                    "size_eur": fill.size_eur,
                    "limit_price_ref": fill.price,
                    "status": "filled",
                    "created_ts_utc": now,
                }
            )

            # Record fill.
            await self.repo.add_fill(
                {
                    "order_id": order_id,
                    "ts_utc": now,
                    "price": fill.price,
                    "size_eur": fill.size_eur,
                    "fee_eur": fill.fee_eur,
                }
            )

            # Upsert position.
            await self.repo.upsert_position(
                {
                    "market_id": candidate.market_id,
                    "side": fill.side,
                    "size_eur": fill.size_eur,
                    "avg_entry_price": fill.price,
                    "last_price": fill.price,
                    "unrealized_pnl": 0.0,
                    "realized_pnl": 0.0,
                    "status": "open",
                    "opened_ts_utc": now,
                    "last_update_ts_utc": now,
                }
            )

            trades_executed += 1
            logger.info(
                "trade_executed",
                market_id=candidate.market_id,
                side=fill.side,
                size=fill.size_eur,
                price=fill.price,
            )

        logger.info("candidate_scan_done", trades=trades_executed)
        return trades_executed

    async def _build_packets_for_candidates(
        self,
        candidates: list,
        markets: list,
        snapshots: dict,
        positions: list,
    ) -> None:
        """Build and store packets for each candidate market."""
        if self.packet_builder is None:
            return

        # Index markets and positions by market_id.
        market_map = {m.market_id: m for m in markets}
        pos_map = {p.market_id: p for p in positions}

        # Get recent evidence and link to markets.
        recent_evidence = await self.repo.get_recent_evidence(
            self.policy.new_evidence_recheck_window_minutes // 60
            if self.policy.new_evidence_recheck_window_minutes >= 60
            else 72
        )
        candidate_markets = [
            market_map[c.market_id]
            for c in candidates
            if c.market_id in market_map
        ]
        linked = await self.evidence_linker.link(recent_evidence, candidate_markets)

        # Proactive search for candidates with sparse evidence.
        sparse_candidates = [
            c
            for c in candidates
            if len(linked.get(c.market_id, [])) < 2
            and c.market_id in market_map
        ]
        if sparse_candidates and self.xai_search_client:
            for candidate in sparse_candidates[:3]:
                if not self.xai_search_client.can_search():
                    break
                market = market_map[candidate.market_id]
                try:
                    articles = await self.xai_search_client.search(
                        market.question, max_results=3
                    )
                    if articles:
                        now_ts = datetime.now(timezone.utc)
                        items = []
                        for a in articles:
                            items.append(
                                {
                                    "ts_utc": now_ts,
                                    "source_type": "xai_search",
                                    "url": a.url,
                                    "title": a.title,
                                    "published_ts_utc": None,
                                    "extracted_text": a.extracted_text[
                                        : self.policy.evidence_excerpt_max_chars
                                    ],
                                    "content_hash": hashlib.sha256(
                                        f"{a.url}|{a.extracted_text}".lower().encode()
                                    ).hexdigest(),
                                }
                            )
                        await self.repo.bulk_upsert_evidence(items)
                        logger.info(
                            "proactive_search_done",
                            market_id=candidate.market_id,
                            articles=len(articles),
                        )
                except Exception:
                    logger.exception(
                        "proactive_search_error",
                        market_id=candidate.market_id,
                    )

            # Re-fetch evidence and re-link for sparse candidates.
            fresh_evidence = await self.repo.get_recent_evidence(
                self.policy.new_evidence_recheck_window_minutes // 60
                if self.policy.new_evidence_recheck_window_minutes >= 60
                else 72
            )
            sparse_markets = [
                market_map[c.market_id]
                for c in sparse_candidates
                if c.market_id in market_map
            ]
            fresh_linked = await self.evidence_linker.link(
                fresh_evidence, sparse_markets
            )
            linked.update(fresh_linked)

        now = datetime.now(timezone.utc)
        for candidate in candidates:
            market = market_map.get(candidate.market_id)
            snap = snapshots.get(candidate.market_id)
            if not market or not snap:
                continue

            evidence = linked.get(candidate.market_id, [])
            position = pos_map.get(candidate.market_id)

            packet = self.packet_builder.build(market, snap, evidence, position)
            packet_hash = self.packet_builder.compute_hash(packet)

            await self.repo.add_packet(
                {
                    "market_id": candidate.market_id,
                    "ts_utc": now,
                    "packet_json": json.loads(packet.model_dump_json()),
                    "packet_hash": packet_hash,
                    "packet_version": packet.packet_version,
                }
            )

            logger.info(
                "packet_built",
                market_id=candidate.market_id,
                packet_hash=packet_hash[:16],
                evidence_count=len(evidence),
            )

    # ------------------------------------------------------------------
    # LLM panel execution
    # ------------------------------------------------------------------

    async def _run_panel_for_candidate(
        self,
        candidate,
        snap,
        snapshots,
        signal_bundle=None,
    ) -> AggregationResult | None:
        """Run the LLM panel for a candidate market and return aggregation."""
        # Check daily panel market limit.
        try:
            panel_count = await self.repo.get_panel_markets_today()
            if panel_count >= self.policy.max_panel_markets_per_day:
                logger.info(
                    "panel_daily_limit",
                    market_id=candidate.market_id,
                    count=panel_count,
                )
                return None
        except Exception:
            logger.exception("panel_market_count_error")

        # Get the latest packet.
        packet_row = await self.repo.get_latest_packet(candidate.market_id)
        if packet_row is None:
            logger.warning("panel_no_packet", market_id=candidate.market_id)
            return None

        from src.packets.schemas import Packet

        packet = Packet.model_validate(packet_row.packet_json)

        # Run the panel with signal bundle.
        panel_result = await self.panel.run_panel(
            packet, signal_bundle=signal_bundle
        )

        # Store model runs.
        now = datetime.now(timezone.utc)
        self._store_panel_model_runs(
            panel_result, candidate.market_id, packet_row.packet_id, now
        )

        if not panel_result.proposals:
            return None

        # Aggregate.
        agg_result = self.aggregator.aggregate(panel_result.proposals, snap.mid)

        # Check escalation with the new system.
        sizing_input_temp = SizingInput(
            p_consensus=agg_result.p_consensus,
            p_market=snap.mid,
            confidence=agg_result.confidence,
            disagreement=agg_result.disagreement,
            best_bid=snap.best_bid or 0.0,
            best_ask=snap.best_ask or 0.0,
        )
        sizing_temp = compute_size(sizing_input_temp, self.policy)
        proposed_size_frac = sizing_temp.clamped_size_eur / self.policy.bankroll_eur

        # Compute odds_move (approximate).
        odds_move = 0.0

        escalation_agent, escalation_trigger = self.panel.determine_escalation(
            panel_result.proposals,
            veto_score=agg_result.veto_score,
            proposed_size_frac=proposed_size_frac,
            odds_move=odds_move,
        )
        if escalation_agent is not None and escalation_trigger is not None:
            logger.info(
                "panel_escalating",
                market_id=candidate.market_id,
                agent_id=escalation_agent.agent_id,
                trigger=escalation_trigger.value,
            )
            escalation_result = await self.panel.run_escalation(
                escalation_agent, escalation_trigger, packet,
                panel_result.proposals,
            )

            # Store escalation model runs.
            for proposal in escalation_result.proposals:
                if proposal.model_id == escalation_agent.agent_id:
                    try:
                        await self.repo.add_model_run(
                            {
                                "run_id": str(uuid.uuid4()),
                                "market_id": candidate.market_id,
                                "packet_id": packet_row.packet_id,
                                "ts_utc": now,
                                "model_id": proposal.model_id,
                                "tier": "escalation",
                                "prompt_version": PROMPT_VERSION,
                                "charter_version": self.panel.charter_hash,
                                "policy_version": self.policy_hash,
                                "raw_response": proposal.model_dump_json(),
                                "parsed_json": {
                                    **proposal.model_dump(mode="json"),
                                    "provider": escalation_agent.provider,
                                    "model": escalation_agent.model,
                                    "trigger": escalation_trigger.value,
                                },
                                "parse_ok": True,
                                "budget_skip": False,
                                "estimated_cost_eur": escalation_result.total_cost_eur,
                            }
                        )
                    except Exception:
                        logger.exception(
                            "model_run_store_error",
                            market_id=candidate.market_id,
                        )

            if escalation_result.proposals:
                agg_result = self.aggregator.aggregate(
                    escalation_result.proposals, snap.mid
                )

        # Store aggregation (enriched with p_market and consensus_side).
        try:
            agg_dict = asdict(agg_result)
            agg_dict["p_market"] = snap.mid
            agg_dict["consensus_side"] = (
                "BUY_YES" if agg_result.p_consensus > snap.mid else "BUY_NO"
            )
            await self.repo.add_aggregation(
                {
                    "market_id": candidate.market_id,
                    "ts_utc": now,
                    "aggregation_json": agg_dict,
                    "policy_version": self.policy_hash,
                }
            )
        except Exception:
            logger.exception(
                "aggregation_store_error", market_id=candidate.market_id
            )

        return agg_result

    # ------------------------------------------------------------------
    # Position lifecycle review
    # ------------------------------------------------------------------

    def _check_review_triggers(
        self,
        position,
        snap,
        market,
        hours_to_res: float | None,
        signal_snapshot=None,
    ) -> list[str]:
        """Return trigger reasons for panel re-analysis (empty = no re-analysis)."""
        triggers: list[str] = []

        if snap and snap.mid is not None:
            odds_move = abs(snap.mid - position.avg_entry_price)
            if odds_move >= self.policy.odds_move_recheck_threshold:
                triggers.append("odds_move")

        if snap and snap.liquidity is not None:
            if snap.liquidity < self.policy.min_liquidity_eur:
                triggers.append("liquidity_drop")

        if hours_to_res is not None:
            if hours_to_res < self.policy.min_hours_to_resolution * 2:
                triggers.append("approaching_resolution")

        # Signal-based triggers.
        if signal_snapshot is not None:
            if (
                signal_snapshot.volume_ratio_24h is not None
                and signal_snapshot.volume_ratio_24h >= self.policy.triage_volume_surge_threshold + 1.0
            ):
                triggers.append("volume_surge")
            if (
                signal_snapshot.google_trends_spike is not None
                and signal_snapshot.google_trends_spike >= self.policy.triage_trends_spike_threshold
            ):
                triggers.append("trends_spike")
            if (
                signal_snapshot.wikipedia_spike is not None
                and signal_snapshot.wikipedia_spike >= self.policy.triage_wiki_spike_threshold
            ):
                triggers.append("wiki_spike")

        return triggers

    async def _check_new_evidence_trigger(self) -> bool:
        """Return True if new evidence exists within the recheck window."""
        try:
            count = await self.repo.count_recent_evidence(
                self.policy.new_evidence_recheck_window_minutes
            )
            return count >= 1
        except Exception:
            logger.exception("new_evidence_trigger_error")
            return False

    async def _build_packet_for_position(self, position, market, snap) -> int | None:
        """Build and store a packet for an open position. Returns packet_id."""
        if self.packet_builder is None:
            return None

        recent_evidence = await self.repo.get_recent_evidence(
            self.policy.new_evidence_recheck_window_minutes // 60
            if self.policy.new_evidence_recheck_window_minutes >= 60
            else 72
        )
        linked = await self.evidence_linker.link(recent_evidence, [market])
        evidence = linked.get(market.market_id, [])

        packet = self.packet_builder.build(market, snap, evidence, position)
        packet_hash = self.packet_builder.compute_hash(packet)

        now = datetime.now(timezone.utc)
        packet_id = await self.repo.add_packet(
            {
                "market_id": market.market_id,
                "ts_utc": now,
                "packet_json": json.loads(packet.model_dump_json()),
                "packet_hash": packet_hash,
                "packet_version": packet.packet_version,
            }
        )
        return packet_id

    def _store_panel_model_runs(
        self, panel_result, market_id: str, packet_id, now
    ) -> None:
        """Store model runs from panel results. Fire-and-forget pattern."""
        for proposal in panel_result.proposals:
            try:
                # Find the agent config to get provider+model info.
                agent_info = self._find_agent_info(proposal.model_id)
                parsed_json = proposal.model_dump(mode="json")
                if agent_info:
                    parsed_json["provider"] = agent_info.provider
                    parsed_json["model"] = agent_info.model

                asyncio.ensure_future(self.repo.add_model_run(
                    {
                        "run_id": str(uuid.uuid4()),
                        "market_id": market_id,
                        "packet_id": packet_id,
                        "ts_utc": now,
                        "model_id": proposal.model_id,
                        "tier": "agent",
                        "prompt_version": PROMPT_VERSION,
                        "charter_version": self.panel.charter_hash,
                        "policy_version": self.policy_hash,
                        "raw_response": proposal.model_dump_json(),
                        "parsed_json": parsed_json,
                        "parse_ok": True,
                        "budget_skip": False,
                        "estimated_cost_eur": panel_result.total_cost_eur
                        / max(len(panel_result.proposals), 1),
                    }
                ))
            except Exception:
                logger.exception(
                    "model_run_store_error", market_id=market_id
                )

    def _find_agent_info(self, agent_id: str) -> PanelAgent | None:
        """Look up agent config by agent_id in panel and escalation lists."""
        for agent in self.panel.default_panel:
            if agent.agent_id == agent_id:
                return agent
        for agent in self.panel.escalation_agents:
            if agent.agent_id == agent_id:
                return agent
        return None

    async def _run_panel_for_position(
        self,
        position,
        snap,
        market,
        packet_row,
        position_exposure_rank: int | None = None,
    ) -> AggregationResult | None:
        """Run the LLM panel for an open position and return aggregation."""
        # Check daily panel market limit.
        try:
            panel_count = await self.repo.get_panel_markets_today()
            if panel_count >= self.policy.max_panel_markets_per_day:
                logger.info(
                    "panel_daily_limit",
                    market_id=position.market_id,
                    count=panel_count,
                )
                return None
        except Exception:
            logger.exception("panel_market_count_error")

        from src.packets.schemas import Packet as PacketSchema

        packet = PacketSchema.model_validate(packet_row.packet_json)

        # Run the panel with exposure rank.
        panel_result = await self.panel.run_panel(
            packet, position_exposure_rank=position_exposure_rank
        )

        # Store model runs.
        now = datetime.now(timezone.utc)
        self._store_panel_model_runs(
            panel_result, position.market_id, packet_row.packet_id, now
        )

        if not panel_result.proposals:
            return None

        # Aggregate.
        agg_result = self.aggregator.aggregate(panel_result.proposals, snap.mid)

        # Check escalation.
        odds_move = abs(snap.mid - position.avg_entry_price) if snap.mid else 0.0
        sizing_input_temp = SizingInput(
            p_consensus=agg_result.p_consensus,
            p_market=snap.mid,
            confidence=agg_result.confidence,
            disagreement=agg_result.disagreement,
            best_bid=(snap.best_bid or 0.0) if snap else 0.0,
            best_ask=(snap.best_ask or 0.0) if snap else 0.0,
        )
        sizing_temp = compute_size(sizing_input_temp, self.policy)
        proposed_size_frac = sizing_temp.clamped_size_eur / self.policy.bankroll_eur

        escalation_agent, escalation_trigger = self.panel.determine_escalation(
            panel_result.proposals,
            veto_score=agg_result.veto_score,
            proposed_size_frac=proposed_size_frac,
            odds_move=odds_move,
            position_exposure_rank=position_exposure_rank,
        )
        if escalation_agent is not None and escalation_trigger is not None:
            logger.info(
                "panel_escalating",
                market_id=position.market_id,
                agent_id=escalation_agent.agent_id,
                trigger=escalation_trigger.value,
            )
            escalation_result = await self.panel.run_escalation(
                escalation_agent, escalation_trigger, packet,
                panel_result.proposals,
            )

            # Store escalation model runs.
            for proposal in escalation_result.proposals:
                if proposal.model_id == escalation_agent.agent_id:
                    try:
                        await self.repo.add_model_run(
                            {
                                "run_id": str(uuid.uuid4()),
                                "market_id": position.market_id,
                                "packet_id": packet_row.packet_id,
                                "ts_utc": now,
                                "model_id": proposal.model_id,
                                "tier": "escalation",
                                "prompt_version": PROMPT_VERSION,
                                "charter_version": self.panel.charter_hash,
                                "policy_version": self.policy_hash,
                                "raw_response": proposal.model_dump_json(),
                                "parsed_json": {
                                    **proposal.model_dump(mode="json"),
                                    "provider": escalation_agent.provider,
                                    "model": escalation_agent.model,
                                    "trigger": escalation_trigger.value,
                                },
                                "parse_ok": True,
                                "budget_skip": False,
                                "estimated_cost_eur": escalation_result.total_cost_eur,
                            }
                        )
                    except Exception:
                        logger.exception(
                            "model_run_store_error",
                            market_id=position.market_id,
                        )

            if escalation_result.proposals:
                agg_result = self.aggregator.aggregate(
                    escalation_result.proposals, snap.mid
                )

        # Store aggregation (enriched with p_market and consensus_side).
        try:
            agg_dict = asdict(agg_result)
            agg_dict["p_market"] = snap.mid
            agg_dict["consensus_side"] = (
                "BUY_YES" if agg_result.p_consensus > snap.mid else "BUY_NO"
            )
            await self.repo.add_aggregation(
                {
                    "market_id": position.market_id,
                    "ts_utc": now,
                    "aggregation_json": agg_dict,
                    "policy_version": self.policy_hash,
                }
            )
        except Exception:
            logger.exception(
                "aggregation_store_error", market_id=position.market_id
            )

        return agg_result

    async def _execute_reduce(self, pos, snap, decision) -> None:
        """Execute a REDUCE action: sell a fraction of the position.

        If the position size is below ``dust_position_eur``, close it
        entirely instead of halving forever (Zeno's paradox guard).
        """
        # Dust guard: close instead of reducing tiny positions.
        if pos.size_eur < self.policy.dust_position_eur:
            logger.info(
                "dust_position_close",
                market_id=pos.market_id,
                size_eur=pos.size_eur,
                threshold=self.policy.dust_position_eur,
            )
            close_price = snap.mid if snap and snap.mid else pos.avg_entry_price
            exit_fee_rate = self._get_fee_rate(pos.market_id)
            realized = calculate_realized_pnl(
                side=pos.side,
                size_eur=pos.size_eur,
                avg_entry_price=pos.avg_entry_price,
                exit_price=close_price,
                fee_eur=pos.size_eur * exit_fee_rate,
            )
            self.daily_realized_pnl += realized
            await self.repo.upsert_position(
                {
                    "market_id": pos.market_id,
                    "side": pos.side,
                    "size_eur": pos.size_eur,
                    "avg_entry_price": pos.avg_entry_price,
                    "last_price": close_price,
                    "unrealized_pnl": 0.0,
                    "realized_pnl": pos.realized_pnl + realized,
                    "status": "closed",
                    "opened_ts_utc": pos.opened_ts_utc,
                    "last_update_ts_utc": datetime.now(timezone.utc),
                }
            )
            return

        reduce_size = pos.size_eur * self.policy.reduce_fraction

        bid = snap.best_bid if snap and snap.best_bid else (snap.mid if snap and snap.mid else pos.avg_entry_price)
        ask = snap.best_ask if snap and snap.best_ask else (snap.mid if snap and snap.mid else pos.avg_entry_price)

        reduce_fee_rate = self._get_fee_rate(pos.market_id)
        ob = await self._fetch_order_book(pos.market_id, side="SELL")
        fill = self.executor.execute(
            side="SELL",
            size_eur=reduce_size,
            best_bid=bid,
            best_ask=ask,
            fee_rate_override=reduce_fee_rate,
            order_book=ob,
        )

        # Calculate realized PnL on the reduced portion.
        partial_pnl = calculate_realized_pnl(
            side=pos.side,
            size_eur=reduce_size,
            avg_entry_price=pos.avg_entry_price,
            exit_price=fill.price,
            fee_eur=fill.fee_eur,
        )

        now = datetime.now(timezone.utc)

        # Record decision.
        decision_id = await self.repo.add_decision(
            {
                "market_id": pos.market_id,
                "ts_utc": now,
                "action": "REDUCE",
                "size_eur": reduce_size,
                "reason_json": {"reasons": decision.reasons},
                "policy_version": self.policy_hash,
            }
        )

        # Record order.
        order_id = await self.repo.add_order(
            {
                "decision_id": decision_id,
                "market_id": pos.market_id,
                "side": "SELL",
                "size_eur": reduce_size,
                "limit_price_ref": fill.price,
                "status": "filled",
                "created_ts_utc": now,
            }
        )

        # Record fill.
        await self.repo.add_fill(
            {
                "order_id": order_id,
                "ts_utc": now,
                "price": fill.price,
                "size_eur": reduce_size,
                "fee_eur": fill.fee_eur,
            }
        )

        # Update position: reduce size, accumulate realized PnL.
        new_size = pos.size_eur - reduce_size
        new_realized = pos.realized_pnl + partial_pnl
        self.daily_realized_pnl += partial_pnl

        close_price = snap.mid if snap and snap.mid else pos.avg_entry_price
        unrealized = calculate_unrealized_pnl(
            side=pos.side,
            size_eur=new_size,
            avg_entry_price=pos.avg_entry_price,
            current_price=close_price,
        )

        await self.repo.upsert_position(
            {
                "market_id": pos.market_id,
                "side": pos.side,
                "size_eur": new_size,
                "avg_entry_price": pos.avg_entry_price,
                "last_price": close_price,
                "unrealized_pnl": unrealized,
                "realized_pnl": new_realized,
                "status": "open",
                "opened_ts_utc": pos.opened_ts_utc,
                "last_update_ts_utc": now,
            }
        )

        logger.info(
            "position_reduced",
            market_id=pos.market_id,
            reduce_size=reduce_size,
            new_size=new_size,
            partial_pnl=partial_pnl,
        )

    async def review_open_positions(self) -> None:
        """Review all open positions and apply lifecycle decisions."""
        async with self._trade_lock:
            await self._review_open_positions_impl()

    async def _review_open_positions_impl(self) -> None:
        """Inner implementation of position review (lock held by caller)."""
        logger.info("position_review_start")
        positions = await self.repo.get_open_positions()
        markets = await self.repo.get_active_markets()
        market_map = {m.market_id: m for m in markets}

        # Check once whether new evidence exists (shared across all positions).
        has_new_evidence = await self._check_new_evidence_trigger()

        # Compute exposure ranks for high-stakes detection.
        positions_sorted = sorted(
            positions, key=lambda p: p.size_eur, reverse=True
        )
        exposure_rank = {p.market_id: i + 1 for i, p in enumerate(positions_sorted)}

        for pos in positions:
            snap = await self.repo.get_latest_snapshot(pos.market_id)
            market = market_map.get(pos.market_id)

            # Calculate hours to resolution.
            hours_to_res: float | None = None
            if market and market.resolution_time_utc:
                delta = market.resolution_time_utc - datetime.now(timezone.utc)
                hours_to_res = delta.total_seconds() / 3600

            # Update unrealized PnL.
            if snap and snap.mid is not None:
                unrealized = calculate_unrealized_pnl(
                    side=pos.side,
                    size_eur=pos.size_eur,
                    avg_entry_price=pos.avg_entry_price,
                    current_price=snap.mid,
                )
                await self.repo.upsert_position(
                    {
                        "market_id": pos.market_id,
                        "side": pos.side,
                        "size_eur": pos.size_eur,
                        "avg_entry_price": pos.avg_entry_price,
                        "last_price": snap.mid,
                        "unrealized_pnl": unrealized,
                        "realized_pnl": pos.realized_pnl,
                        "status": "open",
                        "opened_ts_utc": pos.opened_ts_utc,
                        "last_update_ts_utc": datetime.now(timezone.utc),
                    }
                )

            # Compute and store online score (mark-to-market).
            if snap and snap.mid is not None:
                try:
                    latest_agg = await self.repo.get_latest_aggregation(pos.market_id)
                    if latest_agg and latest_agg.aggregation_json:
                        p_consensus = latest_agg.aggregation_json.get("p_consensus")
                        if p_consensus is not None:
                            from src.evaluation.online_scorer import compute_edge_capture

                            hours_since = (
                                datetime.now(timezone.utc) - pos.opened_ts_utc
                            ).total_seconds() / 3600
                            edge_cap, dir_correct = compute_edge_capture(
                                side=pos.side,
                                p_consensus_at_entry=p_consensus,
                                p_market_at_entry=pos.avg_entry_price,
                                p_market_now=snap.mid,
                            )
                            unrealized_for_score = calculate_unrealized_pnl(
                                side=pos.side,
                                size_eur=pos.size_eur,
                                avg_entry_price=pos.avg_entry_price,
                                current_price=snap.mid,
                            )
                            await self.repo.add_online_score(
                                {
                                    "market_id": pos.market_id,
                                    "position_id": pos.position_id,
                                    "ts_utc": datetime.now(timezone.utc),
                                    "hours_since_entry": hours_since,
                                    "p_consensus_at_entry": p_consensus,
                                    "p_market_at_entry": pos.avg_entry_price,
                                    "p_market_now": snap.mid,
                                    "edge_capture": edge_cap,
                                    "direction_correct": dir_correct,
                                    "unrealized_pnl_eur": unrealized_for_score,
                                }
                            )
                except Exception:
                    logger.warning("online_score_error", market_id=pos.market_id)

            # Fetch latest signal snapshot for this position.
            signal_snap = None
            if self.signal_collector:
                try:
                    signal_snap = await self.repo.get_latest_signal_snapshot(pos.market_id)
                except Exception:
                    logger.warning("signal_snapshot_fetch_error", market_id=pos.market_id)

            # Check triggers for panel re-analysis.
            triggers = self._check_review_triggers(pos, snap, market, hours_to_res, signal_snap)
            if has_new_evidence:
                triggers.append("new_evidence")

            if triggers and self.panel and self.aggregator and self.packet_builder and market and snap:
                # Panel re-analysis path.
                logger.info(
                    "position_review_triggers",
                    market_id=pos.market_id,
                    triggers=triggers,
                )

                # xAI search on odds move.
                if "odds_move" in triggers and self.xai_search_client and self.xai_search_client.can_search():
                    try:
                        query = f"{market.question} current odds {snap.mid:.0%}"
                        articles = await self.xai_search_client.search(query)
                        if articles:
                            now = datetime.now(timezone.utc)
                            evidence_items = []
                            for article in articles:
                                content_hash = f"xai_{hash(article.url + article.extracted_text)}"
                                evidence_items.append(
                                    {
                                        "ts_utc": now,
                                        "source_type": "xai_search",
                                        "url": article.url,
                                        "title": article.title,
                                        "published_ts_utc": None,
                                        "extracted_text": article.extracted_text[
                                            : self.policy.evidence_excerpt_max_chars
                                        ],
                                        "content_hash": content_hash,
                                    }
                                )
                            await self.repo.bulk_upsert_evidence(evidence_items)
                    except Exception:
                        logger.exception("xai_search_review_error", market_id=pos.market_id)

                    # Also search social media for sentiment.
                    try:
                        if self.xai_search_client.can_search():
                            social_articles = await self.xai_search_client.search_social(
                                f"{market.question} latest news opinions",
                                max_results=3,
                            )
                            if social_articles:
                                now_ts = datetime.now(timezone.utc)
                                social_items = []
                                for a in social_articles:
                                    social_items.append(
                                        {
                                            "ts_utc": now_ts,
                                            "source_type": "xai_social",
                                            "url": a.url,
                                            "title": a.title,
                                            "published_ts_utc": None,
                                            "extracted_text": a.extracted_text[
                                                : self.policy.evidence_excerpt_max_chars
                                            ],
                                            "content_hash": hashlib.sha256(
                                                f"{a.url}|{a.extracted_text}".lower().encode()
                                            ).hexdigest(),
                                        }
                                    )
                                await self.repo.bulk_upsert_evidence(social_items)
                    except Exception:
                        logger.exception("xai_social_review_error", market_id=pos.market_id)

                # Build packet and run panel.
                await self._build_packet_for_position(pos, market, snap)
                packet_row = await self.repo.get_latest_packet(pos.market_id)
                if packet_row is None:
                    logger.warning("review_no_packet", market_id=pos.market_id)
                    decision = self.lifecycle.evaluate(
                        position=pos,
                        current_snapshot=snap,
                        entry_snapshot_mid=pos.avg_entry_price,
                        hours_to_resolution=hours_to_res,
                    )
                else:
                    agg_result = await self._run_panel_for_position(
                        pos, snap, market, packet_row,
                        position_exposure_rank=exposure_rank.get(pos.market_id),
                    )
                    if agg_result is None:
                        decision = self.lifecycle.evaluate(
                            position=pos,
                            current_snapshot=snap,
                            entry_snapshot_mid=pos.avg_entry_price,
                            hours_to_resolution=hours_to_res,
                        )
                    else:
                        # Get prior aggregation.
                        prior_agg_row = await self.repo.get_latest_aggregation(pos.market_id)
                        prior_agg: AggregationResult | None = None
                        if prior_agg_row and prior_agg_row.aggregation_json:
                            try:
                                _agg_keys = {f.name for f in dataclass_fields(AggregationResult)}
                                _filtered = {
                                    k: v for k, v in prior_agg_row.aggregation_json.items()
                                    if k in _agg_keys
                                }
                                prior_agg = AggregationResult(**_filtered)
                            except Exception:
                                logger.exception("prior_agg_parse_error", market_id=pos.market_id)

                        decision = self.lifecycle.evaluate_with_aggregation(
                            position=pos,
                            current_snapshot=snap,
                            entry_snapshot_mid=pos.avg_entry_price,
                            hours_to_resolution=hours_to_res,
                            current_agg=agg_result,
                            prior_agg=prior_agg,
                        )
            else:
                # Deterministic fallback.
                decision = self.lifecycle.evaluate(
                    position=pos,
                    current_snapshot=snap,
                    entry_snapshot_mid=pos.avg_entry_price,
                    hours_to_resolution=hours_to_res,
                )

            # Execute decision.
            if decision.action == LifecycleAction.CLOSE:
                logger.info(
                    "lifecycle_action",
                    market_id=pos.market_id,
                    action="CLOSE",
                    reasons=decision.reasons,
                )
                close_price = (
                    snap.mid
                    if snap and snap.mid
                    else pos.avg_entry_price
                )
                exit_fee_rate = self._get_fee_rate(pos.market_id, market)
                realized = calculate_realized_pnl(
                    side=pos.side,
                    size_eur=pos.size_eur,
                    avg_entry_price=pos.avg_entry_price,
                    exit_price=close_price,
                    fee_eur=pos.size_eur * exit_fee_rate,
                )
                self.daily_realized_pnl += realized
                await self.repo.upsert_position(
                    {
                        "market_id": pos.market_id,
                        "side": pos.side,
                        "size_eur": pos.size_eur,
                        "avg_entry_price": pos.avg_entry_price,
                        "last_price": close_price,
                        "unrealized_pnl": 0.0,
                        "realized_pnl": realized,
                        "status": "closed",
                        "opened_ts_utc": pos.opened_ts_utc,
                        "last_update_ts_utc": datetime.now(timezone.utc),
                    }
                )
            elif decision.action == LifecycleAction.REDUCE:
                logger.info(
                    "lifecycle_action",
                    market_id=pos.market_id,
                    action="REDUCE",
                    reasons=decision.reasons,
                )
                await self._execute_reduce(pos, snap, decision)

        logger.info("position_review_done", positions_reviewed=len(positions))

    # ------------------------------------------------------------------
    # Daily scoring (M5)
    # ------------------------------------------------------------------

    async def run_daily_scoring(self) -> int:
        """Score all models on yesterday's resolved markets.

        Returns the number of models scored.
        """
        logger.info("daily_scoring_start")
        try:
            scores = await self.scorer.run_daily_scoring()
            logger.info("daily_scoring_done", models_scored=len(scores))
            return len(scores)
        except Exception:
            logger.exception("daily_scoring_error")
            return 0

    # ------------------------------------------------------------------
    # Daily reset
    # ------------------------------------------------------------------

    def reset_daily(self) -> None:
        """Reset daily tracking (call at start of each day)."""
        self.daily_realized_pnl = 0.0
        logger.info("daily_reset")
