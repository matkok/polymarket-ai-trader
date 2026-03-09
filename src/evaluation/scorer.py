"""Model scoring orchestrator.

Queries the database for resolved markets, computes forecast metrics
and attribution for each model, and stores daily scores.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone

import structlog

from src.config.policy import Policy
from src.db.repository import Repository
from src.evaluation.attribution import (
    ModelAttribution,
    ProposalSummary,
    TradeOutcome,
    compute_attribution,
)
from src.evaluation.metrics import (
    ForecastRecord,
    ModelMetrics,
    compute_model_metrics,
)

logger = structlog.get_logger(__name__)


class ModelScorer:
    """Orchestrates daily model scoring from DB data."""

    def __init__(self, repo: Repository, policy: Policy) -> None:
        self.repo = repo
        self.policy = policy

    async def run_daily_scoring(
        self, scoring_date: date | None = None
    ) -> list[dict]:
        """Score all models for the given date (defaults to yesterday).

        Returns a list of score dicts that were stored.
        """
        if scoring_date is None:
            scoring_date = (datetime.now(timezone.utc) - timedelta(days=1)).date()

        logger.info("daily_scoring_start", date=str(scoring_date))

        # 1. Get resolutions from the scoring window (full day).
        day_start = datetime.combine(scoring_date, datetime.min.time()).replace(
            tzinfo=timezone.utc
        )
        day_end = day_start + timedelta(days=1)
        resolutions = await self.repo.get_resolutions_since(day_start)
        resolutions = [r for r in resolutions if r.resolved_ts_utc < day_end]

        if not resolutions:
            logger.info("daily_scoring_no_resolutions", date=str(scoring_date))
            return []

        # 2. Build forecast records and trade outcomes for each resolved market.
        forecast_records: list[ForecastRecord] = []
        trade_outcomes: list[TradeOutcome] = []
        all_model_ids: set[str] = set()

        for resolution in resolutions:
            outcome_binary = 1 if resolution.outcome == "YES" else 0

            # Get model runs for this market.
            model_runs = await self.repo.get_model_runs_for_market(
                resolution.market_id
            )
            if not model_runs:
                continue

            proposal_summaries: list[ProposalSummary] = []
            for run in model_runs:
                parsed = run.parsed_json or {}
                p_true = parsed.get("p_true")
                if p_true is None:
                    continue

                model_id = run.model_id
                all_model_ids.add(model_id)

                forecast_records.append(
                    ForecastRecord(
                        model_id=model_id,
                        market_id=resolution.market_id,
                        p_true=p_true,
                        outcome=outcome_binary,
                    )
                )

                proposal_summaries.append(
                    ProposalSummary(
                        model_id=model_id,
                        direction=parsed.get("direction", "BUY_YES"),
                        p_true=p_true,
                        confidence=parsed.get("confidence", 0.5),
                        ambiguity_score=parsed.get("ambiguity_score", 0.0),
                        recommended_max_exposure_frac=parsed.get(
                            "recommended_max_exposure_frac", 0.0
                        ),
                    )
                )

            # Check if we had a position in this market.
            position = await self.repo.get_position(resolution.market_id)
            if position and position.status == "closed":
                profitable = position.realized_pnl > 0
                trade_outcomes.append(
                    TradeOutcome(
                        market_id=resolution.market_id,
                        position_side=position.side,
                        realized_pnl=position.realized_pnl,
                        profitable=profitable,
                        model_proposals=proposal_summaries,
                    )
                )

        if not forecast_records:
            logger.info("daily_scoring_no_forecasts", date=str(scoring_date))
            return []

        # 3. Compute metrics and attribution per model.
        scores_stored: list[dict] = []

        for model_id in sorted(all_model_ids):
            metrics = compute_model_metrics(model_id, forecast_records)
            attribution = compute_attribution(
                model_id,
                trade_outcomes,
                ambiguity_threshold=self.policy.ambiguity_veto_threshold,
            )

            calibration_data = [
                {
                    "bin_lower": b.bin_lower,
                    "bin_upper": b.bin_upper,
                    "count": b.count,
                    "mean_predicted": b.mean_predicted,
                    "actual_rate": b.actual_rate,
                }
                for b in metrics.calibration_bins
            ]

            pnl_attrib = {
                "trades_evaluated": attribution.trades_evaluated,
                "support_value": attribution.support_value,
                "dissent_value": attribution.dissent_value,
                "sizing_error": attribution.sizing_error,
            }

            veto_value = {
                "veto_value": attribution.veto_value,
            }

            score_data = {
                "model_id": model_id,
                "score_date": scoring_date,
                "markets_scored": metrics.markets_scored,
                "brier_score": metrics.brier_score,
                "log_loss": metrics.log_loss,
                "calibration_json": calibration_data,
                "pnl_attrib_json": pnl_attrib,
                "veto_value_json": veto_value,
                "notes": None,
            }

            try:
                await self.repo.add_model_score(score_data)
                scores_stored.append(score_data)
                logger.info(
                    "model_scored",
                    model_id=model_id,
                    date=str(scoring_date),
                    brier=metrics.brier_score,
                    log_loss=metrics.log_loss,
                    markets=metrics.markets_scored,
                )
            except Exception:
                logger.exception(
                    "model_score_store_error",
                    model_id=model_id,
                    date=str(scoring_date),
                )

        logger.info(
            "daily_scoring_done",
            date=str(scoring_date),
            models_scored=len(scores_stored),
            resolutions=len(resolutions),
        )
        return scores_stored
