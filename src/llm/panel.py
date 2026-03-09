"""LLM panel orchestrator — runs role-based agents concurrently."""

from __future__ import annotations

import asyncio
import json
import statistics
import uuid
from datetime import datetime, timezone

import structlog

from src.config.budget import BudgetTracker
from src.config.policy import Policy
from src.llm.base import BaseLLMClient
from src.llm.prompt_builder import (
    PROMPT_VERSION,
    build_system_prompt,
    build_user_prompt,
    charter_version_hash,
    get_proposal_schema,
    load_charter,
)
from src.llm.schemas import (
    DEFAULT_PANEL,
    ESCALATION_AGENTS,
    EscalationTrigger,
    ModelProposal,
    PanelAgent,
    PanelResult,
)
from src.packets.schemas import Packet
from src.signals.schemas import MarketSignalBundle

logger = structlog.get_logger(__name__)


class PanelOrchestrator:
    """Runs the multi-agent LLM panel for a single market."""

    def __init__(
        self,
        clients: dict[str, BaseLLMClient],
        budget_tracker: BudgetTracker,
        policy: Policy,
        charter_path: str,
        default_panel: list[PanelAgent] | None = None,
        escalation_agents: list[PanelAgent] | None = None,
    ) -> None:
        self.clients = clients
        self.budget_tracker = budget_tracker
        self.policy = policy
        self.charter_text = load_charter(charter_path)
        self.charter_hash = charter_version_hash(self.charter_text)
        self.proposal_schema = get_proposal_schema()
        self.default_panel = default_panel or DEFAULT_PANEL
        self.escalation_agents = escalation_agents or ESCALATION_AGENTS

    async def run_panel(
        self,
        packet: Packet,
        signal_bundle: MarketSignalBundle | None = None,
        position_exposure_rank: int | None = None,
    ) -> PanelResult:
        """Run the panel using role-based agents, return results."""
        user_prompt = build_user_prompt(packet)

        tasks: dict[str, asyncio.Task] = {}
        skipped: list[str] = []
        agents_used: list[str] = []

        for agent in self.default_panel:
            # Check conditional agents.
            if not agent.always_on:
                if not self._should_run_conditional(
                    agent, signal_bundle, position_exposure_rank
                ):
                    skipped.append(agent.agent_id)
                    logger.info(
                        "panel_agent_skipped",
                        agent_id=agent.agent_id,
                        reason="conditional_not_triggered",
                    )
                    continue

            # Check provider availability.
            client = self.clients.get(agent.provider)
            if client is None:
                skipped.append(agent.agent_id)
                logger.info(
                    "panel_agent_skipped",
                    agent_id=agent.agent_id,
                    reason="no_client",
                )
                continue

            # Tiered model selection.
            selected_model = self.budget_tracker.select_model(agent.provider)
            if selected_model is None:
                skipped.append(agent.agent_id)
                logger.info(
                    "panel_budget_exhausted",
                    agent_id=agent.agent_id,
                    provider=agent.provider,
                )
                continue

            if selected_model != agent.model:
                logger.info(
                    "panel_model_tiered",
                    agent_id=agent.agent_id,
                    default_model=agent.model,
                    selected_model=selected_model,
                    provider_spend=self.budget_tracker.spent.get(
                        agent.provider, 0.0
                    ),
                )

            # Budget check with selected model.
            system_prompt = build_system_prompt(
                self.charter_text, self.proposal_schema, role=agent.role
            )
            estimated_cost = client.estimate_cost_for_model(
                system_prompt, user_prompt, selected_model
            )
            if not self.budget_tracker.can_spend(agent.provider, estimated_cost):
                skipped.append(agent.agent_id)
                logger.info(
                    "panel_budget_skip",
                    agent_id=agent.agent_id,
                    estimated_cost=estimated_cost,
                )
                continue

            agents_used.append(agent.agent_id)
            tasks[agent.agent_id] = asyncio.create_task(
                self._call_agent(
                    agent, system_prompt, user_prompt, packet,
                    model_override=selected_model,
                )
            )

        results = {}
        if tasks:
            done = await asyncio.gather(*tasks.values(), return_exceptions=True)
            for agent_id, result in zip(tasks.keys(), done):
                results[agent_id] = result

        proposals: list[ModelProposal] = []
        total_cost = 0.0

        for agent_id, result in results.items():
            if isinstance(result, Exception):
                logger.error(
                    "panel_agent_error", agent_id=agent_id, error=str(result)
                )
                continue
            proposal, cost = result
            if proposal is not None:
                proposals.append(proposal)
            total_cost += cost

        return PanelResult(
            proposals=proposals,
            agents_used=agents_used,
            skipped_agents=skipped,
            total_cost_eur=total_cost,
        )

    def _should_run_conditional(
        self,
        agent: PanelAgent,
        signal_bundle: MarketSignalBundle | None,
        position_exposure_rank: int | None,
    ) -> bool:
        """Decide whether a conditional agent should run this cycle."""
        if agent.agent_id != "x_signals":
            return False

        # Trigger x_signals when ANY of:
        if signal_bundle and signal_bundle.microstructure:
            if (
                signal_bundle.microstructure.odds_move_6h is not None
                and signal_bundle.microstructure.odds_move_6h
                >= self.policy.odds_move_recheck_threshold
            ):
                return True

        if signal_bundle and signal_bundle.evidence_freshness:
            if signal_bundle.evidence_freshness.credible_evidence_6h == 0:
                return True

        if position_exposure_rank is not None and position_exposure_rank <= 3:
            return True

        return False

    def determine_escalation(
        self,
        proposals: list[ModelProposal],
        veto_score: float,
        proposed_size_frac: float,
        odds_move: float,
        position_exposure_rank: int | None = None,
    ) -> tuple[PanelAgent | None, EscalationTrigger | None]:
        """Decide whether to escalate and which agent to use.

        Priority order:
        1. Rules ambiguity quorum (veto_score >= quorum)
        2. Partial rules ambiguity (0 < veto_score < quorum)
        3. High disagreement (stdev >= 0.12)
        4. High stakes (top-3 position or size > 5%)
        5. Fast odds move (>= threshold)
        """
        if not proposals:
            return None, None

        escalation_map = {a.agent_id: a for a in self.escalation_agents}

        # 1. Rules ambiguity quorum → escalation_anthropic
        if veto_score >= self.policy.veto_quorum:
            agent = escalation_map.get("escalation_anthropic")
            if agent:
                return agent, EscalationTrigger.RULES_AMBIGUITY

        # 2. Partial rules ambiguity → escalation_anthropic
        if 0 < veto_score < self.policy.veto_quorum:
            agent = escalation_map.get("escalation_anthropic")
            if agent:
                return agent, EscalationTrigger.RULES_AMBIGUITY

        # 3. High disagreement → escalation_openai
        p_values = [p.p_true for p in proposals]
        if len(p_values) > 1:
            disagreement = statistics.stdev(p_values)
            if disagreement >= 0.12:
                agent = escalation_map.get("escalation_openai")
                if agent:
                    return agent, EscalationTrigger.DISAGREEMENT

        # 4. High stakes → escalation_google (arbiter)
        high_stakes = (
            (position_exposure_rank is not None and position_exposure_rank <= 3)
            or proposed_size_frac > 0.05
        )
        if high_stakes:
            agent = escalation_map.get("escalation_google")
            if agent:
                return agent, EscalationTrigger.HIGH_STAKES

        # 5. Fast odds move → escalation_openai
        if odds_move >= self.policy.odds_move_recheck_threshold:
            agent = escalation_map.get("escalation_openai")
            if agent:
                return agent, EscalationTrigger.FAST_ODDS_MOVE

        return None, None

    async def run_escalation(
        self,
        agent: PanelAgent,
        trigger: EscalationTrigger,
        packet: Packet,
        existing_proposals: list[ModelProposal],
    ) -> PanelResult:
        """Run a single escalation agent and append its proposal.

        Returns a PanelResult with all proposals (existing + escalation).
        """
        client = self.clients.get(agent.provider)
        if client is None:
            logger.warning(
                "escalation_no_client",
                agent_id=agent.agent_id,
                provider=agent.provider,
            )
            return PanelResult(
                proposals=existing_proposals,
                escalation_trigger=trigger,
                escalation_agent=agent.agent_id,
            )

        # Tiered model selection for escalation.
        selected_model = self.budget_tracker.select_model(agent.provider)
        if selected_model is None:
            logger.info(
                "escalation_budget_exhausted",
                agent_id=agent.agent_id,
                provider=agent.provider,
            )
            return PanelResult(
                proposals=existing_proposals,
                escalation_trigger=trigger,
                escalation_agent=agent.agent_id,
            )

        if selected_model != agent.model:
            logger.info(
                "escalation_model_tiered",
                agent_id=agent.agent_id,
                default_model=agent.model,
                selected_model=selected_model,
                provider_spend=self.budget_tracker.spent.get(
                    agent.provider, 0.0
                ),
            )

        system_prompt = build_system_prompt(
            self.charter_text, self.proposal_schema, role=agent.role
        )
        user_prompt = build_user_prompt(packet)

        estimated_cost = client.estimate_cost_for_model(
            system_prompt, user_prompt, selected_model
        )
        if not self.budget_tracker.can_spend(agent.provider, estimated_cost):
            logger.info(
                "escalation_budget_skip",
                agent_id=agent.agent_id,
                estimated_cost=estimated_cost,
            )
            return PanelResult(
                proposals=existing_proposals,
                escalation_trigger=trigger,
                escalation_agent=agent.agent_id,
            )

        try:
            proposal, cost = await self._call_agent(
                agent, system_prompt, user_prompt, packet,
                model_override=selected_model,
            )
            all_proposals = list(existing_proposals)
            if proposal is not None:
                all_proposals.append(proposal)
            return PanelResult(
                proposals=all_proposals,
                escalation_trigger=trigger,
                escalation_agent=agent.agent_id,
                total_cost_eur=cost,
            )
        except Exception:
            logger.exception(
                "escalation_error", agent_id=agent.agent_id
            )
            return PanelResult(
                proposals=existing_proposals,
                escalation_trigger=trigger,
                escalation_agent=agent.agent_id,
            )

    async def _call_agent(
        self,
        agent: PanelAgent,
        system_prompt: str,
        user_prompt: str,
        packet: Packet,
        model_override: str | None = None,
    ) -> tuple[ModelProposal | None, float]:
        """Call a single agent, parse response, handle retry."""
        client = self.clients[agent.provider]
        model = model_override or agent.model
        run_id = str(uuid.uuid4())

        for attempt in range(2):
            try:
                raw_response, usage = await client.call(
                    system_prompt, user_prompt, model
                )
                cost = client.actual_cost(model, usage)
                self.budget_tracker.record_spend(agent.provider, cost)

                proposal = self._parse_proposal(
                    raw_response, agent, run_id, packet.market_id
                )
                if proposal is not None:
                    return proposal, cost

                if attempt == 0:
                    logger.warning(
                        "panel_parse_retry",
                        agent_id=agent.agent_id,
                        model=agent.model,
                    )
                    continue
                else:
                    logger.warning(
                        "panel_parse_failed",
                        agent_id=agent.agent_id,
                        model=agent.model,
                    )
                    return None, cost

            except Exception:
                logger.exception(
                    "panel_call_error",
                    agent_id=agent.agent_id,
                    model=agent.model,
                    attempt=attempt,
                )
                if attempt == 1:
                    return None, 0.0

        return None, 0.0

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """Strip markdown code fences wrapping JSON."""
        stripped = text.strip()
        if stripped.startswith("```"):
            # Remove opening fence (```json or ```)
            first_newline = stripped.index("\n")
            stripped = stripped[first_newline + 1 :]
        if stripped.endswith("```"):
            stripped = stripped[: -3]
        return stripped.strip()

    def _parse_proposal(
        self,
        raw_response: str,
        agent: PanelAgent,
        run_id: str,
        market_id: str,
    ) -> ModelProposal | None:
        """Parse raw JSON response into a ModelProposal."""
        try:
            cleaned = self._strip_markdown_fences(raw_response)
            data = json.loads(cleaned)
            # Force metadata fields from orchestrator (never trust LLM values).
            data["model_id"] = agent.agent_id
            data["run_id"] = run_id
            data["market_id"] = market_id
            data["ts_utc"] = datetime.now(timezone.utc).isoformat()
            return ModelProposal.model_validate(data)
        except Exception as exc:
            logger.warning(
                "panel_parse_error",
                agent_id=agent.agent_id,
                model=agent.model,
                error=str(exc),
                response_preview=raw_response[:200],
            )
            return None

    # ---- Legacy methods (deprecated, kept for backward compat) ----

    def should_escalate(
        self,
        proposals: list[ModelProposal],
        proposed_size_frac: float,
        odds_move: float,
        hours_to_resolution: float | None,
        current_tier: object = None,
    ) -> None:
        """DEPRECATED: Use determine_escalation() instead. Always returns None."""
        return None
