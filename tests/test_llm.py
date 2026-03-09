"""Tests for src.llm — schemas, prompt builder, clients, panel orchestrator."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.budget import BudgetTracker, DailyBudget
from src.config.policy import Policy
from src.llm.base import BaseLLMClient
from src.llm.panel import PanelOrchestrator
from src.llm.prompt_builder import (
    PROMPT_VERSION,
    ROLE_PREAMBLES,
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
    EvidenceCitation,
    ModelProposal,
    ModelTier,
    PanelAgent,
    PanelResult,
    TIER_MODELS,
)
from src.packets.schemas import Packet, PacketMarketContext


# ---- Helpers ----------------------------------------------------------------


def _make_policy(**overrides) -> Policy:
    return Policy(**overrides)


def _make_proposal_dict(
    model_id: str = "rules_lawyer",
    p_true: float = 0.60,
    confidence: float = 0.80,
) -> dict:
    """Create a valid ModelProposal dict."""
    return {
        "model_id": model_id,
        "run_id": "run-1",
        "market_id": "m1",
        "ts_utc": "2025-06-01T00:00:00Z",
        "p_true": p_true,
        "confidence": confidence,
        "direction": "BUY_YES",
        "rules_ambiguity": 0.10,
        "evidence_ambiguity": 0.05,
        "recommended_max_exposure_frac": 0.05,
        "hold_horizon_hours": 48.0,
        "thesis": "Test thesis",
        "key_risks": ["risk1"],
        "evidence": [],
        "exit_triggers": ["trigger1"],
        "notes": "",
    }


def _make_packet() -> Packet:
    return Packet(
        market_id="m1",
        ts_utc=datetime(2025, 6, 1, tzinfo=timezone.utc),
        market_context=PacketMarketContext(
            question="Will it rain?",
            current_mid=0.50,
            best_bid=0.48,
            best_ask=0.52,
        ),
    )


def _make_panel_agent(
    agent_id: str = "rules_lawyer",
    provider: str = "anthropic",
    model: str = "claude-sonnet-4-20250514",
    role: str = "rules",
    always_on: bool = True,
) -> PanelAgent:
    return PanelAgent(agent_id, provider, model, role, always_on)


def _make_mock_client(
    provider: str, response_json: dict | None = None
) -> MagicMock:
    """Create a mock LLM client."""
    client = MagicMock(spec=BaseLLMClient)
    client.provider = provider
    client.estimate_cost_for_model.return_value = 0.01
    client.actual_cost.return_value = 0.005

    if response_json is None:
        response_json = _make_proposal_dict(model_id=f"{provider}/model")

    client.call = AsyncMock(
        return_value=(
            json.dumps(response_json),
            {"prompt_tokens": 100, "completion_tokens": 50},
        )
    )
    return client


# ---- ModelProposal schema ---------------------------------------------------


class TestModelProposal:
    """ModelProposal validation."""

    def test_valid_proposal(self) -> None:
        data = _make_proposal_dict()
        proposal = ModelProposal.model_validate(data)
        assert proposal.p_true == 0.60
        assert proposal.model_id == "rules_lawyer"

    def test_p_true_out_of_range(self) -> None:
        data = _make_proposal_dict(p_true=1.5)
        with pytest.raises(Exception):
            ModelProposal.model_validate(data)

    def test_confidence_out_of_range(self) -> None:
        data = _make_proposal_dict(confidence=-0.1)
        with pytest.raises(Exception):
            ModelProposal.model_validate(data)

    def test_evidence_citation(self) -> None:
        citation = EvidenceCitation(
            url="https://example.com", claim="test", strength=0.8
        )
        assert citation.strength == 0.8

    def test_evidence_citation_strength_out_of_range(self) -> None:
        with pytest.raises(Exception):
            EvidenceCitation(
                url="https://example.com", claim="test", strength=1.5
            )

    def test_rules_ambiguity_field(self) -> None:
        data = _make_proposal_dict()
        data["rules_ambiguity"] = 0.85
        proposal = ModelProposal.model_validate(data)
        assert proposal.rules_ambiguity == 0.85

    def test_evidence_ambiguity_field(self) -> None:
        data = _make_proposal_dict()
        data["evidence_ambiguity"] = 0.90
        proposal = ModelProposal.model_validate(data)
        assert proposal.evidence_ambiguity == 0.90

    def test_rules_ambiguity_out_of_range(self) -> None:
        data = _make_proposal_dict()
        data["rules_ambiguity"] = 1.5
        with pytest.raises(Exception):
            ModelProposal.model_validate(data)


# ---- Prompt builder ----------------------------------------------------------


class TestPromptBuilder:
    """Prompt construction functions."""

    def test_charter_version_hash_deterministic(self) -> None:
        text = "Test charter content"
        h1 = charter_version_hash(text)
        h2 = charter_version_hash(text)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_charter_version_hash_changes_with_content(self) -> None:
        h1 = charter_version_hash("charter v1")
        h2 = charter_version_hash("charter v2")
        assert h1 != h2

    def test_build_system_prompt_contains_charter(self) -> None:
        charter = "My trading charter."
        schema = get_proposal_schema()
        prompt = build_system_prompt(charter, schema)
        assert "My trading charter." in prompt
        assert "p_true" in prompt

    def test_build_user_prompt_contains_market(self) -> None:
        packet = _make_packet()
        prompt = build_user_prompt(packet)
        assert "Will it rain?" in prompt
        assert "m1" in prompt

    def test_get_proposal_schema_valid_json(self) -> None:
        schema = get_proposal_schema()
        parsed = json.loads(schema)
        assert "properties" in parsed

    def test_prompt_version_set(self) -> None:
        assert PROMPT_VERSION == "m4.0"

    def test_role_specific_prompt_rules(self) -> None:
        """Rules role prompt contains Rules Lawyer identity."""
        charter = "Charter text."
        schema = get_proposal_schema()
        prompt = build_system_prompt(charter, schema, role="rules")
        assert "Rules Lawyer" in prompt

    def test_role_specific_prompt_probability(self) -> None:
        """Probability role prompt contains Probabilist identity."""
        charter = "Charter text."
        schema = get_proposal_schema()
        prompt = build_system_prompt(charter, schema, role="probability")
        assert "Probabilist" in prompt

    def test_role_specific_prompt_sanity(self) -> None:
        """Sanity role prompt contains Sanity Checker identity."""
        charter = "Charter text."
        schema = get_proposal_schema()
        prompt = build_system_prompt(charter, schema, role="sanity")
        assert "Sanity Checker" in prompt

    def test_role_specific_prompt_signals(self) -> None:
        """Signals role prompt contains X Signals Analyst identity."""
        charter = "Charter text."
        schema = get_proposal_schema()
        prompt = build_system_prompt(charter, schema, role="signals")
        assert "X Signals Analyst" in prompt

    def test_role_specific_prompt_arbiter(self) -> None:
        """Arbiter role prompt contains Arbiter identity."""
        charter = "Charter text."
        schema = get_proposal_schema()
        prompt = build_system_prompt(charter, schema, role="arbiter")
        assert "Arbiter" in prompt

    def test_default_role_is_probability(self) -> None:
        """Default role param produces probability preamble."""
        charter = "Charter text."
        schema = get_proposal_schema()
        prompt = build_system_prompt(charter, schema)
        assert "Probabilist" in prompt

    def test_unknown_role_falls_back_to_probability(self) -> None:
        """Unknown role falls back to probability preamble."""
        charter = "Charter text."
        schema = get_proposal_schema()
        prompt = build_system_prompt(charter, schema, role="unknown_role")
        assert "Probabilist" in prompt

    def test_prompt_contains_rules_ambiguity_field(self) -> None:
        """System prompt references rules_ambiguity field."""
        charter = "Charter text."
        schema = get_proposal_schema()
        prompt = build_system_prompt(charter, schema)
        assert "rules_ambiguity" in prompt

    def test_prompt_contains_evidence_ambiguity_field(self) -> None:
        """System prompt references evidence_ambiguity field."""
        charter = "Charter text."
        schema = get_proposal_schema()
        prompt = build_system_prompt(charter, schema)
        assert "evidence_ambiguity" in prompt

    def test_all_role_preambles_exist(self) -> None:
        """All expected roles have preambles."""
        for role in ("rules", "probability", "sanity", "signals", "arbiter"):
            assert role in ROLE_PREAMBLES


# ---- Base LLM client --------------------------------------------------------


class TestBaseLLMClient:
    """Base client methods."""

    def test_get_model_openai_tier1(self) -> None:
        assert TIER_MODELS["openai"]["tier_1"] == "gpt-4.1-mini"
        assert TIER_MODELS["openai"]["tier_2"] == "gpt-4.1"
        assert TIER_MODELS["openai"]["tier_3"] == "o4-mini"

    def test_get_model_anthropic(self) -> None:
        assert TIER_MODELS["anthropic"]["tier_1"] == "claude-haiku-4-5-20251001"

    def test_get_model_gemini(self) -> None:
        assert TIER_MODELS["gemini"]["tier_1"] == "gemini-2.5-flash"

    def test_get_model_xai(self) -> None:
        assert TIER_MODELS["xai"]["tier_1"] == "grok-3-fast"

    def test_estimate_cost_for_model_returns_float(self) -> None:
        from src.llm.openai_client import OpenAILLMClient

        client = MagicMock(spec=OpenAILLMClient)
        client.PRICING = OpenAILLMClient.PRICING
        cost = BaseLLMClient.estimate_cost_for_model(
            client, "system prompt", "user prompt", "gpt-4.1-mini"
        )
        assert isinstance(cost, float)
        assert cost > 0

    def test_estimate_cost_for_model_unknown_model(self) -> None:
        from src.llm.openai_client import OpenAILLMClient

        client = MagicMock(spec=OpenAILLMClient)
        client.PRICING = OpenAILLMClient.PRICING
        cost = BaseLLMClient.estimate_cost_for_model(
            client, "system", "user", "nonexistent-model"
        )
        assert cost == 0.0

    def test_actual_cost_from_usage(self) -> None:
        from src.llm.openai_client import OpenAILLMClient

        client = MagicMock(spec=OpenAILLMClient)
        client.PRICING = OpenAILLMClient.PRICING
        cost = BaseLLMClient.actual_cost(
            client,
            "gpt-4.1-mini",
            {"prompt_tokens": 1000, "completion_tokens": 500},
        )
        expected = (1000 / 1000) * 0.00040 + (500 / 1000) * 0.0016
        assert cost == pytest.approx(expected)


# ---- Panel agent config -----------------------------------------------------


class TestPanelAgentConfig:
    """PanelAgent and DEFAULT_PANEL configuration."""

    def test_default_panel_has_four_agents(self) -> None:
        assert len(DEFAULT_PANEL) == 4

    def test_three_always_on(self) -> None:
        always_on = [a for a in DEFAULT_PANEL if a.always_on]
        assert len(always_on) == 3

    def test_one_conditional(self) -> None:
        conditional = [a for a in DEFAULT_PANEL if not a.always_on]
        assert len(conditional) == 1
        assert conditional[0].agent_id == "x_signals"

    def test_escalation_agents_exist(self) -> None:
        assert len(ESCALATION_AGENTS) == 3

    def test_agent_ids_unique(self) -> None:
        all_ids = [a.agent_id for a in DEFAULT_PANEL + ESCALATION_AGENTS]
        assert len(all_ids) == len(set(all_ids))

    def test_default_panel_roles(self) -> None:
        roles = {a.agent_id: a.role for a in DEFAULT_PANEL}
        assert roles["rules_lawyer"] == "rules"
        assert roles["probabilist"] == "probability"
        assert roles["sanity_checker"] == "sanity"
        assert roles["x_signals"] == "signals"


# ---- Panel orchestrator ------------------------------------------------------


class TestPanelOrchestrator:
    """Panel orchestrator agent-based execution and error handling."""

    def _make_orchestrator(
        self,
        clients: dict | None = None,
        budget: BudgetTracker | None = None,
        panel: list[PanelAgent] | None = None,
    ) -> PanelOrchestrator:
        if clients is None:
            clients = {}
        if budget is None:
            budget = BudgetTracker(DailyBudget())
        if panel is None:
            panel = [
                PanelAgent("agent_a", "openai", "gpt-5-mini", "probability", True),
                PanelAgent("agent_b", "anthropic", "claude-sonnet-4-20250514", "rules", True),
            ]
        with patch("src.llm.panel.load_charter", return_value="test charter"):
            return PanelOrchestrator(
                clients=clients,
                budget_tracker=budget,
                policy=_make_policy(),
                charter_path="charter.md",
                default_panel=panel,
            )

    async def test_runs_all_always_on_agents(self) -> None:
        """All always-on agents are called in parallel."""
        clients = {
            "openai": _make_mock_client("openai"),
            "anthropic": _make_mock_client("anthropic"),
        }
        orch = self._make_orchestrator(clients=clients)
        packet = _make_packet()
        result = await orch.run_panel(packet)

        assert len(result.proposals) == 2
        assert result.total_cost_eur > 0
        assert len(result.agents_used) == 2

    async def test_budget_skip(self) -> None:
        """Agents over budget are skipped."""
        clients = {
            "openai": _make_mock_client("openai"),
            "anthropic": _make_mock_client("anthropic"),
        }
        budget = BudgetTracker(DailyBudget(daily_eur_cap_openai=0.0))
        orch = self._make_orchestrator(clients=clients, budget=budget)
        packet = _make_packet()
        result = await orch.run_panel(packet)

        assert "agent_a" in result.skipped_agents
        assert len(result.proposals) == 1

    async def test_missing_provider_skipped(self) -> None:
        """Agents whose provider is not in clients are skipped."""
        clients = {"anthropic": _make_mock_client("anthropic")}
        orch = self._make_orchestrator(clients=clients)
        packet = _make_packet()
        result = await orch.run_panel(packet)

        assert "agent_a" in result.skipped_agents
        assert len(result.proposals) == 1

    async def test_agent_error_handled(self) -> None:
        """Errors from an agent don't crash the panel."""
        openai_client = _make_mock_client("openai")
        openai_client.call = AsyncMock(side_effect=Exception("API error"))

        anthropic_client = _make_mock_client("anthropic")

        clients = {"openai": openai_client, "anthropic": anthropic_client}
        orch = self._make_orchestrator(clients=clients)
        packet = _make_packet()
        result = await orch.run_panel(packet)

        assert len(result.proposals) == 1

    async def test_parse_failure_retries(self) -> None:
        """Parse failures trigger one retry."""
        client = _make_mock_client("openai")
        valid_response = json.dumps(_make_proposal_dict(model_id="agent_a"))
        client.call = AsyncMock(
            side_effect=[
                (
                    "not valid json",
                    {"prompt_tokens": 100, "completion_tokens": 50},
                ),
                (
                    valid_response,
                    {"prompt_tokens": 100, "completion_tokens": 50},
                ),
            ]
        )
        panel = [
            PanelAgent("agent_a", "openai", "gpt-5-mini", "probability", True)
        ]
        clients = {"openai": client}
        orch = self._make_orchestrator(clients=clients, panel=panel)
        packet = _make_packet()
        result = await orch.run_panel(packet)

        assert len(result.proposals) == 1
        assert client.call.call_count == 2

    async def test_empty_clients(self) -> None:
        """Panel with no clients returns empty result."""
        orch = self._make_orchestrator()
        packet = _make_packet()
        result = await orch.run_panel(packet)

        assert result.proposals == []
        assert result.total_cost_eur == 0.0

    async def test_conditional_agent_skipped_by_default(self) -> None:
        """Conditional x_signals agent is skipped when no triggers."""
        clients = {
            "openai": _make_mock_client("openai"),
            "xai": _make_mock_client("xai"),
        }
        panel = [
            PanelAgent("agent_a", "openai", "gpt-5-mini", "probability", True),
            PanelAgent("x_signals", "xai", "grok-3-fast", "signals", False),
        ]
        orch = self._make_orchestrator(clients=clients, panel=panel)
        packet = _make_packet()
        result = await orch.run_panel(packet)

        assert "x_signals" in result.skipped_agents
        assert len(result.proposals) == 1

    async def test_conditional_agent_runs_with_high_exposure(self) -> None:
        """x_signals runs when position_exposure_rank <= 3."""
        clients = {
            "openai": _make_mock_client("openai"),
            "xai": _make_mock_client("xai"),
        }
        panel = [
            PanelAgent("agent_a", "openai", "gpt-5-mini", "probability", True),
            PanelAgent("x_signals", "xai", "grok-3-fast", "signals", False),
        ]
        orch = self._make_orchestrator(clients=clients, panel=panel)
        packet = _make_packet()
        result = await orch.run_panel(packet, position_exposure_rank=2)

        assert "x_signals" in result.agents_used
        assert len(result.proposals) == 2


# ---- Tiered model selection --------------------------------------------------


class TestPanelTieredModels:
    """Panel orchestrator tiered model selection."""

    def _make_orchestrator(
        self,
        clients: dict | None = None,
        budget: BudgetTracker | None = None,
        panel: list[PanelAgent] | None = None,
    ) -> PanelOrchestrator:
        if clients is None:
            clients = {}
        if budget is None:
            budget = BudgetTracker(DailyBudget())
        if panel is None:
            panel = [
                PanelAgent("agent_a", "openai", "gpt-5-mini", "probability", True),
            ]
        with patch("src.llm.panel.load_charter", return_value="test charter"):
            return PanelOrchestrator(
                clients=clients,
                budget_tracker=budget,
                policy=_make_policy(),
                charter_path="charter.md",
                default_panel=panel,
            )

    async def test_panel_uses_tiered_model(self) -> None:
        """When Tier A is exhausted, panel uses Tier B model."""
        client = _make_mock_client("openai")
        budget = BudgetTracker(DailyBudget())
        budget.record_spend("openai", 2.00)  # past Tier A cap

        panel = [
            PanelAgent("agent_a", "openai", "gpt-5-mini", "probability", True),
        ]
        orch = self._make_orchestrator(
            clients={"openai": client}, budget=budget, panel=panel,
        )
        packet = _make_packet()
        result = await orch.run_panel(packet)

        assert len(result.proposals) == 1
        # Verify client.call was invoked with the Tier B model.
        client.call.assert_called_once()
        call_args = client.call.call_args
        assert call_args[0][2] == "gpt-4.1-mini"

    async def test_panel_skips_exhausted_provider(self) -> None:
        """When all tiers exhausted, agent is skipped."""
        client = _make_mock_client("openai")
        budget = BudgetTracker(DailyBudget())
        budget.record_spend("openai", 4.00)  # past all tiers

        panel = [
            PanelAgent("agent_a", "openai", "gpt-5-mini", "probability", True),
        ]
        orch = self._make_orchestrator(
            clients={"openai": client}, budget=budget, panel=panel,
        )
        packet = _make_packet()
        result = await orch.run_panel(packet)

        assert "agent_a" in result.skipped_agents
        assert result.proposals == []

    async def test_panel_logs_tier_switch(self) -> None:
        """Tier switch emits a structured log event."""
        client = _make_mock_client("openai")
        budget = BudgetTracker(DailyBudget())
        budget.record_spend("openai", 2.00)

        panel = [
            PanelAgent("agent_a", "openai", "gpt-5-mini", "probability", True),
        ]
        orch = self._make_orchestrator(
            clients={"openai": client}, budget=budget, panel=panel,
        )
        packet = _make_packet()

        with patch("src.llm.panel.logger") as mock_logger:
            await orch.run_panel(packet)
            # Find the panel_model_tiered log call.
            tiered_calls = [
                c for c in mock_logger.info.call_args_list
                if c[0][0] == "panel_model_tiered"
            ]
            assert len(tiered_calls) == 1
            assert tiered_calls[0][1]["selected_model"] == "gpt-4.1-mini"


# ---- Conditional Grok triggers -----------------------------------------------


class TestConditionalGrok:
    """Conditional x_signals agent trigger conditions."""

    def _make_orchestrator(self) -> PanelOrchestrator:
        budget = BudgetTracker(DailyBudget())
        with patch("src.llm.panel.load_charter", return_value="test charter"):
            return PanelOrchestrator(
                clients={},
                budget_tracker=budget,
                policy=_make_policy(),
                charter_path="charter.md",
            )

    def test_triggered_by_odds_move(self) -> None:
        from src.signals.schemas import MarketSignalBundle, MicrostructureSignal

        orch = self._make_orchestrator()
        agent = PanelAgent("x_signals", "xai", "grok-3-fast", "signals", False)
        bundle = MarketSignalBundle(
            microstructure=MicrostructureSignal(
                odds_move_6h=0.10, odds_move_1h=0.0, odds_move_24h=0.0
            )
        )
        assert orch._should_run_conditional(agent, bundle, None) is True

    def test_triggered_by_weak_evidence(self) -> None:
        from src.signals.schemas import (
            EvidenceFreshnessSignal,
            MarketSignalBundle,
        )

        orch = self._make_orchestrator()
        agent = PanelAgent("x_signals", "xai", "grok-3-fast", "signals", False)
        bundle = MarketSignalBundle(
            evidence_freshness=EvidenceFreshnessSignal(credible_evidence_6h=0)
        )
        assert orch._should_run_conditional(agent, bundle, None) is True

    def test_triggered_by_high_exposure(self) -> None:
        orch = self._make_orchestrator()
        agent = PanelAgent("x_signals", "xai", "grok-3-fast", "signals", False)
        assert orch._should_run_conditional(agent, None, 3) is True

    def test_not_triggered_when_no_signals(self) -> None:
        orch = self._make_orchestrator()
        agent = PanelAgent("x_signals", "xai", "grok-3-fast", "signals", False)
        assert orch._should_run_conditional(agent, None, None) is False

    def test_not_triggered_for_other_agents(self) -> None:
        orch = self._make_orchestrator()
        agent = PanelAgent("other", "openai", "gpt-5-mini", "probability", False)
        assert orch._should_run_conditional(agent, None, 1) is False


# ---- Escalation (new) -------------------------------------------------------


class TestEscalationNew:
    """New determine_escalation logic."""

    def _make_orchestrator(self) -> PanelOrchestrator:
        budget = BudgetTracker(DailyBudget())
        with patch("src.llm.panel.load_charter", return_value="test charter"):
            return PanelOrchestrator(
                clients={},
                budget_tracker=budget,
                policy=_make_policy(),
                charter_path="charter.md",
            )

    def test_rules_ambiguity_quorum_triggers_anthropic(self) -> None:
        orch = self._make_orchestrator()
        proposals = [
            ModelProposal.model_validate(_make_proposal_dict(p_true=0.60)),
        ]
        agent, trigger = orch.determine_escalation(
            proposals, veto_score=2.0, proposed_size_frac=0.02, odds_move=0.0
        )
        assert agent is not None
        assert agent.agent_id == "escalation_anthropic"
        assert trigger == EscalationTrigger.RULES_AMBIGUITY

    def test_partial_ambiguity_triggers_anthropic(self) -> None:
        orch = self._make_orchestrator()
        proposals = [
            ModelProposal.model_validate(_make_proposal_dict(p_true=0.60)),
        ]
        agent, trigger = orch.determine_escalation(
            proposals, veto_score=1.25, proposed_size_frac=0.02, odds_move=0.0
        )
        assert agent is not None
        assert agent.agent_id == "escalation_anthropic"
        assert trigger == EscalationTrigger.RULES_AMBIGUITY

    def test_high_disagreement_triggers_openai(self) -> None:
        orch = self._make_orchestrator()
        proposals = [
            ModelProposal.model_validate(_make_proposal_dict(p_true=0.30)),
            ModelProposal.model_validate(
                _make_proposal_dict(model_id="b", p_true=0.70)
            ),
        ]
        agent, trigger = orch.determine_escalation(
            proposals, veto_score=0.0, proposed_size_frac=0.02, odds_move=0.0
        )
        assert agent is not None
        assert agent.agent_id == "escalation_openai"
        assert trigger == EscalationTrigger.DISAGREEMENT

    def test_high_stakes_triggers_google(self) -> None:
        orch = self._make_orchestrator()
        proposals = [
            ModelProposal.model_validate(_make_proposal_dict(p_true=0.60)),
        ]
        agent, trigger = orch.determine_escalation(
            proposals,
            veto_score=0.0,
            proposed_size_frac=0.06,
            odds_move=0.0,
        )
        assert agent is not None
        assert agent.agent_id == "escalation_google"
        assert trigger == EscalationTrigger.HIGH_STAKES

    def test_high_stakes_by_exposure_rank(self) -> None:
        orch = self._make_orchestrator()
        proposals = [
            ModelProposal.model_validate(_make_proposal_dict(p_true=0.60)),
        ]
        agent, trigger = orch.determine_escalation(
            proposals,
            veto_score=0.0,
            proposed_size_frac=0.02,
            odds_move=0.0,
            position_exposure_rank=2,
        )
        assert agent is not None
        assert trigger == EscalationTrigger.HIGH_STAKES

    def test_fast_odds_move_triggers_openai(self) -> None:
        orch = self._make_orchestrator()
        proposals = [
            ModelProposal.model_validate(_make_proposal_dict(p_true=0.60)),
        ]
        agent, trigger = orch.determine_escalation(
            proposals,
            veto_score=0.0,
            proposed_size_frac=0.02,
            odds_move=0.10,
        )
        assert agent is not None
        assert agent.agent_id == "escalation_openai"
        assert trigger == EscalationTrigger.FAST_ODDS_MOVE

    def test_no_escalation_when_calm(self) -> None:
        orch = self._make_orchestrator()
        proposals = [
            ModelProposal.model_validate(_make_proposal_dict(p_true=0.60)),
        ]
        agent, trigger = orch.determine_escalation(
            proposals,
            veto_score=0.0,
            proposed_size_frac=0.02,
            odds_move=0.01,
        )
        assert agent is None
        assert trigger is None

    def test_empty_proposals_no_escalation(self) -> None:
        orch = self._make_orchestrator()
        agent, trigger = orch.determine_escalation(
            [], veto_score=0.0, proposed_size_frac=0.02, odds_move=0.0
        )
        assert agent is None

    def test_priority_rules_ambiguity_over_disagreement(self) -> None:
        """Rules ambiguity has higher priority than disagreement."""
        orch = self._make_orchestrator()
        proposals = [
            ModelProposal.model_validate(_make_proposal_dict(p_true=0.30)),
            ModelProposal.model_validate(
                _make_proposal_dict(model_id="b", p_true=0.70)
            ),
        ]
        # Both rules ambiguity AND high disagreement present.
        agent, trigger = orch.determine_escalation(
            proposals, veto_score=2.5, proposed_size_frac=0.02, odds_move=0.0
        )
        assert trigger == EscalationTrigger.RULES_AMBIGUITY

    def test_deprecated_should_escalate_returns_none(self) -> None:
        orch = self._make_orchestrator()
        proposals = [
            ModelProposal.model_validate(_make_proposal_dict(p_true=0.30)),
            ModelProposal.model_validate(
                _make_proposal_dict(model_id="b", p_true=0.80)
            ),
        ]
        result = orch.should_escalate(proposals, 0.10, 0.0, 12.0, ModelTier.TIER_1)
        assert result is None


# ---- Model ID = agent_id ---------------------------------------------------


class TestModelIdAgentId:
    """_parse_proposal sets model_id to agent_id."""

    def _make_orchestrator(self) -> PanelOrchestrator:
        budget = BudgetTracker(DailyBudget())
        with patch("src.llm.panel.load_charter", return_value="test charter"):
            return PanelOrchestrator(
                clients={},
                budget_tracker=budget,
                policy=_make_policy(),
                charter_path="charter.md",
            )

    def test_model_id_is_agent_id(self) -> None:
        orch = self._make_orchestrator()
        data = _make_proposal_dict()
        raw = json.dumps(data)
        agent = PanelAgent("rules_lawyer", "anthropic", "claude-sonnet-4-20250514", "rules", True)
        proposal = orch._parse_proposal(raw, agent, "run-1", "m1")
        assert proposal is not None
        assert proposal.model_id == "rules_lawyer"

    def test_model_id_forced_for_escalation(self) -> None:
        orch = self._make_orchestrator()
        data = _make_proposal_dict()
        raw = json.dumps(data)
        agent = PanelAgent(
            "escalation_anthropic",
            "anthropic",
            "claude-sonnet-4-20250514",
            "rules",
            False,
        )
        proposal = orch._parse_proposal(raw, agent, "run-2", "m2")
        assert proposal is not None
        assert proposal.model_id == "escalation_anthropic"

    async def test_full_panel_produces_agent_ids(self) -> None:
        """End-to-end: panel run produces agent_id as model_id."""
        clients = {
            "openai": _make_mock_client("openai"),
            "anthropic": _make_mock_client("anthropic"),
        }
        budget = BudgetTracker(DailyBudget())
        panel = [
            PanelAgent("rules_lawyer", "anthropic", "claude-sonnet-4-20250514", "rules", True),
            PanelAgent("probabilist", "openai", "gpt-5-mini", "probability", True),
        ]
        with patch("src.llm.panel.load_charter", return_value="test charter"):
            orch = PanelOrchestrator(
                clients=clients,
                budget_tracker=budget,
                policy=_make_policy(),
                charter_path="charter.md",
                default_panel=panel,
            )

        packet = _make_packet()
        result = await orch.run_panel(packet)

        model_ids = {p.model_id for p in result.proposals}
        assert model_ids == {"rules_lawyer", "probabilist"}


# ---- Schema stripping -------------------------------------------------------


class TestSchemaStripping:
    """get_proposal_schema excludes metadata fields."""

    def test_metadata_fields_excluded(self) -> None:
        schema_str = get_proposal_schema()
        parsed = json.loads(schema_str)
        props = parsed.get("properties", {})
        assert "model_id" not in props
        assert "run_id" not in props
        assert "market_id" not in props
        assert "ts_utc" not in props

    def test_metadata_fields_not_in_required(self) -> None:
        schema_str = get_proposal_schema()
        parsed = json.loads(schema_str)
        required = parsed.get("required", [])
        assert "model_id" not in required
        assert "run_id" not in required

    def test_business_fields_still_present(self) -> None:
        schema_str = get_proposal_schema()
        parsed = json.loads(schema_str)
        props = parsed.get("properties", {})
        assert "p_true" in props
        assert "confidence" in props
        assert "direction" in props
        assert "rules_ambiguity" in props
        assert "evidence_ambiguity" in props
        assert "thesis" in props

    def test_old_ambiguity_fields_removed(self) -> None:
        """Old ambiguity_score and ambiguity_reason should not be in schema."""
        schema_str = get_proposal_schema()
        parsed = json.loads(schema_str)
        props = parsed.get("properties", {})
        assert "ambiguity_score" not in props
        assert "ambiguity_reason" not in props


# ---- Gemini truncated JSON repair -------------------------------------------


class TestGeminiJsonRepair:
    """GeminiLLMClient._repair_truncated_json handles truncated responses."""

    def test_valid_json_unchanged(self) -> None:
        """Valid JSON passes through untouched."""
        from src.llm.gemini_client import GeminiLLMClient

        original = '{"p_true": 0.5, "thesis": "hello"}'
        assert GeminiLLMClient._repair_truncated_json(original) == original

    def test_unterminated_string(self) -> None:
        """Truncated mid-string gets closed and braces balanced."""
        from src.llm.gemini_client import GeminiLLMClient

        truncated = '{"p_true": 0.5, "thesis": "this is trunc'
        repaired = GeminiLLMClient._repair_truncated_json(truncated)
        parsed = json.loads(repaired)
        assert parsed["p_true"] == 0.5

    def test_missing_closing_brace(self) -> None:
        """Missing closing brace gets added."""
        from src.llm.gemini_client import GeminiLLMClient

        truncated = '{"p_true": 0.5, "confidence": 0.8'
        repaired = GeminiLLMClient._repair_truncated_json(truncated)
        parsed = json.loads(repaired)
        assert parsed["p_true"] == 0.5
        assert parsed["confidence"] == 0.8

    def test_nested_truncation(self) -> None:
        """Truncated inside nested array gets balanced."""
        from src.llm.gemini_client import GeminiLLMClient

        truncated = '{"items": ["a", "b'
        repaired = GeminiLLMClient._repair_truncated_json(truncated)
        parsed = json.loads(repaired)
        assert "items" in parsed
