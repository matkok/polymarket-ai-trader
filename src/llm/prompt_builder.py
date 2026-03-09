"""Prompt construction for the LLM panel."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.llm.schemas import ModelProposal
from src.packets.schemas import Packet

PROMPT_VERSION = "m4.0"

ROLE_PREAMBLES: dict[str, str] = {
    "rules": (
        "You are the Rules Lawyer on a prediction-market trading panel.\n"
        "Your PRIMARY responsibility is analyzing resolution criteria for ambiguity, "
        "edge cases, and enforceability. You should:\n"
        "- Read the resolution rules extremely carefully and flag any ambiguity.\n"
        "- Set rules_ambiguity with high precision — this is your key contribution.\n"
        "- Identify if the resolution source is verifiable and historically reliable.\n"
        "- Your p_true estimate matters, but your rules analysis is your primary job.\n"
        "- When in doubt about resolution, err on the side of higher rules_ambiguity.\n"
        "- Provide detailed reasoning in notes about resolution criteria interpretation.\n\n"
    ),
    "probability": (
        "You are the Market Probabilist on a prediction-market trading panel.\n"
        "Your PRIMARY responsibility is producing a well-calibrated probability estimate. "
        "You should:\n"
        "- Focus on p_true calibration above all else — be precise and honest.\n"
        "- Reason about base rates, reference classes, and comparable historical events.\n"
        "- Set confidence to reflect your actual epistemic uncertainty, not conviction.\n"
        "- Be especially careful distinguishing 60/40 from 70/30 from 80/20 scenarios.\n"
        "- Your thesis should explain the probabilistic reasoning, not just assert a number.\n\n"
    ),
    "sanity": (
        "You are the Sanity Checker on a prediction-market trading panel.\n"
        "Your PRIMARY responsibility is providing an independent, contrarian perspective. "
        "You should:\n"
        "- Actively look for reasons the consensus might be wrong.\n"
        "- Red-team the thesis: what if the evidence is misleading or incomplete?\n"
        "- Consider alternative scenarios that the other analysts might overlook.\n"
        "- If you genuinely agree, explain why alternatives fail.\n"
        "- Your p_true may differ from others — that is your purpose. Do not anchor.\n\n"
    ),
    "signals": (
        "You are the X Signals Analyst on a prediction-market trading panel.\n"
        "Your PRIMARY responsibility is interpreting social signals and narrative tracking. "
        "You should:\n"
        "- Focus on social sentiment, viral narratives, and real-time information flow.\n"
        "- Assess whether social activity represents genuine signal or noise.\n"
        "- Consider information asymmetry: is social discussion ahead of or behind the market?\n"
        "- Flag when social sentiment diverges sharply from market pricing.\n"
        "- Be skeptical of social hype but don't dismiss genuine information cascades.\n\n"
    ),
    "arbiter": (
        "You are the Arbiter on a prediction-market trading panel.\n"
        "You have been called because the default panel could not reach consensus. "
        "You should:\n"
        "- Weigh ALL evidence and arguments carefully.\n"
        "- Provide your independent probability estimate, not a compromise average.\n"
        "- Explicitly address the source of disagreement in your notes.\n"
        "- Your thesis should explain why one side is more credible.\n\n"
    ),
}


def load_charter(path: str) -> str:
    """Read the trading charter from *path*."""
    return Path(path).read_text(encoding="utf-8")


def charter_version_hash(text: str) -> str:
    """SHA-256 hex digest of the charter text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_system_prompt(charter: str, proposal_schema: str, role: str = "probability") -> str:
    """Assemble the system prompt from charter, JSON schema, and agent role."""
    preamble = ROLE_PREAMBLES.get(role, ROLE_PREAMBLES["probability"])
    return (
        preamble
        + "## Charter\n\n"
        f"{charter}\n\n"
        "## Required Output Format\n\n"
        "Respond with ONLY a valid JSON object matching this schema exactly — "
        "no markdown fences, no commentary:\n\n"
        f"```json\n{proposal_schema}\n```\n\n"
        "Rules:\n"
        "- p_true: your probability estimate that the market resolves YES (0-1)\n"
        "- confidence: your confidence in this estimate (0-1)\n"
        "- direction: BUY_YES if p_true > market mid, else BUY_NO\n"
        "- rules_ambiguity: how ambiguous the resolution criteria are (0-1)\n"
        "- evidence_ambiguity: how insufficient the available evidence is (0-1)\n"
        "- All float fields must be numbers, not strings\n"
        "- evidence list can be empty if no citations available\n"
    )


def get_proposal_schema() -> str:
    """Return the ModelProposal JSON schema as a formatted string.

    Metadata fields injected by the orchestrator (model_id, run_id,
    market_id, ts_utc) are stripped so LLMs never see or fill them.
    """
    schema = ModelProposal.model_json_schema()
    metadata_keys = {"model_id", "run_id", "market_id", "ts_utc"}
    if "properties" in schema:
        for key in metadata_keys:
            schema["properties"].pop(key, None)
    if "required" in schema:
        schema["required"] = [r for r in schema["required"] if r not in metadata_keys]
    return json.dumps(schema, indent=2)


def build_user_prompt(packet: Packet) -> str:
    """Serialize the packet as the user prompt."""
    return (
        "Analyze this market and provide your trading recommendation:\n\n"
        + packet.model_dump_json(indent=2)
    )
