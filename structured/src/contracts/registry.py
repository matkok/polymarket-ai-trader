"""Parser registry and batch classification entry point."""

from __future__ import annotations

from datetime import datetime, timezone

import structlog

from src.contracts.base import ContractParser, ParseResult
from src.contracts.crypto import CryptoParser
from src.contracts.earnings import EarningsParser
from src.contracts.macro import MacroParser
from src.contracts.weather import WeatherParser
from src.db.repository import Repository

logger = structlog.get_logger(__name__)


class ParserRegistry:
    """Holds an ordered list of parsers and dispatches classification."""

    def __init__(self, parsers: list[ContractParser] | None = None) -> None:
        self.parsers: list[ContractParser] = parsers if parsers is not None else [
            WeatherParser(),
            MacroParser(),
            CryptoParser(),
            EarningsParser(),
        ]

    def classify(self, question: str, rules_text: str | None) -> ParseResult:
        """Try each parser in order; first ``can_parse + parse`` match wins."""
        for parser in self.parsers:
            if parser.can_parse(question, rules_text):
                result = parser.parse(question, rules_text)
                if result.matched:
                    return result
        return ParseResult(matched=False, reject_reason="no_parser_matched")


async def classify_markets_batch(
    repo: Repository,
    registry: ParserRegistry,
) -> int:
    """Classify all unparsed active markets.

    Returns the count of markets that were successfully classified (matched).
    """
    unparsed = await repo.get_unparsed_markets()
    if not unparsed:
        logger.info("classify_markets_batch_noop", unparsed=0)
        return 0

    classified = 0
    for market in unparsed:
        result = registry.classify(market.question, market.rules_text)
        now = datetime.now(timezone.utc)

        if result.matched and result.spec is not None:
            # Find the parser name that produced this result.
            parser_name = ""
            for p in registry.parsers:
                if p.category == result.category:
                    parser_name = p.name
                    break

            await repo.upsert_category_assignment({
                "market_id": market.market_id,
                "category": result.category,
                "parser_name": parser_name,
                "parse_status": "parsed",
                "parse_confidence": result.confidence,
                "contract_spec_json": result.spec.to_json(),
                "created_ts_utc": now,
            })
            classified += 1
        else:
            await repo.upsert_category_assignment({
                "market_id": market.market_id,
                "category": "unknown",
                "parser_name": "",
                "parse_status": "rejected",
                "parse_confidence": 0.0,
                "reject_reason": result.reject_reason,
                "created_ts_utc": now,
            })

    logger.info(
        "classify_markets_batch_done",
        unparsed=len(unparsed),
        classified=classified,
        rejected=len(unparsed) - classified,
    )
    return classified
