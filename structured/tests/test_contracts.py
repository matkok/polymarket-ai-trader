"""Tests for src.contracts — contract parsing and market classification."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.contracts.base import ContractSpec, ParseResult
from src.contracts.crypto import CryptoContractSpec, CryptoParser
from src.contracts.earnings import EarningsContractSpec, EarningsParser
from src.contracts.macro import MacroContractSpec, MacroParser
from src.contracts.registry import ParserRegistry, classify_markets_batch
from src.contracts.weather import WeatherContractSpec, WeatherParser
from src.db.models import Market


# ---- Base -------------------------------------------------------------------


class TestContractSpec:
    """ContractSpec and ParseResult basics."""

    def test_to_json_returns_correct_dict(self) -> None:
        spec = ContractSpec(category="weather", raw_fields={"metric": "temp"})
        j = spec.to_json()
        assert j == {"category": "weather", "metric": "temp"}

    def test_parse_result_matched_carries_spec_and_confidence(self) -> None:
        spec = ContractSpec(category="weather")
        r = ParseResult(matched=True, category="weather", spec=spec, confidence=0.9)
        assert r.matched is True
        assert r.spec is spec
        assert r.confidence == 0.9

    def test_parse_result_unmatched_carries_reject_reason(self) -> None:
        r = ParseResult(matched=False, reject_reason="no_parser_matched")
        assert r.matched is False
        assert r.reject_reason == "no_parser_matched"
        assert r.spec is None


# ---- WeatherParser.can_parse ------------------------------------------------


class TestWeatherCanParse:
    """WeatherParser.can_parse keyword checks."""

    def test_temperature_keyword_matches(self) -> None:
        p = WeatherParser()
        assert p.can_parse("Will the temperature exceed 90°F?", None) is True

    def test_rain_keyword_matches(self) -> None:
        p = WeatherParser()
        assert p.can_parse("Will there be rain tomorrow?", None) is True

    def test_no_weather_keywords_returns_false(self) -> None:
        p = WeatherParser()
        assert p.can_parse("Will CPI exceed 3%?", None) is False

    def test_keyword_in_rules_text_matches(self) -> None:
        p = WeatherParser()
        assert p.can_parse(
            "Will this happen?",
            "Resolves based on temperature data.",
        ) is True


# ---- WeatherParser.parse — temperature --------------------------------------


class TestWeatherParseTemperature:
    """WeatherParser temperature pattern extraction."""

    def test_high_temp_exceed(self) -> None:
        p = WeatherParser()
        r = p.parse(
            "Will the high temperature in Dallas exceed 100°F on June 15?",
            None,
        )
        assert r.matched is True
        assert r.confidence >= 0.85
        assert isinstance(r.spec, WeatherContractSpec)
        assert r.spec.metric == "temperature_high"
        assert r.spec.location == "Dallas"
        assert r.spec.threshold == 100.0
        assert r.spec.threshold_unit == "F"
        assert r.spec.comparison == "above"

    def test_low_temp_below(self) -> None:
        p = WeatherParser()
        r = p.parse(
            "Will the low temperature in Minneapolis fall below 0°F on January 20?",
            None,
        )
        assert r.matched is True
        assert isinstance(r.spec, WeatherContractSpec)
        assert r.spec.metric == "temperature_low"
        assert r.spec.comparison == "below"
        assert r.spec.threshold == 0.0

    def test_celsius_unit(self) -> None:
        p = WeatherParser()
        r = p.parse(
            "Will the temperature in London exceed 35°C on July 15?",
            None,
        )
        assert r.matched is True
        assert isinstance(r.spec, WeatherContractSpec)
        assert r.spec.threshold_unit == "C"
        assert r.spec.threshold == 35.0

    def test_negative_threshold(self) -> None:
        p = WeatherParser()
        r = p.parse(
            "Will the low temperature in Anchorage fall below -20°F?",
            None,
        )
        assert r.matched is True
        assert isinstance(r.spec, WeatherContractSpec)
        assert r.spec.threshold == -20.0


# ---- WeatherParser.parse — precipitation ------------------------------------


class TestWeatherParsePrecipitation:
    """WeatherParser precipitation pattern extraction."""

    def test_inches_of_rain(self) -> None:
        p = WeatherParser()
        r = p.parse(
            "Will New York receive more than 2 inches of rain in January 2026?",
            None,
        )
        assert r.matched is True
        assert r.confidence >= 0.80
        assert isinstance(r.spec, WeatherContractSpec)
        assert r.spec.metric == "precipitation"
        assert r.spec.location == "New York"
        assert r.spec.threshold == 2.0
        assert r.spec.comparison == "above"

    def test_inches_of_snow(self) -> None:
        p = WeatherParser()
        r = p.parse(
            "Will Denver receive more than 10 inches of snow in December 2025?",
            None,
        )
        assert r.matched is True
        assert isinstance(r.spec, WeatherContractSpec)
        assert r.spec.metric == "snowfall"

    def test_less_than_comparison(self) -> None:
        p = WeatherParser()
        r = p.parse(
            "Will Seattle receive less than 3 inches of rain in March 2026?",
            None,
        )
        assert r.matched is True
        assert isinstance(r.spec, WeatherContractSpec)
        assert r.spec.comparison == "below"


# ---- WeatherParser.parse — hurricane ----------------------------------------


class TestWeatherParseHurricane:
    """WeatherParser hurricane pattern extraction."""

    def test_category_hurricane_with_threshold(self) -> None:
        p = WeatherParser()
        r = p.parse(
            "Will there be a Category 3+ hurricane in the Atlantic in 2026?",
            None,
        )
        assert r.matched is True
        assert r.confidence >= 0.80
        assert isinstance(r.spec, WeatherContractSpec)
        assert r.spec.metric == "hurricane"
        assert r.spec.threshold == 3.0
        assert r.spec.threshold_unit == "category"

    def test_landfall_with_location(self) -> None:
        p = WeatherParser()
        r = p.parse(
            "Will a hurricane making landfall in Florida in 2026?",
            None,
        )
        assert r.matched is True
        assert isinstance(r.spec, WeatherContractSpec)
        assert "Florida" in (r.spec.location or "")


# ---- WeatherParser.parse — snow occurrence -----------------------------------


class TestWeatherParseSnow:
    """WeatherParser snow occurrence."""

    def test_will_it_snow(self) -> None:
        p = WeatherParser()
        r = p.parse("Will it snow in Chicago on Christmas Day?", None)
        assert r.matched is True
        assert isinstance(r.spec, WeatherContractSpec)
        assert r.spec.metric == "snow_occurrence"
        assert r.spec.location == "Chicago"
        assert r.confidence == 0.80


# ---- WeatherParser edge cases -----------------------------------------------


class TestWeatherEdgeCases:
    """WeatherParser edge cases."""

    def test_keyword_only_no_structure_rejected(self) -> None:
        """Keyword present but no specific pattern → rejected."""
        p = WeatherParser()
        r = p.parse("What weather will tomorrow bring?", None)
        assert r.matched is False
        assert r.reject_reason == "keyword_only_no_structure"

    def test_location_with_comma(self) -> None:
        p = WeatherParser()
        r = p.parse(
            "Will the high temperature in Portland, Oregon exceed 95°F on August 1?",
            None,
        )
        assert r.matched is True
        assert isinstance(r.spec, WeatherContractSpec)
        assert "Portland" in r.spec.location

    def test_to_json_round_trip(self) -> None:
        spec = WeatherContractSpec(
            category="weather",
            metric="temperature_high",
            location="Dallas",
            threshold=100.0,
            threshold_unit="F",
            comparison="above",
        )
        j = spec.to_json()
        assert j["category"] == "weather"
        assert j["metric"] == "temperature_high"
        assert j["location"] == "Dallas"
        assert j["threshold"] == 100.0


# ---- MacroParser.can_parse --------------------------------------------------


class TestMacroCanParse:
    """MacroParser.can_parse keyword checks."""

    def test_cpi_keyword(self) -> None:
        p = MacroParser()
        assert p.can_parse("Will CPI exceed 3%?", None) is True

    def test_fed_keyword(self) -> None:
        p = MacroParser()
        assert p.can_parse("Will the Fed raise rates?", None) is True

    def test_no_macro_keywords_returns_false(self) -> None:
        p = MacroParser()
        assert p.can_parse("Will it rain tomorrow?", None) is False

    def test_keyword_in_rules_text(self) -> None:
        p = MacroParser()
        assert p.can_parse(
            "Will this number be high?",
            "Resolves based on CPI data.",
        ) is True


# ---- MacroParser.parse — indicator threshold ---------------------------------


class TestMacroParseIndicator:
    """MacroParser indicator threshold patterns."""

    def test_cpi_exceed(self) -> None:
        p = MacroParser()
        r = p.parse("Will CPI exceed 3.0% in January 2026?", None)
        assert r.matched is True
        assert r.confidence >= 0.85
        assert isinstance(r.spec, MacroContractSpec)
        assert r.spec.indicator == "cpi"
        assert r.spec.threshold == 3.0
        assert r.spec.comparison == "above"
        assert "January 2026" in r.spec.release_period

    def test_unemployment_below(self) -> None:
        p = MacroParser()
        r = p.parse(
            "Will unemployment be below 4.0% in March 2026?",
            None,
        )
        assert r.matched is True
        assert isinstance(r.spec, MacroContractSpec)
        assert r.spec.indicator == "unemployment"
        assert r.spec.comparison == "below"
        assert r.spec.threshold == 4.0

    def test_core_cpi(self) -> None:
        p = MacroParser()
        r = p.parse("Will core CPI exceed 3.5% in February 2026?", None)
        assert r.matched is True
        assert isinstance(r.spec, MacroContractSpec)
        assert r.spec.indicator == "core_cpi"


# ---- MacroParser.parse — payrolls -------------------------------------------


class TestMacroParsePayrolls:
    """MacroParser nonfarm payrolls pattern."""

    def test_nonfarm_payrolls_exceed(self) -> None:
        p = MacroParser()
        r = p.parse(
            "Will nonfarm payrolls exceed 200K in February 2026?",
            None,
        )
        assert r.matched is True
        assert r.confidence >= 0.85
        assert isinstance(r.spec, MacroContractSpec)
        assert r.spec.indicator == "nonfarm_payrolls"
        assert r.spec.threshold == 200.0
        assert r.spec.threshold_unit == "K"


# ---- MacroParser.parse — fed rate -------------------------------------------


class TestMacroParseFed:
    """MacroParser Fed rate decision patterns."""

    def test_raise_rates(self) -> None:
        p = MacroParser()
        r = p.parse("Will the Fed raise rates in March 2026?", None)
        assert r.matched is True
        assert r.confidence >= 0.80
        assert isinstance(r.spec, MacroContractSpec)
        assert r.spec.indicator == "fed_rate"
        assert r.spec.comparison == "raise"

    def test_cut_by_bps(self) -> None:
        p = MacroParser()
        r = p.parse(
            "Will the Fed cut rates by 25 bps in June 2026?",
            None,
        )
        assert r.matched is True
        assert isinstance(r.spec, MacroContractSpec)
        assert r.spec.indicator == "fed_rate"
        assert r.spec.comparison == "cut"
        assert r.spec.threshold == 25.0
        assert r.spec.threshold_unit == "bps"


# ---- MacroParser edge cases -------------------------------------------------


class TestMacroEdgeCases:
    """MacroParser edge cases."""

    def test_gdp_contraction(self) -> None:
        p = MacroParser()
        r = p.parse("Will GDP growth contract in Q1 2026?", None)
        assert r.matched is True
        assert isinstance(r.spec, MacroContractSpec)
        assert r.spec.indicator == "gdp"

    def test_to_json_round_trip(self) -> None:
        spec = MacroContractSpec(
            category="macro",
            indicator="cpi",
            threshold=3.0,
            threshold_unit="%",
            comparison="above",
            release_period="January 2026",
        )
        j = spec.to_json()
        assert j["category"] == "macro"
        assert j["indicator"] == "cpi"
        assert j["threshold"] == 3.0
        assert j["release_period"] == "January 2026"


# ---- ParserRegistry ---------------------------------------------------------


class TestParserRegistry:
    """ParserRegistry dispatch tests."""

    def test_first_match_wins(self) -> None:
        registry = ParserRegistry()
        result = registry.classify(
            "Will the high temperature in Dallas exceed 100°F?",
            None,
        )
        assert result.matched is True
        assert result.category == "weather"

    def test_no_match_returns_rejected(self) -> None:
        registry = ParserRegistry()
        result = registry.classify(
            "Will the next president be a Democrat?",
            None,
        )
        assert result.matched is False
        assert result.reject_reason == "no_parser_matched"

    def test_custom_parser_order(self) -> None:
        """Macro-first registry classifies CPI before weather would."""
        registry = ParserRegistry(parsers=[MacroParser(), WeatherParser()])
        result = registry.classify("Will CPI exceed 3%?", None)
        assert result.matched is True
        assert result.category == "macro"


# ---- classify_markets_batch -------------------------------------------------


def _mock_market(market_id: str, question: str, rules_text: str | None = None) -> MagicMock:
    m = MagicMock(spec=Market)
    m.market_id = market_id
    m.question = question
    m.rules_text = rules_text
    return m


class TestClassifyMarketsBatch:
    """classify_markets_batch integration tests."""

    async def test_classifies_weather_market(self) -> None:
        repo = AsyncMock()
        repo.get_unparsed_markets.return_value = [
            _mock_market("mkt-1", "Will the high temperature in Dallas exceed 100°F?"),
        ]
        registry = ParserRegistry()

        count = await classify_markets_batch(repo, registry)

        assert count == 1
        repo.upsert_category_assignment.assert_called_once()
        call_data = repo.upsert_category_assignment.call_args[0][0]
        assert call_data["parse_status"] == "parsed"
        assert call_data["category"] == "weather"
        assert call_data["contract_spec_json"]["metric"] == "temperature_high"

    async def test_rejects_unmatched_market(self) -> None:
        repo = AsyncMock()
        repo.get_unparsed_markets.return_value = [
            _mock_market("mkt-2", "Will the next president be a Democrat?"),
        ]
        registry = ParserRegistry()

        count = await classify_markets_batch(repo, registry)

        assert count == 0
        repo.upsert_category_assignment.assert_called_once()
        call_data = repo.upsert_category_assignment.call_args[0][0]
        assert call_data["parse_status"] == "rejected"
        assert call_data["reject_reason"] == "no_parser_matched"

    async def test_empty_unparsed_returns_zero(self) -> None:
        repo = AsyncMock()
        repo.get_unparsed_markets.return_value = []
        registry = ParserRegistry()

        count = await classify_markets_batch(repo, registry)

        assert count == 0
        repo.upsert_category_assignment.assert_not_called()


# ---- Weather precision regression tests ------------------------------------


class TestWeatherFalsePositiveRegression:
    """Regression tests for known false positives.

    These markets were incorrectly classified as weather in production.
    Every test here MUST return matched=False.
    """

    def _assert_not_weather(self, question: str, rules: str | None = None) -> None:
        p = WeatherParser()
        r = p.parse(question, rules)
        assert r.matched is False, (
            f"False positive: {question!r} matched as weather "
            f"(confidence={r.confidence}, reason={r.reject_reason})"
        )

    # -- Substring false positives (Ukraine contains "rain") --

    def test_ukraine_ceasefire_1(self) -> None:
        self._assert_not_weather("Russia x Ukraine ceasefire by end of 2026?")

    def test_ukraine_ceasefire_2(self) -> None:
        self._assert_not_weather("Russia x Ukraine ceasefire by March 31, 2026?")

    def test_ukraine_ceasefire_3(self) -> None:
        self._assert_not_weather("Russia-Ukraine Ceasefire before GTA VI?")

    def test_zelenskyy(self) -> None:
        self._assert_not_weather("Zelenskyy out as Ukraine president by end of 2026?")

    # -- Sports false positives (Carolina Hurricanes, team names) --

    def test_carolina_hurricanes_stanley_cup(self) -> None:
        self._assert_not_weather(
            "Will the Carolina Hurricanes win the 2026 NHL Stanley Cup?"
        )

    def test_miami_heat_nba(self) -> None:
        self._assert_not_weather(
            "Will the Miami Heat win the 2026 NBA Championship?"
        )

    def test_ukraine_fifa(self) -> None:
        self._assert_not_weather(
            "Will Ukraine qualify for the 2026 FIFA World Cup?"
        )

    # -- Other non-weather markets with weather-adjacent keywords --

    def test_trump_impeach(self) -> None:
        """'impeach' is not weather."""
        p = WeatherParser()
        assert p.can_parse("Trump impeached by end of 2026?", None) is False

    def test_election_question(self) -> None:
        p = WeatherParser()
        assert p.can_parse("Will the presidential election be contested?", None) is False

    def test_crypto_market(self) -> None:
        p = WeatherParser()
        assert p.can_parse("Will Bitcoin exceed $100K by end of 2026?", None) is False


class TestWeatherTruePositiveRegression:
    """Regression tests for known true positives.

    These are real weather markets that MUST match.
    """

    def _assert_weather(self, question: str, rules: str | None = None) -> None:
        p = WeatherParser()
        r = p.parse(question, rules)
        assert r.matched is True, (
            f"False negative: {question!r} should match as weather "
            f"(reject_reason={r.reject_reason})"
        )
        assert r.confidence >= 0.80

    def test_temperature_dallas(self) -> None:
        self._assert_weather(
            "Will the high temperature in Dallas exceed 100°F on June 15?"
        )

    def test_temperature_minneapolis(self) -> None:
        self._assert_weather(
            "Will the low temperature in Minneapolis fall below 0°F on January 20?"
        )

    def test_precipitation_nyc(self) -> None:
        self._assert_weather(
            "Will New York receive more than 2 inches of rain in January 2026?"
        )

    def test_hurricane_atlantic(self) -> None:
        self._assert_weather(
            "Will there be a Category 3+ hurricane in the Atlantic in 2026?"
        )

    def test_hurricane_landfall_florida(self) -> None:
        self._assert_weather(
            "Will a hurricane making landfall in Florida in 2026?"
        )

    def test_snow_chicago(self) -> None:
        self._assert_weather(
            "Will it snow in Chicago on Christmas Day?"
        )

    def test_snowfall_denver(self) -> None:
        self._assert_weather(
            "Will Denver receive more than 10 inches of snow in December 2025?"
        )

    def test_temperature_celsius(self) -> None:
        self._assert_weather(
            "Will the temperature in London exceed 35°C on July 15?"
        )


class TestWeatherCanParseWordBoundary:
    """Verify word-boundary matching prevents substring false positives."""

    def test_rain_matches_as_word(self) -> None:
        p = WeatherParser()
        assert p.can_parse("Will there be rain tomorrow?", None) is True

    def test_ukraine_does_not_match_rain(self) -> None:
        p = WeatherParser()
        assert p.can_parse("Russia-Ukraine ceasefire?", None) is False

    def test_window_does_not_match_wind(self) -> None:
        p = WeatherParser()
        assert p.can_parse("The window of opportunity closes?", None) is False

    def test_brain_does_not_match_rain(self) -> None:
        p = WeatherParser()
        assert p.can_parse("Will brain-computer interfaces succeed?", None) is False

    def test_snowball_does_not_match_snow(self) -> None:
        """'snowball' should not trigger snow keyword."""
        p = WeatherParser()
        assert p.can_parse("Will the debt snowball by 2027?", None) is False

    def test_wind_matches_as_word(self) -> None:
        p = WeatherParser()
        assert p.can_parse("Will wind speeds exceed 50 mph?", None) is True

    def test_hurricane_matches_as_word(self) -> None:
        p = WeatherParser()
        assert p.can_parse("Will a hurricane hit Florida?", None) is True

    def test_hurricanes_plural_still_matches(self) -> None:
        """'hurricanes' should still pass can_parse (the \b is after 'hurricane')."""
        p = WeatherParser()
        # "hurricanes" starts with "hurricane" + "s" — \bhurricane\b won't match
        # because 's' is a word char. This is correct: "hurricanes" alone
        # (without tropical context) should not match.
        assert p.can_parse("How many hurricanes this year?", None) is False


# ---- MacroParser precision regression tests --------------------------------


class TestMacroFalsePositiveRegression:
    """Regression tests for macro false positives.

    Markets that MUST NOT match as macro.
    """

    def _assert_not_macro(self, question: str, rules: str | None = None) -> None:
        p = MacroParser()
        r = p.parse(question, rules)
        assert r.matched is False, (
            f"False positive: {question!r} matched as macro "
            f"(confidence={r.confidence}, reason={r.reject_reason})"
        )

    # -- Crypto/DeFi false positives --

    def test_defi_inflation_rate(self) -> None:
        self._assert_not_macro(
            "Will the DeFi token inflation rate exceed 5% this year?"
        )

    def test_yield_farming_rates(self) -> None:
        self._assert_not_macro(
            "Will yield farming rates on Aave exceed 10% APY?"
        )

    def test_staking_inflation(self) -> None:
        self._assert_not_macro(
            "Will staking inflation on Ethereum exceed 3% in 2026?"
        )

    def test_dao_vote_on_rate(self) -> None:
        self._assert_not_macro(
            "Will the DAO vote to cut the protocol rate below 2%?"
        )

    def test_token_inflation_pce(self) -> None:
        self._assert_not_macro(
            "Will token inflation measured by PCE methodology exceed 4%?",
            "This resolves based on DeFi protocol metrics using PCE-like calculation.",
        )

    def test_smart_contract_gdp(self) -> None:
        self._assert_not_macro(
            "Will smart contract GDP tracking token exceed $1B?"
        )

    # -- Keyword-only false positives (no structure) --

    def test_keyword_only_inflation(self) -> None:
        """'inflation' keyword but no pattern → rejected."""
        p = MacroParser()
        r = p.parse("What will happen to inflation next year?", None)
        assert r.matched is False
        assert r.reject_reason == "keyword_only_no_structure"

    def test_keyword_only_the_fed(self) -> None:
        p = MacroParser()
        r = p.parse("What does the Fed think about the economy?", None)
        assert r.matched is False
        assert r.reject_reason == "keyword_only_no_structure"

    def test_keyword_only_fomc(self) -> None:
        p = MacroParser()
        r = p.parse("When is the next FOMC meeting?", None)
        assert r.matched is False
        assert r.reject_reason == "keyword_only_no_structure"


class TestMacroCanParseWordBoundary:
    """Verify word-boundary matching prevents substring false positives."""

    def test_cpi_matches_as_word(self) -> None:
        p = MacroParser()
        assert p.can_parse("Will CPI exceed 3%?", None) is True

    def test_gdp_matches_as_word(self) -> None:
        p = MacroParser()
        assert p.can_parse("Will GDP growth be positive?", None) is True

    def test_gdp_in_compound_no_match(self) -> None:
        """'gdp' embedded in a non-word should not match."""
        p = MacroParser()
        assert p.can_parse("Will dogdp token price rise?", None) is False

    def test_ppi_matches_as_word(self) -> None:
        p = MacroParser()
        assert p.can_parse("Will PPI exceed 2%?", None) is True

    def test_fomc_matches_as_word(self) -> None:
        p = MacroParser()
        assert p.can_parse("Will FOMC cut rates?", None) is True

    def test_no_macro_word_returns_false(self) -> None:
        p = MacroParser()
        assert p.can_parse("Will Bitcoin exceed $100K?", None) is False


class TestMacroTruePositiveRegression:
    """Regression tests for macro true positives.

    Real macro markets that MUST match.
    """

    def _assert_macro(self, question: str, rules: str | None = None) -> None:
        p = MacroParser()
        r = p.parse(question, rules)
        assert r.matched is True, (
            f"False negative: {question!r} should match as macro "
            f"(reject_reason={r.reject_reason})"
        )
        assert r.confidence >= 0.80

    def test_cpi_exceed(self) -> None:
        self._assert_macro("Will CPI exceed 3.5% in March 2026?")

    def test_unemployment_below(self) -> None:
        self._assert_macro("Will unemployment be below 4.0% in February 2026?")

    def test_fed_raise_rates(self) -> None:
        self._assert_macro("Will the Fed raise rates in March 2026?")

    def test_fed_cut_by_bps(self) -> None:
        self._assert_macro("Will the Fed cut rates by 25 bps in June 2026?")

    def test_gdp_contraction(self) -> None:
        self._assert_macro("Will GDP growth contract in Q1 2026?")

    def test_nonfarm_payrolls(self) -> None:
        self._assert_macro("Will nonfarm payrolls exceed 200K in February 2026?")

    def test_core_cpi(self) -> None:
        self._assert_macro("Will core CPI exceed 3.5% in February 2026?")

    def test_pce_exceed(self) -> None:
        self._assert_macro("Will PCE exceed 2.5% in January 2026?")

    def test_ppi_above(self) -> None:
        self._assert_macro("Will PPI be above 3.0% in April 2026?")


# ---- CryptoParser.can_parse ------------------------------------------------


class TestCryptoCanParse:
    """CryptoParser.can_parse keyword checks."""

    def test_btc_keyword(self) -> None:
        p = CryptoParser()
        assert p.can_parse("Will BTC exceed $100,000?", None) is True

    def test_bitcoin_keyword(self) -> None:
        p = CryptoParser()
        assert p.can_parse("Will Bitcoin reach $100K?", None) is True

    def test_eth_keyword(self) -> None:
        p = CryptoParser()
        assert p.can_parse("Will ETH be above $5000?", None) is True

    def test_no_crypto_keywords(self) -> None:
        p = CryptoParser()
        assert p.can_parse("Will CPI exceed 3%?", None) is False

    def test_ethnicity_does_not_match_eth(self) -> None:
        p = CryptoParser()
        assert p.can_parse("Will ethnicity data be released?", None) is False


# ---- CryptoParser.parse — price threshold ----------------------------------


class TestCryptoParsePrice:
    """CryptoParser price threshold patterns."""

    def test_btc_exceed(self) -> None:
        p = CryptoParser()
        r = p.parse("Will BTC exceed $100,000 on Coinbase by June 30?", None)
        assert r.matched is True
        assert r.confidence >= 0.85
        assert isinstance(r.spec, CryptoContractSpec)
        assert r.spec.asset == "BTC"
        assert r.spec.threshold == 100000.0
        assert r.spec.comparison == "above"
        assert r.spec.exchange == "coinbase"

    def test_eth_trading_above(self) -> None:
        p = CryptoParser()
        r = p.parse("Will ETH be trading above $5,000 on December 31?", None)
        assert r.matched is True
        assert isinstance(r.spec, CryptoContractSpec)
        assert r.spec.asset == "ETH"
        assert r.spec.threshold == 5000.0
        assert r.spec.comparison == "above"

    def test_sol_below(self) -> None:
        p = CryptoParser()
        r = p.parse("Will SOL fall below $100 by March 2026?", None)
        assert r.matched is True
        assert isinstance(r.spec, CryptoContractSpec)
        assert r.spec.asset == "SOL"
        assert r.spec.comparison == "below"

    def test_bitcoin_no_dollar_sign(self) -> None:
        p = CryptoParser()
        r = p.parse("Will Bitcoin exceed 100000 by end of 2026?", None)
        assert r.matched is True
        assert isinstance(r.spec, CryptoContractSpec)
        assert r.spec.threshold == 100000.0

    def test_btc_1m_suffix(self) -> None:
        """$1m should parse as 1,000,000."""
        p = CryptoParser()
        r = p.parse("Will bitcoin hit $1m before GTA VI?", None)
        assert r.matched is True
        assert isinstance(r.spec, CryptoContractSpec)
        assert r.spec.asset == "BTC"
        assert r.spec.threshold == 1_000_000.0

    def test_btc_100k_suffix(self) -> None:
        """$100K should parse as 100,000."""
        p = CryptoParser()
        r = p.parse("Will Bitcoin exceed $100K by end of 2026?", None)
        assert r.matched is True
        assert isinstance(r.spec, CryptoContractSpec)
        assert r.spec.threshold == 100_000.0

    def test_eth_10k_suffix(self) -> None:
        """$10k should parse as 10,000."""
        p = CryptoParser()
        r = p.parse("Will ETH reach $10k on Coinbase by 2027?", None)
        assert r.matched is True
        assert isinstance(r.spec, CryptoContractSpec)
        assert r.spec.threshold == 10_000.0

    def test_btc_1b_suffix(self) -> None:
        """$1B should parse as 1,000,000,000."""
        p = CryptoParser()
        r = p.parse("Will Bitcoin hit $1B by 2030?", None)
        assert r.matched is True
        assert isinstance(r.spec, CryptoContractSpec)
        assert r.spec.threshold == 1_000_000_000.0

    def test_trading_pattern_with_suffix(self) -> None:
        """Suffix works with the trading above/below pattern too."""
        p = CryptoParser()
        r = p.parse("Will ETH be trading above $10k on December 31?", None)
        assert r.matched is True
        assert isinstance(r.spec, CryptoContractSpec)
        assert r.spec.threshold == 10_000.0
        assert r.spec.comparison == "above"

    def test_btc_150k_real_polymarket(self) -> None:
        """Real Polymarket market: 'Will Bitcoin hit $150k by March 31, 2026?'"""
        p = CryptoParser()
        r = p.parse("Will Bitcoin hit $150k by March 31, 2026?", None)
        assert r.matched is True
        assert isinstance(r.spec, CryptoContractSpec)
        assert r.spec.asset == "BTC"
        assert r.spec.threshold == 150_000.0
        assert r.spec.comparison == "above"


# ---- CryptoParser.parse — all-time high ------------------------------------


class TestCryptoParseATH:
    """CryptoParser all-time high patterns."""

    def test_ethereum_ath(self) -> None:
        p = CryptoParser()
        r = p.parse("Ethereum all time high by March 31, 2026?", None)
        assert r.matched is True
        assert isinstance(r.spec, CryptoContractSpec)
        assert r.spec.asset == "ETH"
        assert r.spec.threshold == 4890.0
        assert r.spec.comparison == "above"
        assert "March 31, 2026" in r.spec.date_description

    def test_solana_ath(self) -> None:
        p = CryptoParser()
        r = p.parse("Solana all time high by June 30, 2026?", None)
        assert r.matched is True
        assert isinstance(r.spec, CryptoContractSpec)
        assert r.spec.asset == "SOL"
        assert r.spec.threshold == 295.0

    def test_bitcoin_ath_with_hit(self) -> None:
        p = CryptoParser()
        r = p.parse("Will Bitcoin hit all-time high by December 31, 2026?", None)
        assert r.matched is True
        assert isinstance(r.spec, CryptoContractSpec)
        assert r.spec.asset == "BTC"
        assert r.spec.threshold == 109_000.0

    def test_eth_new_ath(self) -> None:
        p = CryptoParser()
        r = p.parse("Will Ethereum reach a new all time high by September 30, 2026?", None)
        assert r.matched is True
        assert isinstance(r.spec, CryptoContractSpec)
        assert r.spec.asset == "ETH"

    def test_ath_keyword_triggers_can_parse(self) -> None:
        p = CryptoParser()
        assert p.can_parse("Ethereum all time high by March 31?", None) is True
        assert p.can_parse("Solana all-time high by June?", None) is True


# ---- CryptoParser edge cases -----------------------------------------------


class TestCryptoEdgeCases:
    """CryptoParser edge cases."""

    def test_keyword_only_no_structure(self) -> None:
        p = CryptoParser()
        r = p.parse("What will happen to Bitcoin next?", None)
        assert r.matched is False
        assert r.reject_reason == "keyword_only_no_structure"

    def test_nft_negative_filter(self) -> None:
        p = CryptoParser()
        r = p.parse("Will ETH NFT trading volume exceed $1B?", None)
        assert r.matched is False

    def test_gas_fee_negative_filter(self) -> None:
        p = CryptoParser()
        r = p.parse("Will ETH gas fee exceed 100 gwei?", None)
        assert r.matched is False

    def test_to_json_round_trip(self) -> None:
        spec = CryptoContractSpec(
            category="crypto",
            asset="BTC",
            threshold=100000.0,
            comparison="above",
            exchange="coinbase",
        )
        j = spec.to_json()
        assert j["asset"] == "BTC"
        assert j["threshold"] == 100000.0
        assert j["exchange"] == "coinbase"

    def test_registry_classifies_crypto(self) -> None:
        registry = ParserRegistry()
        result = registry.classify(
            "Will BTC exceed $100,000 on Coinbase by June 30?", None
        )
        assert result.matched is True
        assert result.category == "crypto"


# ---- EarningsParser.can_parse ------------------------------------------------


class TestEarningsCanParse:
    """EarningsParser.can_parse keyword checks."""

    def test_eps_keyword(self) -> None:
        p = EarningsParser()
        assert p.can_parse("Will Apple's EPS exceed $1.50?", None) is True

    def test_revenue_keyword(self) -> None:
        p = EarningsParser()
        assert p.can_parse("Will Tesla revenue beat $25B?", None) is True

    def test_10k_keyword(self) -> None:
        p = EarningsParser()
        assert p.can_parse("Will Apple file its 10-K by March?", None) is True

    def test_earnings_keyword(self) -> None:
        p = EarningsParser()
        assert p.can_parse("Will Apple beat earnings estimates?", None) is True

    def test_no_earnings_keywords(self) -> None:
        p = EarningsParser()
        assert p.can_parse("Will it rain tomorrow?", None) is False

    def test_keyword_in_rules_text(self) -> None:
        p = EarningsParser()
        assert p.can_parse(
            "Will this company do well?",
            "Resolves based on quarterly earnings data.",
        ) is True


# ---- EarningsParser.parse — EPS/Revenue threshold --------------------------


class TestEarningsParseThreshold:
    """EarningsParser EPS/revenue threshold patterns."""

    def test_eps_exceed(self) -> None:
        p = EarningsParser()
        r = p.parse("Will Apple's EPS exceed $1.50 in Q1 2026?", None)
        assert r.matched is True
        assert r.confidence >= 0.85
        assert isinstance(r.spec, EarningsContractSpec)
        assert r.spec.company == "Apple"
        assert r.spec.metric == "eps"
        assert r.spec.threshold == 1.50
        assert r.spec.comparison == "above"
        assert "Q1 2026" in r.spec.fiscal_period

    def test_revenue_beat(self) -> None:
        p = EarningsParser()
        r = p.parse("Will Tesla's revenue beat $25B in Q4 2025?", None)
        assert r.matched is True
        assert isinstance(r.spec, EarningsContractSpec)
        assert r.spec.company == "Tesla"
        assert r.spec.metric == "revenue"
        assert r.spec.threshold == 25.0
        assert r.spec.threshold_unit == "B"

    def test_miss_comparison(self) -> None:
        p = EarningsParser()
        r = p.parse("Will Amazon's EPS miss $1.20 in Q1 2026?", None)
        assert r.matched is True
        assert isinstance(r.spec, EarningsContractSpec)
        assert r.spec.comparison == "below"


# ---- EarningsParser.parse — filing existence --------------------------------


class TestEarningsParseFiling:
    """EarningsParser filing existence patterns."""

    def test_10k_filing(self) -> None:
        p = EarningsParser()
        r = p.parse("Will Apple file its 10-K by March 2026?", None)
        assert r.matched is True
        assert r.confidence >= 0.80
        assert isinstance(r.spec, EarningsContractSpec)
        assert r.spec.filing_type == "10-K"
        assert r.spec.metric == "filing_10k"

    def test_10q_submit(self) -> None:
        p = EarningsParser()
        r = p.parse("Will Tesla submit its 10-Q for Q1 2026?", None)
        assert r.matched is True
        assert isinstance(r.spec, EarningsContractSpec)
        assert r.spec.filing_type == "10-Q"

    def test_8k_release(self) -> None:
        p = EarningsParser()
        r = p.parse("Will NVIDIA release its 8-K by February 2026?", None)
        assert r.matched is True
        assert isinstance(r.spec, EarningsContractSpec)
        assert r.spec.filing_type == "8-K"


# ---- EarningsParser.parse — ticker-based ------------------------------------


class TestEarningsParseTicker:
    """EarningsParser ticker-based patterns."""

    def test_ticker_eps_beat(self) -> None:
        p = EarningsParser()
        r = p.parse("Will $AAPL EPS beat estimates in Q1 2026?", None)
        assert r.matched is True
        assert r.confidence >= 0.80
        assert isinstance(r.spec, EarningsContractSpec)
        assert r.spec.ticker == "AAPL"
        assert r.spec.metric == "eps"
        assert r.spec.comparison == "above"

    def test_ticker_with_threshold(self) -> None:
        p = EarningsParser()
        r = p.parse("Will TSLA revenue exceed $25B in Q4 2025?", None)
        assert r.matched is True
        assert isinstance(r.spec, EarningsContractSpec)
        assert r.spec.ticker == "TSLA"
        assert r.spec.threshold == 25.0
        assert r.spec.threshold_unit == "B"


# ---- EarningsParser edge cases -----------------------------------------------


class TestEarningsEdgeCases:
    """EarningsParser edge cases."""

    def test_keyword_only_no_structure(self) -> None:
        p = EarningsParser()
        r = p.parse("What will happen with quarterly results?", None)
        assert r.matched is False
        assert r.reject_reason == "keyword_only_no_structure"

    def test_crypto_earnings_negative_filter(self) -> None:
        p = EarningsParser()
        r = p.parse("Will crypto earnings exceed $1B in 2026?", None)
        assert r.matched is False

    def test_staking_earnings_negative_filter(self) -> None:
        p = EarningsParser()
        r = p.parse("Will staking earnings on Ethereum exceed 5%?", None)
        assert r.matched is False

    def test_to_json_round_trip(self) -> None:
        spec = EarningsContractSpec(
            category="earnings",
            company="Apple",
            ticker="AAPL",
            metric="eps",
            threshold=1.50,
            comparison="above",
            fiscal_period="Q1 2026",
        )
        j = spec.to_json()
        assert j["company"] == "Apple"
        assert j["ticker"] == "AAPL"
        assert j["metric"] == "eps"
        assert j["threshold"] == 1.50

    def test_registry_classifies_earnings(self) -> None:
        registry = ParserRegistry()
        result = registry.classify(
            "Will Apple's EPS exceed $1.50 in Q1 2026?", None
        )
        assert result.matched is True
        assert result.category == "earnings"
