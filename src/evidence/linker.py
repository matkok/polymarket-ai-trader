"""Evidence linker — matches evidence to markets using semantic similarity."""

from __future__ import annotations

import structlog

from src.db.models import EvidenceItem, Market
from src.evidence.embedder import EvidenceEmbedder
from src.evidence.quality import compute_quality_score

logger = structlog.get_logger(__name__)

# Minimum cosine similarity to consider a match.
DEFAULT_SIMILARITY_THRESHOLD = 0.35


class EvidenceLinker:
    """Links evidence items to markets using semantic similarity.

    When an embedder is provided, uses OpenAI embeddings + cosine similarity.
    Falls back to keyword matching when no embedder is available.
    """

    def __init__(
        self,
        max_per_market: int = 10,
        embedder: EvidenceEmbedder | None = None,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> None:
        self.max_per_market = max_per_market
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold

    async def link(
        self,
        evidence: list[EvidenceItem],
        markets: list[Market],
    ) -> dict[str, list[EvidenceItem]]:
        """Return ``{market_id: [relevant evidence items]}``."""
        if not evidence or not markets:
            return {m.market_id: [] for m in markets}

        if self.embedder is not None:
            try:
                return await self._link_semantic(evidence, markets)
            except Exception:
                logger.exception("semantic_link_fallback")

        return self._link_keywords(evidence, markets)

    async def _link_semantic(
        self,
        evidence: list[EvidenceItem],
        markets: list[Market],
    ) -> dict[str, list[EvidenceItem]]:
        """Link using embedding cosine similarity."""
        # Build text representations.
        market_texts = [m.question for m in markets]
        evidence_texts = [
            f"{item.title}. {item.extracted_text[:500]}" for item in evidence
        ]

        # Embed everything in batches.
        all_texts = market_texts + evidence_texts
        all_embeddings = await self.embedder.embed_texts(all_texts)

        market_embeddings = all_embeddings[: len(markets)]
        evidence_embeddings = all_embeddings[len(markets) :]

        result: dict[str, list[EvidenceItem]] = {}

        for i, market in enumerate(markets):
            m_emb = market_embeddings[i]
            if not m_emb:
                result[market.market_id] = []
                continue

            scored: list[tuple[float, EvidenceItem]] = []
            for j, item in enumerate(evidence):
                e_emb = evidence_embeddings[j]
                if not e_emb:
                    continue
                sim = EvidenceEmbedder.cosine_similarity(m_emb, e_emb)
                if sim >= self.similarity_threshold:
                    quality = compute_quality_score(item)
                    composite = 0.6 * sim + 0.4 * quality
                    scored.append((composite, item))

            scored.sort(key=lambda x: x[0], reverse=True)
            result[market.market_id] = [
                item for _, item in scored[: self.max_per_market]
            ]

        logger.info(
            "semantic_link_complete",
            markets=len(markets),
            evidence=len(evidence),
            total_links=sum(len(v) for v in result.values()),
        )
        return result

    def _link_keywords(
        self,
        evidence: list[EvidenceItem],
        markets: list[Market],
    ) -> dict[str, list[EvidenceItem]]:
        """Fallback: keyword overlap matching."""
        result: dict[str, list[EvidenceItem]] = {}

        for market in markets:
            keywords = self._extract_keywords(market.question)
            if not keywords:
                result[market.market_id] = []
                continue

            scored: list[tuple[float, EvidenceItem]] = []
            for item in evidence:
                score = self._score(item, keywords)
                if score >= 1:
                    quality = compute_quality_score(item)
                    composite = score + quality
                    scored.append((composite, item))

            scored.sort(key=lambda x: x[0], reverse=True)
            result[market.market_id] = [
                item for _, item in scored[: self.max_per_market]
            ]

        return result

    @staticmethod
    def _extract_keywords(question: str) -> set[str]:
        """Extract meaningful keywords from a market question."""
        stop_words = frozenset({
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "is", "it", "be", "as", "was", "were",
            "are", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", "shall", "can",
            "this", "that", "these", "those", "what", "which", "who", "whom",
            "how", "when", "where", "why", "if", "not", "no", "yes", "so",
            "up", "out", "about", "into", "over", "after", "before", "between",
            "under", "again", "then", "once", "here", "there", "all", "any",
            "each", "every", "both", "few", "more", "most", "other", "some",
            "such", "than", "too", "very",
        })
        words = question.lower().split()
        cleaned = {"".join(c for c in w if c.isalnum()) for w in words}
        return {w for w in cleaned if w and len(w) > 2 and w not in stop_words}

    @staticmethod
    def _score(item: EvidenceItem, keywords: set[str]) -> int:
        """Count keyword matches in evidence title + extracted_text."""
        text = f"{item.title} {item.extracted_text}".lower()
        return sum(1 for kw in keywords if kw in text)
