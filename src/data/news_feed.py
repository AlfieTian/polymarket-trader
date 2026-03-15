"""
News/Event Signal Data Source + LLM Analysis

Fetches news, uses LLM to analyze sentiment/probability for specific markets,
and converts results to Bayesian signals.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import httpx
import numpy as np
from dotenv import load_dotenv

from src.signals.bayesian_engine import Signal, SignalType

logger = logging.getLogger(__name__)

_env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(_env_path)


class Sentiment(str, Enum):
    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"


SENTIMENT_LIKELIHOODS = {
    Sentiment.VERY_BULLISH: (0.90, 0.10),
    Sentiment.BULLISH: (0.70, 0.30),
    Sentiment.NEUTRAL: (0.50, 0.50),
    Sentiment.BEARISH: (0.30, 0.70),
    Sentiment.VERY_BEARISH: (0.10, 0.90),
}


@dataclass
class NewsItem:
    title: str
    source: str
    sentiment: Sentiment
    confidence: float
    relevance: float
    timestamp: float = field(default_factory=time.time)
    url: str = ""
    keywords: list[str] = field(default_factory=list)


@dataclass
class LLMAnalysis:
    """LLM analysis result for a market."""
    market_id: str
    estimated_probability: float  # LLM's estimate of YES probability
    confidence: float  # how confident the LLM is (0-1)
    reasoning: str
    news_headlines: list[str]
    timestamp: float = field(default_factory=time.time)


class NewsFeed:
    """News signal processor with LLM-powered analysis.

    Uses Brave Search for news + Anthropic/OpenAI for analysis.
    Falls back to basic momentum signals if no API keys available.
    """

    def __init__(self):
        self._items: list[NewsItem] = []
        self._llm_cache: dict[str, LLMAnalysis] = {}
        self._cache_ttl = 300  # 5 min cache per market
        self._http: httpx.AsyncClient | None = None

        # API keys for news + LLM
        self._brave_api_key = os.getenv("BRAVE_API_KEY", "")
        self._anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        self._openai_key = os.getenv("OPENAI_API_KEY", "")

        if self._brave_api_key:
            logger.info("📰 Brave Search API available for news")
        if self._anthropic_key or self._openai_key:
            logger.info("🧠 LLM analysis available")

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        return self._http

    # ─── LLM Signal Generation ────────────────────────────────

    async def get_llm_signals(self, market) -> list[Signal]:
        """Generate Bayesian signals for a market using news + LLM analysis.

        Pipeline:
        1. Search recent news related to the market question
        2. Send news + question to LLM for probability estimation
        3. Convert LLM output to Bayesian signals

        Args:
            market: Market object with .id, .question, .yes_price

        Returns:
            List of Signal objects (empty if no APIs or cached)
        """
        # Check cache
        cached = self._llm_cache.get(market.id)
        if cached and (time.time() - cached.timestamp) < self._cache_ttl:
            return []  # Already applied recently, skip

        # Need at least LLM API
        if not (self._anthropic_key or self._openai_key):
            return []

        try:
            # Step 1: Search for relevant news
            headlines = await self._search_news(market.question)

            # Step 2: LLM analysis
            analysis = await self._analyze_with_llm(market, headlines)
            if analysis is None:
                return []

            self._llm_cache[market.id] = analysis

            # Step 3: Convert to Bayesian signal
            signal = self._analysis_to_signal(analysis, market.yes_price)
            if signal:
                logger.info(
                    f"🧠 LLM for {market.id}: p̂={analysis.estimated_probability:.3f} "
                    f"(confidence={analysis.confidence:.2f}) — {analysis.reasoning[:80]}"
                )
                return [signal]

        except Exception as e:
            logger.warning(f"LLM signal failed for {market.id}: {e}")

        return []

    async def _search_news(self, query: str, count: int = 5) -> list[str]:
        """Search recent news headlines related to the query."""
        if not self._brave_api_key:
            return []

        try:
            client = await self._get_http()
            resp = await client.get(
                "https://api.search.brave.com/res/v1/news/search",
                params={"q": query, "count": count, "freshness": "pd"},
                headers={"X-Subscription-Token": self._brave_api_key},
            )
            resp.raise_for_status()
            data = resp.json()

            headlines = []
            for result in data.get("results", []):
                title = result.get("title", "")
                desc = result.get("description", "")
                headlines.append(f"{title}: {desc[:200]}")

            return headlines

        except Exception as e:
            logger.debug(f"News search failed: {e}")
            return []

    async def _analyze_with_llm(self, market, headlines: list[str]) -> LLMAnalysis | None:
        """Use LLM to estimate probability for a market question."""
        news_context = "\n".join(f"- {h}" for h in headlines) if headlines else "No recent news found."

        prompt = f"""You are a prediction market analyst. Estimate the probability of the following outcome.

MARKET QUESTION: {market.question}
CURRENT MARKET PRICE (YES): {market.yes_price:.2f}

RECENT NEWS:
{news_context}

Based on the market question and available information, provide:
1. Your estimated probability of YES (0.00 to 1.00)
2. Your confidence in this estimate (0.00 to 1.00, where 1.0 = very confident)
3. Brief reasoning (1-2 sentences)

Respond in JSON format ONLY:
{{"probability": 0.XX, "confidence": 0.XX, "reasoning": "..."}}"""

        try:
            # Use OpenAI o4-mini (Anthropic OAuth token not compatible with direct REST API)
            if self._openai_key:
                return await self._call_openai(prompt, market.id, headlines)
            elif self._anthropic_key:
                return await self._call_anthropic(prompt, market.id, headlines)
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")

        return None

    async def _call_anthropic(self, prompt: str, market_id: str, headlines: list[str]) -> LLMAnalysis | None:
        client = await self._get_http()
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self._anthropic_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5-20250414",
                "max_tokens": 256,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["content"][0]["text"]
        return self._parse_llm_response(text, market_id, headlines)

    async def _call_openai(self, prompt: str, market_id: str, headlines: list[str]) -> LLMAnalysis | None:
        client = await self._get_http()
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self._openai_key}",
                "Content-type": "application/json",
            },
            json={
                # o4-mini: 10M tokens/day free, strong reasoning for probability estimation
                "model": "o4-mini",
                "max_completion_tokens": 512,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return self._parse_llm_response(text, market_id, headlines)

    def _parse_llm_response(self, text: str, market_id: str, headlines: list[str]) -> LLMAnalysis | None:
        """Parse LLM JSON response into LLMAnalysis."""
        try:
            # Extract JSON from response (handle markdown wrapping)
            text = text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            result = json.loads(text)

            prob = float(result["probability"])
            conf = float(result["confidence"])
            reasoning = str(result.get("reasoning", ""))

            # Sanity check
            prob = np.clip(prob, 0.01, 0.99)
            conf = np.clip(conf, 0.0, 1.0)

            return LLMAnalysis(
                market_id=market_id,
                estimated_probability=prob,
                confidence=conf,
                reasoning=reasoning,
                news_headlines=headlines,
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse LLM response: {e} — text: {text[:200]}")
            return None

    def _analysis_to_signal(self, analysis: LLMAnalysis, market_price: float) -> Signal | None:
        """Convert LLM analysis to a Bayesian signal.

        Only generate signal if LLM estimate differs meaningfully from market price.
        """
        diff = abs(analysis.estimated_probability - market_price)
        if diff < 0.02:  # LLM agrees with market, no signal
            return None

        # Use LLM probability estimate as likelihood
        return Signal(
            signal_type=SignalType.EXPERT_FORECAST,
            likelihood_yes=analysis.estimated_probability,
            likelihood_no=1 - analysis.estimated_probability,
            confidence=analysis.confidence * 0.6,  # discount LLM confidence
            metadata={
                "source": "llm_analysis",
                "reasoning": analysis.reasoning,
                "n_headlines": len(analysis.news_headlines),
            },
        )

    # ─── Manual Signal Methods (still available) ──────────────

    def add_item(self, item: NewsItem) -> None:
        self._items.append(item)

    def to_signal(self, item: NewsItem) -> Signal:
        ll_yes, ll_no = SENTIMENT_LIKELIHOODS[item.sentiment]
        return Signal(
            signal_type=SignalType.NEWS_SENTIMENT,
            likelihood_yes=ll_yes,
            likelihood_no=ll_no,
            confidence=item.confidence * item.relevance,
            timestamp=item.timestamp,
            metadata={"title": item.title, "source": item.source, "url": item.url},
        )

    def get_signals(self, since_timestamp: float = 0, min_relevance: float = 0.3) -> list[Signal]:
        signals = []
        for item in self._items:
            if item.timestamp >= since_timestamp and item.relevance >= min_relevance:
                signals.append(self.to_signal(item))
        return signals

    def create_momentum_signal(self, price_change_1h: float, volume_change_1h: float) -> Signal:
        ll_yes = 1 / (1 + np.exp(-5 * price_change_1h))
        ll_no = 1 - ll_yes
        confidence = min(1.0, 0.3 * volume_change_1h)
        return Signal(
            signal_type=SignalType.MARKET_MOMENTUM,
            likelihood_yes=float(ll_yes),
            likelihood_no=float(ll_no),
            confidence=confidence,
            metadata={"price_change_1h": price_change_1h, "volume_change_1h": volume_change_1h},
        )

    async def close(self):
        if self._http and not self._http.is_closed:
            await self._http.aclose()
