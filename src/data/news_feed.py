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
        self._cache_ttl = 1800  # 30 min cache per market (prediction markets are slow-moving)
        self._http: httpx.AsyncClient | None = None

        # API keys for news + LLM
        self._brave_api_key = os.getenv("BRAVE_API_KEY", "")
        self._anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        self._openai_key = os.getenv("OPENAI_API_KEY", "")

        # Track which analysis version was last applied to the Bayesian engine.
        # Prevents the same cached analysis from being re-applied every cycle,
        # which would cause belief drift and push p̂ to extreme values (0.05/0.95).
        self._applied_cache: dict[str, float] = {}  # market_id -> analysis.timestamp

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
        # Check cache — only apply each analysis ONCE to avoid belief drift.
        # Re-applying the same signal every 15s cycle would push p̂ to 0.05/0.95
        # extremes (signal accumulation bug), causing false large-edge detections.
        cached = self._llm_cache.get(market.id)
        if cached and (time.time() - cached.timestamp) < self._cache_ttl:
            if self._applied_cache.get(market.id) == cached.timestamp:
                # Already applied this analysis in a previous cycle — skip
                return []
            # First time applying this cache entry
            self._applied_cache[market.id] = cached.timestamp
            signal = self._analysis_to_signal(cached, market.yes_price)
            return [signal] if signal else []

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
            self._applied_cache[market.id] = analysis.timestamp  # mark as applied

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

    # ─── RSS sources (free, no API key) ─────────────────────
    RSS_FEEDS = [
        "https://feeds.reuters.com/Reuters/worldNews",
        "https://feeds.reuters.com/reuters/topNews",
        "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://www.aljazeera.com/xml/rss/all.xml",
        "https://feeds.skynews.com/feeds/rss/world.xml",
    ]

    async def _fetch_rss(self, feed_url: str) -> list[str]:
        """Fetch and parse a single RSS feed, return list of title+description strings."""
        try:
            client = await self._get_http()
            resp = await client.get(feed_url, follow_redirects=True,
                                    headers={"User-Agent": "Mozilla/5.0 (compatible; NewsBot/1.0)"})
            resp.raise_for_status()
            import xml.etree.ElementTree as ET
            root = ET.fromstring(resp.text)
            items = []
            for item in root.iter("item"):
                title = (item.findtext("title") or "").strip()
                desc = (item.findtext("description") or "").strip()[:200]
                if title:
                    items.append(f"{title}: {desc}" if desc else title)
            return items[:10]
        except Exception as e:
            logger.debug(f"RSS fetch failed ({feed_url}): {e}")
            return []

    async def _search_news(self, query: str, count: int = 5) -> list[str]:
        """Search recent news headlines related to the query.

        Strategy:
        1. Fetch 3 RSS feeds in parallel (free, unlimited)
        2. Filter by keyword relevance to the query
        3. Fall back to Brave only if no RSS results (and API key exists)
        """
        keywords = {w.lower() for w in query.split() if len(w) > 3}

        # Parallel RSS fetch
        feeds_to_try = self.RSS_FEEDS[:3]  # only 3 feeds to keep latency low
        raw_results = await asyncio.gather(
            *[self._fetch_rss(url) for url in feeds_to_try],
            return_exceptions=True,
        )
        all_items: list[str] = []
        for r in raw_results:
            if isinstance(r, list):
                all_items.extend(r)

        # Score by keyword overlap
        def _score(headline: str) -> int:
            hl = headline.lower()
            return sum(1 for kw in keywords if kw in hl)

        relevant = sorted(all_items, key=_score, reverse=True)
        top = [h for h in relevant if _score(h) > 0][:count]

        if top:
            logger.debug(f"📰 RSS: {len(top)} relevant headlines for query")
            return top

        # Fallback 2: SearXNG (self-hosted, no quota) — try before Brave
        try:
            client = await self._get_http()
            resp = await client.get(
                "http://192.168.55.10:2048/search",
                params={"q": query, "format": "json", "language": "en"},
                timeout=5.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                headlines = []
                for result in data.get("results", [])[:count]:
                    title = result.get("title", "")
                    content = result.get("content", "")[:200]
                    headlines.append(f"{title}: {content}" if content else title)
                if headlines:
                    logger.debug(f"📰 SearXNG: {len(headlines)} headlines for query")
                    return headlines
        except Exception as e:
            logger.debug(f"SearXNG fallback failed: {e}")

        # Fallback 3: Brave (only if key present, RSS came up empty, and not over quota)
        # Note: Brave free tier is limited — skip if we already know it's 402'ing
        if self._brave_api_key and not getattr(self, "_brave_quota_exceeded", False):
            try:
                client = await self._get_http()
                resp = await client.get(
                    "https://api.search.brave.com/res/v1/news/search",
                    params={"q": query, "count": count, "freshness": "pd"},
                    headers={"X-Subscription-Token": self._brave_api_key},
                )
                if resp.status_code == 402:
                    logger.warning("📰 Brave Search quota exceeded (402) — disabling for this session")
                    self._brave_quota_exceeded = True
                    return []
                resp.raise_for_status()
                data = resp.json()
                headlines = []
                for result in data.get("results", []):
                    title = result.get("title", "")
                    desc = result.get("description", "")
                    headlines.append(f"{title}: {desc[:200]}")
                logger.debug(f"📰 Brave fallback: {len(headlines)} headlines")
                return headlines
            except Exception as e:
                logger.debug(f"Brave fallback failed: {e}")

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
        # o4-mini may return reasoning tokens with empty content - handle gracefully
        text = data["choices"][0]["message"].get("content") or ""
        if not text.strip():
            logger.debug(f"Empty LLM response for {market_id} (reasoning model)")
            return None
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

            # Sanity check — cap extremes so LLM overconfidence can't drive p̂ to 0/1
            prob = np.clip(prob, 0.05, 0.95)
            conf = np.clip(conf, 0.0, 0.80)   # hard cap: LLM is never >80% confident

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

        # Use LLM probability estimate as likelihood.
        # Note: confidence is already conservatively capped at 0.80 in _parse_llm_response.
        # Do NOT apply an additional * 0.6 discount here — that would make the maximum
        # achievable signal confidence = 0.48 < min_confidence threshold of 0.50,
        # making it mathematically IMPOSSIBLE to ever execute a trade (Bug #002).
        return Signal(
            signal_type=SignalType.EXPERT_FORECAST,
            likelihood_yes=analysis.estimated_probability,
            likelihood_no=1 - analysis.estimated_probability,
            confidence=analysis.confidence,  # capped at 0.80 by _parse_llm_response
            metadata={
                "source": "llm_analysis",
                "reasoning": analysis.reasoning,
                "n_headlines": len(analysis.news_headlines),
                "raw_confidence": analysis.confidence,  # for debugging/auditing
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
