"""
News/Event Signal Data Source

Fetches and processes news signals for Bayesian updating.
Pluggable signal sources: RSS, APIs, social media.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

from src.signals.bayesian_engine import Signal, SignalType

logger = logging.getLogger(__name__)


class Sentiment(str, Enum):
    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"


# Sentiment → likelihood mapping for YES/NO outcomes
SENTIMENT_LIKELIHOODS = {
    Sentiment.VERY_BULLISH: (0.90, 0.10),
    Sentiment.BULLISH: (0.70, 0.30),
    Sentiment.NEUTRAL: (0.50, 0.50),
    Sentiment.BEARISH: (0.30, 0.70),
    Sentiment.VERY_BEARISH: (0.10, 0.90),
}


@dataclass
class NewsItem:
    """A single news item / event."""

    title: str
    source: str
    sentiment: Sentiment
    confidence: float  # 0-1
    relevance: float  # 0-1 relevance to a specific market
    timestamp: float = field(default_factory=time.time)
    url: str = ""
    keywords: list[str] = field(default_factory=list)


class NewsFeed:
    """News signal processor that converts news items to Bayesian signals."""

    def __init__(self):
        self._items: list[NewsItem] = []

    def add_item(self, item: NewsItem) -> None:
        """Add a news item to the feed."""
        self._items.append(item)
        logger.info(f"News: [{item.source}] {item.title} — {item.sentiment.value}")

    def to_signal(self, item: NewsItem) -> Signal:
        """Convert a news item to a Bayesian signal.

        Maps sentiment to likelihood ratios with confidence weighting.
        """
        ll_yes, ll_no = SENTIMENT_LIKELIHOODS[item.sentiment]

        return Signal(
            signal_type=SignalType.NEWS_SENTIMENT,
            likelihood_yes=ll_yes,
            likelihood_no=ll_no,
            confidence=item.confidence * item.relevance,
            timestamp=item.timestamp,
            metadata={
                "title": item.title,
                "source": item.source,
                "url": item.url,
            },
        )

    def get_signals(
        self,
        since_timestamp: float = 0,
        min_relevance: float = 0.3,
    ) -> list[Signal]:
        """Get all news items as Bayesian signals.

        Args:
            since_timestamp: Only items after this time
            min_relevance: Minimum relevance score

        Returns:
            List of Signal objects for Bayesian engine
        """
        signals = []
        for item in self._items:
            if item.timestamp >= since_timestamp and item.relevance >= min_relevance:
                signals.append(self.to_signal(item))
        return signals

    def create_base_rate_signal(
        self,
        historical_frequency: float,
        sample_size: int = 100,
    ) -> Signal:
        """Create a historical base rate signal.

        Args:
            historical_frequency: How often this type of event happened historically (0-1)
            sample_size: Number of historical observations (affects confidence)

        Returns:
            Signal with base rate as likelihood
        """
        # Confidence scales with sample size (diminishing returns)
        import numpy as np
        confidence = min(1.0, np.log1p(sample_size) / np.log1p(1000))

        return Signal(
            signal_type=SignalType.HISTORICAL_BASE_RATE,
            likelihood_yes=historical_frequency,
            likelihood_no=1 - historical_frequency,
            confidence=confidence,
            metadata={"historical_frequency": historical_frequency, "sample_size": sample_size},
        )

    def create_momentum_signal(
        self,
        price_change_1h: float,
        volume_change_1h: float,
    ) -> Signal:
        """Create a market momentum signal from price/volume changes.

        Args:
            price_change_1h: 1-hour price change (-1 to 1)
            volume_change_1h: 1-hour volume change ratio (e.g., 1.5 = 50% increase)

        Returns:
            Signal encoding momentum direction and strength
        """
        # Convert momentum to likelihoods
        # Positive momentum → higher likelihood of YES
        import numpy as np

        # Sigmoid-like mapping: price_change → likelihood
        ll_yes = 1 / (1 + np.exp(-5 * price_change_1h))
        ll_no = 1 - ll_yes

        # Volume amplifies confidence
        confidence = min(1.0, 0.3 * volume_change_1h)

        return Signal(
            signal_type=SignalType.MARKET_MOMENTUM,
            likelihood_yes=float(ll_yes),
            likelihood_no=float(ll_no),
            confidence=confidence,
            metadata={
                "price_change_1h": price_change_1h,
                "volume_change_1h": volume_change_1h,
            },
        )
