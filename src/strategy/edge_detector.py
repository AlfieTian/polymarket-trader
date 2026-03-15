"""
Edge Detection Module

Identifies profitable trading opportunities by comparing Bayesian posterior
estimates against market prices.

EV = p̂ - p  (where p̂ is our estimate, p is market price)
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TradeOpportunity:
    """A detected edge / trade opportunity."""

    market_id: str
    market_question: str
    side: str  # "YES" or "NO"
    p_hat: float  # our estimated probability
    market_price: float  # current CLOB price
    edge: float  # EV = p_hat - market_price (for YES)
    abs_edge: float
    confidence: float  # signal confidence score
    volume_24h: float = 0.0
    time_to_resolution_hours: float = 0.0

    @property
    def edge_pct(self) -> str:
        return f"{self.abs_edge * 100:.1f}%"


class EdgeDetector:
    """Detect and rank trading opportunities based on edge (EV).

    Filters by:
    - Minimum edge threshold
    - Market volume
    - Time to resolution
    - Signal confidence
    """

    def __init__(
        self,
        min_edge: float = 0.03,
        min_volume_24h: float = 0.0,
        min_confidence: float = 0.0,
        min_time_to_resolution_hours: float = 1.0,
        price_bounds: tuple[float, float] = (0.05, 0.95),
    ):
        self.price_bounds = price_bounds
        self.min_edge = min_edge
        self.min_volume_24h = min_volume_24h
        self.min_confidence = min_confidence
        self.min_time_to_resolution_hours = min_time_to_resolution_hours

    def detect(
        self,
        market_id: str,
        market_question: str,
        p_hat: float,
        market_price_yes: float,
        confidence: float = 1.0,
        volume_24h: float = 0.0,
        time_to_resolution_hours: float = float("inf"),
    ) -> TradeOpportunity | None:
        """Detect edge for a single market.

        Args:
            market_id: Market identifier
            market_question: Human-readable question
            p_hat: Our Bayesian posterior for YES
            market_price_yes: Current YES price on CLOB
            confidence: Signal confidence score (0-1)
            volume_24h: 24h trading volume in USDC
            time_to_resolution_hours: Hours until market resolves

        Returns:
            TradeOpportunity if edge detected, None otherwise
        """
        # Skip extreme prices (already settled or near-certain markets)
        lo, hi = self.price_bounds
        if market_price_yes <= lo or market_price_yes >= hi:
            return None

        # Calculate edge for YES side
        edge_yes = p_hat - market_price_yes
        # Calculate edge for NO side
        edge_no = (1 - p_hat) - (1 - market_price_yes)  # = market_price_yes - p_hat

        # Pick the side with positive edge
        if edge_yes > 0:
            side = "YES"
            edge = edge_yes
        elif edge_no > 0:
            side = "NO"
            edge = edge_no
        else:
            return None

        abs_edge = abs(edge)

        # Apply filters
        if abs_edge < self.min_edge:
            return None
        if volume_24h < self.min_volume_24h:
            logger.debug(f"Skipping {market_id}: volume {volume_24h} < {self.min_volume_24h}")
            return None
        if confidence < self.min_confidence:
            logger.debug(f"Skipping {market_id}: confidence {confidence} < {self.min_confidence}")
            return None
        if time_to_resolution_hours < self.min_time_to_resolution_hours:
            logger.debug(
                f"Skipping {market_id}: resolves in {time_to_resolution_hours}h "
                f"< {self.min_time_to_resolution_hours}h"
            )
            return None

        opp = TradeOpportunity(
            market_id=market_id,
            market_question=market_question,
            side=side,
            p_hat=p_hat,
            market_price=market_price_yes if side == "YES" else (1 - market_price_yes),
            edge=edge,
            abs_edge=abs_edge,
            confidence=confidence,
            volume_24h=volume_24h,
            time_to_resolution_hours=time_to_resolution_hours,
        )

        logger.info(
            f"Edge detected: {market_id} {side} — "
            f"p̂={p_hat:.4f} vs market={market_price_yes:.4f}, "
            f"edge={edge:+.4f} ({opp.edge_pct})"
        )
        return opp

    def scan_markets(
        self,
        markets: list[dict],
    ) -> list[TradeOpportunity]:
        """Scan multiple markets and return ranked opportunities.

        Args:
            markets: List of dicts with keys:
                market_id, question, p_hat, market_price_yes,
                confidence, volume_24h, time_to_resolution_hours

        Returns:
            Opportunities sorted by absolute edge (descending)
        """
        opportunities = []
        for m in markets:
            opp = self.detect(
                market_id=m["market_id"],
                market_question=m.get("question", ""),
                p_hat=m["p_hat"],
                market_price_yes=m["market_price_yes"],
                confidence=m.get("confidence", 1.0),
                volume_24h=m.get("volume_24h", 0.0),
                time_to_resolution_hours=m.get("time_to_resolution_hours", float("inf")),
            )
            if opp is not None:
                opportunities.append(opp)

        # Rank by absolute edge (largest first)
        opportunities.sort(key=lambda o: o.abs_edge, reverse=True)

        if opportunities:
            logger.info(
                f"Scan complete: {len(opportunities)} opportunities "
                f"from {len(markets)} markets"
            )
        return opportunities
