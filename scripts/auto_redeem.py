#!/usr/bin/env python3
"""Auto-redeem check — runs via cron, redeems resolved positions, notifies via openclaw."""

import json
import logging
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from src.execution.redeemer import Redeemer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("auto_redeem")

POSITIONS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs", "positions_state.json"
)

# Known positions that may have been lost from state tracking
# Add condition_ids here for positions the bot opened but lost track of
EXTRA_POSITIONS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs", "pending_redemptions.json"
)


def load_all_positions() -> list[dict]:
    """Load positions from state file + pending redemptions."""
    positions = []

    # Active positions from bot state
    if os.path.exists(POSITIONS_FILE):
        with open(POSITIONS_FILE) as f:
            positions.extend(json.load(f))

    # Extra positions waiting for redemption (lost from state, manually added, etc.)
    if os.path.exists(EXTRA_POSITIONS_FILE):
        with open(EXTRA_POSITIONS_FILE) as f:
            positions.extend(json.load(f))

    return positions


def remove_redeemed(redeemed_cids: set[str]):
    """Remove redeemed positions from pending_redemptions.json."""
    if not os.path.exists(EXTRA_POSITIONS_FILE) or not redeemed_cids:
        return
    with open(EXTRA_POSITIONS_FILE) as f:
        pending = json.load(f)
    remaining = [p for p in pending if p.get("condition_id") not in redeemed_cids]
    with open(EXTRA_POSITIONS_FILE, "w") as f:
        json.dump(remaining, f, indent=2)


def main():
    pk = os.getenv("POLYMARKET_PRIVATE_KEY")
    wallet = os.getenv("POLYMARKET_WALLET_ADDRESS")
    if not pk or not wallet:
        logger.error("Missing POLYMARKET_PRIVATE_KEY or POLYMARKET_WALLET_ADDRESS")
        return

    redeemer = Redeemer(pk, wallet)
    positions = load_all_positions()

    if not positions:
        logger.info("No positions to check")
        return

    logger.info(f"Checking {len(positions)} positions for redemption...")
    results = redeemer.check_and_redeem_all(positions)

    if results:
        total = sum(r["amount"] for r in results)
        details = ", ".join(
            f"{r['market_id']}: +${r['amount']:.2f}" for r in results
        )
        msg = f"💰 Auto-redeem: {len(results)} position(s) redeemed, +${total:.2f} USDC.e total. {details}"
        logger.info(msg)

        # Remove redeemed from pending (regardless of amount — $0 positions are also done)
        redeemed_cids = {r["condition_id"] for r in results}
        remove_redeemed(redeemed_cids)

        # Notify via openclaw
        try:
            subprocess.run(
                ["openclaw", "system", "event", "--text", msg, "--mode", "now"],
                timeout=30, capture_output=True
            )
        except Exception as e:
            logger.warning(f"Failed to notify: {e}")
    else:
        logger.info("No positions ready for redemption")


if __name__ == "__main__":
    main()
