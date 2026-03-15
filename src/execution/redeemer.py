"""Auto-redeem resolved Polymarket positions back to USDC.e.

Checks on-chain payoutDenominator for each held position.
If resolved, calls CTF.redeemPositions (or NegRiskAdapter for neg_risk markets)
to convert conditional tokens back to USDC.e collateral.

For neg_risk (multi-outcome) markets:
- Individual sub-market conditionIds resolve on CTF (payoutDenominator > 0)
- But redemption requires the NegRiskAdapter to also be "determined"
- NegRiskAdapter.getDetermined(questionID) must return True
- We check both: CTF resolution AND NegRiskAdapter determination
- Fallback: try standard CTF first, then NegRiskAdapter, then both
"""

import json
import logging
import os
import urllib.request
from pathlib import Path

from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware

logger = logging.getLogger("trader")

CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
NEG_RISK_ADAPTER = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"
USDC_E = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
RPC = "https://polygon-bor-rpc.publicnode.com"

CTF_ABI = json.loads("""[
  {"inputs":[{"name":"","type":"bytes32"},{"name":"","type":"uint256"}],"name":"payoutNumerators","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
  {"inputs":[{"name":"","type":"bytes32"}],"name":"payoutDenominator","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
  {"inputs":[{"name":"collateralToken","type":"address"},{"name":"parentCollectionId","type":"bytes32"},{"name":"conditionId","type":"bytes32"},{"name":"indexSets","type":"uint256[]"}],"name":"redeemPositions","outputs":[],"stateMutability":"nonpayable","type":"function"},
  {"inputs":[{"name":"account","type":"address"},{"name":"id","type":"uint256"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"}
]""")

NEG_RISK_ABI = json.loads("""[
  {"inputs":[{"name":"_conditionId","type":"bytes32"},{"name":"_amounts","type":"uint256[]"}],"name":"redeemPositions","outputs":[],"stateMutability":"nonpayable","type":"function"},
  {"inputs":[{"name":"questionID","type":"bytes32"}],"name":"getDetermined","outputs":[{"name":"","type":"bool"}],"stateMutability":"view","type":"function"},
  {"inputs":[{"name":"_questionId","type":"bytes32"},{"name":"_outcome","type":"bool"}],"name":"getPositionId","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"}
]""")

USDC_ABI = json.loads('[{"inputs":[{"name":"","type":"address"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"}]')


class Redeemer:
    """Automatically redeem resolved positions to USDC.e."""

    def __init__(self, private_key: str, wallet_address: str):
        self.w3 = Web3(Web3.HTTPProvider(RPC))
        self.w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        self.private_key = private_key
        self.wallet = Web3.to_checksum_address(wallet_address)
        self.ctf = self.w3.eth.contract(
            address=Web3.to_checksum_address(CTF_ADDRESS), abi=CTF_ABI
        )
        self.neg_risk_adapter = self.w3.eth.contract(
            address=Web3.to_checksum_address(NEG_RISK_ADAPTER), abi=NEG_RISK_ABI
        )
        self.usdc = self.w3.eth.contract(
            address=Web3.to_checksum_address(USDC_E), abi=USDC_ABI
        )

    def is_resolved(self, condition_id: str) -> dict | None:
        """Check if a condition is resolved on-chain (CTF level).
        
        Returns dict with payout info if resolved, None otherwise.
        """
        try:
            cid_bytes = bytes.fromhex(condition_id.replace("0x", ""))
            denom = self.ctf.functions.payoutDenominator(cid_bytes).call()
            if denom == 0:
                return None
            p_yes = self.ctf.functions.payoutNumerators(cid_bytes, 0).call()
            p_no = self.ctf.functions.payoutNumerators(cid_bytes, 1).call()
            return {
                "denominator": denom,
                "payout_yes": p_yes,
                "payout_no": p_no,
            }
        except Exception as e:
            logger.debug(f"Failed to check resolution for {condition_id}: {e}")
            return None

    def is_neg_risk_determined(self, question_id: str) -> bool:
        """Check if a neg_risk market's questionID is determined on NegRiskAdapter."""
        try:
            qid_bytes = bytes.fromhex(question_id.replace("0x", ""))
            return self.neg_risk_adapter.functions.getDetermined(qid_bytes).call()
        except Exception as e:
            logger.debug(f"getDetermined failed for {question_id}: {e}")
            return False

    def _lookup_neg_risk_info(self, condition_id: str) -> dict | None:
        """Query gamma API to get neg_risk metadata for a market."""
        try:
            url = f"https://gamma-api.polymarket.com/markets?conditionId={condition_id}"
            data = json.loads(urllib.request.urlopen(url, timeout=10).read())
            if data and isinstance(data, list) and data[0].get("negRisk"):
                return {
                    "neg_risk": True,
                    "question_id": data[0].get("questionID", ""),
                    "neg_risk_market_id": data[0].get("negRiskMarketID", ""),
                    "question": data[0].get("question", ""),
                }
        except Exception as e:
            logger.debug(f"Gamma API lookup failed for {condition_id}: {e}")
        return None

    def can_redeem(self, condition_id: str) -> tuple[bool, bool]:
        """Check if a position can be redeemed.
        
        Returns (redeemable, is_neg_risk).
        For standard markets: redeemable if CTF payoutDenominator > 0.
        For neg_risk markets: redeemable if CTF resolved AND NegRiskAdapter determined.
        """
        resolved = self.is_resolved(condition_id)
        if not resolved:
            return False, False

        # Check if it's a neg_risk market
        neg_info = self._lookup_neg_risk_info(condition_id)
        if neg_info and neg_info["neg_risk"]:
            qid = neg_info.get("question_id", "")
            if qid:
                determined = self.is_neg_risk_determined(qid)
                logger.debug(
                    f"neg_risk market {condition_id[:16]}... "
                    f"CTF resolved=True, NegRisk determined={determined}"
                )
                return determined, True
            return False, True

        return True, False

    def _get_neg_risk_token_balances(self, condition_id: str) -> tuple[int, int]:
        """Query on-chain CTF balances for YES and NO tokens of a NegRisk market.

        Uses NegRiskAdapter.getPositionId to derive token IDs, then checks
        CTF.balanceOf for each.

        Returns:
            (yes_balance_raw, no_balance_raw) in raw units (1e6 decimals)
        """
        try:
            # Look up questionID from gamma API
            neg_info = self._lookup_neg_risk_info(condition_id)
            if not neg_info or not neg_info.get("question_id"):
                logger.warning(f"Cannot find questionID for {condition_id[:16]}...")
                return 0, 0

            qid_bytes = bytes.fromhex(neg_info["question_id"].replace("0x", ""))
            yes_token_id = self.neg_risk_adapter.functions.getPositionId(qid_bytes, True).call()
            no_token_id = self.neg_risk_adapter.functions.getPositionId(qid_bytes, False).call()

            yes_bal = self.ctf.functions.balanceOf(self.wallet, yes_token_id).call()
            no_bal = self.ctf.functions.balanceOf(self.wallet, no_token_id).call()

            logger.debug(
                f"NegRisk balances for {condition_id[:16]}...: "
                f"YES={yes_bal / 1e6:.6f}, NO={no_bal / 1e6:.6f}"
            )
            return yes_bal, no_bal
        except Exception as e:
            logger.error(f"Failed to get NegRisk token balances: {e}")
            return 0, 0

    def redeem(self, condition_id: str, neg_risk: bool = False) -> float:
        """Redeem resolved position. Returns USDC.e amount redeemed."""
        bal_before = self.usdc.functions.balanceOf(self.wallet).call()
        cid_bytes = bytes.fromhex(condition_id.replace("0x", ""))

        nonce = self.w3.eth.get_transaction_count(self.wallet, "pending")
        gas_price = self.w3.eth.gas_price

        try:
            if neg_risk:
                # NegRiskAdapter.redeemPositions(conditionId, amounts)
                # amounts = [yes_amount, no_amount] in raw token units
                yes_bal, no_bal = self._get_neg_risk_token_balances(condition_id)
                if yes_bal == 0 and no_bal == 0:
                    logger.info(f"No tokens to redeem for {condition_id[:16]}...")
                    return 0.0
                amounts = [yes_bal, no_bal]
                logger.info(
                    f"🔄 NegRisk redeem {condition_id[:16]}... "
                    f"amounts=[YES={yes_bal / 1e6:.6f}, NO={no_bal / 1e6:.6f}]"
                )
                tx = self.neg_risk_adapter.functions.redeemPositions(
                    cid_bytes, amounts
                ).build_transaction({
                    "from": self.wallet,
                    "nonce": nonce,
                    "gas": 500000,
                    "gasPrice": int(gas_price * 1.2),
                    "chainId": 137,
                })
            else:
                tx = self.ctf.functions.redeemPositions(
                    Web3.to_checksum_address(USDC_E),
                    b'\x00' * 32,  # parentCollectionId = 0 for top-level
                    cid_bytes,
                    [1, 2],
                ).build_transaction({
                    "from": self.wallet,
                    "nonce": nonce,
                    "gas": 500000,
                    "gasPrice": int(gas_price * 1.2),
                    "chainId": 137,
                })

            signed = self.w3.eth.account.sign_transaction(tx, self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=90)

            if receipt.status == 1:
                bal_after = self.usdc.functions.balanceOf(self.wallet).call()
                redeemed = (bal_after - bal_before) / 1e6
                logger.info(
                    f"💰 Redeemed {condition_id[:16]}... → +${redeemed:.4f} USDC.e "
                    f"(tx: {tx_hash.hex()[:16]}...)"
                )
                return redeemed
            else:
                # Try the other method if first one failed
                if not neg_risk:
                    logger.debug(f"Standard redeem reverted, trying NegRisk adapter...")
                    return self.redeem(condition_id, neg_risk=True)
                logger.warning(f"⚠️ Redeem reverted for {condition_id[:16]}...")
                return 0.0

        except Exception as e:
            logger.error(f"❌ Redeem error for {condition_id[:16]}...: {e}")
            return 0.0

    def check_and_redeem_all(self, positions: list[dict]) -> list[dict]:
        """Check all positions and redeem any that are resolved.
        
        Args:
            positions: list of dicts with at least 'condition_id' key
            
        Returns:
            list of dicts with redemption results
        """
        results = []
        for pos in positions:
            cid = pos.get("condition_id", "")
            if not cid:
                continue

            redeemable, is_neg_risk = self.can_redeem(cid)
            if not redeemable:
                continue

            market_id = pos.get("market_id", cid[:16])
            logger.info(f"🔄 Attempting redemption for {market_id} (neg_risk={is_neg_risk})")
            amount = self.redeem(cid, neg_risk=is_neg_risk)
            results.append({
                "market_id": market_id,
                "condition_id": cid,
                "amount": amount,
                "neg_risk": is_neg_risk,
            })

        return results
