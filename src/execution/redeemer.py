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
import time
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
RPC_LIST = [
    "https://polygon-rpc.com",
    "https://polygon-bor-rpc.publicnode.com",
]

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
        self.private_key = private_key
        self._wallet_address = wallet_address
        # Lazy-init: actual Web3 connection deferred to first use so that
        # temporary RPC outages don't prevent the trader from starting.
        self._w3 = None
        self._w3_checked_at = 0.0
        self._wallet = None
        self._ctf = None
        self._neg_risk_adapter = None
        self._usdc = None

    def _ensure_connected(self) -> bool:
        """Establish Web3 connection on first use. Returns True if connected."""
        if self._w3 is not None:
            # Verify connection is still alive at most once per 60s
            now = time.time()
            last_check = getattr(self, "_w3_checked_at", 0.0)
            if now - last_check > 60:
                try:
                    self._w3.eth.block_number
                    self._w3_checked_at = now
                except Exception:
                    logger.debug("Redeemer Web3 connection lost, reconnecting...")
                    self._w3 = None
        if self._w3 is not None:
            return True
        for rpc in RPC_LIST:
            try:
                candidate = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 10}))
                candidate.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
                if candidate.is_connected():
                    self._w3 = candidate
                    self._wallet = Web3.to_checksum_address(self._wallet_address)
                    self._ctf = self._w3.eth.contract(
                        address=Web3.to_checksum_address(CTF_ADDRESS), abi=CTF_ABI
                    )
                    self._neg_risk_adapter = self._w3.eth.contract(
                        address=Web3.to_checksum_address(NEG_RISK_ADAPTER), abi=NEG_RISK_ABI
                    )
                    self._usdc = self._w3.eth.contract(
                        address=Web3.to_checksum_address(USDC_E), abi=USDC_ABI
                    )
                    logger.debug(f"Redeemer Web3 connected via {rpc}")
                    return True
            except Exception as e:
                logger.debug(f"Redeemer RPC {rpc} failed: {e}")
        logger.warning("All Polygon RPC endpoints unavailable for Redeemer — will retry next cycle")
        return False

    @property
    def w3(self):
        return self._w3

    @property
    def wallet(self):
        return self._wallet

    @property
    def ctf(self):
        return self._ctf

    @property
    def neg_risk_adapter(self):
        return self._neg_risk_adapter

    @property
    def usdc(self):
        return self._usdc

    def is_resolved(self, condition_id: str) -> dict | None:
        """Check if a condition is resolved on-chain (CTF level).

        Returns dict with payout info if resolved, None otherwise.
        """
        if not self._ensure_connected():
            return None
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
        if not self._ensure_connected():
            return False
        try:
            qid_bytes = bytes.fromhex(question_id.replace("0x", ""))
            return self.neg_risk_adapter.functions.getDetermined(qid_bytes).call()
        except Exception as e:
            logger.debug(f"getDetermined failed for {question_id}: {e}")
            return False

    @staticmethod
    def _urlopen_json(url: str, timeout: int = 10) -> dict | list | None:
        """Sync HTTP GET that returns parsed JSON."""
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "polymarket-trader/1.0"})
            return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        except Exception:
            return None

    def _lookup_neg_risk_info(self, condition_id: str, token_id: str = "") -> dict | None:
        """Check if a market uses NegRisk via CLOB API.

        NOTE: The Gamma API conditionId filter is broken (returns unrelated markets).
        We use the CLOB /neg-risk endpoint instead, which works correctly.
        Falls back to Gamma API /markets?clob_token_ids= if token_id is provided.
        """
        base_info = None

        # Method 1: CLOB /neg-risk (fast, reliable)
        if token_id:
            try:
                url = f"https://clob.polymarket.com/neg-risk?token_id={token_id}"
                data = self._urlopen_json(url, timeout=10)
                if data and isinstance(data, dict) and data.get("neg_risk"):
                    base_info = {"neg_risk": True}
            except Exception as e:
                logger.debug(f"CLOB neg-risk check failed for {token_id[:16]}...: {e}")

        # Method 2: Gamma API by token_id for questionID enrichment
        if token_id:
            try:
                url = f"https://gamma-api.polymarket.com/markets?clob_token_ids={token_id}"
                data = self._urlopen_json(url, timeout=10)
                if data and isinstance(data, list):
                    item = data[0]
                    if item.get("negRisk") or base_info:
                        return {
                            "neg_risk": True,
                            "question_id": item.get("questionID", ""),
                            "neg_risk_market_id": item.get("negRiskMarketID", ""),
                            "question": item.get("question", ""),
                        }
            except Exception as e:
                logger.debug(f"Gamma API token lookup failed for {token_id[:16]}...: {e}")

        return base_info

    def can_redeem(self, condition_id: str, token_id: str = "") -> tuple[bool, bool]:
        """Check if a position can be redeemed.
        
        Returns (redeemable, is_neg_risk).
        For standard markets: redeemable if CTF payoutDenominator > 0.
        For neg_risk markets: redeemable if CTF resolved AND NegRiskAdapter determined.
        """
        resolved = self.is_resolved(condition_id)
        if not resolved:
            return False, False

        # Check if it's a neg_risk market
        neg_info = self._lookup_neg_risk_info(condition_id, token_id=token_id)
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

    def _get_neg_risk_token_balances(self, condition_id: str, token_id: str = "") -> tuple[int, int]:
        """Query on-chain CTF balances for YES and NO tokens of a NegRisk market.

        Uses NegRiskAdapter.getPositionId to derive token IDs, then checks
        CTF.balanceOf for each.

        Returns:
            (yes_balance_raw, no_balance_raw) in raw units (1e6 decimals)
        """
        try:
            # Look up questionID from gamma API
            neg_info = self._lookup_neg_risk_info(condition_id, token_id=token_id)
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

    def get_token_balance(self, token_id: str) -> int | None:
        """Return on-chain CTF balance for a token ID, or None if the probe failed."""
        if not self._ensure_connected():
            return None
        try:
            tid = int(token_id)
            return self.ctf.functions.balanceOf(self.wallet, tid).call()
        except Exception as e:
            logger.debug(f"balanceOf failed for token {str(token_id)[:16]}...: {e}")
            return None

    def _build_and_send_redeem_tx(
        self, condition_id: str, neg_risk: bool, token_id: str, cid_bytes: bytes
    ) -> tuple[bool, float]:
        """Build, sign, and send a single redeem transaction.

        Returns (success, tx_hash_or_zero).  Does NOT recurse.
        """
        # Pre-flight balance checks before fetching nonce/gas (avoid wasted RPCs)
        if not neg_risk and token_id:
            bal = self.get_token_balance(token_id)
            if bal is None:
                logger.warning(f"Token balance probe failed for {condition_id[:16]}... (standard)")
                return False, 0.0
            if bal == 0:
                logger.info(f"No tokens to redeem for {condition_id[:16]}... (standard)")
                return False, 0.0

        if neg_risk:
            yes_bal, no_bal = self._get_neg_risk_token_balances(condition_id, token_id=token_id)
            if yes_bal == 0 and no_bal == 0:
                logger.info(f"No tokens to redeem for {condition_id[:16]}...")
                return False, 0.0
            amounts = [yes_bal, no_bal]
            logger.info(
                f"🔄 NegRisk redeem {condition_id[:16]}... "
                f"amounts=[YES={yes_bal / 1e6:.6f}, NO={no_bal / 1e6:.6f}]"
            )
        nonce = self.w3.eth.get_transaction_count(self.wallet, "pending")
        gas_price = self.w3.eth.gas_price

        if neg_risk:
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
        return receipt.status == 1, tx_hash

    def redeem(self, condition_id: str, neg_risk: bool = False, token_id: str = "") -> float:
        """Redeem resolved position. Returns USDC.e amount redeemed.

        If the initial method reverts, retries once with the alternate method
        (standard CTF vs NegRiskAdapter) using a fresh nonce to avoid conflicts.
        """
        if not self._ensure_connected():
            return 0.0
        bal_before = self.usdc.functions.balanceOf(self.wallet).call()
        cid_bytes = bytes.fromhex(condition_id.replace("0x", ""))

        try:
            success, tx_hash = self._build_and_send_redeem_tx(
                condition_id, neg_risk, token_id, cid_bytes
            )

            if success:
                bal_after = self.usdc.functions.balanceOf(self.wallet).call()
                redeemed = (bal_after - bal_before) / 1e6
                logger.info(
                    f"💰 Redeemed {condition_id[:16]}... → +${redeemed:.4f} USDC.e "
                    f"(tx: {tx_hash.hex()[:16]}...)"
                )
                return redeemed

            # First method reverted — try the alternate method with a fresh nonce
            if not neg_risk:
                logger.debug(f"Standard redeem reverted, trying NegRisk adapter (fresh nonce)...")
                try:
                    success2, tx_hash2 = self._build_and_send_redeem_tx(
                        condition_id, neg_risk=True, token_id=token_id, cid_bytes=cid_bytes
                    )
                    if success2:
                        bal_after = self.usdc.functions.balanceOf(self.wallet).call()
                        redeemed = (bal_after - bal_before) / 1e6
                        logger.info(
                            f"💰 Redeemed (NegRisk fallback) {condition_id[:16]}... "
                            f"→ +${redeemed:.4f} USDC.e (tx: {tx_hash2.hex()[:16]}...)"
                        )
                        return redeemed
                except Exception as e2:
                    logger.warning(f"NegRisk fallback also failed: {e2}")

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

            token_id = pos.get("token_id", "")
            redeemable, is_neg_risk = self.can_redeem(cid, token_id=token_id)
            if not redeemable:
                continue

            market_id = pos.get("market_id", cid[:16])
            logger.info(f"🔄 Attempting redemption for {market_id} (neg_risk={is_neg_risk})")
            amount = self.redeem(cid, neg_risk=is_neg_risk, token_id=token_id)
            results.append({
                "market_id": market_id,
                "condition_id": cid,
                "token_id": token_id,
                "amount": amount,
                "neg_risk": is_neg_risk,
            })

        return results
