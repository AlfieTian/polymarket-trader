#!/usr/bin/env python3
"""
On-chain ERC-20 approve: authorize Polymarket contracts to spend USDC.e
Only needs to be run once.
"""
import os, sys, json, time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import httpx
from eth_account import Account
from eth_account.signers.local import LocalAccount

PRIVATE_KEY    = os.environ["POLYMARKET_PRIVATE_KEY"]
WALLET         = os.environ["POLYMARKET_WALLET_ADDRESS"]
RPC            = "https://polygon-bor-rpc.publicnode.com"

# pUSD (new Polymarket collateral) and legacy USDC.e
PUSD           = "0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB"
USDC_E         = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"

# Polymarket contracts (v1 + v2 exchanges)
SPENDERS = [
    ("CTF Exchange v1",          "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"),
    ("Neg Risk CTF Exchange v1", "0xC5d563A36AE78145C45a50134d48A1215220f80a"),
    ("CTF Exchange v2",          "0xE111180000d2663C0091e4f400237545B87B996B"),
    ("Neg Risk CTF Exchange v2", "0xe2222d279d744050d28e00520010520000310F59"),
]

# Collateral tokens to approve (pUSD for new, USDC.e for legacy)
COLLATERAL_TOKENS = [
    ("pUSD",   PUSD),
    ("USDC.e", USDC_E),
]

# Avoid unlimited approval to reduce blast radius if spender is compromised.
# 10,000 USDC.e (6 decimals) is a reasonable upper bound for most bots.
MAX_UINT256 = 10_000 * 10**6


def rpc(method, params):
    resp = httpx.post(RPC, json={"jsonrpc": "2.0", "method": method, "params": params, "id": 1}, timeout=15).json()
    if "error" in resp:
        raise RuntimeError(resp["error"])
    return resp["result"]


def encode_approve(spender: str, amount: int) -> bytes:
    # approve(address,uint256) selector = 0x095ea7b3
    spender_bytes = bytes.fromhex(spender[2:].zfill(64))
    amount_bytes  = amount.to_bytes(32, "big")
    return bytes.fromhex("095ea7b3") + spender_bytes + amount_bytes


def wait_for_receipt(tx_hash: str, timeout: int = 120) -> bool:
    """Poll for a transaction receipt; return True only if it mined with status 1."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            receipt = rpc("eth_getTransactionReceipt", [tx_hash])
        except Exception:
            receipt = None
        if receipt:
            return int(receipt.get("status", "0x0"), 16) == 1
        time.sleep(3)
    return False


def main():
    acct: LocalAccount = Account.from_key(PRIVATE_KEY)
    print(f"Wallet: {acct.address}\n")

    # The transaction is signed by the private key; its nonce must be fetched
    # for the *key's* address, not a (possibly mismatched) env-var address.
    if WALLET and WALLET.lower() != acct.address.lower():
        print(f"⚠️  POLYMARKET_WALLET_ADDRESS ({WALLET}) != key address "
              f"({acct.address}) — using the key address.")
    signer = acct.address

    # Current nonce
    nonce = int(rpc("eth_getTransactionCount", [signer, "pending"]), 16)

    # Current gas price (+20% buffer)
    gas_price = int(int(rpc("eth_gasPrice", []), 16) * 1.2)
    print(f"Gas Price: {gas_price / 1e9:.2f} Gwei\n")

    sent: list[tuple[str, str]] = []   # (label, tx_hash)
    failed = False

    for token_name, token_addr in COLLATERAL_TOKENS:
        for name, spender in SPENDERS:
            label = f"{token_name} → {name}"
            print(f"  Approve {label}...")

            tx = {
                "nonce":    nonce,
                "to":       token_addr,
                "value":    0,
                "data":     "0x" + encode_approve(spender, MAX_UINT256).hex(),
                "gas":      100000,
                "gasPrice": gas_price,
                "chainId":  137,
            }

            signed = acct.sign_transaction(tx)
            raw_hex = "0x" + signed.raw_transaction.hex()

            try:
                tx_hash = rpc("eth_sendRawTransaction", [raw_hex])
                print(f"  Tx broadcast OK: {tx_hash}")
                print(f"     https://polygonscan.com/tx/{tx_hash}")
                sent.append((label, tx_hash))
                nonce += 1   # only advance nonce once a tx is actually accepted
            except Exception as e:
                print(f"  Failed to broadcast {label}: {e}")
                failed = True

            time.sleep(1)

    # Confirm each transaction actually mined with status 1 — a broadcast is
    # not a success; a tx can still revert (e.g. out of gas).
    print("\nConfirming on-chain receipts...")
    for label, tx_hash in sent:
        if wait_for_receipt(tx_hash):
            print(f"  ✅ Confirmed: {label}")
        else:
            print(f"  ❌ NOT confirmed (reverted or timed out): {label}  {tx_hash}")
            failed = True

    if failed:
        print("\n❌ Some approvals failed — check the errors above before trading.")
        sys.exit(1)
    print("\nDone! The bot can now place orders.")


if __name__ == "__main__":
    main()
