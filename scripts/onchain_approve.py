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

# Polygon USDC.e
USDC_E         = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"

# Polymarket contracts (CTF Exchange + Neg Risk CTF Exchange)
SPENDERS = [
    ("CTF Exchange",          "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"),
    ("Neg Risk CTF Exchange", "0xC5d563A36AE78145C45a50134d48A1215220f80a"),
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


def main():
    acct: LocalAccount = Account.from_key(PRIVATE_KEY)
    print(f"Wallet: {acct.address}\n")

    # Current nonce
    nonce = int(rpc("eth_getTransactionCount", [WALLET, "pending"]), 16)

    # Current gas price (+20% buffer)
    gas_price = int(int(rpc("eth_gasPrice", []), 16) * 1.2)
    print(f"Gas Price: {gas_price / 1e9:.2f} Gwei\n")

    for name, spender in SPENDERS:
        print(f"  Approve {name}...")

        tx = {
            "nonce":    nonce,
            "to":       USDC_E,
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
        except Exception as e:
            print(f"  Failed: {e}")

        nonce += 1
        time.sleep(1)

    print("\nWaiting for tx confirmation (~5-10s)...")
    time.sleep(10)
    print("Done! The bot can now place orders.")


if __name__ == "__main__":
    main()
