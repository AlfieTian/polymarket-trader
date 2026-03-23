#!/usr/bin/env python3
"""
One-click Polymarket API credential generation

Prerequisites: fill in POLYMARKET_PRIVATE_KEY and POLYMARKET_WALLET_ADDRESS in .env
Generates API credentials and writes them back to .env
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv, set_key

ENV_PATH = Path(__file__).parent.parent / ".env"


def main():
    if not ENV_PATH.exists():
        print("Error: .env file not found")
        print("   Run: cp .env.template .env")
        print("   Then fill in POLYMARKET_PRIVATE_KEY and POLYMARKET_WALLET_ADDRESS")
        sys.exit(1)

    load_dotenv(ENV_PATH)
    private_key = os.getenv("POLYMARKET_PRIVATE_KEY", "").strip()
    wallet_address = os.getenv("POLYMARKET_WALLET_ADDRESS", "").strip()

    if not private_key:
        print("Error: POLYMARKET_PRIVATE_KEY is empty")
        print("   Please fill in your Polygon wallet private key in .env")
        sys.exit(1)

    if not wallet_address:
        print("Error: POLYMARKET_WALLET_ADDRESS is empty")
        print("   Please fill in your wallet address (0x...) in .env")
        sys.exit(1)

    print("Deriving API credentials via L1->L2 signature...")
    print(f"   Wallet: {wallet_address[:8]}...{wallet_address[-6:]}")

    try:
        from py_clob_client.client import ClobClient

        # Step 1: Create temporary client, derive API credentials
        temp_client = ClobClient(
            host="https://clob.polymarket.com",
            key=private_key,
            chain_id=137,  # Polygon mainnet
        )

        # create_or_derive: returns existing creds if available, otherwise creates new ones
        creds = temp_client.create_or_derive_api_creds()

        api_key = creds.api_key
        api_secret = creds.api_secret
        api_passphrase = creds.api_passphrase

        print(f"\n  API Key:      {api_key[:12]}...")
        print(f"  Secret:       {api_secret[:8]}...")
        print(f"  Passphrase:   {api_passphrase[:8]}...")

        # Write back to .env
        set_key(str(ENV_PATH), "POLYMARKET_API_KEY", api_key)
        set_key(str(ENV_PATH), "POLYMARKET_API_SECRET", api_secret)
        set_key(str(ENV_PATH), "POLYMARKET_API_PASSPHRASE", api_passphrase)
        # Lock down permissions after write (0600)
        os.chmod(ENV_PATH, 0o600)

        print(f"\n  Credentials written to {ENV_PATH}")

        # Step 2: Verify credentials
        print("\n  Verifying credentials...")
        client = ClobClient(
            host="https://clob.polymarket.com",
            key=private_key,
            chain_id=137,
            creds=creds,
            signature_type=0,  # EOA wallet
            funder=wallet_address,
        )

        # Verify by fetching API key list
        try:
            keys = client.get_api_keys()
            print(f"  Verified! {len(keys) if keys else 0} active API key(s)")
        except Exception:
            print("  Credentials generated but verification skipped (may be first-time creation)")

        print("\n" + "=" * 50)
        print("  Setup complete! You can now run:")
        print("   python scripts/run_trader.py")
        print("=" * 50)

    except Exception as e:
        print(f"\n  Generation failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Check private key format (64 hex chars, with or without 0x prefix)")
        print("  2. Ensure wallet has some POL for gas")
        print("  3. Ensure wallet has USDC.e for trading")
        print("  4. Ensure network can reach clob.polymarket.com")
        sys.exit(1)


if __name__ == "__main__":
    main()
