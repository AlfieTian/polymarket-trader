#!/usr/bin/env python3
"""
One-click Polymarket USDC allowance approval.
Must be run once before placing orders.
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv(Path(__file__).parent.parent / ".env")

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, AssetType, BalanceAllowanceParams

HOST           = "https://clob.polymarket.com"
KEY            = os.environ["POLYMARKET_PRIVATE_KEY"]
WALLET_ADDRESS = os.environ.get("POLYMARKET_WALLET_ADDRESS", "")
API_KEY        = os.environ.get("POLYMARKET_API_KEY", "").strip("'\"")
API_SECRET     = os.environ.get("POLYMARKET_API_SECRET", "").strip("'\"")
API_PASSPHRASE = os.environ.get("POLYMARKET_API_PASSPHRASE", "").strip("'\"")

def main():
    print("Connecting to Polymarket CLOB...")
    creds = ApiCreds(
        api_key=API_KEY,
        api_secret=API_SECRET,
        api_passphrase=API_PASSPHRASE,
    )
    client = ClobClient(
        host=HOST,
        key=KEY,
        chain_id=137,
        creds=creds,
        signature_type=0,
        funder=WALLET_ADDRESS,
    )

    print("\nCurrent balance & allowance status:")
    try:
        bal = client.get_balance_allowance(
            params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        )
        print(f"  USDC Balance:   {bal}")
    except Exception as e:
        print(f"  Query failed: {e}")

    print("\nSetting USDC Allowance (COLLATERAL)...")
    try:
        result = client.update_balance_allowance(
            params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        )
        print(f"  Allowance set successfully: {result}")
    except Exception as e:
        print(f"  Failed: {e}")

    print("\nSetting Conditional Token Allowance...")
    try:
        result = client.update_balance_allowance(
            params=BalanceAllowanceParams(asset_type=AssetType.CONDITIONAL)
        )
        print(f"  Conditional Allowance set successfully: {result}")
    except Exception as e:
        print(f"  Failed: {e}")

    print("\nPost-approval balance & status:")
    try:
        bal = client.get_balance_allowance(
            params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        )
        print(f"  {bal}")
    except Exception as e:
        print(f"  Query failed: {e}")

    print("\nDone! You can now place orders.")

if __name__ == "__main__":
    main()
