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

from py_clob_client_v2 import ClobClient, ApiCreds, AssetType, BalanceAllowanceParams

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
        chain_id=137,
        key=KEY,
        creds=creds,
        signature_type=0,
        funder=WALLET_ADDRESS,
    )

    failed = False

    print("\nCurrent balance & allowance status:")
    try:
        bal = client.get_balance_allowance(
            params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        )
        print(f"  pUSD Balance:   {bal}")
    except Exception as e:
        print(f"  Query failed: {e}")

    print("\nSetting pUSD Allowance (COLLATERAL)...")
    try:
        result = client.update_balance_allowance(
            params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        )
        print(f"  Allowance set successfully: {result}")
    except Exception as e:
        print(f"  Failed: {e}")
        failed = True

    # NOTE: Conditional-token (ERC-1155) approval is NOT done here.
    # update_balance_allowance(CONDITIONAL) needs a specific token_id and is a
    # no-op without one — it cannot blanket-approve conditional tokens. The
    # correct operation is the on-chain setApprovalForAll, handled by
    # scripts/fix_ctf_approval.py. Run that to enable selling NO/YES tokens.
    print("\nConditional-token approval: run scripts/fix_ctf_approval.py "
          "(setApprovalForAll) — skipped here.")

    print("\nPost-approval balance & status:")
    try:
        bal = client.get_balance_allowance(
            params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        )
        print(f"  {bal}")
    except Exception as e:
        print(f"  Query failed: {e}")

    if failed:
        print("\n❌ Allowance setup FAILED — fix the error above before trading.")
        sys.exit(1)
    print("\nDone! Collateral allowance set. "
          "Run scripts/fix_ctf_approval.py before selling tokens.")

if __name__ == "__main__":
    main()
