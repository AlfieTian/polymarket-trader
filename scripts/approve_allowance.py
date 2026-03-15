#!/usr/bin/env python3
"""
一键授权 Polymarket USDC Allowance
必须在下单前运行一次
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
    print("🔑 连接 Polymarket CLOB...")
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

    print("\n📊 当前余额 & 授权状态：")
    try:
        bal = client.get_balance_allowance(
            params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        )
        print(f"  USDC Balance:   {bal}")
    except Exception as e:
        print(f"  查询失败: {e}")

    print("\n⚙️  设置 USDC Allowance（COLLATERAL）...")
    try:
        result = client.update_balance_allowance(
            params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        )
        print(f"  ✅ Allowance 设置成功: {result}")
    except Exception as e:
        print(f"  ❌ 失败: {e}")

    print("\n⚙️  设置 Conditional Token Allowance...")
    try:
        result = client.update_balance_allowance(
            params=BalanceAllowanceParams(asset_type=AssetType.CONDITIONAL)
        )
        print(f"  ✅ Conditional Allowance 设置成功: {result}")
    except Exception as e:
        print(f"  ❌ 失败: {e}")

    print("\n📊 授权后余额 & 状态：")
    try:
        bal = client.get_balance_allowance(
            params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        )
        print(f"  {bal}")
    except Exception as e:
        print(f"  查询失败: {e}")

    print("\n🎉 完成！现在可以下单了。")

if __name__ == "__main__":
    main()
