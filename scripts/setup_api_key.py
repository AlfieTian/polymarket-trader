#!/usr/bin/env python3
"""
一键生成 Polymarket API 凭证

前提：在 .env 中填入 POLYMARKET_PRIVATE_KEY
运行后会自动生成 API_KEY / SECRET / PASSPHRASE 并写回 .env
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv, set_key
import os

ENV_PATH = Path(__file__).parent.parent / ".env"


def main():
    if not ENV_PATH.exists():
        print("❌ .env 文件不存在，先运行：cp .env.template .env")
        print("   然后填入 POLYMARKET_PRIVATE_KEY")
        sys.exit(1)

    load_dotenv(ENV_PATH)
    private_key = os.getenv("POLYMARKET_PRIVATE_KEY", "").strip()

    if not private_key:
        print("❌ POLYMARKET_PRIVATE_KEY 为空")
        print("   请在 .env 中填入你的 Polygon 钱包私钥")
        sys.exit(1)

    print("🔑 正在生成 API 凭证...")

    try:
        from py_clob_client.client import ClobClient

        client = ClobClient(
            host="https://clob.polymarket.com",
            key=private_key,
            chain_id=137,  # Polygon mainnet
        )

        creds = client.create_api_key()
        print(f"✅ API Key: {creds['apiKey']}")
        print(f"✅ Secret:  {creds['secret'][:8]}...")
        print(f"✅ Passphrase: {creds['passphrase'][:8]}...")

        # Write back to .env
        set_key(str(ENV_PATH), "POLYMARKET_API_KEY", creds["apiKey"])
        set_key(str(ENV_PATH), "POLYMARKET_API_SECRET", creds["secret"])
        set_key(str(ENV_PATH), "POLYMARKET_API_PASSPHRASE", creds["passphrase"])

        print(f"\n✅ 凭证已写入 {ENV_PATH}")
        print("   现在可以运行 python scripts/run_trader.py 了！")

    except Exception as e:
        print(f"❌ 生成失败: {e}")
        print("   检查私钥是否正确，钱包是否有 MATIC gas")
        sys.exit(1)


if __name__ == "__main__":
    main()
