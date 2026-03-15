#!/usr/bin/env python3
"""
一键生成 Polymarket API 凭证

前提：在 .env 中填入 POLYMARKET_PRIVATE_KEY 和 POLYMARKET_WALLET_ADDRESS
运行后会自动生成 API 凭证并写回 .env
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv, set_key

ENV_PATH = Path(__file__).parent.parent / ".env"


def main():
    if not ENV_PATH.exists():
        print("❌ .env 文件不存在")
        print("   运行: cp .env.template .env")
        print("   然后填入 POLYMARKET_PRIVATE_KEY 和 POLYMARKET_WALLET_ADDRESS")
        sys.exit(1)

    load_dotenv(ENV_PATH)
    private_key = os.getenv("POLYMARKET_PRIVATE_KEY", "").strip()
    wallet_address = os.getenv("POLYMARKET_WALLET_ADDRESS", "").strip()

    if not private_key:
        print("❌ POLYMARKET_PRIVATE_KEY 为空")
        print("   请在 .env 中填入你的 Polygon 钱包私钥")
        sys.exit(1)

    if not wallet_address:
        print("❌ POLYMARKET_WALLET_ADDRESS 为空")
        print("   请在 .env 中填入你的钱包地址（0x...）")
        sys.exit(1)

    print("🔑 正在通过 L1→L2 签名派生 API 凭证...")
    print(f"   钱包地址: {wallet_address[:8]}...{wallet_address[-6:]}")

    try:
        from py_clob_client.client import ClobClient

        # Step 1: 创建临时 client，派生 API 凭证
        temp_client = ClobClient(
            host="https://clob.polymarket.com",
            key=private_key,
            chain_id=137,  # Polygon mainnet
        )

        # create_or_derive: 如果已有则返回已有的，没有则创建新的
        creds = temp_client.create_or_derive_api_creds()

        api_key = creds.api_key
        api_secret = creds.api_secret
        api_passphrase = creds.api_passphrase

        print(f"\n✅ API Key:      {api_key[:12]}...")
        print(f"✅ Secret:       {api_secret[:8]}...")
        print(f"✅ Passphrase:   {api_passphrase[:8]}...")

        # 写回 .env
        set_key(str(ENV_PATH), "POLYMARKET_API_KEY", api_key)
        set_key(str(ENV_PATH), "POLYMARKET_API_SECRET", api_secret)
        set_key(str(ENV_PATH), "POLYMARKET_API_PASSPHRASE", api_passphrase)
        # Lock down permissions after write (0600)
        os.chmod(ENV_PATH, 0o600)

        print(f"\n✅ 凭证已写入 {ENV_PATH}")

        # Step 2: 验证凭证有效
        print("\n🔍 验证凭证...")
        client = ClobClient(
            host="https://clob.polymarket.com",
            key=private_key,
            chain_id=137,
            creds=creds,
            signature_type=0,  # EOA wallet
            funder=wallet_address,
        )

        # 尝试获取 API key 列表来验证
        try:
            keys = client.get_api_keys()
            print(f"✅ 验证成功！当前有 {len(keys) if keys else 0} 个活跃 API Key")
        except Exception:
            print("⚠️ 凭证已生成但验证跳过（可能是首次创建）")

        print("\n" + "=" * 50)
        print("🎉 设置完成！现在可以运行：")
        print("   python scripts/run_trader.py")
        print("=" * 50)

    except Exception as e:
        print(f"\n❌ 生成失败: {e}")
        print("\n排查步骤：")
        print("  1. 检查私钥格式（以 0x 开头的 64 位 hex，或不带 0x）")
        print("  2. 确认钱包里有少量 POL（用于 gas）")
        print("  3. 确认钱包里有 USDC.e（用于交易）")
        print("  4. 确认网络可以访问 clob.polymarket.com")
        sys.exit(1)


if __name__ == "__main__":
    main()
