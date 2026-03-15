#!/usr/bin/env python3
"""
修复 ERC-1155 setApprovalForAll
让 Polymarket 交易所合约可以转移我们钱包里的条件代币（NO/YES tokens）
原因：卖出 NO token 时报 'not enough balance / allowance' = CTF token 转移权限缺失
"""
import argparse
import os, sys, json, time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import httpx
from eth_account import Account

PRIVATE_KEY = os.environ["POLYMARKET_PRIVATE_KEY"]
WALLET      = os.environ["POLYMARKET_WALLET_ADDRESS"]
RPC         = "https://polygon-bor-rpc.publicnode.com"

# ERC-1155 条件代币合约（Polygon mainnet）
CTF_TOKEN = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"

# 需要授权的 operator（交易所合约）
OPERATORS = [
    ("CTF Exchange",           "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"),
    ("Neg Risk CTF Exchange",  "0xC5d563A36AE78145C45a50134d48A1215220f80a"),
    ("NegRisk Adapter",        "0xd91E80cF2Ed2FD526D61a6fdfc9F5ADfA75F0F4b"),
]

def rpc_call(method, params):
    resp = httpx.post(
        RPC,
        json={"jsonrpc": "2.0", "method": method, "params": params, "id": 1},
        timeout=15,
    ).json()
    if "error" in resp:
        raise RuntimeError(resp["error"])
    return resp["result"]

def encode_set_approval_for_all(operator: str, approved: bool) -> str:
    # setApprovalForAll(address,bool) selector = 0xa22cb465
    selector = "a22cb465"
    op_padded  = operator[2:].lower().zfill(64)
    bool_padded = ("01" if approved else "00").zfill(64)
    return "0x" + selector + op_padded + bool_padded

def check_is_approved(operator: str) -> bool:
    # isApprovedForAll(address owner, address operator) = 0xe985e9c5
    selector = "e985e9c5"
    owner_padded = WALLET[2:].lower().zfill(64)
    op_padded    = operator[2:].lower().zfill(64)
    data = "0x" + selector + owner_padded + op_padded
    result = rpc_call("eth_call", [{"to": CTF_TOKEN, "data": data}, "latest"])
    return int(result, 16) == 1

def main():
    parser = argparse.ArgumentParser(description="Set ERC-1155 operator approvals for Polymarket CTF tokens.")
    parser.add_argument(
        "--revoke",
        action="store_true",
        help="Revoke approvals instead of granting them.",
    )
    args = parser.parse_args()
    approve_flag = not args.revoke

    acct = Account.from_key(PRIVATE_KEY)
    print(f"钱包: {acct.address}")
    print(f"CTF Token 合约: {CTF_TOKEN}\n")

    # 先检查哪些已经 approved
    print("🔍 检查当前授权状态...")
    needs_approval = []
    for name, op in OPERATORS:
        try:
            approved = check_is_approved(op)
            status = "✅ 已授权" if approved else "❌ 未授权"
            print(f"  {status}  {name} ({op[:10]}...)")
            if approve_flag and not approved:
                needs_approval.append((name, op))
            if args.revoke and approved:
                needs_approval.append((name, op))
        except Exception as e:
            print(f"  ⚠️  检查失败 {name}: {e}")
            needs_approval.append((name, op))

    if not needs_approval:
        if args.revoke:
            print("\n✅ 所有 operator 已撤销或未授权，无需操作。")
        else:
            print("\n✅ 所有 operator 已授权，无需操作。")
        return

    action_label = "授权" if approve_flag else "撤销授权"
    print(f"\n将要 {action_label} {len(needs_approval)} 个合约：")
    for name, op in needs_approval:
        print(f"  - {name}: {op}")
    confirm = input("\n请输入 'yes' 继续: ").strip().lower()
    if confirm != "yes":
        print("❌ 已取消。")
        return

    print(f"\n开始广播交易（{action_label}）...\n")

    nonce     = int(rpc_call("eth_getTransactionCount", [WALLET, "pending"]), 16)
    gas_price = int(int(rpc_call("eth_gasPrice", []), 16) * 1.2)
    print(f"Gas Price: {gas_price/1e9:.2f} Gwei | 起始 nonce: {nonce}\n")

    for name, op in needs_approval:
        print(f"⚙️  setApprovalForAll → {name}...")
        data = encode_set_approval_for_all(op, approve_flag)
        tx = {
            "nonce":    nonce,
            "to":       CTF_TOKEN,
            "value":    0,
            "data":     data,
            "gas":      80000,
            "gasPrice": gas_price,
            "chainId":  137,
        }
        signed = acct.sign_transaction(tx)
        raw_hex = "0x" + signed.raw_transaction.hex()
        try:
            tx_hash = rpc_call("eth_sendRawTransaction", [raw_hex])
            print(f"  ✅ 广播成功: {tx_hash}")
            print(f"     https://polygonscan.com/tx/{tx_hash}")
        except Exception as e:
            print(f"  ❌ 失败: {e}")
        nonce += 1
        time.sleep(1)

    print("\n⏳ 等待链上确认（约 10 秒）...")
    time.sleep(12)

    # 重新验证
    print("\n🔍 验证授权结果...")
    all_ok = True
    for name, op in OPERATORS:
        try:
            approved = check_is_approved(op)
            status = "✅ 已授权" if approved else "❌ 仍未授权"
            print(f"  {status}  {name}")
            if not approved:
                all_ok = False
        except Exception as e:
            print(f"  ⚠️  验证失败 {name}: {e}")

    if all_ok:
        print("\n🎉 完成！机器人现在可以卖出 NO/YES token 了。")
        print("   958443 NO 持仓应该会在下个 cycle 触发止盈卖出。")
    else:
        print("\n⚠️  部分授权失败，请检查 gas / RPC。")

if __name__ == "__main__":
    main()
