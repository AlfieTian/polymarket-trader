# Polymarket Trading System 📊

基于贝叶斯信号处理与 CLOB 执行的预测市场量化交易框架。

## 架构

```
数据摄取 → 贝叶斯信号引擎 → 边际收益检测 → 分数Kelly仓位 → 风控 → CLOB执行
```

### 模块

| 模块 | 说明 |
|------|------|
| `src/signals/lmsr_pricer.py` | LMSR 参考定价 + 低效检测 |
| `src/signals/bayesian_engine.py` | 顺序贝叶斯更新（对数空间） |
| `src/strategy/edge_detector.py` | EV = p̂ - p 边际过滤 |
| `src/strategy/kelly_sizer.py` | 0.25x 分数 Kelly 仓位计算 |
| `src/data/polymarket_client.py` | Polymarket REST + WebSocket |
| `src/data/news_feed.py` | 新闻信号源 → 贝叶斯信号 |
| `src/execution/clob_executor.py` | CLOB 限价/市价单执行 |
| `src/execution/order_manager.py` | 订单生命周期管理 |
| `src/risk/risk_manager.py` | 敞口/回撤/集中度风控 |

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行测试
pytest tests/ -v

# 启动交易（默认 dry-run 模式）
python scripts/run_trader.py

# 运行回测
python scripts/backtest.py
```

## 配置

编辑 `config.yaml`：

```yaml
polymarket:
  api_key: "your-api-key"
  private_key: "your-wallet-private-key"
  dry_run: true  # ← 改为 false 启用真实交易

strategy:
  min_edge: 0.03        # 最小 EV 阈值
  kelly_fraction: 0.25  # Kelly 系数（0.25 = 四分之一 Kelly）
```

## 核心公式

**LMSR 成本函数：**
$$C(\mathbf{q}) = b \cdot \ln\left(\sum_{i=1}^{n} e^{q_i/b}\right)$$

**贝叶斯更新（对数空间）：**
$$\log P(H|\mathbf{D}) = \log P(H) + \sum_{k=1}^{t} \log P(D_k|H) - \log Z$$

**分数 Kelly：**
$$f^* = \frac{\hat{p} - p}{1 - p} \times \text{kelly\_fraction}$$

## ⚠️ 风险提示

- 默认 `dry_run: true`，不会执行真实交易
- 预测市场有流动性风险和模型风险
- "NEVER full Kelly on 5min markets!" — 文档原文
- 本系统仅供研究和学习使用
