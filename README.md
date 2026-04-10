# community-signal-engine

**9-factor quantitative scoring engine for DeFi community portfolios, built on 364 real Glider.Fi allocations.**

## Overview

Scores DeFi portfolios shared in Discord using market risk metrics computed from 365 days of CoinGecko price history. Ranks portfolios 0–100 across a composite of Sharpe ratio, CVaR, drawdown, volatility, pairwise correlation, diversification entropy, herding (cosine similarity to community consensus), efficient frontier distance, and asset count. Outputs a Discord embed with scorecard, grade, and behavioral classification.

## Architecture

```
Discord #share-portfolio
  → Glider tRPC API          (portfolio allocations, token metadata)
  → CoinGecko historical API (365-day OHLCV, 26 coins)
  → quant_score.py           (all computation local via NumPy/SciPy)
        ├── sharpe_ratio()         annualized log-return Sharpe
        ├── cvar()                 95% conditional value at risk
        ├── max_drawdown()         peak-to-trough on cumulative returns
        ├── annual_volatility()    annualized daily return std
        ├── avg_pairwise_correlation()  mean upper-triangle of correlation matrix
        ├── diversification_entropy()   Shannon entropy of weight vector
        ├── herding_score()        cosine similarity to community mean allocation
        ├── frontier_distance()    excess vol vs Markowitz minimum-variance frontier
        └── composite_score()      weighted sum → 0–100, graded S/A/B/C/D/F
  → arena_bot.py             (Discord bot, real-time reply with embed)
```

## Scoring Weights

| Factor               | Weight |
|----------------------|--------|
| Sharpe ratio         | 20%    |
| CVaR 95%             | 15%    |
| Efficient frontier distance | 10% |
| Max drawdown         | 10%    |
| Annualized volatility | 10%  |
| Pairwise correlation | 10%    |
| Diversification entropy | 10% |
| Asset count          | 10%    |
| Herding score        | 5%     |

Grades: S (85+), A (72+), B (58+), C (42+), D (25+), F (<25)

## Community Findings (364 portfolios)

| Metric            | Value  |
|-------------------|--------|
| Avg composite score | 52.4 / 100 |
| Avg Sharpe        | −0.77  |
| Avg herding score | 0.45   |
| Top assets        | USDC (40%), ETH (34%), cbBTC (24%) |
| Identical portfolios | 6 wallets sharing the same ETH/cbBTC/BRETT/AERO/DEGEN/MORPHO allocation |

## Usage

```bash
git clone https://github.com/JKDasondee/community-signal-engine
cd community-signal-engine
pip install -e .

echo "DISCORD_BOT_TOKEN=your_token" > .env

arena-scrape   # pull portfolio links from Discord history
arena-prices   # fetch CoinGecko price data
arena-score    # score all portfolios
arena-bot      # run live Discord bot
```

## Stack

```
Python 3.12 · NumPy · SciPy · pandas
discord.py 2.x · Ollama (Qwen 2.5 14B, local) · SQLite · Parquet
```

MIT License
