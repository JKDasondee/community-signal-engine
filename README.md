# Portfolio Arena

Real-time community portfolio intelligence for DeFi. Scores portfolios shared in Discord using quantitative risk metrics — CVaR, Sharpe ratio, efficient frontier distance, herding detection.

Built for [Glider.Fi](https://glider.fi) (100K users, a16z CSX + Coinbase Ventures).

## Architecture

```
Discord #share-portfolio
        |
        v
  Glider tRPC API ──> Portfolio data (assets, weights, prices)
        |
        v
  9-Factor Quant Scoring Engine
  ├── Sharpe Ratio (annualized)
  ├── CVaR 95% (tail risk)
  ├── Max Drawdown
  ├── Portfolio Volatility
  ├── Pairwise Correlation
  ├── Diversification Entropy
  ├── Herding Score (vs community consensus)
  ├── Efficient Frontier Distance (Markowitz)
  └── Asset Count
        |
        v
  Ollama Qwen 2.5 14B (local GPU)
  AI-generated portfolio commentary
        |
        v
  Discord Embed Reply (score card + rank + tip)
```

## Quick Start

```bash
git clone https://github.com/JKDasondee/community-signal-engine
cd community-signal-engine
pip install -e .

# set up .env
echo "DISCORD_BOT_TOKEN=your_token" > .env

# scrape all portfolios from Discord history
arena-scrape

# run the bot
arena-bot
```

## Commands

| Command | What |
|---------|------|
| `arena-bot` | Run Discord bot (scores portfolio links in real-time) |
| `arena-scrape` | Scrape all portfolio links from Discord history |
| `arena-score` | Score all portfolios with quant metrics |
| `arena-prices` | Fetch live prices for all tokens |
| `python scripts/backtest_scores.py` | Backtest: do scores predict returns? |
| `python scripts/analyze_portfolios.py` | Community-wide portfolio intelligence report |

## Scoring

9-factor composite score (0-100):

| Factor | Weight | What It Measures |
|--------|--------|-----------------|
| Sharpe Ratio | 20% | Risk-adjusted return |
| CVaR 95% | 15% | Tail risk (worst 5% of days) |
| Max Drawdown | 10% | Worst peak-to-trough loss |
| Volatility | 10% | Annualized price swing |
| Correlation | 10% | Are assets moving together? |
| Diversification | 10% | Shannon entropy of weights |
| Frontier Distance | 10% | Gap from Markowitz optimal |
| Asset Count | 10% | Number of holdings |
| Herding | 5% | Similarity to community consensus |

Grades: S (85+), A (72+), B (58+), C (42+), D (25+), F (<25)

## Community Findings (364 portfolios)

- **Avg score**: 52.4/100
- **Avg Sharpe**: -0.77 (community is net negative risk-adjusted)
- **Avg herding**: 0.45 (moderate convergence)
- **Top assets**: USDC (40%), ETH (34%), cbBTC (24%)
- **6 identical portfolios**: ETH/cbBTC/BRETT/AERO/DEGEN/MORPHO template
- **1 user** running tokenized oil stocks (Chevron, Petrobras, Exxon via Ondo)

## Data Sources

- **Glider tRPC API**: Portfolio allocations, live prices, token metadata
- **CoinGecko API**: 365-day historical prices (26 coins)
- **Discord Gateway**: Message history, real-time portfolio link detection

Covers all Glider-listed tokens: crypto, stablecoins, and Ondo tokenized stocks.

## Tech Stack

- Python 3.12, NumPy, SciPy, Pandas
- discord.py 2.x
- Ollama (Qwen 2.5 14B, local inference)
- SQLite + Parquet

## Docs

- [Methodology](docs/methodology.md) — full scoring model with math and citations

## License

MIT

---

Built by [Jay Dasondee](https://github.com/JKDasondee)
