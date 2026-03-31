# Portfolio Arena — Methodology

**Community Portfolio Intelligence for DeFi**
Jay Dasondee | Glider.Fi | 2026

---

## 1. Overview

Portfolio Arena is a real-time portfolio scoring and community intelligence system for DeFi.
It ingests portfolio allocations shared in Discord, scores them using quantitative risk metrics,
and provides actionable feedback to users via a Discord bot.

The system processes **364+ portfolios** from Glider.Fi's community, covering crypto assets,
tokenized equities (via Ondo Finance), and stablecoins across Ethereum and Base L2.

## 2. Data Pipeline

### 2.1 Portfolio Ingestion
- **Source**: Glider.Fi Discord `#share-portfolio` channel
- **Method**: Discord Gateway API (bot with MESSAGE_CONTENT intent)
- **Parsing**: Regex extraction of `glider.fi/portfolio/{id}` links
- **Resolution**: Glider tRPC API (`strategyInstances.getStrategyInstance`)
- **Coverage**: All text channels scanned, 339 unique portfolio links extracted

### 2.2 Market Data
- **Primary**: Glider tRPC API (`assets.getBasicMarketDataForAsset`)
  - Covers 100% of Glider-listed tokens including Ondo tokenized stocks
  - Real-time price, 24h change, volume, market cap, liquidity
  - 24h sparkline (15-min intervals) for intraday volatility estimation
- **Secondary**: CoinGecko API (365-day daily price history)
  - 26 coins with full historical data
  - Used for Sharpe ratio, CVaR, max drawdown, correlation analysis
- **Token Resolution**: Glider token list API + CoinGecko contract lookup

### 2.3 Storage
- SQLite database with tables: `portfolios`, `allocations`, `token_list`
- Parquet files for historical price data (via strategy-predictor pipeline)

## 3. Scoring Model

### 3.1 Composite Score (0-100)

The portfolio score is a weighted linear combination of 9 normalized factors:

```
S = Σ wᵢ · fᵢ(x)    where Σ wᵢ = 1
```

| Factor | Weight | Source | Normalization |
|--------|--------|--------|---------------|
| Sharpe Ratio | 0.20 | Historical returns | (sharpe + 1) / 4, clamped [0,1] |
| CVaR 95% | 0.15 | Historical returns | 1 + cvar × 10, clamped [0,1] |
| Max Drawdown | 0.10 | Historical returns | 1 + mdd × 2, clamped [0,1] |
| Annual Volatility | 0.10 | Historical returns | 1 - vol, clamped [0,1] |
| Avg Correlation | 0.10 | Pairwise returns | 1 - corr, clamped [0,1] |
| Diversification | 0.10 | Weight entropy | H(w) / H_max, already [0,1] |
| Herding Score | 0.05 | Cosine similarity | 1 - herding, [0,1] |
| Frontier Distance | 0.10 | Markowitz analytical | 1 - dist × 5, clamped [0,1] |
| Asset Count | 0.10 | Portfolio structure | min(n/8, 1) |

### 3.2 Individual Metrics

#### Sharpe Ratio (Sharpe, 1966)

Annualized risk-adjusted return:

```
SR = (μ_p × 365 - r_f) / (σ_p × √365)
```

Where μ_p is mean daily portfolio return, σ_p is daily standard deviation, r_f = 0.

#### Conditional Value at Risk — CVaR 95% (Rockafellar & Uryasev, 2000)

Expected loss in the worst 5% of days:

```
CVaR_α = E[R | R ≤ VaR_α]
```

Where VaR_α is the α-quantile of the return distribution.

#### Maximum Drawdown

Worst peak-to-trough decline:

```
MDD = min_t (V_t / max_{s≤t} V_s - 1)
```

Where V_t is the cumulative portfolio value at time t.

#### Portfolio Volatility

Annualized standard deviation of daily log returns:

```
σ_annual = σ_daily × √365
```

#### Average Pairwise Correlation

Mean of upper-triangular elements of the return correlation matrix:

```
ρ̄ = (2 / n(n-1)) Σᵢ<ⱼ ρᵢⱼ
```

Where ρᵢⱼ = Corr(rᵢ, rⱼ) computed from daily log returns.

#### Diversification Ratio (Shannon Entropy)

Normalized entropy of portfolio weights:

```
D = H(w) / H_max = -Σ wᵢ log(wᵢ) / log(n)
```

D = 0 for single-asset, D = 1 for equal-weight.

#### Herding Score

Cosine similarity between portfolio allocation vector and community consensus:

```
H = (w · w̄) / (‖w‖ · ‖w̄‖)
```

Where w̄ is the average allocation across all community portfolios.
H = 0 means completely unique, H = 1 means identical to consensus.

#### Efficient Frontier Distance (Markowitz, 1952)

Analytical solution for minimum-variance portfolio at the same target return:

```
w* = Σ⁻¹(λμ + γ1)
```

Where λ = (Cμ_p - A)/D, γ = (B - Aμ_p)/D, and A = 1'Σ⁻¹μ, B = μ'Σ⁻¹μ, C = 1'Σ⁻¹1, D = BC - A².

Frontier distance = σ_portfolio - σ_frontier(μ_portfolio).

### 3.3 Grade Scale

| Grade | Score Range | Interpretation |
|-------|-------------|----------------|
| S | 85-100 | Exceptional — near-optimal risk-adjusted construction |
| A | 72-84 | Strong — well-diversified, low herding |
| B | 58-71 | Solid — room to improve on correlation or sector spread |
| C | 42-57 | Average — concentrated or correlated positions |
| D | 25-41 | Weak — single-sector or very few assets |
| F | 0-24 | Minimal — single asset or 100% stablecoins |

## 4. Community Intelligence

### 4.1 Herding Analysis

Community herding is measured as the average cosine similarity of all portfolios
to the community consensus allocation vector. A herding score > 0.5 indicates
significant convergence in investment behavior.

**Finding**: Average herding score across 364 portfolios = 0.45, indicating
moderate convergence. 6 portfolios are exact copies of the same template
(ETH/cbBTC/BRETT/AERO/DEGEN/MORPHO equal-weight).

### 4.2 Portfolio Similarity Network

Jaccard similarity between portfolio asset sets identifies clusters of users
running similar strategies. Portfolios with > 50% asset overlap are linked.

### 4.3 Asset Popularity Distribution

```
USDC:  144 portfolios (40%)
ETH:   124 portfolios (34%)
cbBTC:  88 portfolios (24%)
AERO:   39 portfolios (11%)
uSOL:   32 portfolios (9%)
```

Heavy concentration in top 3 assets suggests systemic risk:
a simultaneous ETH + cbBTC drawdown would affect 58% of community portfolios.

## 5. Backtesting

### 5.1 Methodology

- Score all 364 portfolios using the 9-factor composite model
- Compute actual 7d, 14d, 30d forward returns from price data
- Regress score → actual return
- Perform quintile analysis: do top-scoring portfolios outperform?

### 5.2 Limitations

- CoinGecko free tier provides daily granularity only (not intraday)
- ~40% of community tokens lack CoinGecko price history
- Scoring at a single point in time — no time-series of score evolution
- Survivorship bias: only scored portfolios still accessible via Glider API
- Crypto markets are non-stationary; model may not generalize across regimes

## 6. Technical Architecture

```
Discord Gateway → Message Parser → Glider tRPC API → SQLite
                                         ↓
                                   Portfolio Scoring
                                    (9-factor model)
                                         ↓
                              Ollama Qwen 2.5 14B (local)
                                    AI Commentary
                                         ↓
                                Discord Embed Reply
```

- **Language**: Python 3.12
- **ML/Stats**: NumPy, SciPy, Pandas
- **Bot**: discord.py 2.x
- **AI**: Ollama (Qwen 2.5 14B, local GPU inference)
- **Storage**: SQLite + Parquet
- **Deployment**: Single process, PID file lock

## 7. References

1. Markowitz, H. (1952). Portfolio Selection. *The Journal of Finance*, 7(1), 77-91.
2. Sharpe, W. F. (1966). Mutual Fund Performance. *The Journal of Business*, 39(1), 119-138.
3. Rockafellar, R. T., & Uryasev, S. (2000). Optimization of Conditional Value-at-Risk. *Journal of Risk*, 2(3), 21-42.
4. Engle, R. (2002). Dynamic Conditional Correlation. *Journal of Business & Economic Statistics*, 20(3), 339-350.
5. Lakonishok, J., Shleifer, A., & Vishny, R. W. (1992). The impact of institutional trading on stock prices. *Journal of Financial Economics*, 32(1), 23-43.
6. Shannon, C. E. (1948). A Mathematical Theory of Communication. *Bell System Technical Journal*, 27(3), 379-423.

## 8. Compliance Note

This system performs **portfolio analysis**, not investment advice.
All outputs are historical/statistical assessments with documented uncertainty.
No recommendations to buy, sell, or hold any specific asset.
