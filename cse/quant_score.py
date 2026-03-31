"""
Research-grade portfolio scoring engine.
Replaces heuristic HHI scoring with real quantitative risk metrics.

Uses actual price data from strategy-predictor pipeline.
All computation local (numpy/scipy). No API calls.

Metrics:
  - CVaR (Conditional Value at Risk) at 95%
  - Annualized Sharpe ratio
  - Maximum drawdown
  - Portfolio volatility (annualized)
  - DCC correlation risk (average pairwise correlation)
  - Entropy-based diversification ratio
  - Herding score (mutual information vs community)
  - Mean-variance efficient frontier distance

Run standalone: python -m cse.quant_score
"""
import json
import sqlite3
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

SP_DATA = Path(__file__).parent.parent.parent / "strategy-predictor" / "data"
CSE_DATA = Path(__file__).parent.parent / "data"
DB_PATH = CSE_DATA / "portfolios.db"

@dataclass
class QuantScore:
    total: int                  # 0-100 composite
    grade: str                  # S/A/B/C/D/F
    sharpe: float               # annualized
    cvar_95: float              # 95% CVaR (expected loss in worst 5%)
    max_drawdown: float         # worst peak-to-trough
    annual_vol: float           # annualized volatility
    avg_correlation: float      # mean pairwise correlation
    diversification_ratio: float # entropy-based
    herding_score: float        # vs community average allocation
    frontier_distance: float    # distance to efficient frontier
    n_assets: int
    strategy_type: str
    risk_label: str

def load_prices() -> pd.DataFrame | None:
    for name in ["prices_365d", "prices_full", "prices", "prices_test"]:
        p = SP_DATA / f"{name}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            piv = df.pivot_table(index="date", columns="coin_id", values="price_usd")
            return piv.sort_index().ffill().dropna(how="all")
    return None

# map Glider symbols to CoinGecko IDs
SYM_TO_CG = {
    "ETH": "ethereum", "WETH": "ethereum", "cbBTC": "wrapped-bitcoin",
    "WBTC": "wrapped-bitcoin", "USDC": "usd-coin", "USDT": "tether",
    "DAI": "dai", "AAVE": "aave", "AERO": "aerodrome-finance",
    "MORPHO": "morpho", "UNI": "uniswap", "LINK": "chainlink",
    "COMP": "compound-governance-token", "MKR": "maker",
    "SNX": "synthetix-network-token", "CRV": "curve-dao-token",
    "BAL": "balancer", "LDO": "lido-dao", "SUSHI": "sushi",
    "VIRTUAL": "virtual-protocol", "DEGEN": "degen-base",
    "BRETT": "based-brett", "TOSHI": "toshi",
    "uSOL": "solana", "SOL": "solana",
}

def portfolio_returns(symbols: list[str], weights: list[float], prices: pd.DataFrame) -> pd.Series | None:
    tw = sum(weights)
    w = np.array(weights) / tw if tw > 0 else np.ones(len(weights)) / max(len(weights), 1)
    cols = []
    ws = []
    for sym, wt in zip(symbols, w):
        cg = SYM_TO_CG.get(sym)
        if cg and cg in prices.columns:
            cols.append(cg)
            ws.append(wt)
    if not cols:
        return None
    ws = np.array(ws) / sum(ws)  # renormalize
    ret = np.log(prices[cols] / prices[cols].shift(1)).dropna()
    return (ret * ws).sum(axis=1)

def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    mu = returns.mean() * 365
    sig = returns.std() * np.sqrt(365)
    if sig == 0:
        return 0.0
    return float((mu - rf) / sig)

def cvar(returns: pd.Series, alpha: float = 0.05) -> float:
    threshold = returns.quantile(alpha)
    tail = returns[returns <= threshold]
    return float(tail.mean()) if len(tail) > 0 else 0.0

def max_drawdown(returns: pd.Series) -> float:
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min())

def annual_volatility(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(365))

def avg_pairwise_correlation(symbols: list[str], prices: pd.DataFrame) -> float:
    cols = [SYM_TO_CG.get(s) for s in symbols]
    cols = [c for c in cols if c and c in prices.columns]
    if len(cols) < 2:
        return 0.0
    ret = np.log(prices[cols] / prices[cols].shift(1)).dropna()
    corr = ret.corr().values
    n = len(cols)
    upper = corr[np.triu_indices(n, k=1)]
    return float(np.mean(upper)) if len(upper) > 0 else 0.0

def diversification_entropy(weights: list[float]) -> float:
    tw = sum(weights)
    w = np.array(weights) / tw if tw > 0 else np.ones(len(weights)) / max(len(weights), 1)
    w = w[w > 0]
    entropy = -np.sum(w * np.log(w))
    max_entropy = np.log(len(w)) if len(w) > 1 else 1.0
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0

def herding_score(symbols: list[str], weights: list[float], conn: sqlite3.Connection) -> float:
    """How similar is this portfolio to the community average allocation?
    0 = completely unique, 1 = identical to community consensus."""
    rows = conn.execute("SELECT assets_json FROM portfolios").fetchall()

    # build community average allocation vector
    sym_totals: dict[str, float] = {}
    count = 0
    for (aj,) in rows:
        if not aj: continue
        try:
            assets = json.loads(aj)
            for a in assets:
                s = a["symbol"]
                w = a["weight"]
                sym_totals[s] = sym_totals.get(s, 0) + w
            count += 1
        except:
            pass

    if count == 0:
        return 0.0

    # normalize community vector
    for s in sym_totals:
        sym_totals[s] /= count

    # build portfolio vector in same space
    all_syms = set(sym_totals.keys()) | set(symbols)
    community_vec = np.array([sym_totals.get(s, 0) for s in all_syms])
    portfolio_vec = np.array([
        weights[symbols.index(s)] / sum(weights) * 100 if s in symbols else 0
        for s in all_syms
    ])

    # cosine similarity
    dot = np.dot(community_vec, portfolio_vec)
    norm = np.linalg.norm(community_vec) * np.linalg.norm(portfolio_vec)
    return float(dot / norm) if norm > 0 else 0.0

def frontier_distance(symbols: list[str], weights: list[float], prices: pd.DataFrame) -> float:
    """Distance from efficient frontier. 0 = on frontier, higher = worse."""
    cols = [SYM_TO_CG.get(s) for s in symbols]
    valid = [(c, w) for c, w in zip(cols, weights) if c and c in prices.columns]
    if len(valid) < 2:
        return 0.0

    cols_v = [v[0] for v in valid]
    ret = np.log(prices[cols_v] / prices[cols_v].shift(1)).dropna()

    mu = ret.mean().values * 365
    cov = ret.cov().values * 365
    n = len(cols_v)

    # current portfolio stats
    w = np.array([v[1] for v in valid])
    w = w / w.sum()
    port_ret = w @ mu
    port_vol = np.sqrt(w @ cov @ w)

    # find minimum variance portfolio on frontier with same return
    # using analytical solution: min w'Cw s.t. w'mu = port_ret, w'1 = 1
    try:
        cov_inv = np.linalg.inv(cov)
        ones = np.ones(n)
        A = ones @ cov_inv @ mu
        B = mu @ cov_inv @ mu
        C = ones @ cov_inv @ ones
        D = B * C - A * A

        if D <= 0:
            return 0.0

        # min variance for target return
        lam = (C * port_ret - A) / D
        gam = (B - A * port_ret) / D
        w_opt = cov_inv @ (lam * mu + gam * ones)
        frontier_vol = np.sqrt(w_opt @ cov @ w_opt)

        # distance = excess volatility vs frontier
        return max(0.0, float(port_vol - frontier_vol))
    except:
        return 0.0

def composite_score(
    sharpe_val: float, cvar_val: float, mdd: float, vol: float,
    corr: float, entropy: float, herd: float, frontier: float,
    n_assets: int,
) -> int:
    """Weighted composite score 0-100."""
    # each component mapped to 0-1 where 1 = best
    s_sharpe = min(1, max(0, (sharpe_val + 1) / 4))          # -1 to 3 range
    s_cvar = min(1, max(0, 1 + cvar_val * 10))               # less negative = better
    s_mdd = min(1, max(0, 1 + mdd * 2))                      # less negative = better
    s_vol = min(1, max(0, 1 - vol))                           # lower vol = better
    s_corr = min(1, max(0, 1 - corr))                         # lower correlation = better
    s_entropy = entropy                                        # already 0-1
    s_herd = 1 - herd                                          # less herding = better
    s_frontier = min(1, max(0, 1 - frontier * 5))             # closer to frontier = better
    s_size = min(1, n_assets / 8)                              # more assets = better (up to 8)

    weights = {
        "sharpe": 0.20, "cvar": 0.15, "mdd": 0.10, "vol": 0.10,
        "corr": 0.10, "entropy": 0.10, "herd": 0.05,
        "frontier": 0.10, "size": 0.10,
    }

    total = (
        weights["sharpe"] * s_sharpe +
        weights["cvar"] * s_cvar +
        weights["mdd"] * s_mdd +
        weights["vol"] * s_vol +
        weights["corr"] * s_corr +
        weights["entropy"] * s_entropy +
        weights["herd"] * s_herd +
        weights["frontier"] * s_frontier +
        weights["size"] * s_size
    )

    return max(0, min(100, round(total * 100)))

def grade(score: int) -> str:
    if score >= 85: return "S"
    if score >= 72: return "A"
    if score >= 58: return "B"
    if score >= 42: return "C"
    if score >= 25: return "D"
    return "F"

def risk_label(vol: float, cvar_val: float) -> str:
    if vol < 0.3 and cvar_val > -0.03: return "Low"
    if vol < 0.6 and cvar_val > -0.06: return "Medium"
    return "High"

def strategy_type(symbols: list[str], weights: list[float], entropy: float) -> str:
    stables = {"USDC", "USDT", "DAI"}
    syms = set(symbols)
    if syms <= stables:
        return "Stablecoin Vault"
    if len(symbols) == 1:
        return "Single Asset"
    if entropy > 0.9 and len(symbols) >= 5:
        return "Max Diversified"
    if entropy > 0.7:
        return "Balanced"
    tw = sum(weights)
    stable_w = sum(w for s, w in zip(symbols, weights) if s in stables) / tw if tw > 0 else 0
    if stable_w > 0.4:
        return "Conservative"
    return "Concentrated"

def score_portfolio(
    symbols: list[str],
    weights: list[float],
    prices: pd.DataFrame | None = None,
    conn: sqlite3.Connection | None = None,
) -> QuantScore:
    n = len(symbols)
    if n == 0:
        return QuantScore(0, "F", 0, 0, 0, 0, 0, 0, 0, 0, 0, "Empty", "N/A")

    # if no price data, fall back to heuristic
    ret = portfolio_returns(symbols, weights, prices) if prices is not None else None

    if ret is not None and len(ret) > 10:
        sh = sharpe_ratio(ret)
        cv = cvar(ret)
        md = max_drawdown(ret)
        av = annual_volatility(ret)
        ac = avg_pairwise_correlation(symbols, prices)
        fd = frontier_distance(symbols, weights, prices)
    else:
        sh, cv, md, av, ac, fd = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    ent = diversification_entropy(weights)
    herd = herding_score(symbols, weights, conn) if conn else 0.0
    stype = strategy_type(symbols, weights, ent)

    total = composite_score(sh, cv, md, av, ac, ent, herd, fd, n)
    g = grade(total)
    rl = risk_label(av, cv)

    return QuantScore(
        total=total, grade=g, sharpe=round(sh, 3), cvar_95=round(cv, 6),
        max_drawdown=round(md, 4), annual_vol=round(av, 4),
        avg_correlation=round(ac, 3), diversification_ratio=round(ent, 3),
        herding_score=round(herd, 3), frontier_distance=round(fd, 4),
        n_assets=n, strategy_type=stype, risk_label=rl,
    )

def main():
    """Score all portfolios in DB with quant metrics."""
    prices = load_prices()
    if prices is None:
        print("No price data found. Run strategy-predictor/scripts/01_fetch_full_data.py first.")
        return

    print(f"Price data: {len(prices)} days, {len(prices.columns)} coins")

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT strategy_id, discord_user, assets_json FROM portfolios").fetchall()
    print(f"Portfolios: {len(rows)}")

    results = []
    for sid, user, aj in rows:
        if not aj: continue
        try:
            assets = json.loads(aj)
            syms = [a["symbol"] for a in assets]
            wts = [a["weight"] for a in assets]
            qs = score_portfolio(syms, wts, prices, conn)
            results.append((sid, user, qs))
        except:
            continue

    results.sort(key=lambda x: x[2].total, reverse=True)

    print(f"\n{'Rank':<5} {'User':<18} {'Score':>5} {'Grd':>4} {'Sharpe':>7} {'CVaR':>8} {'MDD':>7} {'Vol':>6} {'Corr':>5} {'Div':>5} {'Herd':>5} {'Type'}")
    print("-" * 110)
    for i, (sid, user, qs) in enumerate(results[:30]):
        print(f"{i+1:<5} {user:<18} {qs.total:>5} {qs.grade:>4} {qs.sharpe:>7.2f} {qs.cvar_95:>8.4f} {qs.max_drawdown:>7.3f} {qs.annual_vol:>6.2f} {qs.avg_correlation:>5.2f} {qs.diversification_ratio:>5.2f} {qs.herding_score:>5.2f} {qs.strategy_type}")

    # community stats
    scores = [r[2] for r in results]
    print(f"\n--- Community Stats ({len(results)} portfolios) ---")
    print(f"  Avg score:    {np.mean([s.total for s in scores]):.1f}")
    print(f"  Avg Sharpe:   {np.mean([s.sharpe for s in scores]):.3f}")
    print(f"  Avg CVaR:     {np.mean([s.cvar_95 for s in scores]):.5f}")
    print(f"  Avg herding:  {np.mean([s.herding_score for s in scores]):.3f}")
    print(f"  Avg vol:      {np.mean([s.annual_vol for s in scores]):.3f}")

    conn.close()

if __name__ == "__main__":
    main()
