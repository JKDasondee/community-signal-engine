"""
Backtest: Do high-scoring portfolios actually outperform?

Regresses portfolio scores against actual forward returns.
If R² > 0 and p < 0.05, the scoring model has predictive power.

Run: python scripts/backtest_scores.py
"""
import sys
sys.path.insert(0, ".")

import json
import sqlite3
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from cse.quant_score import score_portfolio, load_prices, portfolio_returns

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "portfolios.db"
REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

SYM_TO_CG = {
    "ETH": "ethereum", "WETH": "ethereum", "cbBTC": "wrapped-bitcoin",
    "WBTC": "wrapped-bitcoin", "USDC": "usd-coin", "USDT": "tether",
    "DAI": "dai", "AAVE": "aave", "AERO": "aerodrome-finance",
    "MORPHO": "morpho", "UNI": "uniswap", "LINK": "chainlink",
    "VIRTUAL": "virtual-protocol", "DEGEN": "degen-base",
    "BRETT": "based-brett", "TOSHI": "toshi",
    "uSOL": "solana", "SOL": "solana",
}

def main():
    prices = load_prices()
    if prices is None:
        print("No price data. Run strategy-predictor fetch first.")
        return

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT strategy_id, discord_user, assets_json FROM portfolios").fetchall()
    print(f"Portfolios: {len(rows)}, Price data: {len(prices)} days, {len(prices.columns)} coins")

    results = []
    for sid, user, aj in rows:
        if not aj: continue
        try:
            assets = json.loads(aj)
            syms = [a["symbol"] for a in assets]
            wts = [a["weight"] for a in assets]

            qs = score_portfolio(syms, wts, prices, conn)

            # compute actual returns over multiple horizons
            ret = portfolio_returns(syms, wts, prices)
            if ret is None or len(ret) < 30:
                continue

            r7 = float((1 + ret.iloc[-7:]).prod() - 1)
            r14 = float((1 + ret.iloc[-14:]).prod() - 1)
            r30 = float((1 + ret.iloc[-30:]).prod() - 1)

            results.append({
                "sid": sid, "user": user,
                "score": qs.total, "grade": qs.grade,
                "sharpe": qs.sharpe, "cvar": qs.cvar_95,
                "vol": qs.annual_vol, "herding": qs.herding_score,
                "div": qs.diversification_ratio,
                "r7": r7, "r14": r14, "r30": r30,
                "n_assets": qs.n_assets,
                "strategy_type": qs.strategy_type,
            })
        except:
            continue

    conn.close()
    df = pd.DataFrame(results)
    print(f"Backtest sample: {len(df)} portfolios with return data")

    if len(df) < 10:
        print("Not enough portfolios with return data for regression.")
        return

    # regression: score → returns
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    for horizon, col in [("7d", "r7"), ("14d", "r14"), ("30d", "r30")]:
        mask = ~df[col].isna() & ~df["score"].isna()
        x = df.loc[mask, "score"].values
        y = df.loc[mask, col].values

        if len(x) < 5:
            continue

        slope, intercept, r, p, se = stats.linregress(x, y)
        print(f"\n--- Score vs {horizon} Return ---")
        print(f"  N:         {len(x)}")
        print(f"  R:         {r:.4f}")
        print(f"  R2:        {r**2:.4f}")
        print(f"  p-value:   {p:.6f}")
        print(f"  Slope:     {slope:.6f}")
        print(f"  Intercept: {intercept:.6f}")
        print(f"  Significant (p<0.05): {'YES' if p < 0.05 else 'NO'}")

    # quintile analysis
    print(f"\n--- Quintile Analysis (30d returns) ---")
    df["quintile"] = pd.qcut(df["score"], q=5, labels=["Q1(low)", "Q2", "Q3", "Q4", "Q5(high)"])
    qt = df.groupby("quintile")[["r7", "r14", "r30", "score"]].mean()
    print(qt.round(4).to_string())

    # correlation matrix
    print(f"\n--- Factor Correlations with 30d Return ---")
    factors = ["score", "sharpe", "cvar", "vol", "herding", "div", "n_assets"]
    for f in factors:
        mask = ~df[f].isna() & ~df["r30"].isna()
        if mask.sum() < 5:
            continue
        corr, p = stats.pearsonr(df.loc[mask, f], df.loc[mask, "r30"])
        sig = "*" if p < 0.05 else ""
        print(f"  {f:<12} r={corr:+.4f}  p={p:.4f} {sig}")

    # save results
    df.to_csv(REPORTS / "backtest_results.csv", index=False)
    print(f"\nSaved to {REPORTS / 'backtest_results.csv'}")

    # generate summary for methodology doc
    summary = {
        "n_portfolios": len(df),
        "score_vs_r30": {
            "r": float(stats.linregress(df["score"], df["r30"]).rvalue),
            "r2": float(stats.linregress(df["score"], df["r30"]).rvalue ** 2),
            "p": float(stats.linregress(df["score"], df["r30"]).pvalue),
        },
        "quintile_spread": float(qt["r30"].iloc[-1] - qt["r30"].iloc[0]),
        "mean_score": float(df["score"].mean()),
        "mean_r30": float(df["r30"].mean()),
    }
    with open(REPORTS / "backtest_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
