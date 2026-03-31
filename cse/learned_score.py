"""
Learned Portfolio Scoring — replaces hardcoded weights with ML-optimized weights.

Instead of manually setting "Sharpe=20%, CVaR=15%...", we:
1. Compute all 9 factors for each portfolio
2. Use actual forward returns as the target
3. Train XGBoost to learn which factors predict returns
4. Extract feature importances as the optimal weights
5. Use the trained model as the scoring function

This is the difference between "I think Sharpe matters 20%" and
"The data says Sharpe matters 31.4% for predicting 30d returns."

Run: python -m cse.learned_score
"""
import json
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from cse.quant_score import (
    score_portfolio, load_prices, portfolio_returns,
    sharpe_ratio, cvar, max_drawdown, annual_volatility,
    avg_pairwise_correlation, diversification_entropy,
    herding_score, frontier_distance, QuantScore,
)

DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DATA_DIR / "portfolios.db"
REPORTS = Path(__file__).parent.parent / "reports"

@dataclass
class LearnedScore:
    total: int
    grade: str
    factors: dict[str, float]
    factor_contributions: dict[str, float]
    predicted_return: float
    confidence: float
    model_r2: float

def compute_factor_matrix(conn: sqlite3.Connection, prices: pd.DataFrame) -> pd.DataFrame:
    rows = conn.execute("SELECT strategy_id, discord_user, assets_json FROM portfolios").fetchall()

    records = []
    for sid, user, aj in rows:
        if not aj: continue
        try:
            assets = json.loads(aj)
            syms = [a["symbol"] for a in assets]
            wts = [a["weight"] for a in assets]
            if not syms or not wts: continue

            ret = portfolio_returns(syms, wts, prices)
            if ret is None or len(ret) < 30: continue

            w = np.array(wts) / sum(wts)
            n = len(syms)

            r7 = float((1 + ret.iloc[-7:]).prod() - 1)
            r14 = float((1 + ret.iloc[-14:]).prod() - 1)
            r30 = float((1 + ret.iloc[-30:]).prod() - 1)

            sh = sharpe_ratio(ret)
            cv = cvar(ret)
            md = max_drawdown(ret)
            av = annual_volatility(ret)
            ac = avg_pairwise_correlation(syms, prices)
            de = diversification_entropy(wts)
            hs = herding_score(syms, wts, conn)
            fd = frontier_distance(syms, wts, prices)
            hhi = float(np.sum(w ** 2))

            records.append({
                "sid": sid, "user": user,
                "f_sharpe": sh, "f_cvar": cv, "f_mdd": md,
                "f_vol": av, "f_corr": ac, "f_entropy": de,
                "f_herding": hs, "f_frontier": fd, "f_hhi": hhi,
                "f_n_assets": n,
                "r7": r7, "r14": r14, "r30": r30,
            })
        except Exception:
            continue

    return pd.DataFrame(records)

def train_scoring_model(df: pd.DataFrame, target: str = "r30") -> tuple:
    if not HAS_XGB:
        raise ImportError("pip install xgboost")

    feat_cols = [c for c in df.columns if c.startswith("f_")]
    X = df[feat_cols].values
    y = df[target].values

    mask = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
    X, y = X[mask], y[mask]

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    cv_scores = []

    for tr_idx, te_idx in tscv.split(X_s):
        model = xgb.XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42,
        )
        model.fit(X_s[tr_idx], y[tr_idx],
                  eval_set=[(X_s[te_idx], y[te_idx])],
                  verbose=False)
        pred = model.predict(X_s[te_idx])
        cv_scores.append(r2_score(y[te_idx], pred))

    # final model on all data
    final_model = xgb.XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42,
    )
    final_model.fit(X_s, y, verbose=False)

    # feature importance as learned weights
    imp = final_model.feature_importances_
    weights = dict(zip(feat_cols, imp / imp.sum()))

    return final_model, scaler, weights, np.mean(cv_scores), feat_cols

def learned_score(
    symbols: list[str], weights: list[float],
    model, scaler, feat_cols: list[str],
    learned_weights: dict, model_r2: float,
    prices: pd.DataFrame = None, conn = None,
) -> LearnedScore:
    if not symbols:
        return LearnedScore(0, "F", {}, {}, 0.0, 0.0, model_r2)

    w = np.array(weights) / sum(weights)
    n = len(symbols)

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

    de = diversification_entropy(weights)
    hs = herding_score(symbols, weights, conn) if conn else 0.0
    hhi = float(np.sum(w ** 2))

    factors = {
        "f_sharpe": sh, "f_cvar": cv, "f_mdd": md,
        "f_vol": av, "f_corr": ac, "f_entropy": de,
        "f_herding": hs, "f_frontier": fd, "f_hhi": hhi,
        "f_n_assets": float(n),
    }

    # predict return
    X = np.array([[factors[c] for c in feat_cols]])
    X_s = scaler.transform(X)
    pred_return = float(model.predict(X_s)[0])

    # contribution of each factor
    contribs = {}
    for col in feat_cols:
        contribs[col] = round(factors[col] * learned_weights.get(col, 0), 6)

    # score: normalize predicted return to 0-100
    # use sigmoid-like mapping: score = 100 / (1 + exp(-k * pred_return))
    k = 30  # steepness
    raw = 1 / (1 + np.exp(-k * pred_return))
    total = max(0, min(100, round(raw * 100)))

    if total >= 85: grade = "S"
    elif total >= 72: grade = "A"
    elif total >= 58: grade = "B"
    elif total >= 42: grade = "C"
    elif total >= 25: grade = "D"
    else: grade = "F"

    # confidence from CV R²
    confidence = max(0, min(1, model_r2))

    return LearnedScore(
        total=total, grade=grade, factors=factors,
        factor_contributions=contribs, predicted_return=round(pred_return, 6),
        confidence=round(confidence, 4), model_r2=round(model_r2, 4),
    )

def main():
    prices = load_prices()
    if prices is None:
        print("No price data.")
        return

    conn = sqlite3.connect(DB_PATH)
    print("Computing factor matrix...")
    df = compute_factor_matrix(conn, prices)
    print(f"Samples: {len(df)}")

    if len(df) < 20:
        print("Not enough data for ML scoring.")
        return

    print("Training scoring model...")
    model, scaler, weights, cv_r2, feat_cols = train_scoring_model(df)

    print(f"\nCross-validated R2: {cv_r2:.4f}")
    print(f"\nLEARNED FACTOR WEIGHTS (what actually predicts 30d returns):")
    print(f"{'Factor':<15} {'Weight':>8} {'Hardcoded':>10}")
    hardcoded = {
        "f_sharpe": 0.20, "f_cvar": 0.15, "f_mdd": 0.10,
        "f_vol": 0.10, "f_corr": 0.10, "f_entropy": 0.10,
        "f_herding": 0.05, "f_frontier": 0.10, "f_n_assets": 0.10,
    }
    for k, v in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        hc = hardcoded.get(k, 0)
        diff = ">>>" if abs(v - hc) > 0.05 else ""
        print(f"  {k:<15} {v:>7.1%} {hc:>9.0%} {diff}")

    # save model and weights
    import pickle
    model_path = DATA_DIR / "scoring_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "weights": weights,
                      "cv_r2": cv_r2, "feat_cols": feat_cols}, f)
    print(f"\nModel saved to {model_path}")

    # test scoring on sample portfolios
    print(f"\nSample scores using LEARNED model:")
    test = [
        (["ETH", "USDC", "cbBTC", "AAVE", "AERO"], [20, 20, 20, 20, 20]),
        (["ETH"], [100]),
        (["USDC", "USDT"], [50, 50]),
        (["ETH", "cbBTC"], [50, 50]),
    ]
    for syms, wts in test:
        ls = learned_score(syms, wts, model, scaler, feat_cols, weights, cv_r2, prices, conn)
        print(f"  {', '.join(syms):<35} Score:{ls.total:>3} ({ls.grade}) Pred:{ls.predicted_return:+.4f} Conf:{ls.confidence:.2f}")

    # save weights comparison for methodology doc
    comparison = {
        "learned": {k: float(v) for k, v in weights.items()},
        "hardcoded": hardcoded,
        "cv_r2": float(cv_r2),
        "n_samples": len(df),
    }
    REPORTS.mkdir(exist_ok=True)
    with open(REPORTS / "learned_weights.json", "w") as f:
        json.dump(comparison, f, indent=2)

    conn.close()

if __name__ == "__main__":
    main()
