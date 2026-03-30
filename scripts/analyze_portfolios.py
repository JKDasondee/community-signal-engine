"""
Analyze all scraped Glider portfolios using strategy-predictor ML pipeline.
Combines: CSE scraped data + strategy-predictor risk metrics.

Run: python scripts/analyze_portfolios.py
"""
import sys
import json
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "strategy-predictor"))

CSE_DB = Path(__file__).parent.parent / "data" / "portfolios.db"
SP_DATA = Path(__file__).parent.parent.parent / "strategy-predictor" / "data"
OUT = Path(__file__).parent.parent / "reports"
OUT.mkdir(exist_ok=True)

def load_portfolios():
    conn = sqlite3.connect(CSE_DB)
    portfolios = conn.execute("""
        SELECT p.strategy_id, p.discord_user, p.blueprint_name,
               p.current_value_usd, p.total_cost_basis, p.realized_pnl
        FROM portfolios p
    """).fetchall()

    allocs = conn.execute("""
        SELECT strategy_id, symbol, weight_pct FROM allocations
    """).fetchall()
    conn.close()

    alloc_map = {}
    for sid, sym, w in allocs:
        alloc_map.setdefault(sid, []).append((sym, w))

    return portfolios, alloc_map

def concentration_risk(weights):
    """HHI index: 1/n = perfectly diversified, 1 = single asset"""
    if not weights:
        return 1.0
    w = np.array([x / 100 for x in weights])
    w = w / w.sum()
    return float(np.sum(w ** 2))

def correlation_risk(symbols):
    """Estimate correlation risk based on known asset clusters"""
    clusters = {
        "btc_like": {"ETH", "WETH", "cbBTC", "WBTC"},
        "stables": {"USDC", "USDT", "DAI"},
        "base_defi": {"AERO", "MORPHO", "DEGEN", "BRETT"},
        "meme": {"OWB", "KEYCAT", "TOSHI", "DEGEN"},
        "rwa": {"CVXON", "PBRON", "COPON", "OXYON", "XOMON"},
    }
    syms = set(symbols)
    max_overlap = 0
    for cluster_name, cluster_syms in clusters.items():
        overlap = len(syms & cluster_syms)
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap / max(len(syms), 1)

def risk_score(hhi, corr_risk, n_assets):
    """0-100 risk score. Higher = riskier."""
    div_penalty = hhi * 40  # 0-40 points for concentration
    corr_penalty = corr_risk * 30  # 0-30 points for correlation
    size_penalty = max(0, (1 - n_assets / 6)) * 30  # 0-30 for few assets
    return min(100, round(div_penalty + corr_penalty + size_penalty))

def classify_strategy(symbols, weights):
    syms = set(symbols)
    stables = {"USDC", "USDT", "DAI"}
    if syms <= stables:
        return "stablecoin"
    if len(syms) == 1:
        return "single-asset-degen"
    stable_weight = sum(w for s, w in zip(symbols, weights) if s in stables)
    if stable_weight > 50:
        return "conservative"
    if len(syms) >= 5:
        return "diversified"
    return "balanced"

def main():
    portfolios, alloc_map = load_portfolios()
    print(f"Analyzing {len(portfolios)} portfolios...\n")

    results = []
    for sid, user, name, value, cost, rpnl in portfolios:
        allocs = alloc_map.get(sid, [])
        symbols = [s for s, w in allocs]
        weights = [w for s, w in allocs]

        hhi = concentration_risk(weights)
        cr = correlation_risk(symbols)
        rs = risk_score(hhi, cr, len(allocs))
        strat_type = classify_strategy(symbols, weights)

        results.append({
            "strategy_id": sid,
            "discord_user": user,
            "name": name,
            "n_assets": len(allocs),
            "assets": ", ".join(f"{s}({w:.0f}%)" for s, w in allocs),
            "hhi": round(hhi, 3),
            "correlation_risk": round(cr, 2),
            "risk_score": rs,
            "strategy_type": strat_type,
            "value_usd": value,
            "cost_basis": cost,
            "realized_pnl": rpnl,
        })

    # Sort by risk score
    results.sort(key=lambda x: x["risk_score"], reverse=True)

    # Print results
    print(f"{'User':<18} {'Type':<20} {'Assets':>3} {'HHI':>5} {'Risk':>5} {'Allocation'}")
    print("-" * 100)
    for r in results:
        print(f"{r['discord_user']:<18} {r['strategy_type']:<20} {r['n_assets']:>3} {r['hhi']:>5.2f} {r['risk_score']:>5} {r['assets'][:50]}")

    # Community-wide stats
    print(f"\n{'='*60}")
    print("COMMUNITY PORTFOLIO INTELLIGENCE")
    print(f"{'='*60}")

    all_syms = []
    for sid, allocs in alloc_map.items():
        for sym, w in allocs:
            all_syms.append(sym)

    sym_counts = Counter(all_syms)
    print(f"\nTop assets by popularity:")
    for sym, cnt in sym_counts.most_common(10):
        print(f"  {sym:<10} in {cnt} portfolios")

    types = Counter(r["strategy_type"] for r in results)
    print(f"\nStrategy type distribution:")
    for t, cnt in types.most_common():
        print(f"  {t:<25} {cnt} portfolios ({cnt/len(results)*100:.0f}%)")

    avg_risk = np.mean([r["risk_score"] for r in results])
    avg_assets = np.mean([r["n_assets"] for r in results])
    avg_hhi = np.mean([r["hhi"] for r in results])
    print(f"\nCommunity averages:")
    print(f"  Risk score:    {avg_risk:.1f}/100")
    print(f"  Assets/port:   {avg_assets:.1f}")
    print(f"  HHI (conc.):   {avg_hhi:.3f}")

    # Alerts
    print(f"\nRISK ALERTS:")
    high_risk = [r for r in results if r["risk_score"] >= 70]
    for r in high_risk:
        print(f"  [{r['risk_score']}] {r['discord_user']}: {r['strategy_type']} — {r['assets'][:60]}")

    # Save
    with open(OUT / "portfolio_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate markdown report
    md = ["# Glider Community Portfolio Intelligence Report", ""]
    md.append(f"**{len(results)} portfolios analyzed** | Generated automatically from Discord-shared strategies\n")
    md.append("## Risk Leaderboard\n")
    md.append(f"| User | Type | Assets | Risk | Allocation |")
    md.append(f"|------|------|--------|------|------------|")
    for r in results:
        md.append(f"| {r['discord_user']} | {r['strategy_type']} | {r['n_assets']} | {r['risk_score']}/100 | {r['assets'][:40]} |")
    md.append(f"\n## Community Stats\n")
    md.append(f"- Average risk score: **{avg_risk:.1f}/100**")
    md.append(f"- Average assets per portfolio: **{avg_assets:.1f}**")
    md.append(f"- Most popular asset: **{sym_counts.most_common(1)[0][0]}** ({sym_counts.most_common(1)[0][1]} portfolios)")
    md.append(f"- Highest risk: **{results[0]['discord_user']}** ({results[0]['risk_score']}/100, {results[0]['strategy_type']})")
    md.append(f"\n## Risk Alerts\n")
    for r in high_risk:
        md.append(f"- **{r['discord_user']}**: Risk {r['risk_score']}/100 — {r['strategy_type']}, {r['assets'][:50]}")

    report_path = OUT / "portfolio_intelligence.md"
    report_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\nReport saved to {report_path}")
    print(f"JSON saved to {OUT / 'portfolio_analysis.json'}")

if __name__ == "__main__":
    main()
