"""
Fetch live market data for any Glider-listed token via their tRPC API.
Covers everything: crypto, tokenized stocks, memecoins, stablecoins.

Caches results in SQLite to avoid hammering the API.
"""
import json
import time
import sqlite3
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from urllib.parse import quote
from datetime import datetime, timedelta

DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DATA_DIR / "portfolios.db"
BASE = "https://api.glider.fi/v1/trpc"

_price_cache: dict[str, tuple[float, dict]] = {}  # asset_id -> (timestamp, data)
CACHE_TTL = 300  # 5 min

def fetch_market_data(asset_id: str) -> dict | None:
    # check cache
    if asset_id in _price_cache:
        ts, data = _price_cache[asset_id]
        if time.time() - ts < CACHE_TTL:
            return data
    input_json = json.dumps({"json": {"assetId": asset_id}})
    url = f"{BASE}/assets.getBasicMarketDataForAsset?input={quote(input_json)}"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            data = r.json()["result"]["data"]["json"]
            _price_cache[asset_id] = (time.time(), data)
            return data
    except:
        pass
    return None

def get_token_price(asset_id: str) -> dict | None:
    """Get price, 24h change, sparkline for any Glider token."""
    d = fetch_market_data(asset_id)
    if not d:
        return None

    # structure: d["token"] = market data, d["token"]["token"] = metadata
    market = d.get("token", {})
    if not market:
        return None

    meta = market.get("token", {})
    ondo = market.get("ondo", {})

    sparkline = []
    if ondo and "sparkline24h" in ondo:
        sparkline = [(s["timestamp"], s["value"]) for s in ondo["sparkline24h"]]

    symbol = meta.get("symbol", "")
    name = meta.get("name", "")
    token_type = meta.get("explorerData", {}).get("tokenType", "")

    # detect stock from ondo data or token type
    is_stock = bool(ondo) or token_type == "tokenized-equity"
    if ondo:
        tags = ondo.get("tags", [])
        if "stock" in tags or "equities" in tags:
            is_stock = True

    return {
        "asset_id": asset_id,
        "symbol": symbol,
        "name": name,
        "price_usd": float(market.get("priceUSD", 0)),
        "change_1h": float(market.get("change1", 0)),
        "change_24h": float(market.get("change24", 0)),
        "volume_24h": float(market.get("volume24", 0)),
        "market_cap": float(market.get("marketCap", 0)),
        "liquidity": float(market.get("liquidity", 0)),
        "sparkline": sparkline,
        "token_type": "stock" if is_stock else (token_type or "crypto"),
        "is_ondo": bool(ondo),
        "ondo_ticker": ondo.get("ticker", ""),
        "ondo_tags": ondo.get("tags", []),
    }

def fetch_all_portfolio_prices(conn: sqlite3.Connection) -> dict:
    """Fetch prices for all unique assets across all portfolios."""
    rows = conn.execute("SELECT DISTINCT asset_id FROM allocations").fetchall()
    prices = {}
    for (aid,) in rows:
        if not aid or aid.startswith("0x") and len(aid) < 10:
            continue
        p = get_token_price(aid)
        if p:
            prices[aid] = p
            print(f"  {p['symbol']:<10} ${p['price_usd']:<12.4f} 24h: {p['change_24h']:+.2%}  {'[STOCK]' if p['is_ondo'] else ''}")
        time.sleep(0.2)
    return prices

def sparkline_to_returns(sparkline: list[tuple]) -> np.ndarray | None:
    """Convert 24h sparkline to return series."""
    if len(sparkline) < 5:
        return None
    prices = np.array([s[1] for s in sparkline])
    returns = np.diff(np.log(prices))
    return returns

def portfolio_vol_from_sparklines(
    symbols: list[str],
    weights: list[float],
    prices_map: dict,
    asset_id_map: dict,
) -> float:
    """Estimate annualized volatility from 24h sparklines."""
    w = np.array(weights) / sum(weights)
    returns_list = []
    ws = []

    for sym, wt in zip(symbols, w):
        aid = asset_id_map.get(sym, "")
        if aid in prices_map:
            sp = prices_map[aid].get("sparkline", [])
            ret = sparkline_to_returns(sp)
            if ret is not None:
                returns_list.append(ret)
                ws.append(wt)

    if not returns_list:
        return 0.0

    # align lengths
    min_len = min(len(r) for r in returns_list)
    aligned = np.array([r[:min_len] for r in returns_list])
    ws = np.array(ws) / sum(ws)

    port_ret = (aligned.T * ws).sum(axis=1)
    # sparkline is 15-min intervals, ~96 per day, annualize
    vol_15m = np.std(port_ret)
    return float(vol_15m * np.sqrt(96 * 365))

def portfolio_correlation_from_sparklines(
    symbols: list[str],
    prices_map: dict,
    asset_id_map: dict,
) -> float:
    """Estimate average pairwise correlation from sparklines."""
    returns_list = []
    for sym in symbols:
        aid = asset_id_map.get(sym, "")
        if aid in prices_map:
            sp = prices_map[aid].get("sparkline", [])
            ret = sparkline_to_returns(sp)
            if ret is not None:
                returns_list.append(ret)

    if len(returns_list) < 2:
        return 0.0

    min_len = min(len(r) for r in returns_list)
    aligned = np.array([r[:min_len] for r in returns_list])
    corr = np.corrcoef(aligned)
    n = len(returns_list)
    upper = corr[np.triu_indices(n, k=1)]
    return float(np.nanmean(upper))

def enrich_quant_score(
    symbols: list[str],
    weights: list[float],
    asset_ids: list[str],
    prices_map: dict,
) -> dict:
    """Return enriched metrics using Glider live data."""
    aid_map = {sym: aid for sym, aid in zip(symbols, asset_ids)}

    vol = portfolio_vol_from_sparklines(symbols, weights, prices_map, aid_map)
    corr = portfolio_correlation_from_sparklines(symbols, prices_map, aid_map)

    # weighted 24h return
    w = np.array(weights) / sum(weights)
    ret_24h = 0.0
    for sym, wt in zip(symbols, w):
        aid = aid_map.get(sym, "")
        if aid in prices_map:
            ret_24h += prices_map[aid].get("change_24h", 0) * wt

    # total value estimation (if user has amounts — we don't, so skip)
    token_details = []
    for sym, wt in zip(symbols, w):
        aid = aid_map.get(sym, "")
        p = prices_map.get(aid, {})
        token_details.append({
            "symbol": sym,
            "weight": round(wt * 100, 1),
            "price": p.get("price_usd", 0),
            "change_24h": p.get("change_24h", 0),
            "type": "stock" if p.get("is_ondo") else "crypto",
            "liquidity": p.get("liquidity", 0),
        })

    return {
        "vol_annualized": round(vol, 4),
        "avg_correlation": round(corr, 3),
        "return_24h": round(ret_24h, 6),
        "token_details": token_details,
    }

if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    print("Fetching live prices for all portfolio assets...")
    prices = fetch_all_portfolio_prices(conn)
    print(f"\nFetched {len(prices)} token prices")

    # save cache
    cache = {k: {kk: vv for kk, vv in v.items() if kk != "sparkline"} for k, v in prices.items()}
    with open(DATA_DIR / "glider_prices.json", "w") as f:
        json.dump(cache, f, indent=2)
    print(f"Cached to {DATA_DIR / 'glider_prices.json'}")
    conn.close()
