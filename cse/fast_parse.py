"""
Concurrent API fetching for portfolio parsing.
Fetches strategy + history + PnL + prices in parallel using asyncio.

CP technique: batch concurrent IO to minimize wall-clock time.
"""
import asyncio
import json
import aiohttp
from urllib.parse import quote
from typing import Optional

BASE = "https://api.glider.fi/v1/trpc"

async def _fetch(session: aiohttp.ClientSession, endpoint: str, params: dict) -> dict | None:
    input_json = json.dumps({"json": params})
    url = f"{BASE}/{endpoint}?input={quote(input_json)}"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
            if r.status == 200:
                return await r.json()
    except:
        pass
    return None

async def fetch_portfolio_concurrent(sid: str) -> dict:
    """Fetch strategy + history + PnL in parallel. ~1 RTT instead of 3."""
    async with aiohttp.ClientSession() as session:
        strat_task = _fetch(session, "strategyInstances.getStrategyInstance",
                           {"strategyInstanceId": sid})
        hist_task = _fetch(session, "strategyInstances.getAllAssetsInStrategyInstanceHistory",
                          {"strategyInstanceId": sid})
        pnl_task = _fetch(session, "pnl.getCostBasisAndPnL",
                         {"includeTrades": True, "realtime": True, "strategyInstanceId": sid})

        strat, hist, pnl = await asyncio.gather(strat_task, hist_task, pnl_task)

    return {"strategy": strat, "history": hist, "pnl": pnl}

async def fetch_prices_concurrent(asset_ids: list[str]) -> dict:
    """Fetch N token prices in parallel. ~1 RTT instead of N."""
    async with aiohttp.ClientSession() as session:
        tasks = {}
        for aid in asset_ids:
            if aid:
                tasks[aid] = _fetch(session, "assets.getBasicMarketDataForAsset",
                                   {"assetId": aid})

        results = await asyncio.gather(*tasks.values())

    out = {}
    for aid, result in zip(tasks.keys(), results):
        if result:
            try:
                market = result["result"]["data"]["json"].get("token", {})
                meta = market.get("token", {})
                ondo = market.get("ondo", {})
                is_stock = bool(ondo) or meta.get("explorerData", {}).get("tokenType", "") == "tokenized-equity"

                sparkline = []
                if ondo and "sparkline24h" in ondo:
                    sparkline = [(s["timestamp"], s["value"]) for s in ondo["sparkline24h"]]

                out[aid] = {
                    "asset_id": aid,
                    "symbol": meta.get("symbol", ""),
                    "name": meta.get("name", ""),
                    "price_usd": float(market.get("priceUSD", 0)),
                    "change_1h": float(market.get("change1", 0)),
                    "change_24h": float(market.get("change24", 0)),
                    "volume_24h": float(market.get("volume24", 0)),
                    "market_cap": float(market.get("marketCap", 0)),
                    "liquidity": float(market.get("liquidity", 0)),
                    "sparkline": sparkline,
                    "token_type": "stock" if is_stock else "crypto",
                    "is_ondo": bool(ondo),
                    "ondo_ticker": ondo.get("ticker", ""),
                    "ondo_tags": ondo.get("tags", []),
                }
            except:
                pass
    return out
