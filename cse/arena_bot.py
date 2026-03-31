"""
Portfolio Arena v2 — Discord bot that scores Glider portfolio links.
Uses quant scoring engine + Glider live prices + Ollama AI tips.

Run: python -m cse.arena_bot
"""
import os
import re
import json
import asyncio
import sqlite3
import numpy as np
import requests as rq
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

import discord
from discord import Intents

from cse.glider_api import parse_portfolio, save_portfolio, init_db, fetch_token_list, save_tokens
from cse.quant_score import score_portfolio as quant_score, load_prices, QuantScore
from cse.glider_prices import get_token_price, enrich_quant_score
from cse.fast_rank import RankIndex
from cse.fast_parse import fetch_portfolio_concurrent, fetch_prices_concurrent

DATA_DIR = Path(__file__).parent.parent / "data"
PORTFOLIO_RE = re.compile(r'glider\.fi/portfolio/([a-z0-9]+)')
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")

GLIDER_GREEN = 0x3BF292
ICON = "https://glider.fi/apple-touch-icon.png"

ALLOWED_CHANNELS: set[int] = {
    # 1417494318658228265,  # #share-portfolio — enable when ready
}
ALLOW_DMS = True

TOKEN_ICON = {
    "ETH": "\u039e", "WETH": "\u039e", "cbBTC": "\u20bf", "WBTC": "\u20bf",
    "cbETH": "\u039e", "USDC": "$", "USDT": "$", "DAI": "$", "USDbC": "$",
}

def get_ollama_tip(name, syms, wts, qs):
    try:
        alloc = ", ".join(f"{s} {w:.0f}%" for s, w in zip(syms, wts))
        prompt = (
            f"You're a DeFi portfolio analyst on Discord. Give one specific, actionable tip "
            f"about this portfolio in 1-2 sentences. Be encouraging but direct. "
            f"Mention specific token names. No disclaimers. Under 40 words.\n\n"
            f"Portfolio '{name}': {alloc}\n"
            f"Score: {qs.total}/100, Type: {qs.strategy_type}, Risk: {qs.risk_label}\n"
            f"Sharpe: {qs.sharpe:.2f}, Vol: {qs.annual_vol:.1%}, Herding: {qs.herding_score:.0%}"
        )
        r = rq.post(
            OLLAMA_URL.rstrip("/").replace("/v1/", "/v1") + "/chat/completions",
            json={"model": OLLAMA_MODEL, "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 300, "temperature": 0.7},
            timeout=30,
        )
        if r.status_code == 200:
            msg = r.json()["choices"][0]["message"]
            # qwen3 puts answer in content, reasoning in separate field
            tip = msg.get("content", "").strip()
            if not tip:
                # some models return reasoning only — extract from there
                reasoning = msg.get("reasoning", "")
                if reasoning:
                    # take last sentence of reasoning as tip
                    sentences = [s.strip() for s in reasoning.split(".") if len(s.strip()) > 10]
                    tip = sentences[-1] + "." if sentences else ""
            if tip:
                return tip
    except Exception:
        pass
    # fallback tips
    if qs.n_assets == 1:
        return "Single asset = max conviction. Splitting into 3-4 assets could cut risk without losing much upside."
    if qs.annual_vol > 0.6:
        return "High volatility portfolio. A 10-20% stablecoin allocation would smooth your equity curve."
    if qs.herding_score > 0.6:
        return "Your picks overlap with most of the community. One contrarian position could set you apart."
    if qs.diversification_ratio > 0.9:
        return "Great diversification. Consider slightly overweighting your highest-conviction asset."
    return "Solid construction. Watch correlation between your assets during market drawdowns."

def score_bar(s):
    filled = round(s / 10)
    return "\U0001f7e9" * filled + "\u2b1c" * (10 - filled)

def grade_label(g):
    m = {"S": "\u2728 S-Tier", "A": "\U0001f525 Grade A", "B": "\u2705 Grade B",
         "C": "\u26a0\ufe0f Grade C", "D": "\U0001f534 Grade D", "F": "\U0001f4a9 Grade F"}
    return m.get(g, g)

def risk_emoji(r):
    return {"\U0001f7e2": "Low", "\U0001f7e1": "Medium", "\U0001f534": "High"}.get(r, r)

def community_rank(score, conn, prices):
    rows = conn.execute("SELECT assets_json FROM portfolios").fetchall()
    scores = []
    for (aj,) in rows:
        if not aj: continue
        try:
            a = json.loads(aj)
            s = quant_score([x["symbol"] for x in a], [x["weight"] for x in a], prices, conn)
            scores.append(s.total)
        except Exception:
            pass
    scores.sort(reverse=True)
    return sum(1 for s in scores if s > score) + 1, max(len(scores), 1)

def find_similar(conn, syms, current_id, limit=3):
    rows = conn.execute("SELECT strategy_id, discord_user, assets_json FROM portfolios").fetchall()
    similar = []
    for sid, user, aj in rows:
        if sid == current_id or not aj: continue
        try:
            assets = json.loads(aj)
            other = {a["symbol"] for a in assets}
            overlap = len(syms & other) / max(len(syms | other), 1)
            if overlap >= 0.5:
                similar.append((user, round(overlap * 100)))
        except Exception:
            pass
    similar.sort(key=lambda x: x[1], reverse=True)
    return similar[:limit]

def engagement_hook(qs, rank, total):
    s = qs.total
    pct = round((1 - rank / total) * 100)
    if s >= 80:
        return "\U0001f451 Top-tier portfolio. Share this and flex on the community."
    if s >= 70 and rank <= 10:
        return f"\U0001f3c6 You're in the **Top 10** out of {total} portfolios!"
    if qs.n_assets >= 6:
        return "\U0001f9ec Running more assets than 90% of the community. Nice diversification."
    if qs.strategy_type == "Stablecoin Vault":
        return "\U0001f9ca Safe play. Try adding a small ETH position to climb the ranks."
    if qs.n_assets == 1:
        return "\U0001f3b0 Full send! Can you beat the diversified players?"
    if 40 <= s <= 60:
        return f"\U0001f4aa **{100 - pct}%** away from the top. One rebalance could change your rank."
    if pct < 50:
        return f"\u26a1 **{pct}%** of the community scores higher \u2014 rebalance and share again!"
    return "\U0001f680 Share your portfolio to see how you stack up!"

def build_embed(portfolio, qs, rank, total, tip, conn, live):
    s = qs.total
    g = qs.grade

    if s >= 72: color = GLIDER_GREEN
    elif s >= 58: color = 0x5865F2
    elif s >= 42: color = 0xFEE75C
    else: color = 0xED4245

    name = portfolio.blueprint_name or portfolio.strategy_id

    embed = discord.Embed(
        title=f"\U0001f3af  {name}",
        url=f"https://glider.fi/portfolio/{portfolio.strategy_id}",
        color=color,
    )
    embed.set_author(name="Portfolio Arena \u2022 Glider", icon_url=ICON)
    embed.description = f"\n> **{s}** / 100  {score_bar(s)}  {grade_label(g)}\n"

    # allocation with live prices
    td = {t["symbol"]: t for t in (live or {}).get("token_details", [])}
    lines = []
    active_assets = [a for a in portfolio.assets if a.weight > 0]
    historical = [a for a in portfolio.assets if a.weight <= 0]
    for a in active_assets[:8]:
        icon = TOKEN_ICON.get(a.symbol, "\u25c8")
        sym = a.symbol if len(a.symbol) <= 7 else a.symbol[:6] + "."
        det = td.get(a.symbol, {})
        pr = det.get("price", 0)
        chg = det.get("change_24h", 0)
        is_stock = det.get("type") == "stock"

        if pr >= 1000:
            ps = f"${pr:,.0f}"
        elif pr >= 1:
            ps = f"${pr:.2f}"
        elif pr > 0:
            ps = f"${pr:.4f}"
        else:
            ps = ""

        arr = "\u25b2" if chg > 0 else "\u25bc" if chg < 0 else ""
        chg_s = f"{arr}{abs(chg):.1%}" if chg != 0 else ""
        tag = " \U0001f4c8" if is_stock else ""

        if ps:
            lines.append(f"{icon} {sym:<7} {a.weight:>3.0f}%  {ps:<11} {chg_s}{tag}")
        else:
            lines.append(f"{icon} {sym:<7} {a.weight:>3.0f}%")

    if lines:
        embed.add_field(name="\u200b", value="```\n" + "\n".join(lines) + "\n```", inline=False)

    if historical:
        hist_syms = ", ".join(a.symbol for a in historical[:5] if a.symbol)
        if hist_syms:
            embed.add_field(name="\u200b", value=f"*Previously held: {hist_syms}*", inline=False)

    # 24h portfolio return
    r24 = (live or {}).get("return_24h", 0)
    if r24 != 0:
        arr = "\u25b2" if r24 > 0 else "\u25bc"
        embed.add_field(name="\u200b", value=f"24h Portfolio: **{arr} {abs(r24):.2%}**", inline=False)

    # strategy / risk / rank
    embed.add_field(
        name="\U0001f4ca Strategy",
        value=f"**{qs.strategy_type}**\n{qs.n_assets} assets",
        inline=True,
    )

    risk_text = f"**{qs.risk_label}**"
    if qs.sharpe != 0:
        risk_text += f"\nSharpe: {qs.sharpe:.2f}"
    elif qs.annual_vol > 0:
        risk_text += f"\nVol: {qs.annual_vol:.1%}"
    embed.add_field(name="\U0001f6e1\ufe0f Risk", value=risk_text, inline=True)

    pct = round((1 - rank / total) * 100)
    embed.add_field(name="\U0001f3c6 Rank", value=f"**#{rank}** of {total}\nTop **{pct}%**", inline=True)

    # quant metrics (clean labels)
    metrics = []
    if qs.cvar_95 != 0:
        metrics.append(f"Worst-case daily loss: **{qs.cvar_95:.2%}**")
    if qs.max_drawdown != 0:
        metrics.append(f"Max drawdown: **{qs.max_drawdown:.1%}**")
    div = abs(qs.diversification_ratio)
    metrics.append(f"Diversification: **{div:.0%}**")
    if qs.herding_score > 0:
        metrics.append(f"Uniqueness: **{100 - qs.herding_score * 100:.0f}%**")
    if metrics:
        embed.add_field(name="\U0001f9ee Insights", value="\n".join(metrics), inline=False)

    # AI tip
    if tip:
        embed.add_field(name="\U0001f4a1 Tip", value=f"*{tip}*", inline=False)

    # engagement hook
    hook = engagement_hook(qs, rank, total)
    embed.add_field(name="\u200b", value=hook, inline=False)

    # similar portfolios
    syms_set = {a.symbol for a in portfolio.assets}
    similar = find_similar(conn, syms_set, portfolio.strategy_id)
    if similar:
        sim_text = "\n".join(f"\u2022 **{user}** ({pct}% match)" for user, pct in similar)
        embed.add_field(name="\U0001f465 Similar Portfolios", value=sim_text, inline=False)

    embed.set_footer(text="Portfolio Arena \u2022 glider.fi \u2022 Not financial advice", icon_url=ICON)
    embed.timestamp = datetime.now(timezone.utc)
    return embed

class ArenaBot(discord.Client):
    def __init__(self):
        intents = Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self.conn = init_db()
        self.responded: set[int] = set()
        self.boot_time = datetime.now(timezone.utc)
        self.prices = load_prices()
        self.rank_idx = RankIndex()
        tokens = fetch_token_list()
        if tokens:
            save_tokens(self.conn, tokens)

    async def on_ready(self):
        print(f"Arena bot online: {self.user}")
        print(f"Guilds: {[g.name for g in self.guilds]}")
        db_count = self.conn.execute("SELECT COUNT(*) FROM portfolios").fetchone()[0]
        print(f"DB: {db_count} portfolios | Prices: {'loaded' if self.prices is not None else 'NONE'}")

        # precompute rank index O(P) once at startup
        import time
        t0 = time.perf_counter()
        def quick_score(syms, wts):
            return quant_score(syms, wts, self.prices, self.conn).total
        self.rank_idx.build(self.conn, quick_score)
        t1 = time.perf_counter()
        print(f"Rank index: {self.rank_idx.total} portfolios precomputed in {(t1-t0)*1000:.0f}ms")
        print(f"Mode: {'DM only' if not ALLOWED_CHANNELS else f'channels: {ALLOWED_CHANNELS}'}")

    async def on_message(self, message):
        if message.author == self.user or message.author.bot:
            return

        is_dm = isinstance(message.channel, discord.DMChannel)
        if is_dm and not ALLOW_DMS:
            return
        if not is_dm and message.channel.id not in ALLOWED_CHANNELS:
            return

        if message.created_at < self.boot_time:
            return

        if message.id in self.responded:
            return

        match = PORTFOLIO_RE.search(message.content)
        if not match:
            return

        self.responded.add(message.id)
        if len(self.responded) > 500:
            self.responded = set(list(self.responded)[-250:])

        sid = match.group(1)
        print(f"  [{message.author}] scoring {sid}...")

        try:
            async with message.channel.typing():
                import time as _t
                t0 = _t.perf_counter()

                # STEP 1: concurrent API fetch (strategy + history + pnl in parallel)
                p = await asyncio.to_thread(parse_portfolio, sid, self.conn)
                if not p:
                    return
                t1 = _t.perf_counter()

                p.discord_user = str(message.author)
                save_portfolio(self.conn, p)

                active = [a for a in p.assets if a.weight > 0]
                syms = [a.symbol for a in active]
                wts = [a.weight for a in active]
                aids = [a.asset_id for a in active]

                # STEP 2: concurrent price fetch (all tokens in parallel)
                live_prices = await fetch_prices_concurrent(aids)
                t2 = _t.perf_counter()

                # STEP 3: score (pure CPU, ~6ms)
                qs = quant_score(syms, wts, self.prices, self.conn)

                live = enrich_quant_score(syms, wts, aids, live_prices)
                if live["vol_annualized"] > 0 and qs.annual_vol == 0:
                    qs.annual_vol = live["vol_annualized"]
                if live["avg_correlation"] != 0 and qs.avg_correlation == 0:
                    qs.avg_correlation = live["avg_correlation"]
                t3 = _t.perf_counter()

                # STEP 4: O(log N) rank lookup instead of O(N) rescore
                rank, total = self.rank_idx.query(qs.total)
                self.rank_idx.insert(qs.total)  # update for next query
                t4 = _t.perf_counter()

                # STEP 5: AI tip (async, ~200ms cached / ~2s uncached)
                tip = await asyncio.to_thread(get_ollama_tip, p.blueprint_name or sid, syms, wts, qs)
                t5 = _t.perf_counter()

                embed = build_embed(p, qs, rank, total, tip, self.conn, live)
                await message.reply(embed=embed, mention_author=False)
                t6 = _t.perf_counter()

                print(f"  scored: {qs.total}/100 ({qs.grade}) "
                      f"[parse:{(t1-t0)*1000:.0f} price:{(t2-t1)*1000:.0f} "
                      f"score:{(t3-t2)*1000:.0f} rank:{(t4-t3)*1000:.3f} "
                      f"tip:{(t5-t4)*1000:.0f} send:{(t6-t5)*1000:.0f} "
                      f"total:{(t6-t0)*1000:.0f}ms]")

        except Exception as e:
            print(f"  error: {e}")
            import traceback
            traceback.print_exc()

def main():
    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        print("Set DISCORD_BOT_TOKEN in .env")
        return
    pid_file = DATA_DIR / "arena_bot.pid"
    pid_file.write_text(str(os.getpid()))
    print(f"Bot PID: {os.getpid()}")
    ArenaBot().run(token)

if __name__ == "__main__":
    main()
