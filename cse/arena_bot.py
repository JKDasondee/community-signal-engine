"""
Portfolio Arena — Discord bot that auto-scores every shared Glider portfolio.
Scrapes portfolio via tRPC API, scores it, replies with a risk card.

Setup:
  1. Create Discord bot at discord.com/developers
  2. Enable MESSAGE_CONTENT intent
  3. Set DISCORD_BOT_TOKEN env var
  4. python -m cse.arena_bot

Requires: pip install discord.py
"""
import os
import re
import json
import asyncio
import sqlite3
import numpy as np
from datetime import datetime
from pathlib import Path

try:
    import discord
    from discord import Intents
except ImportError:
    print("pip install discord.py")
    exit(1)

from cse.glider_api import parse_portfolio, save_portfolio, init_db, fetch_token_list, save_tokens

DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DATA_DIR / "portfolios.db"
PORTFOLIO_RE = re.compile(r'glider\.fi/portfolio/([a-z0-9]+)')

CLUSTERS = {
    "blue_chip": {"ETH", "WETH", "cbBTC", "WBTC"},
    "stables": {"USDC", "USDT", "DAI"},
    "base_defi": {"AERO", "MORPHO", "COMP", "UNI"},
    "meme": {"OWB", "KEYCAT", "TOSHI", "DEGEN", "BRETT"},
    "rwa": {"CVXON", "PBRON", "COPON", "OXYON", "XOMON"},
}

def score_portfolio(symbols: list[str], weights: list[float]) -> dict:
    n = len(symbols)
    if n == 0:
        return {"score": 0, "grade": "F", "details": "empty portfolio"}

    w = np.array(weights) / sum(weights)

    # HHI
    hhi = float(np.sum(w ** 2))

    # cluster overlap
    sym_set = set(symbols)
    max_cluster_pct = 0
    dominant_cluster = ""
    for cname, csyms in CLUSTERS.items():
        overlap = sym_set & csyms
        if overlap:
            cluster_w = sum(w[i] for i, s in enumerate(symbols) if s in csyms)
            if cluster_w > max_cluster_pct:
                max_cluster_pct = cluster_w
                dominant_cluster = cname

    # diversification score (0-40)
    div_score = (1 - hhi) * 40

    # decorrelation score (0-30)
    n_clusters = len({c for c, syms in CLUSTERS.items() if sym_set & syms})
    decorr_score = min(30, n_clusters * 10)

    # size score (0-30)
    size_score = min(30, n * 5)

    total = round(div_score + decorr_score + size_score)
    total = max(0, min(100, total))

    if total >= 80: grade = "A"
    elif total >= 65: grade = "B"
    elif total >= 50: grade = "C"
    elif total >= 35: grade = "D"
    else: grade = "F"

    # strategy type
    if sym_set <= CLUSTERS["stables"]:
        stype = "Stablecoin Vault"
    elif n == 1:
        stype = "Single Asset YOLO"
    elif max_cluster_pct > 0.8:
        stype = f"{dominant_cluster.replace('_', ' ').title()} Heavy"
    elif n >= 5 and n_clusters >= 3:
        stype = "Multi-Sector Diversified"
    elif n >= 3:
        stype = "Balanced"
    else:
        stype = "Concentrated"

    # suggestion
    has_stable = bool(sym_set & CLUSTERS["stables"])
    suggestion = ""
    if not has_stable and n > 1:
        heaviest = symbols[np.argmax(w)]
        suggestion = f"Adding 10% stablecoin (from {heaviest}) would reduce volatility ~30%"
    elif hhi > 0.5 and n > 1:
        suggestion = "High concentration. Consider spreading across more assets"
    elif n == 1:
        suggestion = "Single asset = maximum risk. Consider diversifying into 3-5 assets"
    elif n_clusters <= 1:
        suggestion = "All assets in one sector. Add assets from different sectors"
    else:
        suggestion = "Well-structured portfolio. Monitor correlations during drawdowns"

    return {
        "score": total,
        "grade": grade,
        "hhi": round(hhi, 3),
        "n_assets": n,
        "n_clusters": n_clusters,
        "dominant_cluster": dominant_cluster,
        "cluster_concentration": round(max_cluster_pct * 100, 1),
        "strategy_type": stype,
        "suggestion": suggestion,
    }

def community_rank(score: int, conn: sqlite3.Connection) -> tuple[int, int]:
    rows = conn.execute("SELECT assets_json FROM portfolios").fetchall()
    scores = []
    for (aj,) in rows:
        if not aj:
            continue
        try:
            assets = json.loads(aj)
            syms = [a["symbol"] for a in assets]
            wts = [a["weight"] for a in assets]
            s = score_portfolio(syms, wts)
            scores.append(s["score"])
        except:
            pass
    scores.sort(reverse=True)
    rank = sum(1 for s in scores if s > score) + 1
    return rank, len(scores)

def format_card(portfolio, analysis, rank, total) -> str:
    assets_str = " | ".join(
        f"{a.symbol} {a.weight:.0f}%"
        for a in portfolio.assets
    )
    pct_beaten = round((1 - rank / max(total, 1)) * 100)

    bars = int(analysis["score"] / 5)
    bar_str = "=" * bars + "-" * (20 - bars)

    # encouraging flavor text
    if analysis["score"] >= 80:
        vibe = "Outstanding portfolio construction!"
    elif analysis["score"] >= 65:
        vibe = "Solid strategy, well thought out."
    elif analysis["score"] >= 50:
        vibe = "Good foundation — a few tweaks could level this up."
    elif analysis["score"] >= 35:
        vibe = "Bold moves! Here's how to tighten it up."
    else:
        vibe = "High conviction play. Consider diversifying for safety."

    lines = [
        f"**PORTFOLIO SCORE: {analysis['score']}/100 ({analysis['grade']})**",
        f"`[{bar_str}]`",
        f"",
        f"*{vibe}*",
        f"",
        f"**{analysis['strategy_type']}**",
        f"`{assets_str}`",
        f"",
        f"Diversification: **{analysis['grade']}** | Sectors: **{analysis['n_clusters']}**",
    ]

    if analysis["dominant_cluster"]:
        lines.append(f"Leaning into: **{analysis['dominant_cluster'].replace('_', ' ')}** ({analysis['cluster_concentration']}%)")

    lines.extend([
        f"",
        f"Community rank: **#{rank}** of {total}",
        f"Top **{pct_beaten}%** of community portfolios",
        f"",
        f"Tip: *{analysis['suggestion']}*",
    ])

    return "\n".join(lines)

class ArenaBot(discord.Client):
    def __init__(self):
        intents = Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self.conn = init_db()
        tokens = fetch_token_list()
        if tokens:
            save_tokens(self.conn, tokens)

    async def on_ready(self):
        print(f"Arena bot online as {self.user}")
        print(f"Watching for glider.fi/portfolio/ links...")

    async def on_message(self, message):
        if message.author == self.user:
            return

        match = PORTFOLIO_RE.search(message.content)
        if not match:
            return

        sid = match.group(1)
        print(f"  scoring {sid} for {message.author}...")

        try:
            p = parse_portfolio(sid, self.conn)
            if not p:
                await message.reply("Couldn't fetch that portfolio. Is the link correct?")
                return

            p.discord_user = str(message.author)
            save_portfolio(self.conn, p)

            syms = [a.symbol for a in p.assets]
            wts = [a.weight for a in p.assets]
            analysis = score_portfolio(syms, wts)
            rank, total = community_rank(analysis["score"], self.conn)

            card = format_card(p, analysis, rank, total)

            embed = discord.Embed(
                title=f"Portfolio Arena — {p.blueprint_name or sid}",
                description=card,
                color=0x00ff88 if analysis["score"] >= 65 else 0xff8800 if analysis["score"] >= 35 else 0xff0000,
            )
            embed.set_footer(text="Portfolio Arena by Jay | Not financial advice")

            await message.reply(embed=embed)
            print(f"  scored: {analysis['score']}/100 ({analysis['grade']})")

        except Exception as e:
            print(f"  error: {e}")
            await message.reply(f"Error analyzing portfolio: {str(e)[:100]}")

def main():
    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        print("Set DISCORD_BOT_TOKEN environment variable")
        print("  1. Go to discord.com/developers/applications")
        print("  2. Create application -> Bot -> Copy token")
        print("  3. Enable MESSAGE_CONTENT intent")
        print("  4. set DISCORD_BOT_TOKEN=your_token_here")
        return
    bot = ArenaBot()
    bot.run(token)

if __name__ == "__main__":
    main()
