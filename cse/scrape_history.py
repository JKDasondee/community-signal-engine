"""
Scrape all portfolio links from Discord channel history.
Fetches every message, extracts glider.fi/portfolio/ links, scores them all.

Run: python -m cse.scrape_history
"""
import re
import json
import time
import sqlite3
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

from cse.glider_api import parse_portfolio, save_portfolio, init_db, fetch_token_list, save_tokens

TOKEN = os.getenv("DISCORD_BOT_TOKEN")
HEADERS = {"Authorization": f"Bot {TOKEN}"}
SHARE_CHANNEL = "1417494318658228265"
PORTFOLIO_RE = re.compile(r'glider\.fi/portfolio/([a-z0-9]+)')
DB_PATH = Path(__file__).parent.parent / "data" / "portfolios.db"

def fetch_messages(channel_id: str, before: str = None, limit: int = 100) -> list:
    url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
    params = {"limit": limit}
    if before:
        params["before"] = before
    r = requests.get(url, headers=HEADERS, params=params, timeout=15)
    if r.status_code == 200:
        return r.json()
    print(f"  fetch error: {r.status_code}")
    return []

def scrape_channel(channel_id: str) -> list[dict]:
    print(f"Scraping channel {channel_id} history...")
    all_links = []
    seen_ids = set()
    before = None
    batch = 0

    while True:
        msgs = fetch_messages(channel_id, before)
        if not msgs:
            break

        batch += 1
        for msg in msgs:
            matches = PORTFOLIO_RE.findall(msg["content"])
            for pid in matches:
                if pid not in seen_ids:
                    seen_ids.add(pid)
                    all_links.append({
                        "portfolio_id": pid,
                        "discord_user": msg["author"]["username"],
                        "discord_user_id": msg["author"]["id"],
                        "message_id": msg["id"],
                        "timestamp": msg["timestamp"],
                        "channel_id": channel_id,
                    })

        before = msgs[-1]["id"]
        print(f"  batch {batch}: {len(msgs)} msgs, {len(all_links)} unique portfolios so far")

        if len(msgs) < 100:
            break
        time.sleep(0.5)

    print(f"Total: {len(all_links)} unique portfolio links found")
    return all_links

def scrape_and_score():
    conn = init_db()

    # fetch token list for symbol resolution
    tokens = fetch_token_list()
    if tokens:
        save_tokens(conn, tokens)

    # scrape channel history
    links = scrape_channel(SHARE_CHANNEL)

    # also check glider-lounge
    lounge_links = scrape_channel("1403574345460355113")
    seen = {l["portfolio_id"] for l in links}
    for l in lounge_links:
        if l["portfolio_id"] not in seen:
            links.append(l)
            seen.add(l["portfolio_id"])

    print(f"\nTotal unique portfolios: {len(links)}")
    print(f"Fetching and scoring each...")

    success = 0
    for i, link in enumerate(links):
        pid = link["portfolio_id"]
        user = link["discord_user"]
        print(f"  [{i+1}/{len(links)}] {pid} ({user})...", end=" ", flush=True)

        try:
            p = parse_portfolio(pid, conn)
            if p:
                p.discord_user = user
                save_portfolio(conn, p)
                syms = [a.symbol for a in p.assets]
                print(f"ok ({len(p.assets)} assets: {', '.join(syms[:4])})")
                success += 1
            else:
                print("no data")
        except Exception as e:
            print(f"error: {e}")
        time.sleep(0.3)

    print(f"\nDone: {success}/{len(links)} portfolios scored")
    print(f"DB: {DB_PATH}")

    # summary
    total = conn.execute("SELECT COUNT(*) FROM portfolios").fetchone()[0]
    total_allocs = conn.execute("SELECT COUNT(*) FROM allocations").fetchone()[0]
    print(f"Total in DB: {total} portfolios, {total_allocs} allocations")

    conn.close()

if __name__ == "__main__":
    scrape_and_score()
