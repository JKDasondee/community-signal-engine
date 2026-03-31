"""
Parse scraped portfolio data from DB.
Extract allocations from API responses and page text.

Run after scraper: python -m cse.parser
"""
import json
import sqlite3
import re
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DATA_DIR / "portfolios.db"

def parse_api_responses(conn: sqlite3.Connection):
    rows = conn.execute(
        "SELECT portfolio_id, url, response_body, response_status FROM network_requests"
    ).fetchall()

    parsed = 0
    for pid, url, body, status in rows:
        if status != 200 or not body:
            continue
        try:
            data = json.loads(body)
        except Exception:
            continue

        allocations = extract_allocations(data, pid)
        if allocations:
            conn.execute("DELETE FROM allocations WHERE portfolio_id = ?", (pid,))
            for a in allocations:
                conn.execute(
                    "INSERT INTO allocations (portfolio_id, asset_symbol, asset_name, weight_pct, token_address, chain) VALUES (?,?,?,?,?,?)",
                    (pid, a["symbol"], a["name"], a["weight"], a.get("address", ""), a.get("chain", ""))
                )
            parsed += 1
            print(f"  {pid}: {len(allocations)} assets")

    conn.commit()
    return parsed

def extract_allocations(data: dict | list, pid: str) -> list[dict]:
    allocations = []

    if isinstance(data, list):
        for item in data:
            allocations.extend(extract_allocations(item, pid))
        return allocations

    if not isinstance(data, dict):
        return []

    # look for common API response patterns
    for key in ["allocations", "holdings", "positions", "assets", "portfolio", "tokens", "balances"]:
        if key in data:
            val = data[key]
            if isinstance(val, list):
                for item in val:
                    a = parse_allocation_item(item)
                    if a:
                        allocations.append(a)
            elif isinstance(val, dict):
                # might be nested
                allocations.extend(extract_allocations(val, pid))

    # check if this dict itself is an allocation item
    if not allocations:
        a = parse_allocation_item(data)
        if a:
            allocations.append(a)

    # recurse into nested dicts
    if not allocations:
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                allocations.extend(extract_allocations(v, pid))

    return allocations

def parse_allocation_item(item: dict) -> dict | None:
    if not isinstance(item, dict):
        return None

    symbol = item.get("symbol") or item.get("token_symbol") or item.get("asset") or item.get("ticker")
    name = item.get("name") or item.get("token_name") or item.get("asset_name") or ""
    weight = (
        item.get("weight") or item.get("allocation") or item.get("percentage")
        or item.get("pct") or item.get("share") or item.get("weight_pct")
    )
    address = item.get("address") or item.get("token_address") or item.get("contract") or ""
    chain = item.get("chain") or item.get("chainId") or item.get("network") or ""

    if symbol and weight is not None:
        try:
            w = float(weight)
            if w > 1 and w <= 100:
                pass  # already in pct
            elif w <= 1:
                w = w * 100  # convert to pct
        except Exception:
            return None
        return {"symbol": str(symbol), "name": str(name), "weight": w, "address": str(address), "chain": str(chain)}
    return None

def parse_page_text(conn: sqlite3.Connection):
    """Fallback: extract from rendered page text if API parsing failed."""
    rows = conn.execute(
        "SELECT id, raw_text FROM portfolios WHERE status = 'scraped'"
    ).fetchall()

    for pid, text in rows:
        if not text:
            continue
        existing = conn.execute(
            "SELECT COUNT(*) FROM allocations WHERE portfolio_id = ?", (pid,)
        ).fetchone()[0]
        if existing > 0:
            continue

        # try to find asset/percentage patterns in page text
        # patterns: "WETH 40.00%", "40% WETH", "WETH: 40%"
        patterns = [
            re.findall(r'([A-Z]{2,10})\s+(\d+(?:\.\d+)?)\s*%', text),
            re.findall(r'(\d+(?:\.\d+)?)\s*%\s+([A-Z]{2,10})', text),
        ]

        for matches in patterns:
            if len(matches) >= 2:
                for sym, pct in matches:
                    if isinstance(pct, str) and sym.isalpha():
                        pass
                    else:
                        sym, pct = pct, sym
                    try:
                        w = float(pct)
                        if 0 < w <= 100:
                            conn.execute(
                                "INSERT INTO allocations (portfolio_id, asset_symbol, weight_pct) VALUES (?,?,?)",
                                (pid, sym, w)
                            )
                    except Exception:
                        pass
                conn.commit()
                count = conn.execute(
                    "SELECT COUNT(*) FROM allocations WHERE portfolio_id = ?", (pid,)
                ).fetchone()[0]
                if count > 0:
                    print(f"  {pid}: {count} assets (from page text)")
                break

def report(conn: sqlite3.Connection):
    total = conn.execute("SELECT COUNT(*) FROM portfolios").fetchone()[0]
    with_alloc = conn.execute(
        "SELECT COUNT(DISTINCT portfolio_id) FROM allocations"
    ).fetchone()[0]
    total_alloc = conn.execute("SELECT COUNT(*) FROM allocations").fetchone()[0]
    api_calls = conn.execute("SELECT COUNT(*) FROM network_requests").fetchone()[0]

    print(f"\n--- Scrape Report ---")
    print(f"  Portfolios scraped: {total}")
    print(f"  With allocations:   {with_alloc}")
    print(f"  Total assets found: {total_alloc}")
    print(f"  API calls captured: {api_calls}")

    print(f"\n--- Top Assets ---")
    rows = conn.execute("""
        SELECT asset_symbol, COUNT(*) as cnt, AVG(weight_pct) as avg_w
        FROM allocations
        GROUP BY asset_symbol
        ORDER BY cnt DESC
        LIMIT 15
    """).fetchall()
    print(f"  {'Symbol':<10} {'Count':>6} {'Avg Weight%':>12}")
    for sym, cnt, avg_w in rows:
        print(f"  {sym:<10} {cnt:>6} {avg_w:>12.1f}")

def main():
    conn = sqlite3.connect(DB_PATH)
    print("Parsing API responses...")
    n = parse_api_responses(conn)
    print(f"  Parsed {n} portfolios from API data")

    print("\nParsing page text (fallback)...")
    parse_page_text(conn)

    report(conn)
    conn.close()

if __name__ == "__main__":
    main()
