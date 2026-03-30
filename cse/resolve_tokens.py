"""
Resolve unknown token addresses to symbols via CoinGecko contract lookup.
Run after scraper: python -m cse.resolve_tokens
"""
import time
import sqlite3
import requests
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DATA_DIR / "portfolios.db"

CHAIN_MAP = {"8453": "base", "1": "ethereum"}

# known tokens not in Glider's list
KNOWN = {
    "0xef5997c2cf2f6c138196f8a6203afc335206b3c1:8453": ("OWB", "OWB"),
    "0x2615a94df961278dcbc41fb0a54fec5f10a693ae:8453": ("VIRTUAL", "Virtuals Protocol"),
    "0x311935cd80b76769bf2ecc9d8ab7635b2139cf82:8453": ("BRETT", "Brett"),
    "0x0b3e328455c4059eeb9e3f84b5543f74e24e7e1b:8453": ("DEGEN", "Degen"),
    "0x12e96c2bfea6e835cf8dd38a5834fa61cf723736:8453": ("TOSHI", "Toshi"),
    "0x4e65fe4dba92790696d040ac24aa414708f5c0ab:8453": ("KEYCAT", "Keyboard Cat"),
    "0x80ac24aa929eaf5013f6436cda2a7ba190f5cc0b:1": ("SNX", "Synthetix"),
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48:1": ("USDC", "USD Coin"),
}

def resolve_via_coingecko(address: str, chain: str) -> tuple[str, str] | None:
    platform = CHAIN_MAP.get(chain)
    if not platform:
        return None
    url = f"https://api.coingecko.com/api/v3/coins/{platform}/contract/{address}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            d = r.json()
            return (d.get("symbol", "").upper(), d.get("name", ""))
    except:
        pass
    return None

def main():
    conn = sqlite3.connect(DB_PATH)

    rows = conn.execute(
        "SELECT DISTINCT asset_id, symbol FROM allocations WHERE symbol LIKE '0x%'"
    ).fetchall()

    print(f"{len(rows)} unresolved tokens")

    for asset_id, old_sym in rows:
        parts = asset_id.split(":")
        addr = parts[0]
        chain = parts[1] if len(parts) > 1 else "8453"

        # check known list first
        if asset_id in KNOWN:
            sym, name = KNOWN[asset_id]
        else:
            print(f"  looking up {addr[:10]}... on chain {chain}", end=" ", flush=True)
            result = resolve_via_coingecko(addr, chain)
            if result:
                sym, name = result
                print(f"-> {sym}")
            else:
                print("-> not found")
                continue
            time.sleep(1.5)

        # update allocations
        conn.execute(
            "UPDATE allocations SET symbol = ?, name = ? WHERE asset_id = ?",
            (sym, name, asset_id)
        )
        # add to token_list
        conn.execute(
            "INSERT OR REPLACE INTO token_list (asset_id, address, chain_id, symbol, name, decimals, logo_uri) VALUES (?,?,?,?,?,?,?)",
            (asset_id, addr, chain, sym, name, 18, "")
        )
        print(f"  {asset_id[:16]}... -> {sym} ({name})")

    conn.commit()

    # print final summary
    print("\n--- All Assets ---")
    rows = conn.execute("""
        SELECT symbol, COUNT(*) as cnt, AVG(weight_pct) as avg_w
        FROM allocations GROUP BY symbol ORDER BY cnt DESC
    """).fetchall()
    print(f"  {'Symbol':<10} {'Count':>6} {'Avg Wt%':>8}")
    for sym, cnt, avg_w in rows:
        print(f"  {sym:<10} {cnt:>6} {avg_w:>8.1f}")

    conn.close()

if __name__ == "__main__":
    main()
