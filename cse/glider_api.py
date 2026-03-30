"""
Direct Glider API client. No browser needed.
tRPC endpoints are public for shared portfolios.

Run: python -m cse.glider_api
"""
import json
import time
import sqlite3
import requests
from pathlib import Path
from datetime import datetime
from urllib.parse import quote
from dataclasses import dataclass, asdict

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "portfolios.db"
BASE = "https://api.glider.fi/v1/trpc"

PORTFOLIO_IDS = [
    "1g06xx9z", "ddllla05", "e4hbkqwg", "ugjd6w86", "mp3ewpon",
    "jal07bw5", "290t48qn", "f8fqym6l", "kcbgvz1h", "s9cwc30j",
    "i6z6q0ws", "tkfq2okp", "zmk99v5s", "nb5onb53", "sr00tq64",
    "ucihnh6j", "idfnwnme", "xant00u4", "tksdau2o", "51sibsj8",
    "o9vhqlaz", "2d29hov7", "65nsrvbu",
]

DISCORD_USERS = {
    "1g06xx9z": "Jadoogar", "ddllla05": "One", "e4hbkqwg": "Simplsimpl",
    "ugjd6w86": "Irina", "mp3ewpon": "DJY", "jal07bw5": "goga2201",
    "290t48qn": "Mawj", "f8fqym6l": "0xliok", "kcbgvz1h": "Grim",
    "s9cwc30j": "RadelPertime", "i6z6q0ws": "stiv", "tkfq2okp": "Irina",
    "zmk99v5s": "mizzy", "nb5onb53": "BoyEgg", "sr00tq64": "SUA FrAi",
    "ucihnh6j": "satya prakash", "idfnwnme": "MS", "xant00u4": "makonaja",
    "tksdau2o": "Richman", "51sibsj8": "Tyo", "o9vhqlaz": "ransen67",
    "2d29hov7": "KOMORAX", "65nsrvbu": "Achmad Effendi",
}

@dataclass
class Asset:
    asset_id: str
    symbol: str
    name: str
    weight: float
    chain_id: str

@dataclass
class Portfolio:
    strategy_id: str
    discord_user: str
    blueprint_name: str
    owner_address: str
    assets: list[Asset]
    current_value_usd: float
    total_cost_basis: float
    realized_pnl: float
    rebalance_interval_ms: int
    scraped_at: str

def trpc_get(endpoint: str, params: dict) -> dict | None:
    input_json = json.dumps({"json": params})
    url = f"{BASE}/{endpoint}?input={quote(input_json)}"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"    error: {e}")
    return None

def fetch_strategy(sid: str) -> dict | None:
    return trpc_get("strategyInstances.getStrategyInstance", {"strategyInstanceId": sid})

def fetch_pnl(sid: str) -> dict | None:
    return trpc_get("pnl.getCostBasisAndPnL", {"includeTrades": True, "realtime": True, "strategyInstanceId": sid})

def fetch_schedule(sid: str) -> dict | None:
    return trpc_get("schedules.getStrategyInstanceSchedule", {"strategyInstanceId": sid})

def fetch_token_list() -> dict | None:
    return trpc_get("assetValidation.getFilteredTokenList", {"chainId": "all"})

def load_token_map(conn: sqlite3.Connection) -> dict[str, dict]:
    rows = conn.execute("SELECT asset_id, symbol, name FROM token_list").fetchall()
    return {r[0]: {"symbol": r[1], "name": r[2]} for r in rows}

_token_map: dict[str, dict] = {}

def resolve_symbol(asset_id: str, conn: sqlite3.Connection | None = None) -> tuple[str, str]:
    global _token_map
    if not _token_map and conn:
        _token_map = load_token_map(conn)
    t = _token_map.get(asset_id, {})
    sym = t.get("symbol", "")
    name = t.get("name", "")
    if not sym and "eeeeeeeeeeeeee" in asset_id:
        sym, name = "ETH", "Ethereum"
    return sym or asset_id[:10], name

def parse_portfolio(sid: str, conn: sqlite3.Connection | None = None) -> Portfolio | None:
    strat = fetch_strategy(sid)
    if not strat:
        return None

    try:
        data = strat["result"]["data"]["json"]
        bp = data["strategyBlueprint"]
        sd = bp["strategy_data"]["entry"]
        children_block = sd.get("children", {})
        children = children_block.get("children", [])
        weight_type = children_block.get("weightType", "equal")
        weightings = children_block.get("weightings", [])

        assets = []
        n = len(children)
        for i, child in enumerate(children):
            aid = child.get("assetId", "")
            parts = aid.split(":")
            chain = parts[1] if len(parts) > 1 else "8453"
            sym, name = resolve_symbol(aid, conn)

            # determine weight
            if weight_type == "equal" or not weightings:
                w = round(100.0 / n, 2) if n > 0 else 0
            else:
                raw = float(weightings[i]) if i < len(weightings) else 0
                # if raw values sum to ~1, they're fractions; if sum to ~100, they're percentages
                raw_sum = sum(float(x) for x in weightings if x)
                if raw_sum > 10:
                    w = round(raw, 2)  # already percentages
                else:
                    w = round(raw * 100, 2)  # fractions → percentages

            assets.append(Asset(asset_id=aid, symbol=sym, name=name, weight=w, chain_id=chain))

        owner = bp.get("owner_address", "")
        name = bp.get("blueprint_name", "")
    except Exception as e:
        print(f"    parse error: {e}")
        return None

    # PnL — also enriches asset symbols from PnL response
    cv, tcb, rpnl = 0.0, 0.0, 0.0
    pnl = fetch_pnl(sid)
    if pnl:
        try:
            pdata = pnl["result"]["data"]["json"]
            pnl_map = {}
            for pa in pdata.get("assets", []):
                aid = pa.get("assetId", "")
                sym = pa.get("symbol", "")
                tcb += pa.get("totalCostBasis", 0)
                rpnl += pa.get("realizedPnL", 0)
                amt = float(pa.get("currentAmount", 0))
                price = float(pa.get("currentPrice", 0))
                cv += amt * price
                if aid and sym:
                    pnl_map[aid] = sym
            # backfill symbols from PnL if missing
            for a in assets:
                if (a.symbol == a.asset_id[:10] or a.symbol == "?") and a.asset_id in pnl_map:
                    a.symbol = pnl_map[a.asset_id]
        except:
            pass

    # Schedule
    rebal_ms = 0
    sched = fetch_schedule(sid)
    if sched:
        try:
            intervals = sched["result"]["data"]["json"]["schedule"]["intervals"]
            if intervals:
                rebal_ms = intervals[0].get("every", 0)
        except:
            pass

    return Portfolio(
        strategy_id=sid,
        discord_user=DISCORD_USERS.get(sid, ""),
        blueprint_name=name,
        owner_address=owner,
        assets=assets,
        current_value_usd=round(cv, 2),
        total_cost_basis=round(tcb, 2),
        realized_pnl=round(rpnl, 2),
        rebalance_interval_ms=rebal_ms,
        scraped_at=datetime.now().isoformat(),
    )

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS portfolios (
            strategy_id TEXT PRIMARY KEY,
            discord_user TEXT,
            blueprint_name TEXT,
            owner_address TEXT,
            current_value_usd REAL,
            total_cost_basis REAL,
            realized_pnl REAL,
            rebalance_interval_ms INTEGER,
            num_assets INTEGER,
            assets_json TEXT,
            scraped_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS allocations (
            strategy_id TEXT,
            asset_id TEXT,
            symbol TEXT,
            name TEXT,
            weight_pct REAL,
            chain_id TEXT,
            UNIQUE(strategy_id, asset_id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS token_list (
            asset_id TEXT PRIMARY KEY,
            address TEXT,
            chain_id TEXT,
            symbol TEXT,
            name TEXT,
            decimals INTEGER,
            logo_uri TEXT
        )
    """)
    conn.commit()
    return conn

def save_portfolio(conn: sqlite3.Connection, p: Portfolio):
    conn.execute(
        """INSERT OR REPLACE INTO portfolios
        (strategy_id, discord_user, blueprint_name, owner_address,
         current_value_usd, total_cost_basis, realized_pnl,
         rebalance_interval_ms, num_assets, assets_json, scraped_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (p.strategy_id, p.discord_user, p.blueprint_name, p.owner_address,
         p.current_value_usd, p.total_cost_basis, p.realized_pnl,
         p.rebalance_interval_ms, len(p.assets),
         json.dumps([asdict(a) for a in p.assets]), p.scraped_at)
    )
    for a in p.assets:
        conn.execute(
            """INSERT OR REPLACE INTO allocations
            (strategy_id, asset_id, symbol, name, weight_pct, chain_id)
            VALUES (?,?,?,?,?,?)""",
            (p.strategy_id, a.asset_id, a.symbol, a.name, a.weight, a.chain_id)
        )
    conn.commit()

def save_tokens(conn: sqlite3.Connection, data: dict):
    try:
        tokens = data["result"]["data"]["json"]
        for t in tokens:
            conn.execute(
                """INSERT OR REPLACE INTO token_list
                (asset_id, address, chain_id, symbol, name, decimals, logo_uri)
                VALUES (?,?,?,?,?,?,?)""",
                (t.get("assetId", ""), t.get("address", ""), t.get("chainId", ""),
                 t.get("symbol", ""), t.get("name", ""), t.get("decimals", 0),
                 t.get("logoURI", ""))
            )
        conn.commit()
        print(f"  saved {len(tokens)} tokens")
    except Exception as e:
        print(f"  token save error: {e}")

def scrape_all():
    conn = init_db()

    print("Fetching token list...")
    tokens = fetch_token_list()
    if tokens:
        save_tokens(conn, tokens)

    print(f"\nScraping {len(PORTFOLIO_IDS)} portfolios...")
    success = 0
    for sid in PORTFOLIO_IDS:
        print(f"  {sid} ({DISCORD_USERS.get(sid, '?')})...", end=" ", flush=True)
        p = parse_portfolio(sid, conn)
        if p:
            save_portfolio(conn, p)
            assets_str = ", ".join(f"{a.symbol}({a.weight}%)" for a in p.assets)
            print(f"${p.current_value_usd:.2f} | {assets_str}")
            success += 1
        else:
            print("failed")
        time.sleep(0.5)

    print(f"\n--- Results ---")
    print(f"  Success: {success}/{len(PORTFOLIO_IDS)}")
    print(f"  DB: {DB_PATH}")

    # Quick summary
    rows = conn.execute("""
        SELECT symbol, COUNT(*) as cnt, AVG(weight_pct) as avg_w
        FROM allocations GROUP BY symbol ORDER BY cnt DESC LIMIT 10
    """).fetchall()
    print(f"\n--- Top Assets Across All Portfolios ---")
    print(f"  {'Symbol':<10} {'Portfolios':>10} {'Avg Weight%':>12}")
    for sym, cnt, avg_w in rows:
        print(f"  {sym:<10} {cnt:>10} {avg_w:>12.1f}")

    conn.close()

if __name__ == "__main__":
    scrape_all()
