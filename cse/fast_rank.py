"""
O(log N) community ranking via sorted array + bisect.
Precomputed on startup, updated incrementally on new portfolios.

CP technique: offline precomputation + binary search for online queries.
"""
import json
import bisect
import sqlite3
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DATA_DIR / "portfolios.db"

class RankIndex:
    __slots__ = ("scores", "total")

    def __init__(self):
        self.scores: list[int] = []
        self.total: int = 0

    def build(self, conn: sqlite3.Connection, score_fn):
        """Precompute all scores once. O(P * S) where P=portfolios, S=score_time."""
        rows = conn.execute("SELECT assets_json FROM portfolios").fetchall()
        self.scores = []
        for (aj,) in rows:
            if not aj:
                continue
            try:
                assets = json.loads(aj)
                syms = [a["symbol"] for a in assets if a.get("weight", 0) > 0]
                wts = [a["weight"] for a in assets if a.get("weight", 0) > 0]
                if syms:
                    s = score_fn(syms, wts)
                    self.scores.append(s)
            except:
                continue
        self.scores.sort()
        self.total = len(self.scores)

    def query(self, score: int) -> tuple[int, int]:
        """O(log N) rank lookup via bisect."""
        # rank = number of portfolios scoring strictly higher + 1
        rank = self.total - bisect.bisect_right(self.scores, score) + 1
        return rank, self.total

    def insert(self, score: int):
        """O(log N) incremental update when new portfolio is scored."""
        bisect.insort(self.scores, score)
        self.total += 1
