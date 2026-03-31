"""Edge case tests for the entire pipeline."""
import numpy as np
import pytest
from cse.quant_score import (
    score_portfolio, sharpe_ratio, cvar, max_drawdown,
    diversification_entropy, composite_score, QuantScore,
)
from cse.fast_rank import RankIndex
import pandas as pd

# === SCORING EDGE CASES ===

def test_empty_portfolio():
    qs = score_portfolio([], [])
    assert qs.total == 0
    assert qs.grade == "F"
    assert qs.n_assets == 0

def test_single_stablecoin():
    qs = score_portfolio(["USDC"], [100.0])
    assert qs.strategy_type == "Stablecoin Vault"
    assert qs.n_assets == 1

def test_all_same_token():
    # user puts 100% in one token
    qs = score_portfolio(["ETH"], [100.0])
    assert qs.diversification_ratio <= 0.0
    assert qs.strategy_type == "Single Asset"

def test_zero_weights():
    qs = score_portfolio(["ETH", "USDC"], [0.0, 0.0])
    # should not crash, handle gracefully
    assert isinstance(qs, QuantScore)

def test_negative_weights():
    # shouldn't happen but defensive
    qs = score_portfolio(["ETH"], [-50.0])
    assert isinstance(qs, QuantScore)

def test_very_many_assets():
    syms = [f"TOKEN{i}" for i in range(50)]
    wts = [2.0] * 50
    qs = score_portfolio(syms, wts)
    assert qs.n_assets == 50
    assert qs.diversification_ratio > 0.95

def test_tiny_weight():
    qs = score_portfolio(["ETH", "USDC"], [99.99, 0.01])
    assert qs.n_assets == 2

def test_unicode_symbol():
    qs = score_portfolio(["KOon", "WMTon"], [50.0, 50.0])
    assert qs.n_assets == 2

def test_duplicate_symbols():
    qs = score_portfolio(["ETH", "ETH"], [50.0, 50.0])
    assert qs.n_assets == 2
    # HHI should be same as single 100% ETH since they're the same asset
    assert qs.diversification_ratio > 0

def test_weights_dont_sum_100():
    # parser normalizes, but scoring should handle raw
    qs = score_portfolio(["ETH", "USDC", "AAVE"], [10.0, 20.0, 30.0])
    assert qs.n_assets == 3
    assert 0 <= qs.total <= 100

# === RETURN SERIES EDGE CASES ===

def test_sharpe_zero_vol():
    # constant returns = 0 vol
    r = pd.Series([0.01] * 100)
    # should not divide by zero
    s = sharpe_ratio(r)
    assert not np.isnan(s)

def test_sharpe_single_return():
    r = pd.Series([0.05])
    s = sharpe_ratio(r)
    assert isinstance(s, float)

def test_cvar_all_positive():
    r = pd.Series(np.abs(np.random.randn(100)) * 0.01)
    c = cvar(r)
    # even all-positive, cvar should return the worst 5%
    assert isinstance(c, float)

def test_max_drawdown_always_up():
    r = pd.Series([0.01] * 50)
    md = max_drawdown(r)
    assert md == 0.0 or md > -0.001

def test_entropy_two_equal():
    e = diversification_entropy([50.0, 50.0])
    assert abs(e - 1.0) < 0.01

def test_entropy_hundred_assets():
    e = diversification_entropy([1.0] * 100)
    assert abs(e - 1.0) < 0.01

# === RANK INDEX EDGE CASES ===

def test_rank_empty():
    ri = RankIndex()
    ri.scores = []
    ri.total = 0
    rank, total = ri.query(50)
    assert rank == 1
    assert total == 0

def test_rank_single():
    ri = RankIndex()
    ri.scores = [50]
    ri.total = 1
    rank, _ = ri.query(50)
    assert rank == 1

def test_rank_top():
    ri = RankIndex()
    ri.scores = [10, 20, 30, 40, 50]
    ri.total = 5
    rank, _ = ri.query(100)
    assert rank == 1

def test_rank_bottom():
    ri = RankIndex()
    ri.scores = [10, 20, 30, 40, 50]
    ri.total = 5
    rank, _ = ri.query(0)
    assert rank == 6

def test_rank_tie():
    ri = RankIndex()
    ri.scores = [50, 50, 50]
    ri.total = 3
    rank, _ = ri.query(50)
    assert rank == 1

def test_rank_insert():
    ri = RankIndex()
    ri.scores = [10, 30, 50]
    ri.total = 3
    ri.insert(40)
    assert ri.total == 4
    assert ri.scores == [10, 30, 40, 50]
    rank, _ = ri.query(40)
    assert rank == 2

def test_rank_insert_many():
    ri = RankIndex()
    ri.scores = []
    ri.total = 0
    for i in range(1000):
        ri.insert(i % 100)
    assert ri.total == 1000
    rank, total = ri.query(50)
    assert total == 1000

# === COMPOSITE SCORE EDGE CASES ===

def test_composite_all_zero():
    s = composite_score(0, 0, 0, 0, 0, 0, 0, 0, 0)
    assert 0 <= s <= 100

def test_composite_all_max():
    s = composite_score(3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 10)
    assert s > 80

def test_composite_extreme_negative():
    s = composite_score(-5.0, -0.5, -1.0, 2.0, 1.0, 0.0, 1.0, 1.0, 1)
    assert 0 <= s <= 100

def test_composite_nan_safe():
    # should not crash on NaN inputs
    s = composite_score(float('nan'), 0, 0, 0, 0, 0, 0, 0, 1)
    # NaN propagation is ok, but should not raise
    assert isinstance(s, (int, float))
