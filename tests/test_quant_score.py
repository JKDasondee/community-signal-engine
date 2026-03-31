import numpy as np
import pandas as pd
import pytest
from cse.quant_score import (
    score_portfolio, sharpe_ratio, cvar, max_drawdown,
    annual_volatility, avg_pairwise_correlation,
    diversification_entropy, composite_score, grade,
    QuantScore,
)

def _returns(n=200, mu=0.001, sigma=0.02):
    np.random.seed(42)
    return pd.Series(np.random.normal(mu, sigma, n))

def test_sharpe_positive():
    r = _returns(mu=0.002, sigma=0.01)
    s = sharpe_ratio(r)
    assert s > 0

def test_sharpe_negative():
    r = _returns(mu=-0.003, sigma=0.02)
    assert sharpe_ratio(r) < 0

def test_cvar_negative():
    r = _returns()
    c = cvar(r)
    assert c < 0

def test_cvar_worse_than_mean():
    r = _returns()
    assert cvar(r) < r.mean()

def test_max_drawdown_negative():
    r = _returns()
    assert max_drawdown(r) < 0

def test_max_drawdown_bounded():
    r = _returns()
    assert max_drawdown(r) >= -1.0

def test_annual_vol_positive():
    r = _returns()
    assert annual_volatility(r) > 0

def test_entropy_single_asset():
    assert diversification_entropy([100.0]) == 0.0

def test_entropy_equal_weight():
    e = diversification_entropy([25.0, 25.0, 25.0, 25.0])
    assert abs(e - 1.0) < 0.01

def test_entropy_concentrated():
    e = diversification_entropy([90.0, 5.0, 5.0])
    assert e < 0.7

def test_composite_score_range():
    s = composite_score(1.0, -0.02, -0.1, 0.3, 0.5, 0.8, 0.3, 0.05, 5)
    assert 0 <= s <= 100

def test_grade_s():
    assert grade(90) == "S"

def test_grade_f():
    assert grade(10) == "F"

def test_score_empty():
    qs = score_portfolio([], [])
    assert qs.total == 0
    assert qs.grade == "F"

def test_score_single_asset():
    qs = score_portfolio(["ETH"], [100.0])
    assert qs.n_assets == 1
    assert qs.strategy_type == "Single Asset"
    # without price data, return-based metrics default neutral
    # with price data, single asset would score lower

def test_score_diversified():
    syms = ["ETH", "cbBTC", "USDC", "AAVE", "AERO"]
    wts = [20.0, 20.0, 20.0, 20.0, 20.0]
    qs = score_portfolio(syms, wts)
    assert qs.n_assets == 5
    assert qs.diversification_ratio > 0.9
    assert qs.total > 40

def test_score_stablecoin():
    qs = score_portfolio(["USDC", "USDT"], [50.0, 50.0])
    assert qs.strategy_type == "Stablecoin Vault"

def test_score_returns_quantscore():
    qs = score_portfolio(["ETH"], [100.0])
    assert isinstance(qs, QuantScore)
    assert hasattr(qs, "sharpe")
    assert hasattr(qs, "cvar_95")
    assert hasattr(qs, "herding_score")
