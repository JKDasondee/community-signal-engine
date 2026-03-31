import numpy as np
import pytest
from cse.glider_prices import (
    sparkline_to_returns, portfolio_vol_from_sparklines,
    portfolio_correlation_from_sparklines, enrich_quant_score,
)

def test_sparkline_to_returns():
    sp = [(i * 1000, 100 + i * 0.5) for i in range(50)]
    r = sparkline_to_returns(sp)
    assert r is not None
    assert len(r) == 49

def test_sparkline_too_short():
    sp = [(0, 100), (1, 101)]
    assert sparkline_to_returns(sp) is None

def test_sparkline_empty():
    assert sparkline_to_returns([]) is None

def test_portfolio_vol_no_data():
    v = portfolio_vol_from_sparklines(["ETH"], [100.0], {}, {})
    assert v == 0.0

def test_portfolio_correlation_single():
    c = portfolio_correlation_from_sparklines(["ETH"], {}, {})
    assert c == 0.0

def test_enrich_empty():
    r = enrich_quant_score(["ETH"], [100.0], ["0x123:8453"], {})
    assert r["vol_annualized"] == 0.0
    assert r["return_24h"] == 0.0
    assert len(r["token_details"]) == 1
