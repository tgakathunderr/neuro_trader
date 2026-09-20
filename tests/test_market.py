import sys
import os
from pathlib import Path

# Ensure root directory is on sys.path
_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pytest
from neuro_trader.data.market import MarketDataFeed


def test_market_feed_loads_default_data():
    feed = MarketDataFeed()
    candles = feed.load_data(limit=100)
    assert len(candles) == 100
    candle = feed.get_candle(0)
    assert "close" in candle and "open" in candle and "volume" in candle
    assert candle["high"] >= candle["low"]


def test_extract_features_shape_and_bounds():
    feed = MarketDataFeed()
    feed.load_data(limit=100)
    features = feed.extract_features(index=20, window=16)
    assert isinstance(features, np.ndarray)
    assert features.shape == (64,)
    assert np.all(np.isfinite(features))
    assert np.all(features >= -5.0) and np.all(features <= 5.0)


def test_market_feed_online_fallback_behavior():
    # Provide a non-existent file path; should gracefully fall back without throwing
    feed = MarketDataFeed(cache_file="non_existent_market_cache_12345.csv")
    assert len(feed.candles) > 0
    c = feed.get_candle(5)
    assert c["close"] > 0.0


def test_bear_regime_generation_and_alpha():
    """Verifies that regime='bear' generates a sustained downtrend for drawdown testing."""
    feed = MarketDataFeed()
    bear_candles = feed.load_data(limit=300, regime="bear")
    assert len(bear_candles) == 300
    p_start = bear_candles[0]["close"]
    p_end = bear_candles[-1]["close"]
    # Bear regime must show significant market drop
    assert p_end < p_start
    pct_drop = (p_end - p_start) / p_start
    assert pct_drop < -0.05  # At least 5% drop in 300 candles
