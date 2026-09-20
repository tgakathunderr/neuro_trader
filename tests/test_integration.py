import sys
import os
from pathlib import Path

# Ensure root directory and BIB-2 are on sys.path
_ROOT = str(Path(__file__).resolve().parent.parent.parent)
_BIB2_DIR = os.path.join(_ROOT, "BIB-2")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _BIB2_DIR not in sys.path:
    sys.path.insert(0, _BIB2_DIR)

import pytest
from bib2.brain import BIB2NervousSystem
from neuro_trader.data.market import MarketDataFeed
from neuro_trader.trading.env import SpotTradingEnv
from neuro_trader.adapter import TradingAdapter
from neuro_trader.trading.sleep_manager import SleepManager


def test_full_trading_loop_100_candles():
    """Verify that data feed, spot env, trading adapter, and sleep manager execute together without errors."""
    brain = BIB2NervousSystem(seed=42)
    feed = MarketDataFeed()
    feed.load_data(limit=150)
    env = SpotTradingEnv(initial_cash=10000.0)
    adapter = TradingAdapter(brain)
    sleep_mgr = SleepManager(brain, session_length=50)

    last_reward = 0.0
    action_counts = {0: 0, 1: 0, 2: 0}
    sleep_count = 0

    for i in range(16, 90):
        candle = feed.get_candle(i)
        features = feed.extract_features(i)

        action = adapter.decide_action(
            market_features=features,
            current_price=candle["close"],
            position_exposure=1.0 if env.btc > 0 else 0.0,
            unrealized_pnl=env.get_unrealized_pnl(candle["close"]),
            last_reward=last_reward
        )
        action_counts[action] += 1

        last_reward, _, info = env.step(action, candle["close"])

        if sleep_mgr.should_sleep(i):
            sleep_stats = sleep_mgr.execute_sws_sleep()
            sleep_count += 1
            assert sleep_stats["downscaled"] is True

    # Confirm healthy execution
    final_equity = env.get_equity(feed.get_candle(89)["close"])
    assert final_equity > 0.0
    assert sum(action_counts.values()) == (90 - 16)
    assert sleep_count >= 1

    # Verify telemetry is clean and bounded
    telemetry = adapter.get_telemetry()
    assert 0.0 <= telemetry["dopamine"] <= 1.0
    assert 0.0 <= telemetry["cortisol"] <= 1.0
    assert 50 <= telemetry["heart_rate"] <= 160
