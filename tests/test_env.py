import sys
from pathlib import Path

# Ensure root directory is on sys.path
_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest
from neuro_trader.trading.env import SpotTradingEnv


def test_initial_state():
    env = SpotTradingEnv(initial_cash=10000.0)
    assert env.cash == 10000.0
    assert env.btc == 0.0
    assert env.get_equity(50000.0) == 10000.0
    assert env.get_unrealized_pnl(50000.0) == 0.0


def test_buy_and_sell_cycle():
    env = SpotTradingEnv(initial_cash=10000.0, fee=0.00075)
    # Buy at 50,000
    reward, done, info = env.step(action=1, current_price=50000.0)
    assert env.btc > 0.0
    assert env.cash < 500.0  # Allocated ~98% of cash
    assert env.entry_price == 50000.0

    # Hold at 52,000 (unrealized profit)
    reward_hold, _, info_hold = env.step(action=0, current_price=52000.0)
    assert info_hold["unrealized_pnl"] > 0.0

    # Sell at 55,000 (10% profit)
    reward_sell, done, info = env.step(action=2, current_price=55000.0)
    assert env.btc == 0.0
    assert env.cash > 10500.0  # Profitable after fees
    assert reward_sell > 0.0
    assert env.total_trades == 1
    assert env.winning_trades == 1


def test_stn_hard_stop_loss():
    env = SpotTradingEnv(initial_cash=10000.0, stn_stop_pct=-0.05)
    env.step(action=1, current_price=50000.0)
    assert env.btc > 0.0

    # Price crashes by 6% to 47,000 (below -5% stop)
    reward, done, info = env.step(action=0, current_price=47000.0)  # Attempting to HOLD
    # STN emergency brake should have overridden action and liquidated BTC to cash
    assert env.btc == 0.0
    assert info.get("stn_triggered") is True
    assert reward < 0.0
    assert env.total_trades == 1
    assert env.winning_trades == 0
