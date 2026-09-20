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

import numpy as np
import pytest
from bib2.brain import BIB2NervousSystem
from neuro_trader.adapter import TradingAdapter


def test_trading_adapter_action_selection():
    brain = BIB2NervousSystem(seed=42)
    adapter = TradingAdapter(brain)
    features = np.zeros(64, dtype=np.float32)
    features[0:16] = 0.5

    action = adapter.decide_action(
        market_features=features,
        current_price=50000.0,
        position_exposure=0.0,
        unrealized_pnl=0.0,
        last_reward=0.0
    )
    assert action in [0, 1, 2]
    assert adapter.last_action == action


def test_trading_adapter_reward_plasticity_and_telemetry():
    brain = BIB2NervousSystem(seed=42)
    adapter = TradingAdapter(brain)
    features = np.ones(64, dtype=np.float32) * 0.2
    initial_da = brain.chemistry.matrix.state.dopamine

    # Feed positive reward from a winning trade
    adapter.decide_action(
        market_features=features,
        current_price=52000.0,
        position_exposure=1.0,
        unrealized_pnl=0.04,
        last_reward=0.5
    )
    assert brain.chemistry.matrix.state.dopamine >= initial_da
    assert len(brain.limbic.hippocampus.ca3.stored_patterns) > 0

    # Feed negative reward / punishment
    adapter.decide_action(
        market_features=features,
        current_price=48000.0,
        position_exposure=1.0,
        unrealized_pnl=-0.05,
        last_reward=-0.6
    )
    assert brain.peripheral.autonomic.sympathetic_tone > 0.25

    # Check telemetry payload
    telemetry = adapter.get_telemetry()
    assert "dopamine" in telemetry
    assert "cortisol" in telemetry
    assert "sympathetic_tone" in telemetry
    assert "d1_weights" in telemetry
    assert "d2_weights" in telemetry
    assert "cerebellar_error" in telemetry
