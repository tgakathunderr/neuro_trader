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
from bib2.sleep.orchestrator import SleepStage
from neuro_trader.trading.sleep_manager import SleepManager


def test_sleep_trigger_and_consolidation():
    brain = BIB2NervousSystem(seed=42)
    manager = SleepManager(brain, session_length=288)

    assert not manager.should_sleep(candle_idx=100)
    assert manager.should_sleep(candle_idx=288)

    # Seed some episodic memories to replay
    for _ in range(5):
        brain.episodic_memory.append(np.ones(64, dtype=np.float32) * 0.5)

    # Accumulate adenosine & run nocturnal sleep consolidation
    brain.chemistry.matrix.state.adenosine = 0.8
    results = manager.execute_sws_sleep()

    assert results["replayed_count"] > 0
    assert results["downscaled"] is True
    # Adenosine should have been cleared by glymphatics
    assert brain.chemistry.matrix.state.adenosine < 0.3
    # Brain should be back in WAKE stage ready for the next day
    assert brain.sleep.current_stage == SleepStage.WAKE
