"""Nocturnal Circadian Sleep Manager: Slow-Wave Sleep (SWS), SWR replay, and Tononi SHY downscaling."""

import os
import sys
from typing import Dict, Any, Optional

# Ensure BIB-2 library is available
_BIM2_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BIB2_PATH = os.path.join(_BIM2_ROOT, "BIB-2")
if _BIB2_PATH not in sys.path:
    sys.path.insert(0, _BIB2_PATH)

from bib2.brain import BIB2NervousSystem
from bib2.sleep.orchestrator import SleepStage


class SleepManager:
    """
    Coordinates nocturnal sleep cycles:
    1. Circadian Day/Night: Every N candles (default 288 = 24 hours of 5m candles).
    2. SWS Replay: Sharp-Wave Ripples (150-250 Hz) replay winning chart patterns into Neocortex association areas.
    3. Tononi SHY Downscaling: -5% synaptic downscaling to prune daytime noise.
    4. Glymphatics: AQP4 convective clearance flushes accumulated adenosine fatigue.
    5. Circadian Dawn: Resets cortisol to morning baseline and transitions back to WAKE.
    """

    def __init__(self, brain: BIB2NervousSystem, session_length: int = 288):
        self.brain = brain
        self.session_length = session_length
        self.total_sleep_cycles = 0
        self.last_sleep_stats: Dict[str, Any] = {}

    def should_sleep(self, candle_idx: int) -> bool:
        """Determines if a trading epoch has completed and night has fallen."""
        return candle_idx > 0 and (candle_idx % self.session_length == 0)

    def execute_sws_sleep(self, n_sleep_ticks: int = 15) -> Dict[str, Any]:
        """
        Executes complete nocturnal Slow-Wave Sleep (SWS) consolidation cycle.
        """
        self.total_sleep_cycles += 1
        initial_adenosine = float(self.brain.chemistry.matrix.state.adenosine)

        # 1. Enter Slow-Wave Sleep
        self.brain.sleep.transition_to(SleepStage.NREM_SWS)

        # 2. Execute SWS consolidation clock ticks (SWR ripples + Tononi SHY downscaling)
        replayed_count = min(5, len(self.brain.episodic_memory))
        for _ in range(n_sleep_ticks):
            self.brain.tick_sleep()

        # 3. Glymphatic final clearance of any residual adenosine
        self.brain.chemistry.matrix.clear_adenosine(amount=initial_adenosine)
        self.brain.chemistry.matrix.state.adenosine = float(
            max(0.0, self.brain.chemistry.matrix.state.adenosine - 0.5)
        )

        # 4. Circadian Dawn: Morning Cortisol Reset & Wake Stage
        self.brain.chemistry.matrix.state.cortisol = 0.15
        self.brain.sleep.transition_to(SleepStage.WAKE)

        stats = {
            "cycle": self.total_sleep_cycles,
            "replayed_count": replayed_count,
            "downscaled": True,
            "initial_adenosine": initial_adenosine,
            "final_adenosine": float(self.brain.chemistry.matrix.state.adenosine),
        }
        self.last_sleep_stats = stats
        return stats
