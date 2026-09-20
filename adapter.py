"""TradingAdapter: Bridges financial market dynamics to the BIB-2 Human Nervous System."""

import os
import sys
from typing import Dict, Any, Optional
import numpy as np

# Ensure BIB-2 library is available without modifying it
_BIM2_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_BIB2_PATH = os.path.join(_BIM2_ROOT, "BIB-2")
if _BIB2_PATH not in sys.path:
    sys.path.insert(0, _BIB2_PATH)

from bib2.brain import BIB2NervousSystem
from bib2.adapters.base import BaseNeuralAdapter


class TradingAdapter(BaseNeuralAdapter):
    """
    Biomimetic trading adapter translating market perception and financial risk into biological circuitry:
    1. Retinal Stream (CN II): 64-dim candlestick geometry, EMAs, and volume profile.
    2. Somatosensory Interoception (C5): Exposure tension and drawdown pain.
    3. Cerebellar Forward Model: Estimates next-candle motion and computes sensory prediction error.
    4. Hippocampal Episodic Attractor (CA3): Recalls past winning chart fingerprints.
    5. Basal Ganglia Action Gating: Gating 3 channels [0: HOLD, 1: BUY, 2: SELL].
    6. Neuroendocrine Plasticity: Dopamine RPE, Lateral Habenula anti-reward, and Cortisol stress.
    """

    def __init__(self, brain: BIB2NervousSystem):
        super().__init__(brain)
        self.n_actions = 3
        self.last_action = 0
        self.last_features = np.zeros(self.brain.dim, dtype=np.float32)
        self.cerebellar_error = 0.0
        self.hippocampal_match = 0.0
        self.last_prediction = 0.0

    def decide_action(
        self,
        market_features: np.ndarray,
        current_price: float,
        position_exposure: float,
        unrealized_pnl: float,
        last_reward: float = 0.0
    ) -> int:
        """
        Ingest reinforcement, update sensory nerves, step biological clock, and select discrete action.
        """
        # Prepare incoming sensory features
        features = np.asarray(market_features, dtype=np.float32).flatten()
        if features.size < self.brain.dim:
            padded = np.zeros(self.brain.dim, dtype=np.float32)
            padded[:features.size] = features
            cn2_input = padded
        else:
            cn2_input = features[:self.brain.dim].copy()

        # 1. Process reinforcement feedback & plasticity
        if last_reward > 0.0:
            # Positive RPE: reinforce selected action via D1 Go pathway
            self.brain.basal_ganglia.reinforce_action(self.last_action, reward_rpe=last_reward)
            self.brain.chemistry.matrix.state.dopamine = float(
                np.clip(self.brain.chemistry.matrix.state.dopamine + last_reward * 0.15, 0.05, 1.0)
            )
            self.brain.peripheral.enteric.satiety = float(
                np.clip(self.brain.peripheral.enteric.satiety + last_reward * 0.1, 0.0, 1.0)
            )
            # Store winning chart vector into Hippocampal CA3 attractor matrix
            pattern_to_store = self.last_features if float(np.dot(self.last_features, self.last_features)) > 0 else cn2_input
            self.brain.limbic.hippocampus.ca3.store(pattern_to_store, lr=0.5)

        elif last_reward < 0.0:
            # Negative RPE / punishment: reinforce D2 NoGo pathway & trigger sympathetic surge
            self.brain.basal_ganglia.reinforce_action(self.last_action, reward_rpe=last_reward)
            self.brain.chemistry.matrix.state.dopamine = float(
                np.clip(self.brain.chemistry.matrix.state.dopamine - abs(last_reward) * 0.2, 0.05, 1.0)
            )
            self.brain.habenula.compute_anti_reward(expected_reward=0.5, received_reward=last_reward)
            self.brain.peripheral.autonomic.trigger_sympathetic_surge(intensity=float(min(1.0, abs(last_reward))))
            # Elevate HPA-axis Cortisol
            self.brain.chemistry.matrix.state.cortisol = float(
                np.clip(self.brain.chemistry.matrix.state.cortisol + abs(last_reward) * 0.1, 0.0, 1.0)
            )

        # 2. Ingest Sensory Inputs into Peripheral Bus

        # Optic Nerve CN II
        self.brain.peripheral.cranial.set_sensory("CN_II", cn2_input)

        # Somatosensory Spinal Dermatome C5
        # Index 0: Position exposure tension [0.0 = Cash, 1.0 = Invested]
        # Index 1: Nociceptive drawdown pain
        somatic = np.zeros(self.brain.dim, dtype=np.float32)
        somatic[0] = float(np.clip(position_exposure, 0.0, 1.0))
        somatic[1] = float(np.clip(max(0.0, -unrealized_pnl * 10.0), 0.0, 2.0))
        self.brain.peripheral.spinal.set_dermatome("C5", somatic)

        # 3. Cerebellar Sensory Prediction Error Check
        # Compare actual incoming price delta to previous prediction
        actual_delta = float(features[50]) if features.size > 50 else 0.0
        self.cerebellar_error = float(abs(actual_delta - self.last_prediction))
        if self.cerebellar_error > 1.5:
            # High surprise: stimulate locus coeruleus norepinephrine
            self.brain.chemistry.matrix.state.norepinephrine = float(
                np.clip(self.brain.chemistry.matrix.state.norepinephrine + 0.1, 0.0, 1.0)
            )

        # 4. Hippocampal CA3 Pattern Completion Match
        if len(self.brain.limbic.hippocampus.ca3.stored_patterns) > 0:
            reconstructed = self.brain.limbic.hippocampus.ca3.complete(cn2_input, iterations=2)
            dot = float(np.dot(reconstructed, cn2_input))
            norm_prod = float(np.linalg.norm(reconstructed) * np.linalg.norm(cn2_input) + 1e-6)
            self.hippocampal_match = float(np.clip(dot / norm_prod, 0.0, 1.0))
        else:
            self.hippocampal_match = 0.0

        # 5. Execute Deterministic 19-Stage Biological Clock Cycle
        self.brain.tick()

        # Update Cerebellar Forward Model for NEXT tick
        dcml_sensory = self.brain.spinal_cord.process_segment_reflex("C5", somatic)
        m1_motor = self.brain.neocortex.registry.get_area("M1_PrimaryMotor").l5_output
        predicted_state, _ = self.brain.cerebellum.step(
            intended_motor_command=m1_motor,
            current_sensory_state=dcml_sensory
        )
        self.last_prediction = float(np.mean(predicted_state[:4]))

        # 6. Action Selection from Prefrontal Cortex (DLPFC) & Basal Ganglia
        dlpfc = self.brain.neocortex.registry.get_area("DLPFC_WorkingMemory").l5_output
        proposals = dlpfc[:self.n_actions].copy()

        # Biological Subthalamic Nucleus (STN) Hyperdirect Emergency Brake:
        # When held position experiences acute nociceptive pain (drawdown), STN fires hyperdirect brake
        if position_exposure > 0.0 and somatic[1] > 0.12:
            self.brain.basal_ganglia.trigger_hyperdirect_stop()
            # Hyperdirect pathway forces emergency motor withdrawal to cash (Channel 2)
            proposals[2] += 12.0
            proposals[1] -= 12.0

        if position_exposure <= 0.0:
            # In Cash: Somatic lack of inventory inhibits SELL
            proposals[2] -= 5.0
            # Visual sensory drive from Optic Nerve V1 cortex + Hippocampal CA3 attractor familiarity
            v1_out = self.brain.neocortex.registry.get_area("V1_Visual").l5_output
            v1_drive = float(np.mean(v1_out[:4])) if v1_out.size >= 4 else 0.0
            dopamine_drive = (float(self.brain.chemistry.matrix.state.dopamine) - 0.5) * 1.5
            proposals[1] += v1_drive * 0.5 + float(self.hippocampal_match * 1.5) + dopamine_drive
        else:
            # In Position: Somatic satiety inhibits re-buying
            proposals[1] -= 5.0
            # Continuous biological drive: Nociceptive pain (STT) + Enteric satiety / profit homeostasis
            pain_drive = somatic[1] * 2.5
            satiety_drive = float(np.clip(unrealized_pnl * 8.0, 0.0, 4.0))
            proposals[2] += pain_drive + satiety_drive

        winner_idx, _ = self.brain.basal_ganglia.select_action(
            proposals, dopamine_level=self.brain.chemistry.matrix.state.dopamine
        )
        action = int(winner_idx % self.n_actions)

        self.last_action = action
        self.last_features = cn2_input.copy()

        return action

    def get_telemetry(self) -> Dict[str, Any]:
        """Collect real-time biological and neurochemical metrics for the HUD."""
        matrix = self.brain.chemistry.matrix.state
        bg = self.brain.basal_ganglia
        autonomic = self.brain.peripheral.autonomic

        # Calculate estimated dynamic heart rate: baseline 70 BPM + sympathetic drive up to 140 BPM
        heart_rate = int(70.0 + autonomic.sympathetic_tone * 70.0)

        return {
            "dopamine": float(matrix.dopamine),
            "cortisol": float(matrix.cortisol),
            "serotonin": float(matrix.serotonin),
            "norepinephrine": float(matrix.norepinephrine),
            "adenosine": float(matrix.adenosine),
            "sympathetic_tone": float(autonomic.sympathetic_tone),
            "vagal_tone": float(autonomic.parasympathetic_tone),
            "heart_rate": heart_rate,
            "d1_weights": [float(w) for w in bg.d1_weights[:self.n_actions]],
            "d2_weights": [float(w) for w in bg.d2_weights[:self.n_actions]],
            "cerebellar_error": self.cerebellar_error,
            "hippocampal_match": self.hippocampal_match,
            "last_action": self.last_action,
            "tick_count": self.brain.tick_count,
        }
