"""Cybernetic Pygame Telemetry Desktop HUD for the BIB-2 Neuro-Trader."""

import math
import time
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pygame

# Cybernetic Theme Colors
BG_DARK = (10, 14, 22)
PANEL_BG = (14, 20, 32)
PANEL_BORDER = (28, 44, 68)
GRID_LINE = (20, 30, 48)

TEXT_WHITE = (235, 242, 255)
TEXT_MUTED = (140, 160, 185)
TEXT_GOLD = (250, 204, 21)

CANDLE_GREEN = (34, 197, 94)
CANDLE_RED = (239, 68, 68)
EMA9_COLOR = (56, 189, 248)
EMA21_COLOR = (251, 146, 60)
EQUITY_COLOR = (52, 211, 153)

ECG_LINE = (34, 211, 153)
ECG_BG = (8, 16, 24)

DA_COLOR = (250, 204, 21)
CORTISOL_COLOR = (244, 63, 94)
SEROTONIN_COLOR = (168, 85, 247)
NE_COLOR = (245, 158, 11)


class CyberneticHUD:
    """
    Renders a responsive 1280x800 60 FPS cybernetic dashboard:
    - Real-time rolling Japanese candlestick chart with 9/21 EMAs and trade flags.
    - Live portfolio equity curve.
    - Dynamic animated ECG oscilloscope (60-140 BPM heart rate tied to autonomic sympathetic tone).
    - Endocrine & neuromodulatory matrix meters (DA, Cortisol, 5-HT, NE).
    - Basal Ganglia D1/D2 channel activation bars.
    - SWS nocturnal sleep overlay screen.
    """

    def __init__(self, width: int = 1280, height: int = 800):
        pygame.init()
        pygame.display.set_caption("BIB-2 Neuro-Trader: Biomimetic Market Organism")
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_title = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_body = pygame.font.SysFont("Consolas", 13)
        self.font_small = pygame.font.SysFont("Consolas", 11)
        self.font_large = pygame.font.SysFont("Consolas", 22, bold=True)

        # ECG state
        self.ecg_points: List[float] = [0.0] * 120
        self.ecg_phase = 0.0

        # Sleep state
        self.sleep_overlay_alpha = 0

    def _generate_ecg_sample(self, phase: float) -> float:
        """Authentic P-Q-R-S-T cardiac voltage waveform generator."""
        p = phase % (2.0 * math.pi)
        val = 0.0
        # P-wave
        if 0.5 <= p < 1.0:
            val += 0.15 * math.sin((p - 0.5) * 2.0 * math.pi)
        # Q-dip
        elif 1.4 <= p < 1.6:
            val -= 0.15 * math.sin((p - 1.4) * 5.0 * math.pi)
        # R-spike
        elif 1.6 <= p < 1.9:
            val += 1.0 * math.sin((p - 1.6) * (math.pi / 0.3))
        # S-dip
        elif 1.9 <= p < 2.1:
            val -= 0.25 * math.sin((p - 1.9) * 5.0 * math.pi)
        # T-wave
        elif 2.6 <= p < 3.4:
            val += 0.25 * math.sin((p - 2.6) * (math.pi / 0.8))
        return float(val)

    def render(
        self,
        candles: List[Dict[str, float]],
        current_candle_idx: int,
        env_info: Dict[str, Any],
        telemetry: Dict[str, Any],
        trades: List[Dict[str, Any]],
        equity_history: List[float],
        speed_text: str = "1x",
        paused: bool = False,
        is_sleeping: bool = False,
        sleep_stats: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Draw one complete frame. Returns False if window close is requested.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        self.screen.fill(BG_DARK)

        # 1. Top Header Banner
        self._render_header(env_info, current_candle_idx, speed_text, paused, is_sleeping)

        # 2. Main Market Candlestick Chart (Left 62%)
        chart_rect = pygame.Rect(15, 60, 780, 460)
        self._render_candlesticks(chart_rect, candles, current_candle_idx, trades)

        # 3. Portfolio Equity Curve (Below Candlestick Chart)
        equity_rect = pygame.Rect(15, 535, 780, 215)
        self._render_equity_curve(equity_rect, equity_history, env_info)

        # 4. Upper Right: Biometrics & ECG Oscilloscope
        bio_rect = pygame.Rect(810, 60, 455, 340)
        self._render_biometrics(bio_rect, telemetry)

        # 5. Lower Right: Neural Circuits & Basal Ganglia Gating
        neural_rect = pygame.Rect(810, 415, 455, 335)
        self._render_neural_telemetry(neural_rect, telemetry, env_info)

        # 6. Bottom Status & Keybind Help Bar
        self._render_footer(paused)

        # 7. Nocturnal Sleep Overlay (if active)
        if is_sleeping:
            self._render_sleep_screen(sleep_stats)

        pygame.display.flip()
        self.clock.tick(60)
        return True

    def _render_header(self, env_info: Dict[str, Any], candle_idx: int, speed: str, paused: bool, sleeping: bool) -> None:
        pygame.draw.rect(self.screen, PANEL_BG, (15, 10, self.width - 30, 40), border_radius=6)
        pygame.draw.rect(self.screen, PANEL_BORDER, (15, 10, self.width - 30, 40), 1, border_radius=6)

        title = self.font_title.render("BIB-2 NEURO-TRADER  |  BTC/USDT 5M", True, EMA9_COLOR)
        self.screen.blit(title, (30, 20))

        equity = env_info.get("equity", 10000.0)
        ret = ((equity - 10000.0) / 10000.0) * 100.0
        ret_color = CANDLE_GREEN if ret >= 0 else CANDLE_RED
        ret_str = f"EQUITY: ${equity:,.2f} ({ret:+.2f}%)"
        ret_surf = self.font_title.render(ret_str, True, ret_color)
        self.screen.blit(ret_surf, (440, 20))

        day = (candle_idx // 288) + 1
        status_color = TEXT_GOLD if paused else (SEROTONIN_COLOR if sleeping else CANDLE_GREEN)
        status_text = "PAUSED" if paused else ("SWS SLEEP" if sleeping else f"SPEED: {speed}")
        stat_surf = self.font_body.render(f"DAY: {day} | {status_text}", True, status_color)
        self.screen.blit(stat_surf, (820, 22))

        win_rate = env_info.get("win_rate", 0.0) * 100.0
        wr_surf = self.font_body.render(f"WIN: {win_rate:.1f}% | TRADES: {env_info.get('total_trades', 0)}", True, TEXT_WHITE)
        self.screen.blit(wr_surf, (1060, 22))

    def _render_candlesticks(self, rect: pygame.Rect, candles: List[Dict[str, float]], current_idx: int, trades: List[Dict[str, Any]]) -> None:
        pygame.draw.rect(self.screen, PANEL_BG, rect, border_radius=6)
        pygame.draw.rect(self.screen, PANEL_BORDER, rect, 1, border_radius=6)

        window_size = 32
        start_idx = max(0, current_idx - window_size)
        display_candles = candles[start_idx:current_idx]

        if len(display_candles) < 2:
            return

        highs = [c["high"] for c in display_candles]
        lows = [c["low"] for c in display_candles]
        max_p = max(highs) * 1.001
        min_p = min(lows) * 0.999
        p_range = max(1e-6, max_p - min_p)

        def price_to_y(p: float) -> int:
            norm = (p - min_p) / p_range
            return int(rect.bottom - 20 - norm * (rect.height - 40))

        # Draw Grid Lines
        for step in range(5):
            p_val = min_p + (p_range * step / 4.0)
            y = price_to_y(p_val)
            pygame.draw.line(self.screen, GRID_LINE, (rect.left, y), (rect.right, y), 1)
            lbl = self.font_small.render(f"${p_val:,.1f}", True, TEXT_MUTED)
            self.screen.blit(lbl, (rect.right - 70, y - 6))

        # Compute and draw 9 & 21 EMAs
        closes = [c["close"] for c in display_candles]
        c_w = (rect.width - 90) / window_size

        ema9_points = []
        ema21_points = []
        for i in range(len(display_candles)):
            x = int(rect.left + 20 + i * c_w)
            sub_c = closes[: i + 1]
            e9 = float(np.mean(sub_c[-9:]))
            e21 = float(np.mean(sub_c[-21:] if len(sub_c) >= 21 else sub_c))
            ema9_points.append((x, price_to_y(e9)))
            ema21_points.append((x, price_to_y(e21)))

        if len(ema9_points) > 1:
            pygame.draw.lines(self.screen, EMA9_COLOR, False, ema9_points, 2)
        if len(ema21_points) > 1:
            pygame.draw.lines(self.screen, EMA21_COLOR, False, ema21_points, 2)

        # Draw Candlesticks
        for i, c in enumerate(display_candles):
            x = int(rect.left + 20 + i * c_w)
            o = c["open"]
            close = c["close"]
            h = c["high"]
            l = c["low"]

            color = CANDLE_GREEN if close >= o else CANDLE_RED
            y_h = price_to_y(h)
            y_l = price_to_y(l)
            y_o = price_to_y(o)
            y_c = price_to_y(close)

            # Wick
            pygame.draw.line(self.screen, color, (x, y_h), (x, y_l), 1)
            # Body
            top_y = min(y_o, y_c)
            b_height = max(2, abs(y_o - y_c))
            b_w = max(4, int(c_w * 0.7))
            pygame.draw.rect(self.screen, color, (x - b_w // 2, top_y, b_w, b_height))

            # Check if this candle had a trade exit/entry
            candle_global_idx = start_idx + i
            for tr in trades[-10:]:
                if tr.get("idx") == candle_global_idx:
                    if tr.get("action") == 1:  # BUY
                        pygame.draw.polygon(self.screen, CANDLE_GREEN, [(x, y_l + 12), (x - 6, y_l + 20), (x + 6, y_l + 20)])
                    elif tr.get("action") == 2:  # SELL
                        pygame.draw.polygon(self.screen, CANDLE_RED, [(x, y_h - 12), (x - 6, y_h - 20), (x + 6, y_h - 20)])

        # Legend
        l1 = self.font_small.render("— EMA 9", True, EMA9_COLOR)
        l2 = self.font_small.render("— EMA 21", True, EMA21_COLOR)
        self.screen.blit(l1, (rect.left + 15, rect.top + 10))
        self.screen.blit(l2, (rect.left + 85, rect.top + 10))

    def _render_equity_curve(self, rect: pygame.Rect, equity_history: List[float], env_info: Dict[str, Any]) -> None:
        pygame.draw.rect(self.screen, PANEL_BG, rect, border_radius=6)
        pygame.draw.rect(self.screen, PANEL_BORDER, rect, 1, border_radius=6)

        lbl = self.font_title.render("PORTFOLIO PERFORMANCE & EQUITY CURVE", True, TEXT_WHITE)
        self.screen.blit(lbl, (rect.left + 15, rect.top + 10))

        if len(equity_history) < 2:
            return

        window = equity_history[-100:]
        max_e = max(window) * 1.002
        min_e = min(window) * 0.998
        e_range = max(1.0, max_e - min_e)

        points = []
        step_x = (rect.width - 30) / (len(window) - 1)
        for i, val in enumerate(window):
            x = int(rect.left + 15 + i * step_x)
            norm = (val - min_e) / e_range
            y = int(rect.bottom - 20 - norm * (rect.height - 60))
            points.append((x, y))

        if len(points) > 1:
            pygame.draw.lines(self.screen, EQUITY_COLOR, False, points, 2)

        cash = env_info.get("cash", 10000.0)
        btc = env_info.get("btc", 0.0)
        dd = env_info.get("max_drawdown", 0.0) * 100.0
        sub_str = f"CASH: ${cash:,.2f} | BTC: {btc:.4f} | MAX DRAWDOWN: {dd:.2f}%"
        sub_surf = self.font_small.render(sub_str, True, TEXT_MUTED)
        self.screen.blit(sub_surf, (rect.left + 15, rect.bottom - 20))

    def _render_biometrics(self, rect: pygame.Rect, telemetry: Dict[str, Any]) -> None:
        pygame.draw.rect(self.screen, PANEL_BG, rect, border_radius=6)
        pygame.draw.rect(self.screen, PANEL_BORDER, rect, 1, border_radius=6)

        # Title & Heart Rate
        hr = telemetry.get("heart_rate", 70)
        hr_color = CANDLE_RED if hr > 110 else (TEXT_GOLD if hr > 90 else CANDLE_GREEN)
        hr_state = "TACHYCARDIA (STRESS)" if hr > 110 else ("ELEVATED" if hr > 90 else "HOMEOSTATIC")

        lbl = self.font_title.render("AUTONOMIC BIOMETRICS & ECG", True, TEXT_WHITE)
        hr_surf = self.font_body.render(f"HEART RATE: {hr} BPM [{hr_state}]", True, hr_color)
        self.screen.blit(lbl, (rect.left + 15, rect.top + 10))
        self.screen.blit(hr_surf, (rect.left + 15, rect.top + 32))

        # ECG Oscilloscope Box
        ecg_rect = pygame.Rect(rect.left + 15, rect.top + 55, rect.width - 30, 95)
        pygame.draw.rect(self.screen, ECG_BG, ecg_rect, border_radius=4)
        pygame.draw.rect(self.screen, PANEL_BORDER, ecg_rect, 1, border_radius=4)

        # Advance ECG Phase based on current heart rate
        speed_factor = (hr / 60.0) * 0.15
        self.ecg_phase += speed_factor
        new_sample = self._generate_ecg_sample(self.ecg_phase)
        self.ecg_points.append(new_sample)
        if len(self.ecg_points) > ecg_rect.width:
            self.ecg_points.pop(0)

        mid_y = ecg_rect.centery
        pts = []
        for i, val in enumerate(self.ecg_points):
            x = ecg_rect.left + i
            y = int(mid_y - val * 35.0)
            pts.append((x, y))

        if len(pts) > 1:
            pygame.draw.lines(self.screen, ECG_LINE, False, pts, 2)
            # Glowing lead tip
            pygame.draw.circle(self.screen, (200, 255, 230), pts[-1], 3)

        # Endocrine Matrix Meters (DA, Cortisol, 5-HT, NE)
        y_meter = rect.top + 165
        meters = [
            ("DOPAMINE (REWARD/SEEK)", telemetry.get("dopamine", 0.5), DA_COLOR),
            ("CORTISOL (STRESS/HPA)", telemetry.get("cortisol", 0.2), CORTISOL_COLOR),
            ("SEROTONIN (PATIENCE)", telemetry.get("serotonin", 0.5), SEROTONIN_COLOR),
            ("NOREPINEPHRINE (AROUSAL)", telemetry.get("norepinephrine", 0.4), NE_COLOR),
        ]

        for name, val, col in meters:
            t_surf = self.font_small.render(f"{name}: {val:.2f}", True, TEXT_WHITE)
            self.screen.blit(t_surf, (rect.left + 15, y_meter))

            bar_bg = pygame.Rect(rect.left + 220, y_meter + 2, 200, 10)
            pygame.draw.rect(self.screen, (20, 30, 48), bar_bg, border_radius=3)
            fill_w = int(max(0.0, min(1.0, val)) * 200)
            if fill_w > 0:
                pygame.draw.rect(self.screen, col, (rect.left + 220, y_meter + 2, fill_w, 10), border_radius=3)
            y_meter += 22

        # Sympatho-Vagal Balance
        sym = telemetry.get("sympathetic_tone", 0.3)
        vag = telemetry.get("vagal_tone", 0.7)
        vagal_str = f"SYMPATHETIC: {sym:.2f}  |  PARASYMPATHETIC (VAGUS): {vag:.2f}"
        self.screen.blit(self.font_small.render(vagal_str, True, TEXT_MUTED), (rect.left + 15, y_meter + 5))

    def _render_neural_telemetry(self, rect: pygame.Rect, telemetry: Dict[str, Any], env_info: Dict[str, Any]) -> None:
        pygame.draw.rect(self.screen, PANEL_BG, rect, border_radius=6)
        pygame.draw.rect(self.screen, PANEL_BORDER, rect, 1, border_radius=6)

        lbl = self.font_title.render("NEURAL SUB-SYSTEMS & ACTION GATING", True, TEXT_WHITE)
        self.screen.blit(lbl, (rect.left + 15, rect.top + 10))

        # Basal Ganglia 3 Channels (HOLD, BUY, SELL)
        actions = ["0: HOLD", "1: BUY", "2: SELL"]
        d1 = telemetry.get("d1_weights", [1.0, 1.0, 1.0])
        d2 = telemetry.get("d2_weights", [1.0, 1.0, 1.0])
        last_action = telemetry.get("last_action", 0)

        y_bg = rect.top + 38
        self.screen.blit(self.font_body.render("BASAL GANGLIA TRIPARTITE GATING:", True, TEXT_GOLD), (rect.left + 15, y_bg))
        y_bg += 22

        for i, act_name in enumerate(actions):
            is_active = (i == last_action)
            border_col = TEXT_GOLD if is_active else PANEL_BORDER
            badge_bg = (30, 48, 75) if is_active else PANEL_BG

            ch_rect = pygame.Rect(rect.left + 15, y_bg, rect.width - 30, 26)
            pygame.draw.rect(self.screen, badge_bg, ch_rect, border_radius=4)
            pygame.draw.rect(self.screen, border_col, ch_rect, 1, border_radius=4)

            tag = self.font_body.render(act_name, True, TEXT_WHITE if is_active else TEXT_MUTED)
            self.screen.blit(tag, (rect.left + 25, y_bg + 5))

            w1 = d1[i] if i < len(d1) else 1.0
            w2 = d2[i] if i < len(d2) else 1.0
            stat = self.font_small.render(f"D1 (Go): {w1:.2f} | D2 (NoGo): {w2:.2f}", True, EMA9_COLOR if is_active else TEXT_MUTED)
            self.screen.blit(stat, (rect.left + 160, y_bg + 7))

            if is_active:
                act_icon = self.font_small.render("[ACTIVE DISINHIBITION]", True, TEXT_GOLD)
                self.screen.blit(act_icon, (rect.right - 180, y_bg + 7))
            y_bg += 32

        # Cerebellar Forward Prediction Error
        c_err = telemetry.get("cerebellar_error", 0.0)
        c_status = "SURPRISE / HIGH ERROR" if c_err > 1.5 else "CALIBRATED PREDICTION"
        c_col = CANDLE_RED if c_err > 1.5 else CANDLE_GREEN
        c_txt = self.font_body.render(f"CEREBELLUM SMITH PREDICTOR: {c_status}", True, c_col)
        self.screen.blit(c_txt, (rect.left + 15, y_bg + 10))

        # Hippocampal CA3 Episodic Match
        hip_match = telemetry.get("hippocampal_match", 0.0)
        hip_col = SEROTONIN_COLOR if hip_match > 0.6 else TEXT_MUTED
        hip_txt = self.font_body.render(f"HIPPOCAMPUS CA3 PATTERN MATCH: {hip_match * 100:.1f}%", True, hip_col)
        self.screen.blit(hip_txt, (rect.left + 15, y_bg + 32))

        # STN Emergency Hard Stop Alarm
        stn_trig = env_info.get("stn_triggered", False)
        unrealized = env_info.get("unrealized_pnl", 0.0) * 100.0
        if stn_trig:
            stn_txt = self.font_body.render("STN EMERGENCY BRAKE: FIRED! (LIQUIDATED)", True, CANDLE_RED)
        else:
            stn_txt = self.font_body.render(f"STN STOP-LOSS: ARMED (-5.0% LIMIT) [OPEN: {unrealized:+.2f}%]", True, TEXT_MUTED)
        self.screen.blit(stn_txt, (rect.left + 15, y_bg + 54))

    def _render_footer(self, paused: bool) -> None:
        y_bot = self.height - 35
        guide = "HOTKEYS: [SPACE] Pause/Resume  |  [1-4] Speed (1x, 5x, 20x, MAX)  |  [S] Nocturnal Sleep  |  [K] Shock  |  [Q] Exit"
        surf = self.font_body.render(guide, True, TEXT_MUTED)
        self.screen.blit(surf, (30, y_bot))

    def _render_sleep_screen(self, sleep_stats: Optional[Dict[str, Any]]) -> None:
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((6, 12, 28, 220))  # Deep nocturnal blue
        self.screen.blit(overlay, (0, 0))

        title = self.font_large.render("NOCTURNAL SLOW-WAVE SLEEP (NREM SWS)", True, EMA9_COLOR)
        self.screen.blit(title, (self.width // 2 - title.get_width() // 2, 250))

        sub = self.font_body.render("Hippocampal Sharp-Wave Ripples (150-250 Hz) replaying profitable trades into Neocortex...", True, TEXT_WHITE)
        self.screen.blit(sub, (self.width // 2 - sub.get_width() // 2, 300))

        # Draw undulating delta waves
        pts = []
        t = time.time() * 3.0
        for x in range(200, self.width - 200, 4):
            y = int(390 + 30.0 * math.sin(x * 0.02 + t) + 15.0 * math.sin(x * 0.05 - t))
            pts.append((x, y))
        if len(pts) > 1:
            pygame.draw.lines(self.screen, EMA9_COLOR, False, pts, 3)

        shy_txt = self.font_body.render("Tononi Synaptic Homeostasis (SHY): -5.0% Synaptic Downscaling Applied", True, TEXT_GOLD)
        self.screen.blit(shy_txt, (self.width // 2 - shy_txt.get_width() // 2, 450))

        gly_txt = self.font_body.render("Glymphatics: Astrocytic AQP4 Convective Clearance Flushed Adenosine", True, CANDLE_GREEN)
        self.screen.blit(gly_txt, (self.width // 2 - gly_txt.get_width() // 2, 480))
