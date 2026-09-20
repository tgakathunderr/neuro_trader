"""Main entrypoint executable for the BIB-2 Neuro-Trader biomimetic market organism."""

import argparse
import os
import sys
import time
from typing import List, Dict, Any
import numpy as np

# Ensure root directory and BIB-2 are on path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_BIB2_PATH = os.path.join(_ROOT, "BIB-2")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _BIB2_PATH not in sys.path:
    sys.path.insert(0, _BIB2_PATH)

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from bib2.brain import BIB2NervousSystem
from neuro_trader.data.market import MarketDataFeed
from neuro_trader.trading.env import SpotTradingEnv
from neuro_trader.adapter import TradingAdapter
from neuro_trader.trading.sleep_manager import SleepManager


def run_trader(
    headless: bool = False,
    limit: int = 1500,
    speed: int = 5,
    initial_cash: float = 10000.0,
    cache_file: str = None,
    regime: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the Neuro-Trader simulation loop either headlessly or with the Pygame HUD."""
    regime_str = f" [Regime: {regime.upper()}]" if regime else ""
    print("=" * 80)
    print(f"[BRAIN] BIB-2 NEURO-TRADER: BIOMIMETIC MARKET ORGANISM{regime_str}")
    print(f"Mode: {'HEADLESS BACKTEST' if headless else 'INTERACTIVE CYBERNETIC HUD'}")
    print(f"Target Candles: {limit} (5m)  |  Initial Capital: ${initial_cash:,.2f} USDT")
    print("=" * 80)

    # 1. Initialize Subsystems
    feed = MarketDataFeed(cache_file=cache_file)
    candles = feed.load_data(limit=limit, regime=regime)
    n_candles = len(candles)
    print(f"Loaded {n_candles} historical 5-minute candles.")

    env = SpotTradingEnv(initial_cash=initial_cash)
    brain = BIB2NervousSystem(feature_dim=64, seed=42)
    adapter = TradingAdapter(brain)
    sleep_mgr = SleepManager(brain, session_length=288)

    hud = None
    if not headless:
        import pygame
        from neuro_trader.hud import CyberneticHUD
        hud = CyberneticHUD(width=1280, height=800)

    # State tracking
    equity_history: List[float] = [initial_cash]
    trade_markers: List[Dict[str, Any]] = []
    last_reward = 0.0
    paused = False
    is_sleeping = False
    sleep_frames = 0
    sleep_stats = None

    speed_map = {1: "1x", 5: "5x", 20: "20x", 100: "MAX"}
    current_speed = speed
    speed_text = speed_map.get(current_speed, f"{current_speed}x")

    t_start = time.time()
    candle_idx = 16  # Warmup window

    while candle_idx < n_candles:
        # Handle Pygame interactive events
        if hud is not None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    candle_idx = n_candles
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                        candle_idx = n_candles
                        break
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_1:
                        current_speed = 1
                        speed_text = "1x"
                    elif event.key == pygame.K_2:
                        current_speed = 5
                        speed_text = "5x"
                    elif event.key == pygame.K_3:
                        current_speed = 20
                        speed_text = "20x"
                    elif event.key == pygame.K_4:
                        current_speed = 100
                        speed_text = "MAX"
                    elif event.key == pygame.K_s:
                        # Force sleep cycle
                        is_sleeping = True
                        sleep_frames = 60
                        sleep_stats = sleep_mgr.execute_sws_sleep()
                    elif event.key == pygame.K_k:
                        # Inject artificial volatility flash shock
                        print("⚡ Manual Market Flash Crash Shock Injected!")
                        brain.peripheral.autonomic.trigger_sympathetic_surge(intensity=1.0)
                        brain.habenula.compute_anti_reward(expected_reward=0.8, received_reward=-0.8)

        if paused:
            if hud is not None:
                telemetry = adapter.get_telemetry()
                hud.render(
                    candles=candles,
                    current_candle_idx=candle_idx,
                    env_info=info if 'info' in locals() else {"equity": initial_cash},
                    telemetry=telemetry,
                    trades=trade_markers,
                    equity_history=equity_history,
                    speed_text=speed_text,
                    paused=True,
                    is_sleeping=is_sleeping,
                    sleep_stats=sleep_stats,
                )
            time.sleep(0.05)
            continue

        # Handle Sleep Display Timeout
        if is_sleeping:
            sleep_frames -= 1
            if sleep_frames <= 0:
                is_sleeping = False

        # Step Simulation
        candle = feed.get_candle(candle_idx)
        current_price = candle["close"]
        features = feed.extract_features(candle_idx)

        # Check automated nocturnal sleep cycle (every 288 candles = 24 hours)
        if sleep_mgr.should_sleep(candle_idx):
            is_sleeping = True
            sleep_frames = 45 if not headless else 0
            sleep_stats = sleep_mgr.execute_sws_sleep()
            if headless or candle_idx % 288 == 0:
                print(f"[NIGHT SWS] Day {(candle_idx // 288)} Ended: Slow-Wave Sleep Consolidation Executed (Replayed {sleep_stats['replayed_count']} trades, Tononi SHY downscaled).")

        # 1. Biological Decision Gating
        action = adapter.decide_action(
            market_features=features,
            current_price=current_price,
            position_exposure=1.0 if env.btc > 0 else 0.0,
            unrealized_pnl=env.get_unrealized_pnl(current_price),
            last_reward=last_reward
        )

        # 2. Financial Order Execution
        prev_btc = env.btc
        reward, _, info = env.step(action, current_price, volatility_atr=features[49] / 100.0)
        last_reward = reward

        # Record trade marker
        if env.btc > 0 and prev_btc == 0.0:
            trade_markers.append({"idx": candle_idx, "action": 1, "price": current_price})
        elif env.btc == 0.0 and prev_btc > 0.0:
            trade_markers.append({"idx": candle_idx, "action": 2, "price": current_price})

        equity = info["equity"]
        equity_history.append(equity)

        # Render HUD if visual mode
        if hud is not None:
            telemetry = adapter.get_telemetry()
            hud_alive = hud.render(
                candles=candles,
                current_candle_idx=candle_idx,
                env_info=info,
                telemetry=telemetry,
                trades=trade_markers,
                equity_history=equity_history,
                speed_text=speed_text,
                paused=False,
                is_sleeping=is_sleeping,
                sleep_stats=sleep_stats,
            )
            if not hud_alive:
                break

            # Speed rate limiting
            if current_speed == 1:
                time.sleep(0.5)
            elif current_speed == 5:
                time.sleep(0.08)
            elif current_speed == 20:
                time.sleep(0.01)

        elif candle_idx % 100 == 0:
            ret = ((equity - initial_cash) / initial_cash) * 100.0
            print(f"Candle {candle_idx}/{n_candles} | Price: ${current_price:,.2f} | Equity: ${equity:,.2f} ({ret:+.2f}%) | Trades: {env.total_trades} (Win: {env.winning_trades})")

        candle_idx += 1

    # Cleanup GUI
    if hud is not None:
        import pygame
        pygame.quit()

    duration = max(0.001, time.time() - t_start)
    final_equity = env.get_equity(candles[min(candle_idx, n_candles - 1)]["close"])
    total_return = ((final_equity - initial_cash) / initial_cash) * 100.0
    bh_return = ((candles[min(candle_idx, n_candles - 1)]["close"] - candles[16]["close"]) / candles[16]["close"]) * 100.0
    alpha = total_return - bh_return

    print("\n" + "=" * 80)
    print("[SUMMARY] BIB-2 NEURO-TRADER PERFORMANCE REPORT")
    print("=" * 80)
    print(f"Candles Processed:     {candle_idx - 16} in {duration:.2f}s ({((candle_idx - 16) / duration):.1f} candles/sec)")
    print(f"Initial Capital:       ${initial_cash:,.2f} USDT")
    print(f"Ending Equity:         ${final_equity:,.2f} USDT")
    print(f"Neuro-Trader Return:   {total_return:+.2f}%")
    print(f"Buy & Hold Return:     {bh_return:+.2f}%")
    print(f"Alpha (Excess Return): {alpha:+.2f}%")
    print(f"Max Drawdown:          {env.max_drawdown * 100.0:.2f}%")
    print(f"Total Trades Closed:   {env.total_trades}")
    print(f"Winning Trades:        {env.winning_trades} ({((env.winning_trades / env.total_trades) * 100.0 if env.total_trades > 0 else 0.0):.1f}%)")
    print(f"Sleep Cycles Held:     {sleep_mgr.total_sleep_cycles}")
    print(f"Final Dopamine Level:  {brain.chemistry.matrix.state.dopamine:.3f}")
    print(f"Final Cortisol Level:  {brain.chemistry.matrix.state.cortisol:.3f}")
    print(f"Basal Ganglia D1/D2:   D1={brain.basal_ganglia.d1_weights[:3]}  D2={brain.basal_ganglia.d2_weights[:3]}")
    print("=" * 80)

    return {
        "initial_cash": initial_cash,
        "final_equity": final_equity,
        "total_return": total_return,
        "bh_return": bh_return,
        "alpha": alpha,
        "max_drawdown": env.max_drawdown,
        "total_trades": env.total_trades,
        "winning_trades": env.winning_trades,
        "sleep_cycles": sleep_mgr.total_sleep_cycles,
    }


def main():
    parser = argparse.ArgumentParser(description="BIB-2 Neuro-Trader: Biomimetic Market Organism")
    parser.add_argument("--headless", action="store_true", help="Run in fast headless backtest mode without Pygame window")
    parser.add_argument("--limit", type=int, default=1000, help="Number of 5-minute candles to process (default: 1000)")
    parser.add_argument("--speed", type=int, default=5, choices=[1, 5, 20, 100], help="Simulation speed multiplier (1, 5, 20, 100)")
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial USDT capital (default: 10000.0)")
    parser.add_argument("--csv", type=str, default=None, help="Custom historical CSV path")
    parser.add_argument("--regime", type=str, default=None, choices=["mixed", "bear", "bull"], help="Market regime simulation (mixed, bear, bull)")
    args = parser.parse_args()

    run_trader(
        headless=args.headless,
        limit=args.limit,
        speed=args.speed,
        initial_cash=args.capital,
        cache_file=args.csv,
        regime=args.regime,
    )


if __name__ == "__main__":
    main()
