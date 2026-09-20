"""Spot long-only trading execution environment with institutional fees, slippage, and STN stops."""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np


class SpotTradingEnv:
    """
    Spot Long-Only Execution Environment:
    - Capital: Initial USDT cash balance (default: $10,000.00).
    - Actions: 0 (HOLD), 1 (BUY), 2 (SELL / CASH).
    - STN Stop: Subthalamic Nucleus emergency hard stop (involuntary liquidation on drawdown <= -5%).
    - Friction: 0.075% maker/taker trading fee and volatility slippage.
    """

    def __init__(
        self,
        initial_cash: float = 10000.0,
        fee: float = 0.00075,
        stn_stop_pct: float = -0.05,
        base_slippage: float = 0.0002,
    ):
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.btc = 0.0
        self.fee_rate = float(fee)
        self.stn_stop_pct = float(stn_stop_pct)
        self.base_slippage = float(base_slippage)

        self.entry_price = 0.0
        self.peak_equity = float(initial_cash)
        self.max_drawdown = 0.0

        self.trade_history: List[Dict[str, float]] = []
        self.total_trades = 0
        self.winning_trades = 0

    def reset(self) -> None:
        """Reset account state to initial capital."""
        self.cash = self.initial_cash
        self.btc = 0.0
        self.entry_price = 0.0
        self.peak_equity = self.initial_cash
        self.max_drawdown = 0.0
        self.trade_history = []
        self.total_trades = 0
        self.winning_trades = 0

    def get_equity(self, current_price: float) -> float:
        """Calculate total liquidation value in USDT."""
        return float(self.cash + self.btc * current_price)

    def get_unrealized_pnl(self, current_price: float) -> float:
        """Return percentage unrealized return of currently held position."""
        if self.btc <= 0.0 or self.entry_price <= 0.0:
            return 0.0
        return float((current_price - self.entry_price) / self.entry_price)

    def step(
        self,
        action: int,
        current_price: float,
        volatility_atr: float = 0.002
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Execute trade action at current candle close:
        0: HOLD
        1: BUY (Allocates 98% of available cash into BTC)
        2: SELL (Liquidates 100% of BTC into cash)
        """
        reward = 0.0
        stn_triggered = False
        effective_slippage = self.base_slippage * (1.0 + 10.0 * volatility_atr)

        # 1. Check Subthalamic Nucleus (STN) Hyperdirect Emergency Stop
        unrealized = self.get_unrealized_pnl(current_price)
        if self.btc > 0.0 and unrealized <= self.stn_stop_pct:
            action = 2  # Involuntary override to SELL / CASH
            stn_triggered = True

        # 2. Execute Actions
        if action == 1 and self.btc == 0.0 and self.cash > 10.0:
            # BUY: allocate 98% of cash to allow for fee room
            invest_cash = self.cash * 0.98
            exec_price = current_price * (1.0 + effective_slippage)
            fee = invest_cash * self.fee_rate
            net_cash = invest_cash - fee

            self.btc = net_cash / exec_price
            self.cash -= invest_cash
            self.entry_price = current_price

        elif action == 2 and self.btc > 0.0:
            # SELL / CASH: liquidate all BTC
            exec_price = current_price * (1.0 - effective_slippage)
            gross_cash = self.btc * exec_price
            fee = gross_cash * self.fee_rate
            net_cash = gross_cash - fee

            trade_pnl = (exec_price - self.entry_price) / (self.entry_price + 1e-6)
            self.cash += net_cash
            self.btc = 0.0

            self.total_trades += 1
            if trade_pnl > 0:
                self.winning_trades += 1
            self.trade_history.append({
                "pnl": float(trade_pnl),
                "exit_price": float(exec_price),
                "stn": stn_triggered
            })

            # Reward Prediction Error (RPE) feedback
            # Scaled so 5% return gives ~0.5 RPE reward; loss gives negative RPE
            reward = float(np.clip(trade_pnl * 10.0, -1.0, 1.0))
            self.entry_price = 0.0

        elif action == 0 and self.btc > 0.0:
            # HOLD: Small continuous somatic feedback on open unrealized position
            reward = float(np.clip(unrealized * 0.05, -0.2, 0.2))

        # 3. Update Portfolio Drawdown Stats
        equity = self.get_equity(current_price)
        if equity > self.peak_equity:
            self.peak_equity = equity
        dd = (self.peak_equity - equity) / (self.peak_equity + 1e-6)
        if dd > self.max_drawdown:
            self.max_drawdown = float(dd)

        info = {
            "equity": equity,
            "cash": self.cash,
            "btc": self.btc,
            "unrealized_pnl": self.get_unrealized_pnl(current_price),
            "max_drawdown": self.max_drawdown,
            "stn_triggered": stn_triggered,
            "win_rate": (self.winning_trades / self.total_trades) if self.total_trades > 0 else 0.0,
            "total_trades": self.total_trades,
        }

        return reward, False, info
