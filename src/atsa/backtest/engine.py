"""Event-accurate bar backtester with a Numba-compiled core.

Execution model (deliberately conservative, no look-ahead):
  * Signals are computed on the CLOSE of bar t.
  * Orders fill at the OPEN of bar t+1 (gaps are paid, not skipped).
  * Stop-loss / take-profit are checked intrabar against high/low; if both could trigger in the
    same bar the stop is assumed to hit first. A gap through the stop fills at the open.
  * Every change in position pays (fee + slippage) per unit of turnover; a long->short flip pays twice.

Exit semantics when exit_long / exit_short is null:
  * with max_hold_bars -> hold for max_hold_bars (event trades, e.g. "buy the shock, hold 3 bars")
  * without            -> exit when the entry condition stops being true (state trades)
Stops, take-profits and an opposite-side entry always apply.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data.base import Bars
from ..dsl.expr import Evaluator
from ..dsl.indicators import jit
from ..dsl.strategy import Strategy


@jit
def simulate(open_, high, low, close, entry_l, exit_l, entry_s, exit_s, use_exit_l, use_exit_s,
             stop_loss, take_profit, max_hold, cost):
    """Returns (equity, position, trade_returns, n_trades, next_position).

    position[t] is held at the CLOSE of bar t; next_position is what to hold from the next open."""
    n = len(close)
    equity = np.ones(n)
    position = np.zeros(n, dtype=np.int8)
    trade_rets = np.zeros(n)
    n_trades = 0
    eq = 1.0
    pos = 0
    desired = 0
    entry_px = 0.0
    trade_start_eq = 1.0
    held = 0
    for t in range(1, n):
        start_px = close[t - 1]
        # 1) execute yesterday's decision at today's open
        if desired != pos:
            if pos != 0:
                eq *= 1.0 + pos * (open_[t] / close[t - 1] - 1.0)
                eq *= 1.0 - cost
                trade_rets[n_trades] = eq / trade_start_eq - 1.0
                n_trades += 1
            if desired != 0:
                eq *= 1.0 - cost
                trade_start_eq = eq / (1.0 - cost)  # trade P&L includes its entry cost
                entry_px = open_[t]
                held = 0
            pos = desired
            start_px = open_[t]
        # 2) intrabar stop-loss / take-profit, then mark to close
        if pos != 0:
            held += 1
            exited = False
            if pos == 1:
                if stop_loss > 0.0 and low[t] <= entry_px * (1.0 - stop_loss):
                    px = min(open_[t], entry_px * (1.0 - stop_loss))
                    eq *= px / start_px
                    exited = True
                elif take_profit > 0.0 and high[t] >= entry_px * (1.0 + take_profit):
                    px = max(open_[t], entry_px * (1.0 + take_profit))
                    eq *= px / start_px
                    exited = True
            else:
                if stop_loss > 0.0 and high[t] >= entry_px * (1.0 + stop_loss):
                    px = max(open_[t], entry_px * (1.0 + stop_loss))
                    eq *= 1.0 - (px / start_px - 1.0)
                    exited = True
                elif take_profit > 0.0 and low[t] <= entry_px * (1.0 - take_profit):
                    px = min(open_[t], entry_px * (1.0 - take_profit))
                    eq *= 1.0 - (px / start_px - 1.0)
                    exited = True
            if exited:
                eq *= 1.0 - cost
                trade_rets[n_trades] = eq / trade_start_eq - 1.0
                n_trades += 1
                pos = 0
                desired = 0
            else:
                eq *= 1.0 + pos * (close[t] / start_px - 1.0)
        if eq < 0.0:
            eq = 0.0
        equity[t] = eq
        position[t] = pos
        # 3) decide at the close what to hold from the next open
        if pos == 1:
            desired = 1
            leave = exit_l[t] if use_exit_l else (max_hold == 0 and not entry_l[t])
            if leave or (max_hold > 0 and held >= max_hold):
                desired = 0
            if entry_s[t] and not entry_l[t]:
                desired = -1
        elif pos == -1:
            desired = -1
            leave = exit_s[t] if use_exit_s else (max_hold == 0 and not entry_s[t])
            if leave or (max_hold > 0 and held >= max_hold):
                desired = 0
            if entry_l[t] and not entry_s[t]:
                desired = 1
        else:
            if entry_l[t] and not entry_s[t]:
                desired = 1
            elif entry_s[t] and not entry_l[t]:
                desired = -1
            else:
                desired = 0
    # mark an open trade at the final close
    if pos != 0:
        trade_rets[n_trades] = eq / trade_start_eq - 1.0
        n_trades += 1
    return equity, position, trade_rets[:n_trades], n_trades, desired


@dataclass
class BacktestResult:
    equity: np.ndarray
    returns: np.ndarray
    position: np.ndarray
    trade_returns: np.ndarray
    n_trades: int
    next_position: int = 0


def _signals(ev: Evaluator, strategy: Strategy, allow_short: bool):
    n = ev.n
    false = np.zeros(n, dtype=np.bool_)
    el = ev.condition(strategy.entry_long)
    xl = ev.condition(strategy.exit_long)
    es = ev.condition(strategy.entry_short) if allow_short else None
    xs = ev.condition(strategy.exit_short) if allow_short else None
    return (
        el if el is not None else false,
        xl if xl is not None else false,
        es if es is not None else false,
        xs if xs is not None else false,
        xl is not None,
        xs is not None,
    )


def run_backtest(bars: Bars, strategy: Strategy, fee_bps: float = 5.0, slippage_bps: float = 5.0,
                 allow_short: bool = True, evaluator: Evaluator | None = None) -> BacktestResult:
    ev = evaluator or Evaluator(bars)
    el, xl, es, xs, use_xl, use_xs = _signals(ev, strategy, allow_short)
    cost = (fee_bps + slippage_bps) / 1e4
    equity, position, trade_rets, n_trades, nxt = simulate(
        bars.open, bars.high, bars.low, bars.close, el, xl, es, xs, use_xl, use_xs,
        float(strategy.stop_loss or 0.0), float(strategy.take_profit or 0.0), int(strategy.max_hold_bars or 0),
        float(cost),
    )
    rets = np.zeros(len(equity))
    with np.errstate(divide="ignore", invalid="ignore"):
        rets[1:] = np.where(equity[:-1] > 0, equity[1:] / equity[:-1] - 1.0, 0.0)
    return BacktestResult(equity, rets, position, trade_rets, int(n_trades), int(nxt))
