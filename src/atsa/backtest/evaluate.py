"""Turn one strategy into a full scorecard: in-sample, out-of-sample, walk-forward, cost stress."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import BacktestConfig
from ..data.base import Bars
from ..dsl.expr import Evaluator, max_lookback
from ..dsl.indicators import jit
from ..dsl.strategy import RULE_FIELDS, Strategy
from .engine import run_backtest
from .metrics import per_period_sharpe, sharpe, summarize


@dataclass
class Evaluation:
    strategy: Strategy
    symbol: str
    full: dict = field(default_factory=dict)
    in_sample: dict = field(default_factory=dict)
    out_of_sample: dict = field(default_factory=dict)
    wf_sharpes: list[float] = field(default_factory=list)
    stress_sharpe: float = 0.0
    benchmark_sharpe: float = 0.0
    sr_per_period: float = 0.0
    returns: np.ndarray | None = None
    equity_curve: list[float] = field(default_factory=list)
    current_position: int = 0
    next_position: int = 0
    error: str | None = None
    # filled by the gate
    dsr: float = 0.0
    trials_eff: int = 0
    cluster: int = -1
    variants: int = 1
    checks: dict = field(default_factory=dict)
    passed: bool = False
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)

    @property
    def wf_positive(self) -> float:
        return float(np.mean([s > 0 for s in self.wf_sharpes])) if self.wf_sharpes else 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.strategy.id, "symbol": self.symbol, "name": self.strategy.name,
            "agent": self.strategy.agent, "generation": self.strategy.generation,
            "spec": self.strategy.spec(), "full": self.full, "in_sample": self.in_sample,
            "out_of_sample": self.out_of_sample, "wf_sharpes": self.wf_sharpes,
            "wf_positive": self.wf_positive, "stress_sharpe": self.stress_sharpe,
            "benchmark_sharpe": self.benchmark_sharpe, "dsr": self.dsr, "trials_eff": self.trials_eff,
            "cluster": self.cluster, "variants": self.variants, "checks": self.checks, "passed": self.passed,
            "score": self.score, "reasons": self.reasons, "current_position": self.current_position,
            "next_position": self.next_position,
            "error": self.error,
        }


@jit
def _slice_trades_kernel(returns, position):
    n = len(position)
    out = np.empty(n)
    k = 0
    t = 0
    while t < n:
        if position[t] != 0:
            side = position[t]
            s = t
            while t < n and position[t] == side:
                t += 1
            end = t + 1 if t + 1 < n else n
            acc = 1.0
            for i in range(s, end):
                acc *= 1.0 + returns[i]
            out[k] = acc - 1.0
            k += 1
        else:
            t += 1
    return out[:k]


def _slice_trades(returns: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Approximate per-trade returns inside a slice from position runs."""
    return _slice_trades_kernel(np.ascontiguousarray(returns, dtype=np.float64), np.ascontiguousarray(position))


def _downsample(equity: np.ndarray, points: int = 240) -> list[float]:
    if len(equity) <= points:
        return [round(float(x), 5) for x in equity]
    idx = np.linspace(0, len(equity) - 1, points).astype(int)
    return [round(float(x), 5) for x in equity[idx]]


def evaluate_strategy(bars: Bars, strategy: Strategy, cfg: BacktestConfig, periods_per_year: float,
                      evaluator: Evaluator | None = None) -> Evaluation:
    ev = Evaluation(strategy=strategy, symbol=bars.symbol)
    try:
        warmup = max(max_lookback(getattr(strategy, f)) for f in RULE_FIELDS if getattr(strategy, f))
        if warmup >= len(bars) * 0.5:
            raise ValueError(f"indicator look-back ({warmup} bars) too long for {len(bars)} bars of data")
        evaluator = evaluator or Evaluator(bars)
        res = run_backtest(bars, strategy, cfg.fee_bps, cfg.slippage_bps, cfg.allow_short, evaluator)
    except Exception as e:  # invalid strategy -> recorded, never crashes the swarm
        ev.error = f"{type(e).__name__}: {e}"
        ev.reasons = [f"error: {ev.error}"]
        return ev

    n = len(bars)
    start = min(warmup, n - 2)
    rets = res.returns[start + 1:]
    pos = res.position[start + 1:]
    split = int(len(rets) * (1.0 - cfg.oos_fraction))

    ev.full = summarize(rets, pos, res.trade_returns, periods_per_year)
    ev.in_sample = summarize(rets[:split], pos[:split], _slice_trades(rets[:split], pos[:split]), periods_per_year)
    ev.out_of_sample = summarize(rets[split:], pos[split:], _slice_trades(rets[split:], pos[split:]),
                                 periods_per_year)
    folds = max(2, cfg.wf_folds)
    ev.wf_sharpes = [round(sharpe(chunk, periods_per_year), 3) for chunk in np.array_split(rets, folds)]
    stressed = run_backtest(bars, strategy, cfg.fee_bps * 2, cfg.slippage_bps * 2, cfg.allow_short, evaluator)
    ev.stress_sharpe = sharpe(stressed.returns[start + 1:], periods_per_year)
    bh = np.diff(bars.close[start:]) / bars.close[start:-1]
    ev.benchmark_sharpe = sharpe(bh, periods_per_year)
    ev.sr_per_period = per_period_sharpe(rets)
    ev.returns = rets
    ev.equity_curve = _downsample(res.equity[start:] / res.equity[start] if res.equity[start] > 0 else res.equity[start:])
    ev.current_position = int(res.position[-1])
    ev.next_position = int(res.next_position)
    return ev
