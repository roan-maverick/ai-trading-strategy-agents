"""The Robustness Gate: only strategies that survive every test are called winners.

Tests (all configurable in `gate:`):
  1. enough trades to mean anything
  2. Sharpe in-sample AND on the unseen out-of-sample period
  3. max drawdown cap
  4. walk-forward stability: positive Sharpe in most time slices, not one lucky year
  5. cost stress: still profitable with 2x fees + slippage
  6. Deflated Sharpe Ratio: adjusts for how many strategies the swarm tried (anti data-mining)
"""

from __future__ import annotations

import numpy as np

from ..config import GateConfig
from .evaluate import Evaluation
from .metrics import deflated_sharpe, effective_trials


def raw_score(ev: Evaluation) -> float:
    if ev.error or not ev.full:
        return -99.0
    oos = ev.out_of_sample.get("sharpe", 0.0)
    ins = min(ev.in_sample.get("sharpe", 0.0), 3.0)
    calmar = float(np.clip(ev.full.get("calmar", 0.0), -3.0, 3.0))
    return float((0.5 * oos + 0.3 * ins + 0.2 * calmar) * (0.5 + 0.5 * ev.wf_positive))


def trial_universe(evals: list[Evaluation]) -> tuple[np.ndarray, int]:
    """Sharpe ratios of the *independent* ideas tried (correlated variants are clustered together).
    Also stamps each evaluation with its idea cluster so near-duplicate winners can be grouped."""
    valid = [e for e in evals if not e.error and e.returns is not None and len(e.returns) > 10]
    if not valid:
        return np.array([]), 0
    length = min(len(e.returns) for e in valid)
    matrix = np.vstack([e.returns[-length:] for e in valid])
    srs, n, labels = effective_trials(matrix, return_labels=True)
    for e, lab in zip(valid, labels):
        e.cluster = int(lab)
    return srs, n


def distinct(evals: list[Evaluation]) -> list[Evaluation]:
    """Best evaluation per (symbol, idea cluster), each annotated with how many variants it represents."""
    best: dict[tuple, Evaluation] = {}
    counts: dict[tuple, int] = {}
    for e in sorted(evals, key=lambda x: x.score, reverse=True):
        key = (e.symbol, e.cluster if e.cluster >= 0 else e.strategy.id)
        counts[key] = counts.get(key, 0) + 1
        best.setdefault(key, e)
    for key, e in best.items():
        e.variants = counts[key]
    return list(best.values())


def apply_gate(evals: list[Evaluation], cfg: GateConfig, trial_sharpes: np.ndarray | None = None) -> list[Evaluation]:
    """Score + pass/fail every evaluation.

    trial_sharpes: per-period Sharpes of every independent idea tried on this symbol. Default: computed
    from `evals` with correlation clustering. Used by the Deflated Sharpe Ratio (multiple-testing control).
    """
    n_eff = len(trial_sharpes) if trial_sharpes is not None else 0
    if trial_sharpes is None:
        trial_sharpes, n_eff = trial_universe(evals)
    for ev in evals:
        if ev.error or ev.returns is None:
            ev.passed, ev.score, ev.dsr = False, -99.0, 0.0
            if not ev.reasons:
                ev.reasons = [f"error: {ev.error}"]
            continue
        ev.dsr = deflated_sharpe(ev.returns, trial_sharpes)
        ev.trials_eff = n_eff
        reasons = []
        f, i, o = ev.full, ev.in_sample, ev.out_of_sample
        if f["trades"] < cfg.min_trades:
            reasons.append(f"too few trades ({f['trades']} < {cfg.min_trades})")
        if i["sharpe"] < cfg.min_sharpe_is:
            reasons.append(f"in-sample Sharpe {i['sharpe']:.2f} < {cfg.min_sharpe_is}")
        if o["sharpe"] < cfg.min_sharpe_oos:
            reasons.append(f"out-of-sample Sharpe {o['sharpe']:.2f} < {cfg.min_sharpe_oos}")
        if f["max_drawdown"] > cfg.max_drawdown:
            reasons.append(f"max drawdown {f['max_drawdown']:.0%} > {cfg.max_drawdown:.0%}")
        if ev.wf_positive < cfg.min_wf_positive:
            reasons.append(f"walk-forward: only {ev.wf_positive:.0%} of periods profitable")
        if ev.stress_sharpe < cfg.min_stress_sharpe:
            reasons.append(f"dies under 2x costs (Sharpe {ev.stress_sharpe:.2f})")
        if ev.dsr < cfg.min_dsr:
            reasons.append(f"deflated Sharpe {ev.dsr:.2f} < {cfg.min_dsr} (likely luck)")
        ev.reasons = reasons
        ev.passed = not reasons
        ev.score = raw_score(ev) * max(ev.dsr, 0.0)
    return evals
