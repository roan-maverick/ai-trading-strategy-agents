"""Performance metrics and overfitting statistics (PSR / Deflated Sharpe Ratio)."""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

_N = NormalDist()
EULER_GAMMA = 0.5772156649015329


def sharpe(returns: np.ndarray, periods_per_year: float) -> float:
    r = returns[np.isfinite(returns)]
    if len(r) < 3:
        return 0.0
    sd = r.std(ddof=1)
    if sd <= 1e-12:
        return 0.0
    return float(r.mean() / sd * math.sqrt(periods_per_year))


def sortino(returns: np.ndarray, periods_per_year: float) -> float:
    r = returns[np.isfinite(returns)]
    if len(r) < 3:
        return 0.0
    downside = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))
    if downside <= 1e-12:
        return 0.0
    return float(r.mean() / downside * math.sqrt(periods_per_year))


def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, 1.0 - equity / peak, 0.0)
    return float(np.max(dd))


def cagr(equity: np.ndarray, periods_per_year: float) -> float:
    if len(equity) < 2 or equity[0] <= 0:
        return 0.0
    years = (len(equity) - 1) / periods_per_year
    if years <= 0:
        return 0.0
    growth = equity[-1] / equity[0]
    if growth <= 0:
        return -1.0
    return float(growth ** (1.0 / years) - 1.0)


def summarize(returns: np.ndarray, position: np.ndarray, trade_returns: np.ndarray,
              periods_per_year: float) -> dict:
    """Metrics for one return stream (full sample, in-sample or out-of-sample slice)."""
    equity = np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    ann_vol = float(np.std(returns, ddof=1) * math.sqrt(periods_per_year)) if len(returns) > 2 else 0.0
    mdd = max_drawdown(equity)
    c = cagr(equity, periods_per_year)
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]
    pf = float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else (float("inf") if len(wins) else 0.0)
    turnover = float(np.abs(np.diff(position.astype(np.float64))).sum()) if len(position) > 1 else 0.0
    years = max(len(returns) / periods_per_year, 1e-9)
    return {
        "total_return": float(equity[-1] - 1.0),
        "cagr": c,
        "ann_vol": ann_vol,
        "sharpe": sharpe(returns, periods_per_year),
        "sortino": sortino(returns, periods_per_year),
        "max_drawdown": mdd,
        "calmar": float(c / mdd) if mdd > 1e-9 else 0.0,
        "trades": int(len(trade_returns)),
        "win_rate": float(len(wins) / len(trade_returns)) if len(trade_returns) else 0.0,
        "profit_factor": min(pf, 99.0),
        "avg_trade": float(trade_returns.mean()) if len(trade_returns) else 0.0,
        "exposure": float(np.mean(position != 0)) if len(position) else 0.0,
        "turnover_per_year": turnover / years,
    }


def _moments(returns: np.ndarray) -> tuple[float, float, float]:
    r = returns[np.isfinite(returns)]
    if len(r) < 4 or r.std() <= 1e-12:
        return 0.0, 0.0, 3.0
    sr = r.mean() / r.std(ddof=1)
    z = (r - r.mean()) / r.std()
    return float(sr), float(np.mean(z ** 3)), float(np.mean(z ** 4))


def probabilistic_sharpe(returns: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """PSR (Bailey & Lopez de Prado 2012): P(true per-period Sharpe > sr_benchmark)."""
    sr, skew, kurt = _moments(returns)
    t = int(np.isfinite(returns).sum())
    if t < 4:
        return 0.0
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    if denom <= 1e-12:
        return 0.0
    return float(_N.cdf((sr - sr_benchmark) * math.sqrt(t - 1) / math.sqrt(denom)))


def expected_max_sharpe(trial_sharpes: np.ndarray) -> float:
    """Expected maximum per-period Sharpe among N unskilled trials (False Strategy Theorem)."""
    s = np.asarray(trial_sharpes, dtype=float)
    s = s[np.isfinite(s)]
    n = len(s)
    if n < 2:
        return 0.0
    var = float(np.var(s, ddof=1))
    if var <= 0:
        return 0.0
    a = _N.inv_cdf(1.0 - 1.0 / n)
    b = _N.inv_cdf(1.0 - 1.0 / (n * math.e))
    return float(math.sqrt(var) * ((1.0 - EULER_GAMMA) * a + EULER_GAMMA * b))


def deflated_sharpe(returns: np.ndarray, trial_sharpes: np.ndarray) -> float:
    """DSR (Bailey & Lopez de Prado 2014): PSR against the Sharpe you'd expect from luck alone,
    given how many strategies were tried. ~0.95+ means the edge very likely isn't a data-mining fluke."""
    return probabilistic_sharpe(returns, expected_max_sharpe(trial_sharpes))


def effective_trials(return_matrix: np.ndarray, corr_threshold: float = 0.7,
                     return_labels: bool = False):
    """Cluster highly correlated trials (variants of the same idea) so DSR counts *independent* ideas.

    Greedy correlation clustering in the spirit of Lopez de Prado (2019), 'A Data Science Solution to the
    Multiple-Testing Crisis in Financial Research'. Returns (cluster Sharpe per cluster, n_clusters) and,
    with return_labels=True, also a cluster label per input row (-1 = flat / no returns).
    """
    m = np.asarray(return_matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] == 0:
        return (np.array([]), 0, np.array([], dtype=int)) if return_labels else (np.array([]), 0)
    m = np.nan_to_num(m)
    sd = m.std(axis=1)
    live = sd > 1e-12
    srs = np.array([per_period_sharpe(row) for row in m])
    labels = np.full(m.shape[0], -1, dtype=int)
    if live.sum() < 2:
        labels[live] = 0
        return (srs, int(m.shape[0]), labels) if return_labels else (srs, int(m.shape[0]))
    z = (m[live] - m[live].mean(axis=1, keepdims=True)) / sd[live][:, None]
    corr = (z @ z.T) / z.shape[1]
    idx = np.flatnonzero(live)
    unassigned = set(range(len(idx)))
    cluster_srs = []
    order = np.argsort(-np.abs(srs[idx]))
    for i in order:
        if i not in unassigned:
            continue
        members = [j for j in unassigned if corr[i, j] >= corr_threshold]
        unassigned.difference_update(members)
        labels[idx[members]] = len(cluster_srs)
        cluster_srs.append(float(np.mean(srs[idx][members])))
    if return_labels:
        return np.array(cluster_srs), len(cluster_srs), labels
    return np.array(cluster_srs), len(cluster_srs)


def per_period_sharpe(returns: np.ndarray) -> float:
    return _moments(returns)[0]
