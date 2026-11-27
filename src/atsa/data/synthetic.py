"""Deterministic synthetic market generator for zero-setup demos and tests.

It produces a regime-switching market (bull trend, bear trend, mean-reverting range, volatility
shock) with fat tails, plus PLANTED EDGES that a good research desk should discover:

  * a calendar effect: one weekday (daily bars) or one hour (intraday) carries a hidden drift
  * overreaction: after a > 1.8-sigma down day, part of the drop is recovered over the next 3 bars
  * range regimes snap back toward an anchor price

`bars.meta["planted"]` tells you what was hidden, so you can check whether the swarm found it.
It is NOT real market data: winners here prove the machinery works, not that a strategy will make
money. Use csv / yahoo / binance sources for real research.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .base import Bars

_MS_PER_DAY = 86_400_000
_INTERVAL_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000, "1h": 3_600_000,
                "4h": 14_400_000, "1d": _MS_PER_DAY, "1w": 7 * _MS_PER_DAY}


def _seed_for(symbol: str, seed: int | None) -> int:
    h = int(hashlib.sha256(symbol.encode()).hexdigest()[:8], 16)
    return h if seed is None else (h ^ seed) & 0xFFFFFFFF


def generate_bars(symbol: str = "DEMO", n: int = 2520, interval: str = "1d", seed: int | None = None,
                  start_ms: int = 1_420_070_400_000) -> Bars:
    rng = np.random.default_rng(_seed_for(symbol, seed))
    step = _INTERVAL_MS.get(interval, _MS_PER_DAY)
    scale = np.sqrt(step / _MS_PER_DAY)

    # regime: 0 bull, 1 bear, 2 range (mean reverting), 3 shock
    mu = np.array([0.0012, -0.0010, 0.0, -0.002]) * (step / _MS_PER_DAY)
    sig = np.array([0.011, 0.016, 0.009, 0.035]) * scale
    phi = np.array([0.10, 0.08, -0.25, 0.0])
    trans = np.array([
        [0.985, 0.004, 0.009, 0.002],
        [0.006, 0.978, 0.012, 0.004],
        [0.008, 0.006, 0.983, 0.003],
        [0.10, 0.25, 0.15, 0.50],
    ])
    if step == _MS_PER_DAY:  # business days only for daily bars
        ts = _business_days(start_ms, n)
    else:
        ts = start_ms + step * np.arange(n, dtype=np.int64)
    dt = ts.astype("datetime64[ms]")
    if step >= _MS_PER_DAY:
        bucket = (dt.astype("datetime64[D]").astype(np.int64) + 3) % 7
        edge_bucket = int(rng.integers(0, 5))
        names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        edge_label = f"dow={names[edge_bucket]}"
    else:
        bucket = dt.astype("datetime64[h]").astype(np.int64) % 24
        edge_bucket = int(rng.integers(0, 24))
        edge_label = f"hour={edge_bucket:02d}"
    edge_size = float(rng.uniform(0.0030, 0.0040)) * scale * (1 if rng.random() < 0.75 else -1)
    calendar = np.where(bucket == edge_bucket, edge_size, 0.0)

    regime = np.empty(n, dtype=np.int64)
    r = np.zeros(n)
    logp = np.zeros(n)
    logp[0] = np.log(50 + 150 * rng.random())
    anchor = logp[0]
    state = 0
    rebound_left = 0
    rebound_amt = 0.0
    shocks = rng.standard_t(df=4, size=n) / np.sqrt(2.0)  # unit-variance-ish fat tails
    u = rng.random(n)
    for t in range(1, n):
        state = int(np.searchsorted(np.cumsum(trans[state]), u[t]))
        state = min(state, 3)
        regime[t] = state
        rt = mu[state] + phi[state] * r[t - 1] + sig[state] * shocks[t] + calendar[t]
        if rebound_left > 0:
            rt += rebound_amt
            rebound_left -= 1
        if shocks[t] < -1.8 and state != 3:
            rebound_left = 3
            rebound_amt = 0.30 * sig[state] * -shocks[t]
        if state == 2:
            rt += -0.08 * (logp[t - 1] - anchor)
        else:
            anchor = 0.97 * anchor + 0.03 * logp[t - 1]
        r[t] = rt
        logp[t] = logp[t - 1] + rt
    regime[0] = 0
    close = np.exp(logp)
    gap = rng.normal(0, 0.25, n) * sig[regime]
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1] * np.exp(gap[1:])
    wick = np.abs(rng.normal(0, 0.6, (2, n))) * sig[regime]
    high = np.maximum(open_, close) * np.exp(wick[0])
    low = np.minimum(open_, close) * np.exp(-wick[1])
    volume = np.exp(rng.normal(13.0, 0.3, n) + 20.0 * np.abs(r))

    planted = {"calendar_effect": edge_label, "calendar_drift_bps": round(edge_size * 1e4, 1),
               "overreaction_rebound": "after >1.8 sigma down bar, +3 bar recovery",
               "range_regime_mean_reversion": True}
    return Bars(symbol, interval, ts, open_, high, low, close, volume, source="synthetic",
                meta={"regimes": regime, "planted": planted})


def _business_days(start_ms: int, n: int) -> np.ndarray:
    days = np.datetime64(start_ms, "ms").astype("datetime64[D]")
    out = np.busday_offset(days, np.arange(n), roll="forward")
    return out.astype("datetime64[ms]").astype(np.int64)
