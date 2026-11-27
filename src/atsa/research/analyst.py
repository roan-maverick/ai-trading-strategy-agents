"""Quant Analyst: turns raw bars into a live research brief the LLM agents reason over.

No LLM here - pure statistics, so agents get hard numbers instead of vibes:
trend, volatility regime, autocorrelation, variance ratio, Hurst exponent, mean-reversion
half-life, several fair-value estimates with mispricing z-scores, and calendar seasonality.
"""

from __future__ import annotations

import math

import numpy as np

from ..data.base import Bars
from ..dsl import indicators as ind


def _f(x, nd: int = 4):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if not math.isfinite(x) else round(x, nd)


def hurst_exponent(log_prices: np.ndarray, max_lag: int = 50) -> float:
    """Variance-of-lagged-differences estimator. <0.5 mean-reverting, ~0.5 random walk, >0.5 trending."""
    x = log_prices[np.isfinite(log_prices)]
    lags = np.arange(2, min(max_lag, len(x) // 4))
    if len(lags) < 5:
        return float("nan")
    tau = np.array([np.std(x[lag:] - x[:-lag]) for lag in lags])
    ok = tau > 0
    if ok.sum() < 5:
        return float("nan")
    slope = np.polyfit(np.log(lags[ok]), np.log(tau[ok]), 1)[0]
    return float(slope)


def half_life(log_prices: np.ndarray) -> float:
    """Ornstein-Uhlenbeck mean-reversion half-life in bars (inf if not mean reverting)."""
    x = log_prices[np.isfinite(log_prices)]
    if len(x) < 30:
        return float("nan")
    y = np.diff(x)
    xl = x[:-1] - x[:-1].mean()
    beta = np.dot(xl, y) / np.dot(xl, xl) if np.dot(xl, xl) > 0 else 0.0
    if beta >= 0:
        return float("inf")
    return float(-math.log(2) / beta)


def variance_ratio(returns: np.ndarray, q: int = 5) -> float:
    r = returns[np.isfinite(returns)]
    if len(r) < q * 10:
        return float("nan")
    v1 = np.var(r, ddof=1)
    rq = np.convolve(r, np.ones(q), mode="valid")
    vq = np.var(rq, ddof=1)
    return float(vq / (q * v1)) if v1 > 0 else float("nan")


def _autocorr(r: np.ndarray, lag: int = 1) -> float:
    r = r[np.isfinite(r)]
    if len(r) <= lag + 10:
        return float("nan")
    return float(np.corrcoef(r[lag:], r[:-lag])[0, 1])


def event_study(close: np.ndarray, returns: np.ndarray, z_threshold: float = 1.8, horizon: int = 3,
                lookback: int = 20) -> dict:
    """What happens after shock bars? Forward `horizon`-bar return (entered at the next bar) after
    returns more than `z_threshold` rolling standard deviations below / above their mean."""
    z = ind.zscore(returns, lookback)
    n = len(close)
    out = {}
    for label, mask in (("after_big_down_bar", z < -z_threshold), ("after_big_up_bar", z > z_threshold)):
        idx = np.flatnonzero(mask)
        idx = idx[idx + 1 + horizon < n]
        if len(idx) < 15:
            out[label] = {"events": int(len(idx))}
            continue
        fwd = close[idx + 1 + horizon] / close[idx + 1] - 1.0
        sd = fwd.std(ddof=1)
        out[label] = {"events": int(len(idx)), "fwd_bars": horizon, "mean_fwd_return_bps": _f(fwd.mean() * 1e4, 1),
                      "hit_rate": _f((fwd > 0).mean(), 2),
                      "t_stat": _f(fwd.mean() / (sd / math.sqrt(len(fwd))), 2) if sd > 0 else None}
    out["definition"] = f"z = zscore(ret,{lookback}); event when |z| > {z_threshold}; entry next bar, hold {horizon} bars"
    return out


def _bars_per_day(bars: Bars) -> float:
    if len(bars) < 3:
        return 1.0
    step = float(np.median(np.diff(bars.ts[-200:])))
    return max(86_400_000.0 / step, 1e-9) if step > 0 else 1.0


def analyze(bars: Bars, periods_per_year: float = 252.0) -> dict:
    c = bars.close
    n = len(c)
    logp = np.log(c)
    r = np.zeros(n)
    r[1:] = c[1:] / c[:-1] - 1.0
    bpd = _bars_per_day(bars)

    def ret_over(k):
        k = int(round(k))
        return _f(c[-1] / c[-1 - k] - 1.0) if 0 < k < n else None

    vol_20 = ind.rolling_std(r, 20)
    vol_series = vol_20[np.isfinite(vol_20)]
    vol_pct = float((vol_series <= vol_series[-1]).mean()) if len(vol_series) else float("nan")
    ann = math.sqrt(periods_per_year)

    sma50, sma200 = ind.sma(c, 50), ind.sma(c, 200)
    slope100 = ind.rolling_linreg(logp, 100, True)
    lookback = min(n, 750)
    recent_logp = logp[-lookback:]
    recent_r = r[-lookback:]

    fv_reg = ind.fair_value(c, 100)[-1]
    fv_ema = ind.ema(c, 50)[-1]
    fv_vwap = ind.rolling_vwap((bars.high + bars.low + bars.close) / 3.0, bars.volume, 50)[-1] \
        if bars.volume.sum() > 0 else float("nan")
    estimates = [v for v in (fv_reg, fv_ema, fv_vwap) if np.isfinite(v)]
    composite = float(np.mean(estimates)) if estimates else float("nan")
    gap = ind.fv_gap(c, 60)
    z20 = ind.zscore(c, 20)

    hurst = hurst_exponent(recent_logp)
    vr = variance_ratio(recent_r)
    ac1 = _autocorr(recent_r)
    hl = half_life(recent_logp[-250:])

    trend_up = np.isfinite(sma200[-1]) and c[-1] > sma200[-1] and sma50[-1] > sma200[-1]
    trend_dn = np.isfinite(sma200[-1]) and c[-1] < sma200[-1] and sma50[-1] < sma200[-1]
    trend = "uptrend" if trend_up else "downtrend" if trend_dn else "range/transition"
    vol_regime = "high" if vol_pct > 0.8 else "low" if vol_pct < 0.2 else "normal"
    if np.isfinite(hurst) and np.isfinite(vr):
        character = "trending (momentum-friendly)" if hurst > 0.55 or vr > 1.1 else \
            "mean-reverting (fade extremes)" if hurst < 0.45 or vr < 0.9 else "near random walk (edges are thin)"
    else:
        character = "unknown"

    seasonality = {}
    dt = bars.ts.astype("datetime64[ms]")
    if bpd <= 1.5:
        dow = (dt.astype("datetime64[D]").astype(np.int64) + 3) % 7
        names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        for d in range(7):
            m = (dow == d) & np.isfinite(r)
            m[0] = False
            if m.sum() >= 30:
                rr = r[m]
                seasonality[names[d]] = {"mean_bps": _f(rr.mean() * 1e4, 2),
                                         "t_stat": _f(rr.mean() / (rr.std(ddof=1) / math.sqrt(len(rr))), 2)}
    else:
        hour = dt.astype("datetime64[h]").astype(np.int64) % 24
        for h in range(24):
            m = (hour == h)
            m[0] = False
            if m.sum() >= 30:
                rr = r[m]
                seasonality[f"{h:02d}h"] = {"mean_bps": _f(rr.mean() * 1e4, 2),
                                            "t_stat": _f(rr.mean() / (rr.std(ddof=1) / math.sqrt(len(rr))), 2)}

    tail = r[-lookback:]
    tail = tail[np.isfinite(tail)]
    mu, sd = tail.mean(), tail.std()
    skew = float(np.mean(((tail - mu) / sd) ** 3)) if sd > 0 else 0.0
    kurt = float(np.mean(((tail - mu) / sd) ** 4)) if sd > 0 else 3.0
    eq = c[-min(n, int(periods_per_year)):]
    mdd_1y = float(np.max(1 - eq / np.maximum.accumulate(eq)))

    return {
        "symbol": bars.symbol, "interval": bars.interval, "source": bars.source,
        "bars": n, "from": bars.start_date, "to": bars.end_date, "last_price": _f(c[-1], 6),
        "returns": {"1w": ret_over(5 * bpd), "1m": ret_over(21 * bpd), "3m": ret_over(63 * bpd),
                    "1y": ret_over(252 * bpd)},
        "volatility": {"ann_vol_20": _f(vol_20[-1] * ann), "ann_vol_full": _f(np.nanstd(r) * ann),
                       "vol_percentile": _f(vol_pct, 2), "regime": vol_regime},
        "trend": {"state": trend, "price_vs_sma200": _f(c[-1] / sma200[-1] - 1.0),
                  "sma50_vs_sma200": _f(sma50[-1] / sma200[-1] - 1.0),
                  "slope_100_ann": _f(slope100[-1] * periods_per_year), "adx_14": _f(ind.adx(bars.high, bars.low, c, 14)[-1], 1)},
        "market_character": {"hurst": _f(hurst, 3), "variance_ratio_5": _f(vr, 3), "autocorr_lag1": _f(ac1, 3),
                             "half_life_bars": _f(hl, 1), "verdict": character},
        "fair_value": {"regression_100": _f(fv_reg, 6), "ema_50": _f(fv_ema, 6), "vwap_50": _f(fv_vwap, 6),
                       "composite": _f(composite, 6), "gap_vs_composite": _f(c[-1] / composite - 1.0) if composite else None,
                       "mispricing_z_60": _f(gap[-1], 2), "zscore_20": _f(z20[-1], 2),
                       "rsi_14": _f(ind.wilder_rsi(c, 14)[-1], 1)},
        "risk": {"skew": _f(skew, 2), "kurtosis": _f(kurt, 2), "max_drawdown_1y": _f(mdd_1y)},
        "seasonality": seasonality,
        "event_study": event_study(c, r),
    }


def brief_markdown(brief: dict) -> str:
    b = brief
    fv = b["fair_value"]
    mc = b["market_character"]
    lines = [
        f"### {b['symbol']} ({b['interval']}, {b['bars']} bars {b['from']} -> {b['to']})",
        f"- Price {b['last_price']} | trend: **{b['trend']['state']}** | vol regime: **{b['volatility']['regime']}** "
        f"(ann vol {b['volatility']['ann_vol_20']})",
        f"- Character: **{mc['verdict']}** (Hurst {mc['hurst']}, VR5 {mc['variance_ratio_5']}, "
        f"AC1 {mc['autocorr_lag1']}, half-life {mc['half_life_bars']} bars)",
        f"- Fair value (composite) {fv['composite']} -> price is {_pct(fv['gap_vs_composite'])} vs fair; "
        f"mispricing z {fv['mispricing_z_60']}, RSI14 {fv['rsi_14']}",
    ]
    es = b.get("event_study", {}).get("after_big_down_bar", {})
    if es.get("t_stat") is not None:
        lines.append(f"- After shock down-bars: {es['mean_fwd_return_bps']} bps over next {es['fwd_bars']} bars "
                     f"(t={es['t_stat']}, {es['events']} events, hit rate {es['hit_rate']})")
    return "\n".join(lines)


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:+.1%}"
