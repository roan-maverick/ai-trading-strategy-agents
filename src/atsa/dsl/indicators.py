"""Causal technical indicators, JIT-compiled with Numba (falls back to pure Python if Numba is missing).

Every function only looks backwards: output[i] depends on input[:i+1]. Warm-up bars are NaN.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - import guard
    from numba import njit

    HAS_NUMBA = True

    def jit(fn):
        return njit(cache=True, nogil=True, fastmath=False)(fn)
except Exception:  # pragma: no cover
    HAS_NUMBA = False

    def jit(fn):
        return fn


@jit
def rolling_mean(x, n):
    m = len(x)
    out = np.full(m, np.nan)
    s = 0.0
    bad = 0
    for i in range(m):
        v = x[i]
        if np.isnan(v):
            bad += 1
        else:
            s += v
        if i >= n:
            o = x[i - n]
            if np.isnan(o):
                bad -= 1
            else:
                s -= o
        if i >= n - 1 and bad == 0:
            out[i] = s / n
    return out


@jit
def rolling_std(x, n):
    m = len(x)
    out = np.full(m, np.nan)
    if n < 2:
        return out
    for i in range(n - 1, m):
        s = 0.0
        ok = True
        for j in range(i - n + 1, i + 1):
            if np.isnan(x[j]):
                ok = False
                break
            s += x[j]
        if not ok:
            continue
        mu = s / n
        ss = 0.0
        for j in range(i - n + 1, i + 1):
            d = x[j] - mu
            ss += d * d
        out[i] = np.sqrt(ss / (n - 1))
    return out


@jit
def rolling_max(x, n):
    m = len(x)
    out = np.full(m, np.nan)
    for i in range(n - 1, m):
        best = -np.inf
        ok = True
        for j in range(i - n + 1, i + 1):
            v = x[j]
            if np.isnan(v):
                ok = False
                break
            if v > best:
                best = v
        if ok:
            out[i] = best
    return out


@jit
def rolling_min(x, n):
    m = len(x)
    out = np.full(m, np.nan)
    for i in range(n - 1, m):
        best = np.inf
        ok = True
        for j in range(i - n + 1, i + 1):
            v = x[j]
            if np.isnan(v):
                ok = False
                break
            if v < best:
                best = v
        if ok:
            out[i] = best
    return out


@jit
def ewm(x, alpha, min_periods):
    """Exponential moving average starting at the first valid value."""
    m = len(x)
    out = np.full(m, np.nan)
    started = False
    val = 0.0
    count = 0
    for i in range(m):
        v = x[i]
        if np.isnan(v):
            if started:
                out[i] = np.nan
            continue
        if not started:
            val = v
            started = True
        else:
            val = alpha * v + (1.0 - alpha) * val
        count += 1
        if count >= min_periods:
            out[i] = val
    return out


@jit
def wilder_rsi(x, n):
    m = len(x)
    out = np.full(m, np.nan)
    avg_g = 0.0
    avg_l = 0.0
    seen = 0
    for i in range(1, m):
        if np.isnan(x[i]) or np.isnan(x[i - 1]):
            continue
        ch = x[i] - x[i - 1]
        g = ch if ch > 0 else 0.0
        lo = -ch if ch < 0 else 0.0
        seen += 1
        if seen <= n:
            avg_g += g / n
            avg_l += lo / n
            if seen < n:
                continue
        else:
            avg_g = (avg_g * (n - 1) + g) / n
            avg_l = (avg_l * (n - 1) + lo) / n
        if avg_l == 0.0:
            out[i] = 100.0 if avg_g > 0 else 50.0
        else:
            out[i] = 100.0 - 100.0 / (1.0 + avg_g / avg_l)
    return out


@jit
def true_range(high, low, close):
    m = len(close)
    out = np.empty(m)
    out[0] = high[0] - low[0]
    for i in range(1, m):
        a = high[i] - low[i]
        b = abs(high[i] - close[i - 1])
        c = abs(low[i] - close[i - 1])
        out[i] = max(a, max(b, c))
    return out


@jit
def wilder_smooth(x, n):
    m = len(x)
    out = np.full(m, np.nan)
    acc = 0.0
    seen = 0
    for i in range(m):
        v = x[i]
        if np.isnan(v):
            continue
        seen += 1
        if seen <= n:
            acc += v / n
            if seen == n:
                out[i] = acc
        else:
            acc = (acc * (n - 1) + v) / n
            out[i] = acc
    return out


@jit
def adx(high, low, close, n):
    m = len(close)
    plus_dm = np.zeros(m)
    minus_dm = np.zeros(m)
    for i in range(1, m):
        up = high[i] - high[i - 1]
        dn = low[i - 1] - low[i]
        if up > dn and up > 0:
            plus_dm[i] = up
        if dn > up and dn > 0:
            minus_dm[i] = dn
    tr = true_range(high, low, close)
    atr_ = wilder_smooth(tr, n)
    pdi = 100.0 * wilder_smooth(plus_dm, n) / atr_
    mdi = 100.0 * wilder_smooth(minus_dm, n) / atr_
    dx = np.full(m, np.nan)
    for i in range(m):
        s = pdi[i] + mdi[i]
        if not np.isnan(s) and s > 0:
            dx[i] = 100.0 * abs(pdi[i] - mdi[i]) / s
    return wilder_smooth(dx, n)


@jit
def rolling_linreg(y, n, want_slope):
    """Rolling OLS of y on time. Returns fitted value at the window end (or the slope)."""
    m = len(y)
    out = np.full(m, np.nan)
    sx = n * (n - 1) / 2.0
    sxx = (n - 1) * n * (2 * n - 1) / 6.0
    den = n * sxx - sx * sx
    if den == 0:
        return out
    for i in range(n - 1, m):
        sy = 0.0
        sxy = 0.0
        ok = True
        for k in range(n):
            v = y[i - n + 1 + k]
            if np.isnan(v):
                ok = False
                break
            sy += v
            sxy += k * v
        if not ok:
            continue
        slope = (n * sxy - sx * sy) / den
        if want_slope:
            out[i] = slope
        else:
            out[i] = (sy - slope * sx) / n + slope * (n - 1)
    return out


@jit
def rolling_pct_rank(x, n):
    m = len(x)
    out = np.full(m, np.nan)
    for i in range(n - 1, m):
        cur = x[i]
        if np.isnan(cur):
            continue
        le = 0
        ok = True
        for j in range(i - n + 1, i + 1):
            v = x[j]
            if np.isnan(v):
                ok = False
                break
            if v <= cur:
                le += 1
        if ok:
            out[i] = le / n
    return out


def lag(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if n < len(x):
        out[n:] = x[: len(x) - n]
    return out


def sma(x, n):
    return rolling_mean(x, n)


def ema(x, n):
    return ewm(x, 2.0 / (n + 1.0), n)


def zscore(x, n):
    sd = rolling_std(x, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (x - rolling_mean(x, n)) / sd
    z[~np.isfinite(z)] = np.nan
    return z


def roc(x, n):
    with np.errstate(divide="ignore", invalid="ignore"):
        return x / lag(x, n) - 1.0


def atr(high, low, close, n):
    return wilder_smooth(true_range(high, low, close), n)


def fair_value(x, n):
    """Rolling regression 'fair value' of a series."""
    return rolling_linreg(x, n, False)


def fv_gap(x, n):
    """Mispricing z-score: distance of price from its rolling regression fair value, in residual std units."""
    resid = x - rolling_linreg(x, n, False)
    sd = rolling_std(resid, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = resid / sd
    z[~np.isfinite(z)] = np.nan
    return z


def rolling_vwap(price, volume, n):
    pv = rolling_mean(price * volume, n)
    vv = rolling_mean(volume, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = pv / vv
    out[~np.isfinite(out)] = np.nan
    return out


def stoch_k(high, low, close, n):
    hh = rolling_max(high, n)
    ll = rolling_min(low, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = 100.0 * (close - ll) / (hh - ll)
    k[~np.isfinite(k)] = np.nan
    return k
