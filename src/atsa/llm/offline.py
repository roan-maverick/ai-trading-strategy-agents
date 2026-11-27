"""Offline 'quant template' provider: no API key, no network, fully deterministic with a seed.

It plays every agent persona with a library of classic quant templates whose parameters are
sampled and biased by the live research brief, and it evolves leaders by mutation. Use it to try the
whole pipeline in seconds, in CI, or as a cheap baseline that your LLM has to beat.
"""

from __future__ import annotations

import json
import random
import re

from ..config import LLMConfig
from .base import LLMProvider

_NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)")
_FIXED = re.compile(r"(dow|hour|month)\s*==\s*$")


def _ch(rng: random.Random, xs):
    return xs[rng.randrange(len(xs))]


def _momentum(rng, short):
    k = rng.randrange(4)
    if k == 0:
        f, s = _ch(rng, [(10, 50), (20, 100), (20, 50), (50, 200), (10, 30)])
        a = _ch(rng, [15, 20, 25])
        st = {"name": f"EMA {f}/{s} trend + ADX>{a}",
              "thesis": "Time-series momentum: trends persist once established and confirmed by trend strength.",
              "entry_long": f"ema(close,{f}) > ema(close,{s}) and adx(14) > {a}"}
        if short:
            st["entry_short"] = f"ema(close,{f}) < ema(close,{s}) and adx(14) > {a}"
    elif k == 1:
        n = _ch(rng, [20, 40, 60, 120])
        th = _ch(rng, [0.0, 0.02, 0.05])
        st = {"name": f"ROC({n}) momentum", "thesis": "Recent winners keep winning over medium horizons.",
              "entry_long": f"roc(close,{n}) > {th}", "exit_long": "roc(close,%d) < 0" % n}
        if short:
            st["entry_short"] = f"roc(close,{n}) < -{th}"
            st["exit_short"] = "roc(close,%d) > 0" % n
    elif k == 2:
        n, m = _ch(rng, [(20, 10), (55, 20), (40, 20), (100, 50)])
        st = {"name": f"Donchian {n}/{m} breakout", "thesis": "Turtle-style breakouts capture the start of new trends.",
              "entry_long": f"close > lag(highest(high,{n}),1)", "exit_long": f"close < lag(lowest(low,{m}),1)"}
        if short:
            st["entry_short"] = f"close < lag(lowest(low,{n}),1)"
            st["exit_short"] = f"close > lag(highest(high,{m}),1)"
    else:
        n = _ch(rng, [50, 100, 200])
        st = {"name": f"Price above SMA{n} with rising slope",
              "thesis": "Only hold when both price and the regression slope confirm the trend.",
              "entry_long": f"close > sma(close,{n}) and linreg_slope(close,{n // 2}) > 0"}
        if short:
            st["entry_short"] = f"close < sma(close,{n}) and linreg_slope(close,{n // 2}) < 0"
    st["stop_loss"] = _ch(rng, [None, 0.05, 0.08, 0.1])
    return st


def _shock_reversal(rng, short, brief):
    es = (brief.get("event_study") or {})
    down, up = es.get("after_big_down_bar") or {}, es.get("after_big_up_bar") or {}
    th = _ch(rng, [1.5, 1.8, 2.0, 2.5])
    hold = _ch(rng, [2, 3, 5])
    st = {"name": f"Shock reversal: buy {th}-sigma down bars, hold {hold}",
          "thesis": f"Event study shows forced selling overshoots: after shock down-bars the next bars recover "
                    f"(t={down.get('t_stat')}).",
          "entry_long": f"zscore(ret,20) < -{th}", "max_hold_bars": hold}
    if rng.random() < 0.5:
        st["entry_long"] += f" and close > sma(close,{_ch(rng, [100, 200])})"
        st["name"] += " (uptrend only)"
    st["exit_long"] = "ret > 0.0" if rng.random() < 0.3 else None
    if short and (up.get("t_stat") or 0) < -2:
        st["entry_short"] = f"zscore(ret,20) > {th}"
    st["stop_loss"] = _ch(rng, [None, 0.05, 0.08])
    return st


def _evidence(brief, key) -> float:
    return float(((brief.get("event_study") or {}).get(key) or {}).get("t_stat") or 0.0)


def _mean_reversion(rng, short, brief=None):
    brief = brief or {}
    if _evidence(brief, "after_big_down_bar") > 2.0 and rng.random() < 0.4:
        return _shock_reversal(rng, short, brief)
    k = rng.randrange(4)
    if k == 0:
        n, lo, ex, tr = _ch(rng, [2, 3, 5, 14]), _ch(rng, [10, 20, 25, 30]), _ch(rng, [50, 60, 70]), _ch(rng, [100, 200])
        st = {"name": f"RSI({n})<{lo} dip-buy above SMA{tr}",
              "thesis": "Short-term oversold dips inside a long-term uptrend tend to revert up.",
              "entry_long": f"rsi(close,{n}) < {lo} and close > sma(close,{tr})", "exit_long": f"rsi(close,{n}) > {ex}"}
        if short:
            st["entry_short"] = f"rsi(close,{n}) > {100 - lo} and close < sma(close,{tr})"
            st["exit_short"] = f"rsi(close,{n}) < {100 - ex}"
    elif k == 1:
        n, z = _ch(rng, [10, 20, 30]), _ch(rng, [1.5, 2.0, 2.5])
        st = {"name": f"Z-score({n}) fade at {z} sigma", "thesis": "Statistical overreactions revert to the rolling mean.",
              "entry_long": f"zscore(close,{n}) < -{z}", "exit_long": f"zscore(close,{n}) > 0"}
        if short:
            st["entry_short"] = f"zscore(close,{n}) > {z}"
            st["exit_short"] = f"zscore(close,{n}) < 0"
    elif k == 2:
        n, kk = _ch(rng, [20, 30]), _ch(rng, [2.0, 2.5])
        st = {"name": f"Bollinger({n},{kk}) snap-back",
              "thesis": "Closes outside the bands are stretched and snap back to the mid-band.",
              "entry_long": f"close < bb_lower(close,{n},{kk})", "exit_long": f"close > sma(close,{n})"}
        if short:
            st["entry_short"] = f"close > bb_upper(close,{n},{kk})"
            st["exit_short"] = f"close < sma(close,{n})"
    else:
        n, lo = _ch(rng, [5, 14]), _ch(rng, [10, 20])
        st = {"name": f"Stochastic({n}) oversold", "thesis": "Closing near the bottom of the recent range signals exhaustion.",
              "entry_long": f"stoch_k({n}) < {lo}", "exit_long": f"stoch_k({n}) > 50"}
        if short:
            st["entry_short"] = f"stoch_k({n}) > {100 - lo}"
            st["exit_short"] = f"stoch_k({n}) < 50"
    st["max_hold_bars"] = _ch(rng, [None, 10, 20])
    st["stop_loss"] = _ch(rng, [None, 0.05, 0.08])
    return st


def _vol_breakout(rng, short):
    k = rng.randrange(2)
    if k == 0:
        n, m, c = _ch(rng, [10, 20]), _ch(rng, [60, 100]), _ch(rng, [0.7, 0.8])
        b = _ch(rng, [10, 20])
        st = {"name": f"Vol squeeze {n}/{m} + {b}-bar breakout",
              "thesis": "Volatility compression precedes expansion; trade the direction of the breakout.",
              "entry_long": f"lag(stdev(ret,{n}),1) < lag(stdev(ret,{m}),1) * {c} and close > lag(highest(close,{b}),1)",
              "exit_long": f"close < sma(close,{b})"}
        if short:
            st["entry_short"] = f"lag(stdev(ret,{n}),1) < lag(stdev(ret,{m}),1) * {c} and close < lag(lowest(close,{b}),1)"
            st["exit_short"] = f"close > sma(close,{b})"
    else:
        n, kk = _ch(rng, [20, 50]), _ch(rng, [1.0, 1.5, 2.0])
        st = {"name": f"ATR channel breakout ({n}, {kk}x)",
              "thesis": "A move of more than k ATRs from the mean is an impulse that tends to continue.",
              "entry_long": f"close > ema(close,{n}) + {kk} * atr(14)", "exit_long": f"close < ema(close,{n})"}
        if short:
            st["entry_short"] = f"close < ema(close,{n}) - {kk} * atr(14)"
            st["exit_short"] = f"close > ema(close,{n})"
    st["stop_loss"] = _ch(rng, [None, 0.06, 0.1])
    return st


def _fair_value(rng, short, brief):
    hl = (brief.get("market_character") or {}).get("half_life_bars") or 30
    try:
        hl = float(hl)
    except (TypeError, ValueError):
        hl = 30.0
    base = 60 if hl == float("inf") or hl > 120 else max(20, int(round(hl * 2 / 10.0)) * 10)
    k = rng.randrange(3)
    if k == 0:
        n, z = _ch(rng, [base, 40, 60, 100]), _ch(rng, [1.5, 2.0, 2.5])
        st = {"name": f"Fair-value gap({n}) at {z} sigma",
              "thesis": "Price far below its regression fair value is mispriced and converges back.",
              "entry_long": f"fv_gap(close,{n}) < -{z}", "exit_long": f"fv_gap(close,{n}) > 0"}
        if short:
            st["entry_short"] = f"fv_gap(close,{n}) > {z}"
            st["exit_short"] = f"fv_gap(close,{n}) < 0"
    elif k == 1:
        n, d = _ch(rng, [20, 50]), _ch(rng, [0.03, 0.05, 0.08])
        st = {"name": f"VWAP({n}) discount {d:.0%}", "thesis": "Deep discounts to volume-weighted value attract buyers.",
              "entry_long": f"close < vwap({n}) * {round(1 - d, 3)}", "exit_long": f"close > vwap({n})"}
        if short:
            st["entry_short"] = f"close > vwap({n}) * {round(1 + d, 3)}"
            st["exit_short"] = f"close < vwap({n})"
    else:
        n, kk = _ch(rng, [50, 100]), _ch(rng, [2.0, 3.0])
        st = {"name": f"Fair value ({n}) - {kk} ATR", "thesis": "Volatility-scaled mispricing vs trend fair value.",
              "entry_long": f"close < fair_value(close,{n}) - {kk} * atr(14)", "exit_long": f"close > fair_value(close,{n})"}
        if short:
            st["entry_short"] = f"close > fair_value(close,{n}) + {kk} * atr(14)"
            st["exit_short"] = f"close < fair_value(close,{n})"
    st["max_hold_bars"] = _ch(rng, [None, 20, 40])
    st["stop_loss"] = _ch(rng, [None, 0.08, 0.12])
    return st


def _regime(rng, short):
    k = rng.randrange(3)
    if k == 0:
        n, m, z = _ch(rng, [100, 200]), _ch(rng, [10, 20]), _ch(rng, [1.0, 1.5, 2.0])
        st = {"name": f"Buy-the-dip in uptrend (SMA{n}, z{m}<-{z})",
              "thesis": "In confirmed uptrends, pullbacks are bought; outside them, stay flat.",
              "entry_long": f"close > sma(close,{n}) and zscore(close,{m}) < -{z}", "exit_long": f"zscore(close,{m}) > 0"}
        if short:
            st["entry_short"] = f"close < sma(close,{n}) and zscore(close,{m}) > {z}"
            st["exit_short"] = f"zscore(close,{m}) < 0"
    elif k == 1:
        a, n = _ch(rng, [20, 25, 30]), _ch(rng, [20, 50])
        st = {"name": f"ADX>{a} trend regime follow (EMA{n})",
              "thesis": "Follow trends only when the regime is strongly trending; sit out chop.",
              "entry_long": f"adx(14) > {a} and close > ema(close,{n}) and rising(ema(close,{n}),5)",
              "exit_long": f"close < ema(close,{n})"}
        if short:
            st["entry_short"] = f"adx(14) > {a} and close < ema(close,{n}) and falling(ema(close,{n}),5)"
            st["exit_short"] = f"close > ema(close,{n})"
    else:
        n = _ch(rng, [50, 100])
        st = {"name": f"Calm-uptrend carry (SMA{n})", "thesis": "Low-volatility uptrends have the best risk-adjusted drift.",
              "entry_long": f"close > sma(close,{n}) and stdev(ret,20) < stdev(ret,100)",
              "exit_long": f"close < sma(close,{n}) or stdev(ret,20) > stdev(ret,100) * 1.5"}
    st["stop_loss"] = _ch(rng, [None, 0.06, 0.1])
    return st


def _seasonality(rng, short, brief):
    seas = brief.get("seasonality") or {}
    ranked = sorted(seas.items(), key=lambda kv: abs((kv[1] or {}).get("t_stat") or 0), reverse=True)
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    if ranked and ranked[0][0] in names:
        pick = ranked[min(rng.randrange(2), len(ranked) - 1)]
        day = names.index(pick[0])
        sign = 1 if ((pick[1] or {}).get("mean_bps") or 0) >= 0 else -1
        prev = (day - 1) % 7
        if brief.get("source") != "binance" and day == 0:
            prev = 4  # Friday close -> Monday
        side = "entry_long" if sign > 0 or not short else "entry_short"
        n = _ch(rng, [50, 100, 200])
        filt = f" and close > sma(close,{n})" if side == "entry_long" and rng.random() < 0.5 else ""
        return {"name": f"{pick[0]} effect ({'long' if side == 'entry_long' else 'short'})",
                "thesis": f"{pick[0]} returns show a persistent calendar bias (t={pick[1].get('t_stat')}).",
                side: f"dow == {prev}{filt}", "max_hold_bars": 1}
    if ranked:
        h = int(ranked[0][0][:2])
        return {"name": f"{h:02d}h session drift", "thesis": "Intraday session flow creates a time-of-day drift.",
                "entry_long": f"hour == {(h - 1) % 24}", "max_hold_bars": 1}
    return _regime(rng, short)


def _mutate_expr(expr: str | None, rng: random.Random, mapping: dict[str, str]) -> str | None:
    """Re-tune numeric literals. The same literal maps to the same new value across all rules of a
    strategy, so `zscore(close,10)` in entry and exit stays consistent."""
    if not expr:
        return expr

    def sub(m: re.Match) -> str:
        if _FIXED.search(expr[: m.start()]):
            return m.group(0)
        txt = m.group(1)
        if txt in mapping:
            return mapping[txt]
        if "." in txt:
            v = float(txt)
            v *= rng.uniform(0.98, 1.02) if 0.5 < v < 1.5 else rng.uniform(0.8, 1.25)
            new = f"{round(v, 3 if 0.5 < v < 1.5 else 2)}"
        else:
            v = int(txt)
            new = txt if v <= 1 else str(min(max(2, int(round(v * rng.uniform(0.7, 1.4)))), 400))
        mapping[txt] = new
        return new

    return _NUM.sub(sub, expr)


def mutate(spec: dict, rng: random.Random) -> dict:
    child = {k: spec.get(k) for k in ("entry_long", "exit_long", "entry_short", "exit_short", "stop_loss",
                                      "take_profit", "max_hold_bars")}
    roll = rng.random()
    if roll < 0.6:
        mapping: dict[str, str] = {}
        for k in ("entry_long", "exit_long", "entry_short", "exit_short"):
            child[k] = _mutate_expr(child[k], rng, mapping)
        how = "re-tuned parameters"
    elif roll < 0.75 and child.get("entry_long") and child.get("entry_short"):
        side = _ch(rng, ["long", "short"])
        child[f"entry_{side}"] = None
        child[f"exit_{side}"] = None
        how = f"dropped the {side} side"
    elif roll < 0.9:
        n = _ch(rng, [100, 200])
        if child.get("entry_long"):
            child["entry_long"] = f"({child['entry_long']}) and close > sma(close,{n})"
        if child.get("entry_short"):
            child["entry_short"] = f"({child['entry_short']}) and close < sma(close,{n})"
        how = f"added SMA{n} regime filter"
    else:
        child["stop_loss"] = _ch(rng, [0.04, 0.06, 0.08, 0.12])
        child["max_hold_bars"] = _ch(rng, [None, 10, 20, 40])
        how = "new risk management"
    base_name = re.sub(r"\s*\[.*\]$", "", spec.get("name", "strategy"))[:70]
    child["name"] = f"{base_name} [{how}]"
    parent_thesis = re.sub(r"^Evolved from a leader: [^.]*\. ", "", spec.get("thesis") or "")
    child["thesis"] = f"Evolved from a leader: {how}. " + parent_thesis[:200]
    return child


class OfflineProvider(LLMProvider):
    name = "offline"
    default_model = "quant-templates-v1"
    needs_key = False

    def __init__(self, cfg: LLMConfig, client=None, seed: int | None = None):
        super().__init__(cfg, client)
        self.model = self.default_model
        self.seed = seed

    async def complete(self, system: str, user: str, meta: dict | None = None) -> str:
        meta = meta or {}
        persona = meta.get("persona", "momentum_hunter")
        n = int(meta.get("n", 5))
        brief = meta.get("brief") or {}
        short = bool(meta.get("allow_short", True))
        gen = int(meta.get("generation", 1))
        base_seed = self.seed if self.seed is not None else random.randrange(1 << 30)
        rng = random.Random(f"{base_seed}:{persona}:{gen}:{brief.get('symbol')}")
        self.usage["calls"] += 1
        leaders = meta.get("leaders") or []
        out = []
        for _ in range(n):
            if leaders and (persona == "evolver" or rng.random() < 0.4):
                parent = _ch(rng, leaders)
                child = mutate(parent.get("spec", parent), rng)
                child["parent_id"] = parent.get("id")
                out.append(child)
                continue
            if persona == "momentum_hunter":
                out.append(_momentum(rng, short))
            elif persona == "mean_reversion_sniper":
                out.append(_mean_reversion(rng, short, brief))
            elif persona == "volatility_breakout":
                out.append(_vol_breakout(rng, short))
            elif persona == "fair_value_arbitrageur":
                out.append(_fair_value(rng, short, brief))
            elif persona == "regime_switcher":
                out.append(_regime(rng, short))
            elif persona == "seasonality_miner":
                out.append(_seasonality(rng, short, brief))
            else:
                fn = _ch(rng, [_momentum, _mean_reversion, _vol_breakout, _regime])
                out.append(fn(rng, short))
        return json.dumps({"strategies": out})
