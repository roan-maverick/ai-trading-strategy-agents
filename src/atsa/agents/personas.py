"""The research desk. Each agent hunts a different kind of market inefficiency."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Persona:
    key: str
    title: str
    mandate: str


PERSONAS: dict[str, Persona] = {
    "momentum_hunter": Persona(
        "momentum_hunter", "Momentum Hunter",
        "Find persistent trends and time-series momentum. Use moving-average structure, rate of change, "
        "breakouts of recent highs/lows and trend-strength filters (adx). Ride winners, cut losers fast.",
    ),
    "mean_reversion_sniper": Persona(
        "mean_reversion_sniper", "Mean-Reversion Sniper",
        "Find overreactions that snap back. Use rsi, zscore, Bollinger bands and stochastics to fade "
        "short-term extremes, ideally only in the direction of the longer-term trend. Exit quickly at the mean.",
    ),
    "volatility_breakout": Persona(
        "volatility_breakout", "Volatility Breakout Trader",
        "Exploit volatility clustering: compression is followed by expansion. Combine volatility squeezes "
        "(stdev/atr falling vs their longer average) with range breakouts. Use atr-aware exits.",
    ),
    "fair_value_arbitrageur": Persona(
        "fair_value_arbitrageur", "Fair-Value Arbitrageur",
        "Estimate fair value (fair_value regression, vwap, ema) and trade mispricings: buy when price is "
        "statistically far below fair value (fv_gap strongly negative), sell/short when far above, exit when "
        "the gap closes. Size the threshold to the measured half-life and volatility.",
    ),
    "regime_switcher": Persona(
        "regime_switcher", "Regime Strategist",
        "Detect the market regime (trending vs ranging, calm vs volatile) with adx, linreg_slope, stdev "
        "ratios and long moving averages, and only trade the style that fits the regime "
        "(e.g. buy dips only in uptrends, follow breakouts only when adx is high).",
    ),
    "seasonality_miner": Persona(
        "seasonality_miner", "Seasonality Miner",
        "Look for calendar and time-of-day effects (dow, hour, month) that are statistically meaningful in "
        "the brief (|t_stat| > 2), and combine them with a simple trend or volatility filter. "
        "Remember signals fire at the close and fill at the next open.",
    ),
    "evolver": Persona(
        "evolver", "Strategy Evolver",
        "You receive the current best strategies with their backtest results. Produce improved variants: "
        "tune parameters to more robust (rounder, less fragile) values, add a regime or volatility filter, "
        "add sensible stops, or remove the weak side (long/short). Never just copy a parent.",
    ),
}


def get_persona(key: str) -> Persona:
    if key in PERSONAS:
        return PERSONAS[key]
    # unknown keys become a custom agent with a free-form mandate = the key itself
    return Persona(key, key.replace("_", " ").title(), f"Research strategies in the style of: {key.replace('_', ' ')}.")
