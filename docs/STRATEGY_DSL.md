# Strategy language

Agents (and you) write strategies as JSON with plain-text rules:

```json
{
  "name": "RSI(2) dip-buy in uptrend",
  "thesis": "Short-term oversold dips inside a long-term uptrend revert.",
  "entry_long": "rsi(close,2) < 10 and close > sma(close,200)",
  "exit_long": "rsi(close,2) > 70",
  "entry_short": null,
  "exit_short": null,
  "stop_loss": 0.08,
  "take_profit": null,
  "max_hold_bars": null
}
```

## Semantics

* Rules are evaluated at each bar's **close**; orders fill at the **next open**.
* `exit_long: null` + `max_hold_bars: N` → hold N bars (event trade).
* `exit_long: null` and no `max_hold_bars` → exit when `entry_long` turns false (state trade).
* `stop_loss` / `take_profit` are fractions of the entry price (`0.05` = 5%; `5` is also accepted as 5%).
* If both the long and short entries are true on the same bar, nothing happens.

## Variables

| Name | Meaning |
|---|---|
| `open high low close volume` | Bar prices and volume |
| `hl2`, `hlc3` | (high+low)/2, (high+low+close)/3 |
| `ret` | 1-bar close-to-close return |
| `dow` | Day of week, 0=Mon … 6=Sun (UTC) |
| `hour` | Hour of day 0–23 (UTC) |
| `month` | 1–12 |

## Functions

`x` = any expression, `n` = integer literal period (1–1000), `k` = numeric literal.

| Function | Meaning |
|---|---|
| `sma(x,n)` `ema(x,n)` `wilder(x,n)` | Moving averages |
| `rsi(x,n)` | Wilder RSI, 0–100 |
| `macd(x,fast,slow)` `macd_signal(x,fast,slow,signal)` | MACD line / signal line |
| `stoch_k(n)` | Stochastic %K, 0–100 |
| `adx(n)` | Trend strength, 0–100 |
| `atr(n)` | Average true range |
| `stdev(x,n)` | Rolling standard deviation |
| `zscore(x,n)` | (x − mean) / std over n bars |
| `roc(x,n)` | x / x[n bars ago] − 1 |
| `pct_rank(x,n)` | Percentile of x in the last n bars, 0–1 |
| `highest(x,n)` `lowest(x,n)` | Rolling max / min, **including the current bar**. Use `lag(highest(high,20),1)` for breakouts. |
| `bb_upper(x,n,k)` `bb_lower(x,n,k)` | Bollinger bands |
| `fair_value(x,n)` | Rolling linear-regression fair value |
| `fv_gap(x,n)` | Mispricing z-score of x vs its regression fair value |
| `linreg_slope(x,n)` | Rolling regression slope |
| `vwap(n)` | Rolling volume-weighted average price |
| `lag(x,n)` `diff(x,n)` | x n bars ago / x − x n bars ago |
| `rising(x,n)` `falling(x,n)` | x above / below its value n bars ago |
| `streak(cond,n)` | cond true on each of the last n bars |
| `cross_above(a,b)` `cross_below(a,b)` | Crossover on this bar |
| `abs(x)` `log(x)` `max(a,b)` `min(a,b)` | Math |

Operators: `+ - * /`, `< <= > >= == !=`, `and or not`, parentheses. `&&` and `||` are accepted too.

Anything else (attributes, imports, indexing, strings, keyword arguments, non-literal periods) is rejected
before backtesting, with an error message the agents learn from.

## Add an indicator

Add a function to `src/atsa/dsl/indicators.py` (make it causal), register it in `FUNCTIONS` in `src/atsa/dsl/expr.py`
with an argument signature (`"si"` = series + int), and it automatically shows up in the agents' prompts.
