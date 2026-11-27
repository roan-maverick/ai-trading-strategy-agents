# Architecture

```
src/atsa/
├── cli.py                 # `atsa` command line (Typer + Rich)
├── config.py              # Settings: atsa.yaml + env vars + .env
├── pipeline.py            # Orchestrator: one hunt cycle, champion re-validation, reports
├── data/
│   ├── base.py            # Bars: columnar float64 OHLCV arrays
│   ├── sources.py         # Binance (public), Yahoo, CSV/Parquet adapters
│   ├── synthetic.py       # Deterministic demo market with planted edges
│   └── __init__.py        # load_bars() + Parquet cache
├── research/analyst.py    # Quant Analyst: the live research brief (no LLM)
├── agents/
│   ├── personas.py        # The research desk (add your own)
│   ├── prompts.py         # Strict JSON-only prompts
│   └── swarm.py           # Parallel agents + validation
├── llm/
│   ├── base.py            # LLMProvider interface, retries, JSON extraction
│   ├── providers.py       # Gemini, OpenAI-compatible (10+ presets), Anthropic
│   └── offline.py         # Key-less quant-template provider + mutation-based evolver
├── dsl/
│   ├── expr.py            # Safe expression language (AST whitelist) + memoised evaluator
│   ├── indicators.py      # Numba-JIT causal indicators
│   └── strategy.py        # Strategy spec (the JSON contract)
├── backtest/
│   ├── engine.py          # Numba-JIT event simulation
│   ├── evaluate.py        # Scorecard: full / IS / OOS / walk-forward / 2x cost
│   ├── metrics.py         # Sharpe, Sortino, DD, PSR, DSR, trial clustering
│   ├── gate.py            # Robustness gate + scoring + idea de-duplication
│   └── parallel.py        # Multi-core evaluation
├── store.py               # SQLite leaderboard + run history
├── report.py              # Markdown report + bot hand-off JSON
└── dashboard/             # FastAPI + single-file HTML dashboard
```

## One hunt cycle

```
for each symbol:
    bars   = load_bars()                    # Parquet cache, refreshed when older than interval_minutes
    brief  = analyze(bars)                  # trend, vol regime, Hurst, VR, half-life, fair value,
                                            # mispricing z, event study, seasonality
    champs = leaderboard winners            # re-tested on fresh data (decay -> demotion)
    for generation in 1..G:
        ideas  = await swarm.research(brief, leaders, rejects)    # N agents in parallel
        valid  = Strategy.model_validate(...)                    # AST whitelist, no code execution
        evals  = evaluate_many(valid)                            # JIT backtests, shared indicator cache
        apply_gate(all evals so far)                             # DSR over independent idea clusters
        leaders, rejects = top-k, best near-misses               # fed to the next generation
    store + report
```

## Execution model (no look-ahead)

* Indicators are causal: `output[t]` uses only `input[:t+1]` (tested in `tests/test_dsl.py`).
* Signals are evaluated at the **close of bar t** and filled at the **open of bar t+1**.
* Stops and take-profits are checked intrabar against high/low. A gap through the stop fills at the open,
  and if both could trigger in the same bar, the stop is assumed to hit first.
* Costs: `(fee_bps + slippage_bps)` per side, per unit of turnover. A flip pays twice.
* If an exit rule is null and `max_hold_bars` is set, the trade is held for that many bars (event trades).
  If both are null, the trade exits when its entry condition turns false (state trades).

## The robustness gate

| Statistic | How |
|---|---|
| In-sample / out-of-sample | Fixed chronological split (`oos_fraction`, default last 30%). Indicators warm up before the split. |
| Walk-forward stability | Sharpe ratio in `wf_folds` consecutive time slices; the share that are positive. |
| Cost stress | Full re-simulation at 2x fees and slippage. |
| Deflated Sharpe Ratio | PSR against the expected maximum Sharpe of N unskilled trials (False Strategy Theorem). N and the trial-Sharpe variance come from **correlation-clustered** trials, so 30 re-tunes of one idea count as one idea, not 30. |
| Score | `DSR × (0.5·SR_oos + 0.3·min(SR_is,3) + 0.2·clip(Calmar)) × (0.5 + 0.5·WF_positive)` |

Winners are grouped by idea cluster when they are reported, so your bot gets distinct edges instead of 20 copies of one.

## Why a DSL instead of letting the LLM write Python?

1. **Safety:** LLM output is never executed. The parser rejects attribute access, imports, subscripts and lambdas.
2. **Speed:** indicator calls are memoised per dataset, so 1,000 strategies sharing `sma(close,200)` compute it once.
3. **Portability:** rules are readable strings that any bot, language or exchange integration can implement.
4. **No look-ahead by construction:** every function is causal, and negative lags are impossible.

## 24/7 mode

`atsa run --forever` loops: hunt, sleep `interval_minutes`, then hunt again. Each cycle re-downloads data when the
cache is stale, re-validates the champions and adds new ones. SQLite runs in WAL mode, so the dashboard can read
while the runner writes. `SIGTERM` finishes the current cycle and exits cleanly (Docker-friendly).
