import numpy as np
import pytest

from atsa.backtest import apply_gate, evaluate_many, evaluate_strategy, run_backtest
from atsa.backtest.metrics import (
    deflated_sharpe,
    effective_trials,
    max_drawdown,
    probabilistic_sharpe,
    sharpe,
)
from atsa.config import BacktestConfig, GateConfig
from atsa.data import Bars, generate_bars
from atsa.dsl import Strategy


@pytest.fixture(scope="module")
def bars():
    return generate_bars("BT", n=1500)


def test_always_long_equals_buy_and_hold(bars):
    r = run_backtest(bars, Strategy(entry_long="close > 0"), 0, 0)
    # signal at close of bar 1 -> fill at open of bar 2
    assert r.equity[-1] == pytest.approx(bars.close[-1] / bars.open[2], rel=1e-10)
    assert r.n_trades == 1


def test_costs_are_charged(bars):
    free = run_backtest(bars, Strategy(entry_long="close > 0"), 0, 0).equity[-1]
    paid = run_backtest(bars, Strategy(entry_long="close > 0"), 5, 5).equity[-1]
    assert paid == pytest.approx(free * (1 - 0.001), rel=1e-10)


def _toy(prices, opens=None, highs=None, lows=None):
    n = len(prices)
    c = np.asarray(prices, float)
    o = np.asarray(opens if opens is not None else c, float)
    h = np.asarray(highs if highs is not None else np.maximum(o, c), float)
    lo = np.asarray(lows if lows is not None else np.minimum(o, c), float)
    return Bars("TOY", "1d", np.arange(n) * 86_400_000, o, h, lo, c, np.ones(n))


def test_stop_loss_fills_at_stop_or_gap():
    # enter at open of bar 2 (=100); bar 3 low touches 90 -> 5% stop fills at 95
    b = _toy([100, 100, 100, 97, 97], opens=[100, 100, 100, 99, 97], lows=[100, 100, 100, 90, 97])
    r = run_backtest(b, Strategy(entry_long="close > 0", exit_long="close < 0", stop_loss=0.05), 0, 0)
    assert r.equity[3] == pytest.approx(0.95)
    # gap below the stop fills at the open, not at the stop
    b = _toy([100, 100, 100, 80, 80], opens=[100, 100, 100, 85, 80], lows=[100, 100, 100, 79, 80])
    r = run_backtest(b, Strategy(entry_long="close > 0", exit_long="close < 0", stop_loss=0.05), 0, 0)
    assert r.equity[3] == pytest.approx(0.85)


def test_max_hold_without_exit_holds_n_bars():
    b = _toy([100, 100, 101, 102, 103, 104, 105, 106])
    r = run_backtest(b, Strategy(entry_long="dow == dow and close == 100", max_hold_bars=3), 0, 0)
    held = r.position[r.position != 0]
    assert len(held) == 3


def test_short_profits_when_price_falls():
    b = _toy([100, 100, 90, 80, 70])
    r = run_backtest(b, Strategy(entry_short="close > 0"), 0, 0)
    assert r.equity[-1] > 1.0


def test_metrics_basics():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 2000)
    assert sharpe(r, 252) == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(252))
    assert max_drawdown(np.array([1, 2, 1, 3])) == pytest.approx(0.5)
    assert probabilistic_sharpe(r) > 0.95
    noise = rng.normal(0, 0.01, (200, 1000))
    trials = np.array([x.mean() / x.std() for x in noise])
    # the best of 200 random strategies looks great but deflates to luck
    best = noise[np.argmax(trials)]
    assert probabilistic_sharpe(best) > 0.9
    assert deflated_sharpe(best, trials) < 0.6


def test_effective_trials_clusters_duplicates():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 0.01, (5, 500))
    dupes = np.vstack([base + rng.normal(0, 0.001, base.shape) for _ in range(4)])
    _, n = effective_trials(dupes)
    assert n == 5


def test_evaluate_and_gate(bars):
    cfg = BacktestConfig()
    strategies = [Strategy(name=f"z{n}", entry_long=f"zscore(close,{n}) < -2", exit_long=f"zscore(close,{n}) > 0")
                  for n in (10, 20, 30, 40)] + [Strategy(name="bad", entry_long="close > sma(close, 999)")]
    evals = evaluate_many(bars, strategies, cfg, 252, workers=1)
    apply_gate(evals, GateConfig())
    assert evals[-1].error and not evals[-1].passed  # look-back longer than half the data
    for e in evals[:-1]:
        assert e.full and e.in_sample and e.out_of_sample and len(e.wf_sharpes) == cfg.wf_folds
        assert 0.0 <= e.dsr <= 1.0
        assert e.passed == (not e.reasons)


def test_parallel_matches_serial(bars):
    cfg = BacktestConfig()
    strategies = [Strategy(name=f"s{n}", entry_long=f"close > sma(close,{n})") for n in range(10, 110)]
    a = evaluate_many(bars, strategies, cfg, 252, workers=1)
    b = evaluate_many(bars, strategies, cfg, 252, workers=2)
    assert [round(x.full["sharpe"], 10) for x in a] == [round(x.full["sharpe"], 10) for x in b]


def test_single_strategy_evaluation(bars):
    ev = evaluate_strategy(bars, Strategy(entry_long="rsi(close,2) < 10", max_hold_bars=3), BacktestConfig(), 252)
    assert ev.error is None and ev.full["trades"] > 10 and len(ev.equity_curve) > 10
