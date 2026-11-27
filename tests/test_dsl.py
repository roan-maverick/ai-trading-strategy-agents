import numpy as np
import pytest

from atsa.data import generate_bars
from atsa.dsl import Evaluator, ExprError, Strategy, parse
from atsa.dsl import indicators as ind


@pytest.fixture(scope="module")
def bars():
    return generate_bars("TEST", n=600)


@pytest.mark.parametrize("bad", [
    '__import__("os")', "close.real", "close.real()", "open[0]", "lambda: 1", "sma(close, x)",
    "sma(close, 0)", "lag(close, -1)", "foo(close)", "sma(close=1)", '"text"', "[1, 2]", "close if 1 else 2",
    "sma(close, 5000)",
])
def test_unsafe_or_invalid_expressions_rejected(bad):
    with pytest.raises(ExprError):
        parse(bad)


@pytest.mark.parametrize("good", [
    "rsi(close,14) < 30 and close > sma(close,200)",
    "cross_above(ema(close,12), ema(close,26)) or adx(14) > 25",
    "fv_gap(close,60) < -2", "dow == 0", "not (close > vwap(20))", "zscore(ret,20) < -1.8",
    "macd(close,12,26) > macd_signal(close,12,26,9)", "streak(close > sma(close,10), 3)",
    "close < bb_lower(close,20,2.0)", "stoch_k(14) < 20 && rising(close,3)",
])
def test_valid_expressions_evaluate(bars, good):
    out = Evaluator(bars).condition(good)
    assert out.dtype == np.bool_ and len(out) == len(bars)


def test_indicators_match_naive(bars):
    c = bars.close
    np.testing.assert_allclose(ind.sma(c, 20)[19:], np.convolve(c, np.ones(20) / 20, mode="valid"), rtol=1e-9)
    assert np.isnan(ind.sma(c, 20)[:19]).all()
    sd = ind.rolling_std(c, 20)
    assert abs(sd[50] - np.std(c[31:51], ddof=1)) < 1e-9
    assert ind.rolling_max(c, 10)[30] == c[21:31].max()
    rsi = ind.wilder_rsi(c, 14)
    assert np.nanmin(rsi) >= 0 and np.nanmax(rsi) <= 100


def test_indicators_are_causal(bars):
    """Changing the future must not change the past (no look-ahead)."""
    ev_full = Evaluator(bars).eval("fv_gap(close,60) + rsi(close,14) + adx(14) + zscore(close,20)")
    cut = bars.slice(0, 400)
    ev_cut = Evaluator(cut).eval("fv_gap(close,60) + rsi(close,14) + adx(14) + zscore(close,20)")
    np.testing.assert_allclose(ev_full[:400], ev_cut, equal_nan=True)


def test_strategy_validation_and_id():
    s = Strategy(name="a", entry_long="rsi(close,14) < 30", stop_loss=5)  # percent -> fraction
    assert s.stop_loss == pytest.approx(0.05)
    same = Strategy(name="different name", entry_long="rsi(close,14) < 30", stop_loss=0.05)
    assert s.id == same.id
    with pytest.raises(ValueError):
        Strategy(name="no entry")
    with pytest.raises(ValueError):
        Strategy(name="bad", entry_long="import os")
