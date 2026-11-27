import httpx
import numpy as np
import polars as pl

from atsa.config import DataConfig, infer_periods_per_year
from atsa.data import generate_bars, load_bars
from atsa.data.sources import load_binance, read_ohlcv_file
from atsa.research import analyze, brief_markdown


def test_synthetic_is_deterministic_and_valid():
    a, b = generate_bars("X", n=500), generate_bars("X", n=500)
    np.testing.assert_array_equal(a.close, b.close)
    assert (a.high >= np.maximum(a.open, a.close) - 1e-9).all()
    assert (a.low <= np.minimum(a.open, a.close) + 1e-9).all()
    assert "calendar_effect" in a.meta["planted"]


def test_csv_loader_handles_dates_and_missing_columns(tmp_path):
    f = tmp_path / "SPY.csv"
    pl.DataFrame({"Date": ["2024-01-03", "2024-01-02", "2024-01-04"], "Close": [101.0, 100.0, 102.0]}).write_csv(f)
    bars = read_ohlcv_file(f, "SPY", "1d")
    assert list(bars.close) == [100.0, 101.0, 102.0]  # sorted
    cfg = DataConfig(source="csv", path=str(tmp_path))
    assert len(load_bars(cfg, "SPY")) == 3


def test_binance_pagination_and_parsing():
    day = 86_400_000
    start = 1_600_000_000_000

    def kline(i):
        t = start + i * day
        return [t, "1", "2", "0.5", str(1 + i), "10", t + day - 1, "0", 0, "0", "0", "0"]

    def handler(req: httpx.Request):
        st = int(req.url.params["startTime"])
        i0 = (st - start) // day if st >= start else 0
        rows = [kline(i) for i in range(i0, min(i0 + 1000, 1500))]
        return httpx.Response(200, json=rows)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    bars = load_binance("BTC/USDT", "1d", "2020-09-13", None, client=client)
    assert len(bars) == 1500 and bars.close[-1] == 1500.0


def test_periods_per_year():
    assert infer_periods_per_year("1d", "yahoo") == 252
    assert infer_periods_per_year("1d", "binance") == 365
    assert infer_periods_per_year("1h", "binance") == 365 * 24
    assert infer_periods_per_year("1mo", "yahoo") == 12


def test_research_brief_has_the_essentials():
    brief = analyze(generate_bars("DEMO-B"))
    for key in ("trend", "volatility", "market_character", "fair_value", "seasonality", "event_study"):
        assert key in brief
    assert brief["event_study"]["after_big_down_bar"]["t_stat"] > 2  # planted overreaction edge is visible
    assert "DEMO-B" in brief_markdown(brief)
