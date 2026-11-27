"""Market data adapters: CSV/Parquet files, Yahoo Finance (stocks/ETFs/FX/futures), Binance (crypto, no key)."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import polars as pl

from .base import Bars

_TS_CANDIDATES = ("ts", "timestamp", "time", "date", "datetime", "open_time", "opentime")


def _to_epoch_ms(df: pl.DataFrame, col: str) -> pl.DataFrame:
    dtype = df.schema[col]
    if dtype.is_numeric():
        s = df[col].cast(pl.Float64)
        mx = float(s.max() or 0)
        mult = 1.0 if mx > 1e11 else 1000.0  # seconds vs milliseconds
        return df.with_columns((pl.col(col).cast(pl.Float64) * mult).cast(pl.Int64).alias("ts"))
    if dtype == pl.Utf8 or dtype == pl.String:
        parsed = pl.col(col).str.to_datetime(strict=False, time_zone="UTC")
        return df.with_columns(parsed.dt.epoch("ms").alias("ts"))
    if isinstance(dtype, (pl.Datetime, pl.Date)) or dtype in (pl.Datetime, pl.Date):
        return df.with_columns(pl.col(col).cast(pl.Datetime("ms")).dt.epoch("ms").alias("ts"))
    raise ValueError(f"Cannot interpret timestamp column {col!r} of type {dtype}")


def read_ohlcv_file(path: str | Path, symbol: str, interval: str) -> Bars:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        df = pl.read_parquet(path)
    else:
        df = pl.read_csv(path, try_parse_dates=False, infer_schema_length=10_000)
    df = df.rename({c: c.strip().lower().replace(" ", "_") for c in df.columns})
    if "adj_close" in df.columns and "close" not in df.columns:
        df = df.rename({"adj_close": "close"})
    ts_col = next((c for c in _TS_CANDIDATES if c in df.columns), None)
    if ts_col is None:
        raise ValueError(f"{path}: no timestamp column (expected one of {_TS_CANDIDATES})")
    df = _to_epoch_ms(df, ts_col)
    if "close" not in df.columns:
        raise ValueError(f"{path}: no 'close' column")
    for c in ("open", "high", "low"):
        if c not in df.columns:
            df = df.with_columns(pl.col("close").alias(c))
    if "volume" not in df.columns:
        df = df.with_columns(pl.lit(0.0).alias("volume"))
    return Bars.from_polars(df, symbol, interval, source="csv")


def load_csv(symbol: str, interval: str, path: str | None) -> Bars:
    if not path:
        raise ValueError("data.path is required for the csv source (a file, or a folder of SYMBOL.csv files)")
    p = Path(path)
    if p.is_dir():
        for ext in (".parquet", ".csv"):
            f = p / f"{symbol}{ext}"
            if f.exists():
                return read_ohlcv_file(f, symbol, interval)
        raise FileNotFoundError(f"No {symbol}.csv or {symbol}.parquet in {p}")
    return read_ohlcv_file(p, symbol, interval)


def load_yahoo(symbol: str, interval: str, start: str | None, end: str | None) -> Bars:
    try:
        import yfinance as yf
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError("Yahoo source needs yfinance: pip install 'ai-trading-strategy-agents[yahoo]'") from e
    raw = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=True, progress=False,
                      threads=False)
    if raw is None or len(raw) == 0:
        raise ValueError(f"Yahoo returned no data for {symbol}")
    if hasattr(raw.columns, "nlevels") and raw.columns.nlevels > 1:
        raw.columns = raw.columns.get_level_values(0)
    import pandas as pd  # yfinance depends on pandas

    raw = raw.reset_index()
    ts_col = raw.columns[0]
    ts = pd.to_datetime(raw[ts_col], utc=True)
    df = pl.DataFrame({
        "ts": ts.dt.tz_convert(None).to_numpy().astype("datetime64[ms]").astype("int64"),
        "open": raw["Open"].to_numpy(dtype=float),
        "high": raw["High"].to_numpy(dtype=float),
        "low": raw["Low"].to_numpy(dtype=float),
        "close": raw["Close"].to_numpy(dtype=float),
        "volume": raw["Volume"].to_numpy(dtype=float) if "Volume" in raw else np.zeros(len(raw)),
    })
    return Bars.from_polars(df, symbol, interval, source="yahoo")


BINANCE_ENDPOINTS = ("https://api.binance.com", "https://data-api.binance.vision")


def _date_ms(s: str | None) -> int | None:
    if not s:
        return None
    return int(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp() * 1000)


def parse_binance_klines(rows: list[list]) -> pl.DataFrame:
    return pl.DataFrame({
        "ts": [int(r[0]) for r in rows],
        "open": [float(r[1]) for r in rows],
        "high": [float(r[2]) for r in rows],
        "low": [float(r[3]) for r in rows],
        "close": [float(r[4]) for r in rows],
        "volume": [float(r[5]) for r in rows],
        "close_time": [int(r[6]) for r in rows],
    })


def load_binance(symbol: str, interval: str, start: str | None, end: str | None,
                 client: httpx.Client | None = None, max_bars: int = 200_000) -> Bars:
    """Public Binance spot klines (no API key). Symbol: BTCUSDT or BTC/USDT."""
    pair = symbol.replace("/", "").replace("-", "").upper()
    start_ms = _date_ms(start) or 1_500_000_000_000
    end_ms = _date_ms(end)
    own = client is None
    client = client or httpx.Client(timeout=30.0)
    frames: list[pl.DataFrame] = []
    try:
        base = None
        cursor = start_ms
        fetched = 0
        while fetched < max_bars:
            params = {"symbol": pair, "interval": interval, "startTime": cursor, "limit": 1000}
            if end_ms:
                params["endTime"] = end_ms
            rows = None
            last_err: Exception | None = None
            for ep in ([base] if base else BINANCE_ENDPOINTS):
                try:
                    resp = client.get(f"{ep}/api/v3/klines", params=params)
                    resp.raise_for_status()
                    rows = resp.json()
                    base = ep
                    break
                except Exception as e:  # try the next public endpoint
                    last_err = e
            if rows is None:
                raise RuntimeError(f"Binance request failed for {pair}: {last_err}")
            if not rows:
                break
            df = parse_binance_klines(rows)
            frames.append(df)
            fetched += len(rows)
            cursor = int(rows[-1][0]) + 1
            if len(rows) < 1000:
                break
            time.sleep(0.05)
    finally:
        if own:
            client.close()
    if not frames:
        raise ValueError(f"Binance returned no data for {pair} {interval}")
    df = pl.concat(frames)
    now_ms = int(time.time() * 1000)
    df = df.filter(pl.col("close_time") < now_ms)  # drop the still-forming candle
    return Bars.from_polars(df, symbol, interval, source="binance")
