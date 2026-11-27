"""Columnar OHLCV container. Everything downstream works on contiguous float64 NumPy arrays."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl


@dataclass
class Bars:
    symbol: str
    interval: str
    ts: np.ndarray  # int64 epoch milliseconds (UTC)
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    source: str = "unknown"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.ts = np.ascontiguousarray(self.ts, dtype=np.int64)
        for name in ("open", "high", "low", "close", "volume"):
            setattr(self, name, np.ascontiguousarray(getattr(self, name), dtype=np.float64))
        n = len(self.ts)
        for name in ("open", "high", "low", "close", "volume"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"Bars column {name!r} has wrong length")

    def __len__(self) -> int:
        return len(self.ts)

    def slice(self, start: int, stop: int | None = None) -> "Bars":
        s = slice(start, stop)
        return Bars(self.symbol, self.interval, self.ts[s], self.open[s], self.high[s], self.low[s],
                    self.close[s], self.volume[s], self.source, dict(self.meta))

    def tail(self, n: int) -> "Bars":
        return self.slice(max(0, len(self) - n))

    @property
    def start_date(self) -> str:
        return str(np.datetime64(int(self.ts[0]), "ms"))[:10] if len(self) else ""

    @property
    def end_date(self) -> str:
        return str(np.datetime64(int(self.ts[-1]), "ms"))[:10] if len(self) else ""

    def to_polars(self) -> pl.DataFrame:
        return pl.DataFrame({
            "ts": self.ts, "open": self.open, "high": self.high, "low": self.low,
            "close": self.close, "volume": self.volume,
        })

    @classmethod
    def from_polars(cls, df: pl.DataFrame, symbol: str, interval: str, source: str) -> "Bars":
        df = clean_frame(df)
        return cls(symbol, interval, df["ts"].to_numpy(), df["open"].to_numpy(), df["high"].to_numpy(),
                   df["low"].to_numpy(), df["close"].to_numpy(), df["volume"].to_numpy(), source)


def clean_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Sort, de-duplicate, drop non-positive prices and repair inconsistent high/low."""
    df = (
        df.select(["ts", "open", "high", "low", "close", "volume"])
        .with_columns(pl.col("ts").cast(pl.Int64), pl.col(["open", "high", "low", "close", "volume"]).cast(pl.Float64))
        .drop_nulls(["ts", "close"])
        .unique(subset="ts", keep="last")
        .sort("ts")
        .filter(pl.col("close") > 0)
        .with_columns(
            pl.col("open").fill_null(pl.col("close")),
            pl.col("volume").fill_null(0.0),
        )
        .with_columns(
            pl.max_horizontal("open", "high", "close").alias("high"),
            pl.min_horizontal("open", "low", "close").alias("low"),
        )
    )
    return df
