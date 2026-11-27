"""Unified data loading with a Parquet cache (fast re-loads, works offline after first fetch)."""

from __future__ import annotations

import time
from pathlib import Path

import polars as pl

from ..config import DataConfig
from .base import Bars
from .sources import load_binance, load_csv, load_yahoo
from .synthetic import generate_bars

__all__ = ["Bars", "load_bars", "generate_bars"]


def _cache_path(cfg: DataConfig, symbol: str) -> Path:
    safe = symbol.replace("/", "_").replace("^", "_").replace("=", "_")
    return Path(cfg.cache_dir) / f"{cfg.source}_{safe}_{cfg.interval}_{cfg.start or 'na'}_{cfg.end or 'live'}.parquet"


def load_bars(cfg: DataConfig, symbol: str, refresh: bool = False, max_age_minutes: float | None = None,
              seed: int | None = None) -> Bars:
    """Load bars for one symbol. Remote sources are cached as Parquet.

    max_age_minutes: re-download when the cache is older than this (live / 24-7 mode).
    """
    if cfg.source == "synthetic":
        return generate_bars(symbol, n=cfg.bars, interval=cfg.interval, seed=seed)
    if cfg.source == "csv":
        return load_csv(symbol, cfg.interval, cfg.path)

    path = _cache_path(cfg, symbol)
    fresh = path.exists() and not refresh
    if fresh and max_age_minutes is not None and cfg.end is None:
        fresh = (time.time() - path.stat().st_mtime) / 60.0 < max_age_minutes
    if fresh:
        return Bars.from_polars(pl.read_parquet(path), symbol, cfg.interval, source=cfg.source)

    if cfg.source == "yahoo":
        bars = load_yahoo(symbol, cfg.interval, cfg.start, cfg.end)
    elif cfg.source == "binance":
        bars = load_binance(symbol, cfg.interval, cfg.start, cfg.end)
    else:
        raise ValueError(f"Unknown data source {cfg.source!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    bars.to_polars().write_parquet(path)
    return bars
