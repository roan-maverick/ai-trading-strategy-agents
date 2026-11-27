"""Configuration: one YAML file + environment variables (.env supported, no extra deps)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env loader. Existing environment variables always win."""
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


class LLMConfig(BaseModel):
    provider: str = "gemini"
    model: Optional[str] = None  # None -> provider default (gemini -> gemini-4-neon)
    temperature: Optional[float] = 0.8
    max_output_tokens: int = 8192
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None  # override which env var holds the key
    thinking_level: Optional[str] = None  # Gemini 4: low | medium | high | xhigh
    timeout_s: float = 180.0
    max_retries: int = 4
    json_mode: Optional[bool] = None  # None -> provider decides


class DataConfig(BaseModel):
    source: Literal["synthetic", "csv", "yahoo", "binance"] = "synthetic"
    symbols: list[str] = Field(default_factory=lambda: ["DEMO-A", "DEMO-B"])
    interval: str = "1d"
    start: Optional[str] = "2016-01-01"
    end: Optional[str] = None
    path: Optional[str] = None  # csv/parquet file or folder (csv source)
    bars: int = 2520  # synthetic only (~10 years of daily bars)
    cache_dir: str = ".atsa/cache"
    periods_per_year: Optional[float] = None  # None -> inferred from interval + source


class SwarmConfig(BaseModel):
    agents: list[str] = Field(
        default_factory=lambda: [
            "momentum_hunter",
            "mean_reversion_sniper",
            "volatility_breakout",
            "fair_value_arbitrageur",
            "regime_switcher",
            "seasonality_miner",
        ]
    )
    strategies_per_agent: int = 6
    generations: int = 3
    evolve_top_k: int = 8
    concurrency: int = 6
    seed: Optional[int] = None


class BacktestConfig(BaseModel):
    fee_bps: float = 5.0
    slippage_bps: float = 5.0
    allow_short: bool = True
    oos_fraction: float = 0.3
    wf_folds: int = 5


class GateConfig(BaseModel):
    min_trades: int = 20
    min_sharpe_is: float = 0.5
    min_sharpe_oos: float = 0.5
    max_drawdown: float = 0.35
    min_wf_positive: float = 0.6
    min_dsr: float = 0.90
    min_stress_sharpe: float = 0.25


class RuntimeConfig(BaseModel):
    db_path: str = ".atsa/atsa.db"
    reports_dir: str = "reports"
    workers: int = 0  # 0 -> auto (cpu count)
    parallel_symbols: int = 1  # hunt several symbols at once (LLM latency bound -> big speed-up)
    interval_minutes: float = 60.0  # pause between cycles in --forever mode


class Settings(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    swarm: SwarmConfig = Field(default_factory=SwarmConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    gate: GateConfig = Field(default_factory=GateConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)


_ENV_MAP = {
    "ATSA_PROVIDER": ("llm", "provider"),
    "ATSA_MODEL": ("llm", "model"),
    "ATSA_BASE_URL": ("llm", "base_url"),
    "ATSA_THINKING_LEVEL": ("llm", "thinking_level"),
    "ATSA_SOURCE": ("data", "source"),
    "ATSA_INTERVAL": ("data", "interval"),
    "ATSA_DB": ("runtime", "db_path"),
}


def load_settings(path: str | Path | None = None) -> Settings:
    """Load settings from YAML (atsa.yaml by default if present) then apply env overrides."""
    load_dotenv()
    data: dict = {}
    candidates = [Path(path)] if path else [Path("atsa.yaml"), Path("atsa.yml")]
    for c in candidates:
        if c.exists():
            data = yaml.safe_load(c.read_text()) or {}
            break
    else:
        if path:
            raise FileNotFoundError(f"Config file not found: {path}")
    settings = Settings.model_validate(data)
    for env, (section, key) in _ENV_MAP.items():
        val = os.environ.get(env)
        if val:
            setattr(getattr(settings, section), key, val)
    if os.environ.get("ATSA_SYMBOLS"):
        settings.data.symbols = [s.strip() for s in os.environ["ATSA_SYMBOLS"].split(",") if s.strip()]
    return settings


def infer_periods_per_year(interval: str, source: str) -> float:
    """Annualisation factor. Crypto trades 24/7; everything else uses US equity sessions."""
    iv = interval.strip()
    if iv.lower().endswith("mo"):
        return 12.0
    unit = iv[-1]
    try:
        n = float(iv[:-1] or 1)
    except ValueError:
        n = 1.0
    always_open = source == "binance"
    days = 365.0 if always_open else 252.0
    hours_per_day = 24.0 if always_open else 6.5
    if unit in ("d", "D"):
        return days / n
    if unit in ("w", "W"):
        return 52.0 / n
    if unit in ("h", "H"):
        return days * hours_per_day / n
    if unit == "M":  # Binance uses 1M for monthly
        return 12.0
    if unit == "m":
        return days * hours_per_day * 60.0 / n
    return 252.0
