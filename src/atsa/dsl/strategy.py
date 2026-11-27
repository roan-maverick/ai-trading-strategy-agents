"""Strategy spec: the JSON contract between LLM agents, the backtester and your trading bot."""

from __future__ import annotations

import hashlib
import json
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from .expr import ExprError, parse

RULE_FIELDS = ("entry_long", "exit_long", "entry_short", "exit_short")


class Strategy(BaseModel):
    name: str = Field(default="unnamed", max_length=120)
    thesis: str = Field(default="", max_length=1200)
    agent: str = "manual"
    generation: int = 1
    parent_id: Optional[str] = None

    entry_long: Optional[str] = None
    exit_long: Optional[str] = None
    entry_short: Optional[str] = None
    exit_short: Optional[str] = None

    stop_loss: Optional[float] = Field(default=None, description="fraction, e.g. 0.05 = 5%")
    take_profit: Optional[float] = Field(default=None, description="fraction, e.g. 0.10 = 10%")
    max_hold_bars: Optional[int] = None

    @field_validator("entry_long", "exit_long", "entry_short", "exit_short", mode="before")
    @classmethod
    def _blank_to_none(cls, v):
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("rules must be strings")
        v = v.strip()
        return v or None

    @field_validator("stop_loss", "take_profit", mode="before")
    @classmethod
    def _fraction(cls, v):
        if v in (None, "", 0, 0.0):
            return None
        v = float(v)
        if v >= 1.0:  # agents sometimes send percent (5 -> 5%)
            v = v / 100.0
        if not 0.0005 <= v <= 0.9:
            raise ValueError("stop_loss/take_profit must be between 0.05% and 90%")
        return v

    @field_validator("max_hold_bars", mode="before")
    @classmethod
    def _hold(cls, v):
        if v in (None, "", 0):
            return None
        v = int(v)
        if v < 1:
            raise ValueError("max_hold_bars must be >= 1")
        return v

    @model_validator(mode="after")
    def _check_rules(self):
        if not self.entry_long and not self.entry_short:
            raise ValueError("strategy needs entry_long and/or entry_short")
        for f in RULE_FIELDS:
            expr = getattr(self, f)
            if expr:
                try:
                    parse(expr)
                except ExprError as e:
                    raise ValueError(f"{f}: {e}") from None
        return self

    @property
    def id(self) -> str:
        """Content hash of the trading logic (name/thesis excluded) -> dedupes identical ideas."""
        core = {f: getattr(self, f) for f in RULE_FIELDS}
        core.update(stop_loss=self.stop_loss, take_profit=self.take_profit, max_hold_bars=self.max_hold_bars)
        return hashlib.sha1(json.dumps(core, sort_keys=True).encode()).hexdigest()[:12]

    @property
    def direction(self) -> str:
        if self.entry_long and self.entry_short:
            return "long_short"
        return "long" if self.entry_long else "short"

    def spec(self) -> dict:
        return self.model_dump(exclude_none=True)


STRATEGY_JSON_SCHEMA_HINT = """{
  "name": "short descriptive name",
  "thesis": "1-2 sentences: which inefficiency/mispricing this exploits and why it should persist",
  "entry_long": "expression or null",
  "exit_long": "expression or null (null = hold max_hold_bars, or exit when entry_long turns false)",
  "entry_short": "expression or null",
  "exit_short": "expression or null",
  "stop_loss": 0.05,
  "take_profit": null,
  "max_hold_bars": null
}"""
