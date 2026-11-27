"""The swarm: N specialist agents research in parallel, every idea is validated before backtesting."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable

from pydantic import ValidationError

from ..config import BacktestConfig, SwarmConfig
from ..dsl.strategy import Strategy
from ..llm.base import LLMProvider, extract_json
from .personas import get_persona
from .prompts import system_prompt, user_prompt

log = logging.getLogger("atsa.swarm")


@dataclass
class AgentReport:
    agent: str
    proposed: int = 0
    valid: list[Strategy] = field(default_factory=list)
    invalid: list[str] = field(default_factory=list)
    error: str | None = None
    seconds: float = 0.0


def parse_strategies(text: str, agent: str, generation: int) -> tuple[list[Strategy], list[str], int]:
    data = extract_json(text)
    items = data.get("strategies", []) if isinstance(data, dict) else data
    if isinstance(items, dict):
        items = [items]
    valid, invalid = [], []
    for item in items or []:
        if not isinstance(item, dict):
            invalid.append("not an object")
            continue
        item = {**item, "agent": agent, "generation": generation}
        try:
            valid.append(Strategy.model_validate(item))
        except ValidationError as e:
            msg = "; ".join(err.get("msg", "") for err in e.errors())[:300]
            invalid.append(f"{item.get('name', '?')}: {msg}")
    return valid, invalid, len(items or [])


class Swarm:
    def __init__(self, provider: LLMProvider, swarm_cfg: SwarmConfig, bt_cfg: BacktestConfig,
                 on_event: Callable[..., None] | None = None, symbol: str = ""):
        self.provider = provider
        self.cfg = swarm_cfg
        self.bt = bt_cfg
        self.symbol = symbol
        self._emit = on_event or (lambda *a, **k: None)
        self._sem = asyncio.Semaphore(max(1, swarm_cfg.concurrency))

    async def _run_agent(self, agent: str, brief: dict, n: int, generation: int,
                         leaders: list[dict], failures: list[dict]) -> AgentReport:
        persona = get_persona(agent)
        rep = AgentReport(agent=agent)
        t0 = time.perf_counter()
        async with self._sem:
            self._emit("agent_start", symbol=self.symbol, agent=agent, generation=generation)
            try:
                text = await self.provider.complete(
                    system_prompt(persona),
                    user_prompt(brief, n, self.bt.allow_short, self.bt.fee_bps + self.bt.slippage_bps,
                                generation, leaders, failures),
                    meta={"persona": agent, "n": n, "brief": brief, "generation": generation,
                          "allow_short": self.bt.allow_short, "leaders": leaders},
                )
                rep.valid, rep.invalid, rep.proposed = parse_strategies(text, agent, generation)
            except Exception as e:  # one bad agent never kills the desk
                rep.error = f"{type(e).__name__}: {e}"
                log.warning("agent %s failed: %s", agent, rep.error)
        rep.seconds = time.perf_counter() - t0
        self._emit("agent_done", symbol=self.symbol, agent=agent, generation=generation, report=rep)
        return rep

    async def research(self, brief: dict, generation: int = 1, leaders: list[dict] | None = None,
                       failures: list[dict] | None = None) -> list[AgentReport]:
        agents = list(self.cfg.agents)
        if generation > 1 and leaders and "evolver" not in agents:
            agents.append("evolver")
        tasks = [self._run_agent(a, brief, self.cfg.strategies_per_agent, generation, leaders or [],
                                 failures or []) for a in agents]
        return list(await asyncio.gather(*tasks))
