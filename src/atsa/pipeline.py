"""The orchestrator: research -> swarm ideas -> validate -> backtest -> gate -> evolve -> store -> report.

          +-------------------+
 bars --> |  Quant Analyst    |  live brief: trend, vol regime, Hurst, half-life, fair value, mispricing
          +---------+---------+
                    v
   +----------------+-----------------+
   | Agent swarm (parallel LLM calls) |  momentum / mean-rev / vol / fair-value / regime / seasonality
   +----------------+-----------------+
                    v
          safe DSL validation  ->  JIT backtests (shared indicator cache, multi-core)
                    v
          Robustness gate: IS + OOS Sharpe, walk-forward, 2x costs, max DD, Deflated Sharpe
                    v
          leaders + rejects fed back to the swarm  ->  next generation (evolution)
                    v
          SQLite leaderboard  ->  reports/ + winners_latest.json  ->  your bot / dashboard
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from rich.console import Console

from .agents.swarm import Swarm
from .backtest.evaluate import Evaluation
from .backtest.gate import apply_gate, distinct, raw_score
from .backtest.parallel import evaluate_many
from .config import Settings, infer_periods_per_year
from .data import load_bars
from .dsl.expr import Evaluator
from .dsl.strategy import Strategy
from .llm import LLMProvider, get_provider
from .report import write_reports
from .research.analyst import analyze
from .store import Store

log = logging.getLogger("atsa")


@dataclass
class SymbolResult:
    symbol: str
    brief: dict
    evaluations: list[Evaluation] = field(default_factory=list)
    agent_errors: list[str] = field(default_factory=list)
    invalid: int = 0

    @property
    def winners(self) -> list[Evaluation]:
        return sorted([e for e in self.evaluations if e.passed], key=lambda e: e.score, reverse=True)


@dataclass
class CycleResult:
    run_id: int
    symbols: list[SymbolResult]
    seconds: float
    report_md: str | None = None
    report_json: str | None = None

    @property
    def tested(self) -> int:
        return sum(len(s.evaluations) for s in self.symbols)

    @property
    def winners(self) -> list[Evaluation]:
        return sorted([w for s in self.symbols for w in s.winners], key=lambda e: e.score, reverse=True)

    @property
    def distinct_winners(self) -> list[Evaluation]:
        """One best strategy per independent idea (near-duplicate variants folded together)."""
        return sorted(distinct(self.winners), key=lambda e: e.score, reverse=True)


class Pipeline:
    def __init__(self, settings: Settings, provider: LLMProvider | None = None, console: Console | None = None,
                 quiet: bool = False):
        self.s = settings
        self.provider = provider or get_provider(settings.llm, seed=settings.swarm.seed)
        self.console = console or Console(quiet=quiet)
        self.store = Store(settings.runtime.db_path)

    def _ppy(self) -> float:
        return self.s.data.periods_per_year or infer_periods_per_year(self.s.data.interval, self.s.data.source)

    def _champions(self, symbol: str) -> list[Strategy]:
        """Yesterday's winners are re-validated on fresh data every cycle (and demoted if they decay)."""
        out = []
        for row in self.store.leaderboard(limit=self.s.swarm.evolve_top_k, symbol=symbol):
            try:
                spec = {**row["spec"], "generation": 0}
                out.append(Strategy.model_validate(spec))
            except Exception:  # noqa: BLE001 - old/invalid rows are skipped
                continue
        return out

    async def hunt_symbol(self, symbol: str, run_id: int) -> SymbolResult:
        c = self.console
        cfg = self.s
        live_age = cfg.runtime.interval_minutes if cfg.data.end is None else None
        bars = await asyncio.to_thread(load_bars, cfg.data, symbol, False, live_age)
        ppy = self._ppy()
        brief = analyze(bars, ppy)
        res = SymbolResult(symbol=symbol, brief=brief)
        mc = brief["market_character"]
        c.print(f"\n[bold cyan]>> {symbol}[/] {len(bars)} bars {bars.start_date}->{bars.end_date} | "
                f"{brief['trend']['state']} | vol {brief['volatility']['regime']} | {mc['verdict']} | "
                f"mispricing z={brief['fair_value']['mispricing_z_60']}")

        evaluator = Evaluator(bars)
        seen: set[str] = set()
        swarm = Swarm(self.provider, cfg.swarm, cfg.backtest)

        champions = [s for s in self._champions(symbol) if s.id not in seen]
        if champions:
            seen.update(s.id for s in champions)
            evals = await asyncio.to_thread(evaluate_many, bars, champions, cfg.backtest, ppy,
                                            cfg.runtime.workers, evaluator)
            res.evaluations.extend(evals)
            c.print(f"   [dim]re-validating {len(champions)} champion(s) from previous cycles on fresh data[/]")

        leaders: list[dict] = []
        failures: list[dict] = []
        for gen in range(1, cfg.swarm.generations + 1):
            t0 = time.perf_counter()
            reports = await swarm.research(brief, gen, leaders, failures)
            fresh: list[Strategy] = []
            for rep in reports:
                if rep.error:
                    res.agent_errors.append(f"{rep.agent}: {rep.error}")
                res.invalid += len(rep.invalid)
                for s in rep.valid:
                    if s.id not in seen:
                        seen.add(s.id)
                        fresh.append(s)
            t_llm = time.perf_counter() - t0
            t1 = time.perf_counter()
            evals = await asyncio.to_thread(evaluate_many, bars, fresh, cfg.backtest, ppy, cfg.runtime.workers,
                                            evaluator)
            t_bt = time.perf_counter() - t1
            res.evaluations.extend(evals)
            apply_gate(res.evaluations, cfg.gate)
            ranked = sorted([e for e in res.evaluations if not e.error], key=raw_score, reverse=True)
            leaders = [e.to_dict() for e in ranked[: cfg.swarm.evolve_top_k]]
            gen_fails = sorted([e for e in evals if not e.passed and not e.error], key=raw_score, reverse=True)
            failures = [e.to_dict() for e in gen_fails[:6]]
            n_pass = sum(e.passed for e in res.evaluations)
            best = ranked[0] if ranked else None
            errs = sum(1 for r in reports if r.error)
            n_invalid = sum(len(r.invalid) for r in reports)
            extra = (f", {n_invalid} invalid" if n_invalid else "") + (f", {errs} agent errors" if errs else "")
            c.print(f"   gen {gen}: {sum(r.proposed for r in reports)} ideas from {len(reports)} agents "
                    f"({len(fresh)} new{extra}) | research {t_llm:.1f}s | backtests {t_bt * 1000:.0f}ms | "
                    f"passing: [bold green]{n_pass}[/]"
                    + (f" | best OOS SR {best.out_of_sample.get('sharpe', 0):.2f}" if best else ""))
            if errs and not fresh:
                for r in reports:
                    if r.error:
                        c.print(f"   [red]{r.agent}: {r.error[:200]}[/]")
                if gen == 1 and errs == len(reports):
                    break

        # final gate: Deflated Sharpe counts every independent idea tried on this symbol this cycle
        apply_gate(res.evaluations, cfg.gate)
        return res

    async def run_cycle(self) -> CycleResult:
        cfg = self.s
        t0 = time.perf_counter()
        run_id = self.store.start_run(self.provider.name, self.provider.model or "", cfg.data.source, cfg.data.symbols)
        self.console.print(f"[bold magenta]AI Trading Strategy Agents[/] | run #{run_id} | model "
                           f"[bold]{self.provider.describe()}[/] | {len(cfg.swarm.agents)} agents x "
                           f"{cfg.swarm.generations} generations | data: {cfg.data.source} {cfg.data.interval}")
        results: list[SymbolResult] = []
        status = "ok"
        try:
            for sym in cfg.data.symbols:
                try:
                    results.append(await self.hunt_symbol(sym, run_id))
                except Exception as e:  # noqa: BLE001 - keep hunting other symbols
                    status = "partial"
                    self.console.print(f"[red]   {sym} failed: {type(e).__name__}: {e}[/]")
                    log.exception("symbol %s failed", sym)
            for r in results:
                for ev in r.evaluations:
                    if ev.error:
                        continue
                    self.store.upsert(run_id, ev.to_dict(), ev.equity_curve)
            self.store.commit()
        finally:
            await self.provider.aclose()
        seconds = time.perf_counter() - t0
        cyc = CycleResult(run_id, results, seconds)
        winners = [w.to_dict() for w in cyc.distinct_winners]
        near = sorted([e for r in results for e in r.evaluations if not e.passed and not e.error],
                      key=raw_score, reverse=True)
        header = {"provider": self.provider.name, "model": self.provider.model, "source": cfg.data.source,
                  "interval": cfg.data.interval, "fee_bps": cfg.backtest.fee_bps,
                  "slippage_bps": cfg.backtest.slippage_bps, "oos_fraction": cfg.backtest.oos_fraction}
        md, js = write_reports(cfg.runtime.reports_dir, run_id, header, [r.brief for r in results], winners,
                               [e.to_dict() for e in near[:8]],
                               {"tested": cyc.tested, "seconds": seconds, "passed": len(cyc.winners)})
        cyc.report_md, cyc.report_json = str(md), str(js)
        self.store.finish_run(run_id, cyc.tested, len(cyc.winners), status, dict(self.provider.usage), str(md))
        return cyc

    def close(self) -> None:
        self.store.close()
