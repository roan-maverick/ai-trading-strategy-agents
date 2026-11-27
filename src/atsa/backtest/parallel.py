"""Parallel evaluation: shared indicator cache in-process, multi-core fan-out for big batches."""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

from ..config import BacktestConfig
from ..data.base import Bars
from ..dsl.expr import Evaluator
from ..dsl.strategy import Strategy
from .evaluate import Evaluation, evaluate_strategy

_W_BARS: Bars | None = None
_W_EVAL: Evaluator | None = None
_W_CFG: BacktestConfig | None = None
_W_PPY: float = 252.0

PROCESS_THRESHOLD = 96  # below this, one process with a hot indicator cache is faster


def _init_worker(bars: Bars, cfg: BacktestConfig, ppy: float) -> None:
    global _W_BARS, _W_EVAL, _W_CFG, _W_PPY
    _W_BARS, _W_EVAL, _W_CFG, _W_PPY = bars, Evaluator(bars), cfg, ppy


def _work(chunk: list[dict]) -> list[Evaluation]:
    out = []
    for spec in chunk:
        s = Strategy.model_validate(spec)
        ev = evaluate_strategy(_W_BARS, s, _W_CFG, _W_PPY, _W_EVAL)
        out.append(ev)
    return out


def resolve_workers(workers: int) -> int:
    return workers if workers and workers > 0 else max(1, (os.cpu_count() or 2) - 1)


def evaluate_many(bars: Bars, strategies: list[Strategy], cfg: BacktestConfig, periods_per_year: float,
                  workers: int = 0, evaluator: Evaluator | None = None) -> list[Evaluation]:
    workers = resolve_workers(workers)
    if workers <= 1 or len(strategies) < PROCESS_THRESHOLD:
        evaluator = evaluator or Evaluator(bars)
        return [evaluate_strategy(bars, s, cfg, periods_per_year, evaluator) for s in strategies]
    specs = [s.model_dump() for s in strategies]
    size = max(8, len(specs) // (workers * 4))
    chunks = [specs[i:i + size] for i in range(0, len(specs), size)]
    results: list[Evaluation] = []
    methods = mp.get_all_start_methods()
    ctx = mp.get_context("forkserver" if "forkserver" in methods else "spawn")  # fork-safe with JIT threads
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init_worker,
                             initargs=(bars, cfg, periods_per_year)) as pool:
        for part in pool.map(_work, chunks):
            results.extend(part)
    return results
