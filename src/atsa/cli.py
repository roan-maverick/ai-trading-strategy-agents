"""`atsa` command line: demo, run (24/7), research, backtest, leaderboard, signals, export, dashboard, bench."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .config import Settings, infer_periods_per_year, load_settings

app = typer.Typer(add_completion=False, no_args_is_help=True, rich_markup_mode="rich",
                  help="[bold magenta]AI Trading Strategy Agents[/] - a swarm of AI quants that finds, "
                       "backtests and filters trading strategies 24/7. Works with any LLM.")
console = Console()

BANNER = r"""[bold magenta]
    ___    ____   ______               ___                 _____ __             __
   /   |  /  _/  /_  __/________ _____/ (_)___  ____ _   / ___// /__________ _/ /____  ____ ___  __
  / /| |  / /     / / / ___/ __ `/ __  / / __ \/ __ `/   \__ \/ __/ ___/ __ `/ __/ _ \/ __ `/ / / /
 / ___ |_/ /     / / / /  / /_/ / /_/ / / / / / /_/ /   ___/ / /_/ /  / /_/ / /_/  __/ /_/ / /_/ /
/_/  |_/___/    /_/ /_/   \__,_/\__,_/_/_/ /_/\__, /   /____/\__/_/   \__,_/\__/\___/\__, /\__, /
                                             /____/   [white]A G E N T S[/]                 /____//____/
[/]"""


def _settings(config: Optional[Path], provider: Optional[str], model: Optional[str], symbols: Optional[str],
              source: Optional[str], interval: Optional[str], generations: Optional[int],
              start: Optional[str] = None) -> Settings:
    s = load_settings(config)
    if provider:
        s.llm.provider = provider
    if model:
        s.llm.model = model
    if symbols:
        s.data.symbols = [x.strip() for x in symbols.split(",") if x.strip()]
    if source:
        s.data.source = source  # type: ignore[assignment]
    if interval:
        s.data.interval = interval
    if generations:
        s.swarm.generations = generations
    if start:
        s.data.start = start
    return s


def _load_or_exit(s: Settings, symbol: str):
    from .data import load_bars

    try:
        return load_bars(s.data, symbol)
    except Exception as e:  # noqa: BLE001 - friendly CLI error instead of a traceback
        console.print(f"[red]Could not load {symbol} from {s.data.source}: {e}[/]")
        raise typer.Exit(1) from None


def _pos(p: int) -> str:
    return {1: "[green]LONG[/]", -1: "[red]SHORT[/]", 0: "[dim]FLAT[/]"}.get(int(p), "FLAT")


def _winners_table(rows: list[dict], title: str) -> Table:
    t = Table(title=title, header_style="bold cyan", expand=False)
    for col in ("#", "Symbol", "Strategy", "Agent", "SR in", "SR out", "DSR", "CAGR", "MaxDD", "Trades",
                "Var", "Signal"):
        if col == "Strategy":
            t.add_column(col, no_wrap=True, overflow="ellipsis", max_width=32)
        elif col == "Agent":
            t.add_column(col, no_wrap=True, overflow="ellipsis", max_width=14)
        elif col == "Symbol":
            t.add_column(col, no_wrap=True, min_width=6)
        else:
            t.add_column(col, justify="right", no_wrap=True, min_width=len(col))
    for i, r in enumerate(rows, 1):
        res = r.get("result", r)
        t.add_row(str(i), res["symbol"], res["name"][:48], res["agent"], f"{res['in_sample']['sharpe']:.2f}",
                  f"[bold]{res['out_of_sample']['sharpe']:.2f}[/]", f"{res['dsr']:.2f}",
                  f"{res['full']['cagr']:+.1%}", f"{res['full']['max_drawdown']:.0%}", str(res['full']['trades']),
                  str(res.get("variants", 1)), _pos(res.get("next_position", 0)))
    return t


LEGEND = ("[dim]SR = Sharpe ratio | in = in-sample | out = out-of-sample (never seen during selection) | "
          "DSR = Deflated Sharpe | Var = correlated variants that also passed[/]")


def _print_cycle(cyc) -> None:
    winners = [w.to_dict() for w in cyc.distinct_winners]
    console.print()
    if winners:
        console.print(_winners_table(winners[:15], f"WINNERS: {len(cyc.winners)} of {cyc.tested} strategies survived "
                                                   f"every robustness test ({len(winners)} distinct "
                                                   f"idea{'s' if len(winners) != 1 else ''})"))
        console.print(LEGEND)
    else:
        console.print(Panel(f"0 of {cyc.tested} strategies survived the robustness gate this cycle.\n"
                            "That's the gate protecting you from overfit junk. Run more generations/symbols, "
                            "or plug in a stronger model.", border_style="yellow"))
    console.print(f"[dim]report: {cyc.report_md} | bot hand-off: {cyc.report_json} | {cyc.seconds:.1f}s[/]")


@app.command()
def demo(generations: int = typer.Option(4, help="evolution rounds"),
         seed: int = typer.Option(7, help="random seed (deterministic demo)"),
         workdir: Path = typer.Option(Path(".atsa-demo"), help="where the demo DB + reports go")):
    """60-second zero-setup demo: offline agents on a synthetic market with planted edges. No API key."""
    console.print(BANNER)
    s = _demo_settings(workdir, generations, seed, ["DEMO-A", "DEMO-B", "DEMO-C", "DEMO-D"])
    from .data import generate_bars
    from .pipeline import Pipeline

    pipe = Pipeline(s, console=console)
    cyc = asyncio.run(pipe.run_cycle())
    pipe.close()
    _print_cycle(cyc)
    planted = {sym: generate_bars(sym).meta["planted"] for sym in s.data.symbols}
    console.print("\n[bold]Hidden edges planted in the synthetic market[/] (did the swarm find them?):")
    for sym, p in planted.items():
        console.print(f"  {sym}: {p['calendar_effect']} drift {p['calendar_drift_bps']}bps, {p['overreaction_rebound']}")
    db = s.runtime.db_path
    console.print(f"\nExplore: [bold]atsa leaderboard --db {db}[/] | [bold]atsa dashboard --db {db}[/] | "
                  f"[bold]atsa signals --db {db}[/]")
    console.print("Go live: [bold]atsa run --provider gemini --source binance --symbols BTCUSDT,ETHUSDT --forever[/]")


@app.command()
def run(config: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML config (default ./atsa.yaml)"),
        provider: Optional[str] = typer.Option(None, help="gemini | openai | anthropic | openrouter | ollama | offline ..."),
        model: Optional[str] = typer.Option(None, help="model id, e.g. gemini-4-neon"),
        symbols: Optional[str] = typer.Option(None, help="comma separated, e.g. BTCUSDT,ETHUSDT or SPY,QQQ"),
        source: Optional[str] = typer.Option(None, help="synthetic | csv | yahoo | binance"),
        interval: Optional[str] = typer.Option(None, help="1d, 4h, 1h ..."),
        start: Optional[str] = typer.Option(None, help="history start date YYYY-MM-DD"),
        generations: Optional[int] = typer.Option(None, help="evolution rounds per cycle"),
        forever: bool = typer.Option(False, "--forever", help="run 24/7: hunt, sleep, refresh data, repeat"),
        every: Optional[float] = typer.Option(None, help="minutes between cycles in --forever mode"),
        parallel: Optional[int] = typer.Option(None, help="symbols hunted at the same time")):
    """Run the strategy hunt (one cycle, or 24/7 with --forever)."""
    s = _settings(config, provider, model, symbols, source, interval, generations, start)
    if every:
        s.runtime.interval_minutes = every
    if parallel:
        s.runtime.parallel_symbols = parallel
    from .pipeline import Pipeline

    console.print(BANNER)
    pipe = Pipeline(s, console=console)
    stop = {"flag": False}

    def _stop(*_):
        stop["flag"] = True
        console.print("\n[yellow]Stopping after this cycle...[/]")

    if forever:
        signal.signal(signal.SIGTERM, _stop)
    cycle = 0
    try:
        while True:
            cycle += 1
            cyc = asyncio.run(pipe.run_cycle())
            _print_cycle(cyc)
            if not forever or stop["flag"]:
                break
            wait = s.runtime.interval_minutes * 60
            console.print(f"[dim]cycle {cycle} done. Next hunt in {s.runtime.interval_minutes:g} min "
                          f"(Ctrl+C to stop)...[/]")
            end = time.time() + wait
            while time.time() < end and not stop["flag"]:
                time.sleep(min(5.0, end - time.time()))
            if stop["flag"]:
                break
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped.[/]")
    finally:
        pipe.close()


def _demo_settings(workdir: Path, generations: int, seed: int, symbols: list[str]) -> Settings:
    s = Settings()
    s.llm.provider = "offline"
    s.swarm.seed = seed
    s.swarm.generations = generations
    s.data.source = "synthetic"
    s.data.symbols = symbols
    s.runtime.db_path = str(workdir / "atsa.db")
    s.runtime.reports_dir = str(workdir / "reports")
    return s


@app.command()
def live(config: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML config (default ./atsa.yaml)"),
         provider: Optional[str] = typer.Option(None, help="gemini | openai | anthropic | openrouter | ollama | offline ..."),
         model: Optional[str] = typer.Option(None, help="model id, e.g. gemini-4-neon"),
         symbols: Optional[str] = typer.Option(None, help="comma separated, e.g. BTCUSDT,ETHUSDT,SOLUSDT"),
         source: Optional[str] = typer.Option(None, help="synthetic | csv | yahoo | binance"),
         interval: Optional[str] = typer.Option(None, help="1d, 4h, 1h ..."),
         generations: Optional[int] = typer.Option(None, help="evolution rounds per cycle"),
         parallel: int = typer.Option(3, help="symbols hunted at the same time"),
         forever: bool = typer.Option(False, "--forever", help="keep hunting 24/7"),
         demo_mode: bool = typer.Option(False, "--demo", help="no API key: offline agents on the synthetic market"),
         fps: float = typer.Option(12.0, help="screen refresh rate")):
    """Full-screen live command center: market tape, agent swarm, backtest feed, gate funnel, winners."""
    from .llm import get_provider
    from .tui import run_live

    if demo_mode:
        s = _demo_settings(Path(".atsa-demo"), generations or 3, 7,
                           ["DEMO-A", "DEMO-B", "DEMO-C", "DEMO-D", "DEMO-E", "DEMO-F"])
    else:
        s = _settings(config, provider, model, symbols, source, interval, generations)
    s.runtime.parallel_symbols = max(1, parallel)
    prov = get_provider(s.llm, seed=s.swarm.seed)
    if demo_mode:
        prov.latency = (0.25, 0.9)  # show the agents "thinking" like a real LLM would
    cycles = asyncio.run(run_live(s, prov, fps=fps, forever=forever))
    for cyc in cycles[-1:]:
        _print_cycle(cyc)


@app.command()
def research(symbol: str = typer.Argument(..., help="symbol to analyse"),
             config: Optional[Path] = typer.Option(None, "--config", "-c"),
             source: Optional[str] = typer.Option(None), interval: Optional[str] = typer.Option(None),
             as_json: bool = typer.Option(False, "--json", help="print the raw brief JSON")):
    """Live market research brief: regime, fair value, mispricing, seasonality (what the agents see)."""
    from .research import analyze, brief_markdown

    s = _settings(config, None, None, None, source, interval, None)
    bars = _load_or_exit(s, symbol)
    brief = analyze(bars, s.data.periods_per_year or infer_periods_per_year(s.data.interval, s.data.source))
    if as_json:
        print(json.dumps(brief, indent=2))
        return
    console.print(Panel(brief_markdown(brief), title=f"Research brief: {symbol}", border_style="cyan"))
    console.print_json(json.dumps(brief))


@app.command()
def backtest(strategy_file: Path = typer.Argument(..., help="JSON file with one strategy (or a list)"),
             symbol: Optional[str] = typer.Option(None, help="symbol (default: first configured)"),
             config: Optional[Path] = typer.Option(None, "--config", "-c"),
             source: Optional[str] = typer.Option(None), interval: Optional[str] = typer.Option(None)):
    """Backtest your own strategy JSON through the same robustness gate."""
    from .backtest import apply_gate, evaluate_many
    from .dsl import Strategy

    s = _settings(config, None, None, None, source, interval, None)
    raw = json.loads(strategy_file.read_text())
    items = raw.get("strategies", raw) if isinstance(raw, dict) else raw
    items = items if isinstance(items, list) else [items]
    strategies = [Strategy.model_validate(x) for x in items]
    sym = symbol or s.data.symbols[0]
    bars = _load_or_exit(s, sym)
    ppy = s.data.periods_per_year or infer_periods_per_year(s.data.interval, s.data.source)
    evals = evaluate_many(bars, strategies, s.backtest, ppy, workers=1)
    apply_gate(evals, s.gate)
    for ev in evals:
        if ev.error:
            console.print(f"[red]{ev.strategy.name}: {ev.error}[/]")
            continue
        t = Table(title=f"{ev.strategy.name} on {sym}", header_style="bold cyan")
        t.add_column("metric")
        for col in ("full", "in-sample", "out-of-sample"):
            t.add_column(col, justify="right")
        for k in ("sharpe", "sortino", "cagr", "max_drawdown", "calmar", "trades", "win_rate", "profit_factor",
                  "exposure"):
            vals = [ev.full[k], ev.in_sample[k], ev.out_of_sample[k]]
            fmt = (lambda v: f"{v:.1%}") if k in ("cagr", "max_drawdown", "win_rate", "exposure") else \
                (lambda v: f"{v:.0f}") if k == "trades" else (lambda v: f"{v:.2f}")
            t.add_row(k, *[fmt(v) for v in vals])
        console.print(t)
        verdict = "[bold green]PASSED[/]" if ev.passed else "[bold red]REJECTED[/]"
        console.print(f"walk-forward Sharpes {ev.wf_sharpes} | 2x-cost Sharpe {ev.stress_sharpe:.2f} | "
                      f"deflated Sharpe {ev.dsr:.2f} | buy&hold Sharpe {ev.benchmark_sharpe:.2f} | signal now "
                      f"{_pos(ev.next_position)} -> {verdict}")
        for r in ev.reasons:
            console.print(f"  [red]- {r}[/]")


@app.command()
def leaderboard(top: int = typer.Option(20, help="rows"), symbol: Optional[str] = typer.Option(None),
                all_: bool = typer.Option(False, "--all", help="include rejected strategies"),
                config: Optional[Path] = typer.Option(None, "--config", "-c"),
                db: Optional[str] = typer.Option(None, help="path to atsa.db (default from config)")):
    """Best strategies found so far (persisted across runs)."""
    from .store import Store

    s = load_settings(config)
    st = Store(db or s.runtime.db_path)
    rows = st.leaderboard(top, symbol, winners_only=not all_)
    c = st.counts()
    console.print(_winners_table(rows, f"Leaderboard - {c['winners']} winners / {c['tested']} strategies tested"))
    console.print(LEGEND)
    st.close()


@app.command()
def signals(top: int = typer.Option(10), config: Optional[Path] = typer.Option(None, "--config", "-c"),
            as_json: bool = typer.Option(False, "--json"),
            db: Optional[str] = typer.Option(None, help="path to atsa.db (default from config)")):
    """What should each winning strategy hold right now? (re-computed on the latest data)."""
    from .backtest import evaluate_strategy
    from .data import load_bars
    from .dsl import Strategy
    from .store import Store

    s = load_settings(config)
    st = Store(db or s.runtime.db_path)
    rows = st.leaderboard(top)
    st.close()
    ppy = s.data.periods_per_year or infer_periods_per_year(s.data.interval, s.data.source)
    out = []
    cache: dict = {}
    for r in rows:
        sym = r["symbol"]
        if sym not in cache:
            cache[sym] = load_bars(s.data, sym, max_age_minutes=s.runtime.interval_minutes)
        ev = evaluate_strategy(cache[sym], Strategy.model_validate(r["spec"]), s.backtest, ppy)
        out.append({"id": r["id"], "symbol": sym, "name": r["name"], "as_of": cache[sym].end_date,
                    "holding": {1: "LONG", -1: "SHORT", 0: "FLAT"}[ev.current_position],
                    "next_open": {1: "LONG", -1: "SHORT", 0: "FLAT"}[ev.next_position],
                    "last_price": float(cache[sym].close[-1])})
    if as_json:
        print(json.dumps(out, indent=2))
        return
    t = Table(title="Live signals from winning strategies", header_style="bold cyan")
    for col in ("Symbol", "Strategy", "As of", "Holding", "Target next open", "Last price"):
        t.add_column(col)
    for o in out:
        t.add_row(o["symbol"], o["name"][:50], o["as_of"], o["holding"], f"[bold]{o['next_open']}[/]",
                  f"{o['last_price']:.4g}")
    console.print(t)


@app.command()
def export(path: Path = typer.Argument(Path("winners.json")), top: int = typer.Option(50),
           config: Optional[Path] = typer.Option(None, "--config", "-c"),
           db: Optional[str] = typer.Option(None, help="path to atsa.db (default from config)")):
    """Export winning strategies as JSON for your trading bot."""
    from .report import winners_payload
    from .store import Store

    s = load_settings(config)
    st = Store(db or s.runtime.db_path)
    rows = [r["result"] for r in st.leaderboard(top)]
    st.close()
    payload = winners_payload(rows, {"source": s.data.source, "interval": s.data.interval})
    path.write_text(json.dumps(payload, indent=2))
    console.print(f"[green]Exported {len(rows)} strategies -> {path}[/]")


@app.command()
def dashboard(host: str = typer.Option("127.0.0.1"), port: int = typer.Option(8000),
              config: Optional[Path] = typer.Option(None, "--config", "-c"),
              db: Optional[str] = typer.Option(None, help="path to atsa.db (default from config)")):
    """Live web dashboard: leaderboard, equity curves, run history."""
    try:
        import uvicorn
    except ImportError:
        console.print("[red]Install the dashboard extra: pip install 'ai-trading-strategy-agents[dashboard]'[/]")
        raise typer.Exit(1) from None
    from .dashboard import create_app

    s = load_settings(config)
    console.print(f"[bold magenta]Dashboard[/] -> http://{host}:{port}")
    uvicorn.run(create_app(db or s.runtime.db_path), host=host, port=port, log_level="warning")


@app.command()
def bench(strategies: int = typer.Option(1000, help="how many strategies to backtest"),
          bars: int = typer.Option(2520, help="bars per backtest (2520 = 10y daily)"),
          workers: int = typer.Option(0, help="processes (0 = all cores)")):
    """Measure raw backtest throughput on this machine."""
    import random

    import numpy as np

    from .backtest import evaluate_many
    from .backtest.engine import simulate
    from .config import BacktestConfig
    from .data import generate_bars
    from .dsl import Evaluator, Strategy
    from .llm.offline import _mean_reversion, _momentum, _regime, _vol_breakout

    rng = random.Random(1)
    fns = [_momentum, _mean_reversion, _vol_breakout, _regime]
    specs, seen = [], set()
    while len(specs) < strategies:
        st = Strategy.model_validate(rng.choice(fns)(rng, True))
        st = st.model_copy(update={"stop_loss": round(rng.uniform(0.02, 0.15), 3)})
        if st.id not in seen:
            seen.add(st.id)
            specs.append(st)
    b = generate_bars("BENCH", n=bars)
    cfg = BacktestConfig()
    console.print(f"[bold]Benchmark[/]: {strategies} strategies x {bars} bars (each = full + 2x-cost re-run)")
    ev = Evaluator(b)
    el = ev.condition("rsi(close,14) < 30")
    xl = ev.condition("rsi(close,14) > 60")
    z = np.zeros(bars, dtype=bool)
    simulate(b.open, b.high, b.low, b.close, el, xl, z, z, True, False, 0.05, 0.0, 0, 0.001)
    t0 = time.perf_counter()
    reps = 2000
    for _ in range(reps):
        simulate(b.open, b.high, b.low, b.close, el, xl, z, z, True, False, 0.05, 0.0, 0, 0.001)
    core_us = (time.perf_counter() - t0) / reps * 1e6
    evaluate_many(b, specs[:20], cfg, 252, workers=1)  # warm JIT + cache
    t0 = time.perf_counter()
    evaluate_many(b, specs, cfg, 252, workers=1)
    single = time.perf_counter() - t0
    from .backtest.parallel import resolve_workers

    w = resolve_workers(workers)
    t0 = time.perf_counter()
    evaluate_many(b, specs, cfg, 252, workers=w)
    multi = time.perf_counter() - t0
    t = Table(header_style="bold cyan")
    t.add_column("stage")
    t.add_column("result", justify="right")
    t.add_row("core simulation kernel (1 backtest)", f"{core_us:.0f} us")
    t.add_row("full scorecard, 1 core (signals+IS/OOS/WF/stress)", f"{strategies / single:,.0f} strategies/s")
    t.add_row(f"full scorecard, {w} processes", f"{strategies / multi:,.0f} strategies/s")
    t.add_row("strategies per minute (best)", f"{60 * strategies / min(single, multi):,.0f}")
    console.print(t)


@app.command()
def providers():
    """List supported LLM providers and whether their API key is set."""
    from .config import load_dotenv
    from .llm import ALL_PROVIDERS

    load_dotenv()
    t = Table(title="LLM providers", header_style="bold cyan")
    for col in ("provider", "default model", "key env", "key set"):
        t.add_column(col)
    for name, cls in sorted(ALL_PROVIDERS.items()):
        envs = getattr(cls, "key_envs", ())
        needs = getattr(cls, "needs_key", True)
        has = any(os.environ.get(e) for e in envs)
        t.add_row(name, getattr(cls, "default_model", None) or "[dim](set --model)[/]", ", ".join(envs) or "-",
                  "[green]yes[/]" if has else ("[dim]not needed[/]" if not needs else "[red]no[/]"))
    console.print(t)


@app.command()
def init(force: bool = typer.Option(False, help="overwrite existing files")):
    """Create atsa.yaml and .env in the current folder."""
    from .templates import ENV_TEMPLATE, YAML_TEMPLATE

    for name, content in (("atsa.yaml", YAML_TEMPLATE), (".env", ENV_TEMPLATE)):
        p = Path(name)
        if p.exists() and not force:
            console.print(f"[yellow]{name} exists (use --force to overwrite)[/]")
            continue
        p.write_text(content)
        console.print(f"[green]created {name}[/]")
    console.print("Add your API key to .env, then: [bold]atsa run[/]")


@app.command()
def version():
    """Print version."""
    console.print(f"ai-trading-strategy-agents {__version__}")


if __name__ == "__main__":  # pragma: no cover
    app()
