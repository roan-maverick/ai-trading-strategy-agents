"""`atsa live`: full-screen terminal command center for the strategy hunt.

  * market tape  - live prices (Binance) or a fast market replay (other sources) with sparklines
  * agent swarm  - every agent's state in real time: thinking / ready, ideas, latest idea
  * engine feed  - every backtest as it lands: PASS / FAIL, Sharpe in/out, Deflated Sharpe, why it died
  * gate funnel  - how many strategies survive each robustness test
  * leaderboard  - best distinct winners so far
"""

from __future__ import annotations

import asyncio
import io
import time
from collections import OrderedDict, deque
from datetime import datetime
from pathlib import Path

import httpx
from rich import box
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .config import Settings
from .data import load_bars
from .llm import LLMProvider, get_provider
from .pipeline import Pipeline

SPARK = "▁▂▃▄▅▆▇█"
SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
FUNNEL = [
    ("trades", "20+ trades"),
    ("sharpe_is", "Sharpe in-sample"),
    ("sharpe_oos", "Sharpe out-of-sample"),
    ("drawdown", "Drawdown cap"),
    ("walk_forward", "Walk-forward stable"),
    ("cost_stress", "Survives 2x costs"),
    ("deflated_sharpe", "Deflated Sharpe"),
]
ACCENT = "#b48cff"
UP = "#3fd68f"
DOWN = "#ff5f6d"
WARN = "#ffcc4d"
DIM = "#7d8590"
CYAN = "#58c4ff"


def title(name: str, note: str = "") -> Text:
    return Text.assemble((f" {name} ", f"bold {CYAN}"), (f" {note} " if note else "", DIM))


def sparkline(values: list[float], width: int) -> Text:
    vals = list(values)[-width:]
    if len(vals) < 2:
        return Text(" " * width)
    lo, hi = min(vals), max(vals)
    span = hi - lo or 1.0
    t = Text()
    for i, v in enumerate(vals):
        ch = SPARK[min(7, int((v - lo) / span * 7.999))]
        up = i == 0 or v >= vals[i - 1]
        t.append(ch, style=UP if up else DOWN)
    return Text(" " * (width - len(vals))) + t


class Tape:
    """Price tape: polls Binance live tickers, otherwise replays historical bars at speed."""

    def __init__(self, symbols: list[str], window: int = 60):
        self.symbols = symbols
        self.prices: dict[str, deque] = {s: deque(maxlen=window) for s in symbols}
        self.last_move: dict[str, int] = {s: 0 for s in symbols}
        self.mode = "replay"

    def push(self, sym: str, price: float) -> None:
        q = self.prices[sym]
        if q:
            self.last_move[sym] = 1 if price > q[-1] else -1 if price < q[-1] else 0
        q.append(price)

    async def run(self, settings: Settings, tick_hz: float, stop: asyncio.Event) -> None:
        cfg = settings.data
        if cfg.source == "binance":
            if await self._binance(stop):
                return
        series = {}
        for s in self.symbols:
            try:
                bars = await asyncio.to_thread(load_bars, cfg, s)
                series[s] = bars.close[-1500:]
            except Exception:  # noqa: BLE001 - tape is cosmetic, never fatal
                continue
        self.mode = f"replay {cfg.interval} bars"
        i = 0
        while not stop.is_set() and series:
            for s, closes in series.items():
                self.push(s, float(closes[i % len(closes)]))
            i += 1
            await asyncio.sleep(1.0 / tick_hz)

    async def _binance(self, stop: asyncio.Event) -> bool:
        pairs = [s.replace("/", "").replace("-", "").upper() for s in self.symbols]
        async with httpx.AsyncClient(timeout=5.0) as client:
            failures = 0
            while not stop.is_set():
                try:
                    r = await client.get("https://data-api.binance.vision/api/v3/ticker/price",
                                         params={"symbols": "[" + ",".join(f'"{p}"' for p in pairs) + "]"})
                    r.raise_for_status()
                    px = {d["symbol"]: float(d["price"]) for d in r.json()}
                    for s, p in zip(self.symbols, pairs):
                        if p in px:
                            self.push(s, px[p])
                    self.mode = "LIVE binance"
                    failures = 0
                except Exception:  # noqa: BLE001
                    failures += 1
                    if failures >= 3 and self.mode != "LIVE binance":
                        return False  # unreachable -> fall back to replay
                await asyncio.sleep(1.0)
        return True


class LiveState:
    def __init__(self, settings: Settings, model: str):
        self.s = settings
        self.model = model
        self.t0 = time.perf_counter()
        self.run_id = 0
        self.cycle_done = False
        agents = list(settings.swarm.agents) + ["evolver"]
        self.agents: OrderedDict[str, dict] = OrderedDict(
            (a, {"active": [], "ideas": 0, "valid": 0, "invalid": 0, "last": "", "errors": 0}) for a in agents)
        self.symbols: OrderedDict[str, dict] = OrderedDict(
            (s, {"status": "queued", "gen": 0, "brief": None}) for s in settings.data.symbols)
        self.all_evals: list = []
        self.pending: deque = deque()
        self.feed: deque = deque(maxlen=60)
        self.bt_count = 0
        self.bt_seconds = 0.0
        self.briefs: list[tuple[str, dict]] = []

    # pipeline events --------------------------------------------------------------------------
    def on_event(self, event: str, **d) -> None:
        if event == "cycle_start":
            self.run_id = d["run_id"]
            self.cycle_done = False
        elif event == "symbol_start":
            st = self.symbols.setdefault(d["symbol"], {"status": "", "gen": 0, "brief": None})
            st.update(status="researching", brief=d["brief"])
            self.briefs.append((d["symbol"], d["brief"]))
        elif event == "agent_start":
            a = self.agents.setdefault(d["agent"], {"active": [], "ideas": 0, "valid": 0, "invalid": 0,
                                                    "last": "", "errors": 0})
            a["active"].append(d["symbol"])
            self.symbols[d["symbol"]].update(status="hunting", gen=d["generation"])
        elif event == "agent_done":
            a = self.agents[d["agent"]]
            if d["symbol"] in a["active"]:
                a["active"].remove(d["symbol"])
            rep = d["report"]
            a["ideas"] += rep.proposed
            a["valid"] += len(rep.valid)
            a["invalid"] += len(rep.invalid)
            if rep.error:
                a["errors"] += 1
            if rep.valid:
                a["last"] = rep.valid[-1].name
        elif event == "evals":
            for ev in d["evals"]:
                self.all_evals.append(ev)
                self.pending.append(ev)
            if d.get("seconds"):
                self.bt_count += len(d["evals"])
                self.bt_seconds += d["seconds"]
        elif event == "symbol_done":
            res = d["result"]
            self.symbols[d["symbol"]].update(status="done", winners=len(res.winners))
        elif event == "symbol_error":
            self.symbols[d["symbol"]].update(status="error")
        elif event == "cycle_done":
            self.cycle_done = True

    # rendering --------------------------------------------------------------------------------
    def _drain(self) -> None:
        k = max(2, len(self.pending) // 5)
        now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        for _ in range(min(k, len(self.pending))):
            self.feed.append((now, self.pending.popleft()))

    def header(self) -> Panel:
        el = time.perf_counter() - self.t0
        in_flight = sum(len(a["active"]) for a in self.agents.values())
        valid = [e for e in self.all_evals if not e.error]
        winners = sum(1 for e in valid if e.passed)
        speed = self.bt_count / self.bt_seconds if self.bt_seconds > 0 else 0.0
        dot = Text("●", style=f"bold {UP}") if int(el * 2) % 2 == 0 or self.cycle_done else Text("●", style=DIM)
        t = Table.grid(expand=True)
        for _ in range(7):
            t.add_column(justify="center")
        t.add_row(
            Text.assemble(dot, " ", ("AI TRADING STRATEGY AGENTS", f"bold {ACCENT}")),
            Text.assemble(("model ", DIM), (self.model, "bold white")),
            Text.assemble(("cycle ", DIM), (f"#{self.run_id}", "bold white"), ("  t+", DIM), (f"{el:5.1f}s", "white")),
            Text.assemble(("agents in flight ", DIM), (f"{in_flight:2d}", f"bold {WARN}")),
            Text.assemble(("tested ", DIM), (f"{len(self.all_evals):,}", "bold white")),
            Text.assemble(("engine ", DIM), (f"{speed:,.0f}", f"bold {CYAN}"), (" strat/s", DIM)),
            Text.assemble(("winners ", DIM), (f"{winners}", f"bold {UP}")),
        )
        return Panel(t, box=box.HEAVY, border_style=ACCENT, padding=(0, 1))

    def tape_panel(self, tape: Tape) -> Panel:
        t = Table(box=None, expand=True, padding=(0, 1), header_style=DIM)
        t.add_column("SYMBOL", style="bold white", no_wrap=True, min_width=8)
        t.add_column("LAST", justify="right", no_wrap=True, min_width=9)
        t.add_column("CHG", justify="right", no_wrap=True, min_width=7)
        t.add_column("TREND", no_wrap=True, width=18)
        t.add_column("HUNT", no_wrap=True, min_width=10)
        for s in tape.symbols:
            q = tape.prices.get(s) or []
            last = q[-1] if q else float("nan")
            chg = (q[-1] / q[0] - 1.0) if len(q) > 1 and q[0] else 0.0
            mv = tape.last_move.get(s, 0)
            col = UP if mv > 0 else DOWN if mv < 0 else "white"
            st = self.symbols.get(s, {})
            status = st.get("status", "")
            if status == "hunting":
                hunt = Text(f"gen {st.get('gen', 0)}/{self.s.swarm.generations}", style=WARN)
            elif status == "done":
                w = st.get("winners", 0)
                hunt = Text(f"done {w:>2} win", style=UP if w else DIM)
            elif status == "researching":
                hunt = Text("research", style=CYAN)
            else:
                hunt = Text(status or "-", style=DIM)
            t.add_row(s, Text(f"{last:,.2f}", style=col), Text(f"{chg:+.2%}", style=UP if chg >= 0 else DOWN),
                      sparkline(list(q), 18), hunt)
        return Panel(t, title=title("MARKET", tape.mode), title_align="left",
                     border_style="#30363d", box=box.ROUNDED)

    def agents_panel(self) -> Panel:
        t = Table(box=None, expand=True, padding=(0, 1), header_style=DIM)
        t.add_column("AGENT", no_wrap=True, style="white", min_width=22)
        t.add_column("STATUS", no_wrap=True, min_width=14)
        t.add_column("IDEAS", justify="right", no_wrap=True, min_width=5)
        t.add_column("LATEST IDEA", no_wrap=True, overflow="ellipsis", ratio=1)
        frame = int(time.perf_counter() * 12)
        for name, a in self.agents.items():
            if a["active"]:
                status = Text(f"{SPIN[frame % len(SPIN)]} thinking x{len(a['active'])}", style=f"bold {WARN}")
            elif a["ideas"]:
                status = Text("● ready", style=UP)
            else:
                status = Text("○ idle", style=DIM)
            t.add_row(name.replace("_", " "), status, f"{a['valid']}", Text(a["last"] or "-", style=DIM))
        return Panel(t, title=title("AGENT SWARM", "parallel LLM research"), title_align="left",
                     border_style="#30363d", box=box.ROUNDED)

    def feed_panel(self, rows: int) -> Panel:
        self._drain()
        lines = []
        for ts, ev in list(self.feed)[-rows:]:
            line = Text(no_wrap=True, overflow="ellipsis")
            line.append(ts[3:] + " ", style=DIM)
            line.append(f"{ev.symbol:<7}", style="white")
            if ev.error:
                line.append(" ERR  ", style=f"bold {DOWN}")
                line.append(ev.error[:60], style=DIM)
            else:
                line.append(" PASS " if ev.passed else " FAIL ",
                            style=f"bold black on {UP}" if ev.passed else f"bold {DOWN}")
                line.append(f" SR {ev.in_sample.get('sharpe', 0):5.2f}/{ev.out_of_sample.get('sharpe', 0):5.2f}",
                            style="white")
                line.append(f"  DSR {ev.dsr:4.2f}  ", style=CYAN)
                line.append(ev.strategy.name[:44], style="white" if ev.passed else DIM)
                if not ev.passed and ev.reasons:
                    line.append("  x " + ev.reasons[0].split("(")[0].strip()[:34], style=DIM)
            lines.append(line)
        while len(lines) < rows:
            lines.insert(0, Text(""))
        return Panel(Group(*lines), title=title("BACKTEST ENGINE", "JIT | next-open fills | fees + slippage"),
                     title_align="left", border_style="#30363d", box=box.ROUNDED)

    def funnel_panel(self) -> Panel:
        valid = [e for e in self.all_evals if not e.error and e.checks]
        n = len(valid)
        bar = 26
        t = Table(box=None, expand=True, padding=(0, 1), show_header=False)
        t.add_column(no_wrap=True, style="white", min_width=21)
        t.add_column(no_wrap=True, width=bar)
        t.add_column(justify="right", no_wrap=True, min_width=6)
        t.add_row("Strategies tested", Text("█" * bar, style=CYAN), f"{n:,}")
        alive = valid
        for key, label in FUNNEL:
            alive = [e for e in alive if e.checks.get(key)]
            frac = len(alive) / n if n else 0.0
            filled = max(1 if alive else 0, int(round(frac * bar)))
            col = UP if key == "deflated_sharpe" else CYAN
            t.add_row(label, Text("█" * filled, style=col) + Text("░" * (bar - filled), style="#30363d"),
                      Text(f"{len(alive):,}", style=f"bold {UP}" if key == "deflated_sharpe" else "white"))
        return Panel(t, title=title("ROBUSTNESS GATE", "all 7 tests must pass"), title_align="left",
                     border_style="#30363d", box=box.ROUNDED)

    def leaders_panel(self) -> Panel:
        best: dict[tuple, object] = {}
        for e in self.all_evals:
            if e.error or not e.passed:
                continue
            k = (e.symbol, e.cluster)
            if k not in best or e.score > best[k].score:
                best[k] = e
        top = sorted(best.values(), key=lambda e: e.score, reverse=True)[:6]
        t = Table(box=None, expand=True, padding=(0, 1), header_style=DIM)
        t.add_column("#", no_wrap=True, style=DIM)
        t.add_column("SYMBOL", no_wrap=True)
        t.add_column("STRATEGY", no_wrap=True, overflow="ellipsis", ratio=1)
        t.add_column("SR IN", justify="right", no_wrap=True)
        t.add_column("SR OUT", justify="right", no_wrap=True)
        t.add_column("DSR", justify="right", no_wrap=True)
        t.add_column("CAGR", justify="right", no_wrap=True)
        t.add_column("MAX DD", justify="right", no_wrap=True)
        for i, e in enumerate(top, 1):
            t.add_row(str(i), e.symbol, e.strategy.name, f"{e.in_sample['sharpe']:.2f}",
                      Text(f"{e.out_of_sample['sharpe']:.2f}", style=f"bold {UP}"),
                      f"{e.dsr:.2f}", f"{e.full['cagr']:+.1%}", f"{e.full['max_drawdown']:.1%}")
        if not top:
            t.add_row("", Text("hunting...", style=DIM), "", "", "", "", "", "")
        return Panel(t, title=title("WINNERS", "best variant per idea, live"), title_align="left",
                     border_style="#30363d", box=box.ROUNDED)

    def footer(self) -> Panel:
        if self.briefs:
            idx = int((time.perf_counter() - self.t0) / 2.5) % len(self.briefs)
            sym, b = self.briefs[idx]
            es = (b.get("event_study") or {}).get("after_big_down_bar") or {}
            fv = b["fair_value"]
            txt = Text.assemble(
                ("RESEARCH ", f"bold {ACCENT}"), (f"{sym}  ", "bold white"),
                (f"{b['trend']['state']}", "white"), ("  |  vol ", DIM), (b["volatility"]["regime"], "white"),
                ("  |  Hurst ", DIM), (f"{b['market_character']['hurst']}", "white"),
                ("  |  vs fair value ", DIM),
                (f"{(fv.get('gap_vs_composite') or 0):+.1%}", "white"),
                ("  |  mispricing z ", DIM), (f"{fv['mispricing_z_60']}", "white"),
                ("  |  after crash bars ", DIM),
                (f"{es.get('mean_fwd_return_bps', 'n/a')} bps (t={es.get('t_stat', 'n/a')})", "white"),
            )
        else:
            txt = Text("RESEARCH  loading market data...", style=DIM)
        return Panel(txt, box=box.ROUNDED, border_style="#30363d", padding=(0, 1))

    def render(self, tape: Tape, height: int) -> Layout:
        root = Layout()
        root.split_column(Layout(self.header(), size=3), Layout(name="body"), Layout(self.footer(), size=3))
        body = root["body"]
        body.split_row(Layout(name="left", ratio=5), Layout(name="right", ratio=7))
        body_h = max(12, height - 6)
        tape_h = len(tape.symbols) + 3
        agents_h = len(self.agents) + 3
        body["left"].split_column(Layout(self.tape_panel(tape), size=tape_h),
                                  Layout(self.agents_panel(), size=agents_h), Layout(self.funnel_panel()))
        winners_h = 10
        feed_h = max(6, body_h - winners_h)
        body["right"].split_column(Layout(self.feed_panel(feed_h - 2), size=feed_h),
                                   Layout(self.leaders_panel(), size=winners_h))
        return root


async def run_live(settings: Settings, provider: LLMProvider | None = None, fps: float = 12.0, tick_hz: float = 25.0,
                   forever: bool = False, record_dir: str | None = None, record_seconds: float | None = None,
                   size: tuple[int, int] = (150, 42), linger: float = 1.5) -> list:
    """Run hunt cycle(s) behind the live dashboard. Returns the CycleResults.

    record_dir: write one SVG per frame (+ timings.txt) instead of drawing to the terminal (for videos).
    """
    provider = provider or get_provider(settings.llm, seed=settings.swarm.seed)
    state = LiveState(settings, provider.describe())
    tape = Tape(list(settings.data.symbols))
    pipe = Pipeline(settings, provider=provider, console=Console(file=io.StringIO()), on_event=state.on_event)
    stop = asyncio.Event()
    cycles: list = []

    async def hunt() -> None:
        try:
            while True:
                cycles.append(await pipe.run_cycle())
                if not forever:
                    break
                end = time.monotonic() + settings.runtime.interval_minutes * 60
                while time.monotonic() < end and not stop.is_set():
                    await asyncio.sleep(0.5)
                if stop.is_set():
                    break
        finally:
            await asyncio.sleep(linger)
            stop.set()

    tape_task = asyncio.create_task(tape.run(settings, tick_hz, stop))
    hunt_task = asyncio.create_task(hunt())
    try:
        if record_dir:
            out = Path(record_dir)
            out.mkdir(parents=True, exist_ok=True)
            w, h = size
            con = Console(record=True, width=w, height=h, force_terminal=True, color_system="truecolor",
                          file=io.StringIO(), legacy_windows=False)
            stamps = []
            frame = 0
            start = time.perf_counter()
            while not stop.is_set():
                if record_seconds and time.perf_counter() - start > record_seconds:
                    break
                con.print(state.render(tape, h))
                con.save_svg(str(out / f"frame_{frame:05d}.svg"), title="atsa live", clear=True)
                stamps.append(time.perf_counter() - start)
                frame += 1
                await asyncio.sleep(max(0.0, 1.0 / fps - 0.005))
            (out / "timings.txt").write_text("\n".join(f"{t:.4f}" for t in stamps))
        else:
            console = Console()
            with Live(console=console, screen=True, auto_refresh=False, transient=False) as live:
                while not stop.is_set():
                    live.update(state.render(tape, console.size.height), refresh=True)
                    await asyncio.sleep(1.0 / fps)
    finally:
        stop.set()
        hunt_task.cancel() if not hunt_task.done() else None
        tape_task.cancel()
        for task in (hunt_task, tape_task):
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        pipe.close()
    return cycles
