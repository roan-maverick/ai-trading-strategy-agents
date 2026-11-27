import asyncio
from pathlib import Path

from rich.console import Console

from atsa.cli import _demo_settings
from atsa.llm import get_provider
from atsa.tui import LiveState, Tape, run_live, sparkline


def test_sparkline_width():
    assert len(sparkline([1, 2, 3, 2, 5], 10)) == 10
    assert len(sparkline([1], 6)) == 6


def test_live_dashboard_records_frames(tmp_path):
    s = _demo_settings(tmp_path / "state", 1, 7, ["DEMO-A", "DEMO-C"])
    s.runtime.parallel_symbols = 2
    s.runtime.workers = 1
    provider = get_provider(s.llm, seed=7)
    cycles = asyncio.run(run_live(s, provider, fps=6, record_dir=str(tmp_path / "frames"), size=(150, 36),
                                  linger=0.2))
    assert cycles and cycles[0].tested > 20
    frames = sorted(Path(tmp_path / "frames").glob("frame_*.svg"))
    assert frames and (tmp_path / "frames" / "timings.txt").exists()


def test_render_with_events_only():
    s = _demo_settings(Path("/tmp/unused"), 1, 7, ["DEMO-A"])
    state = LiveState(s, "test:model")
    state.on_event("cycle_start", run_id=1, model="m", symbols=["DEMO-A"])
    state.on_event("agent_start", symbol="DEMO-A", agent="momentum_hunter", generation=1)
    con = Console(width=150, height=36, record=True, force_terminal=True)
    con.print(state.render(Tape(["DEMO-A"]), 36))
    text = con.export_text()
    assert "AGENT SWARM" in text and "thinking" in text and "ROBUSTNESS GATE" in text
