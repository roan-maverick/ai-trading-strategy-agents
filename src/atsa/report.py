"""Human report (Markdown) + machine hand-off file (JSON) for trading bots."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .research.analyst import brief_markdown

DISCLAIMER = ("_Research output, not financial advice. Backtests are hypothetical and past performance does not "
              "guarantee future results. Paper-trade before risking capital._")


def _pct(x: float) -> str:
    return f"{x:+.1%}"


def _pos(p: int) -> str:
    return {1: "LONG", -1: "SHORT", 0: "FLAT"}.get(int(p), "FLAT")


def winners_payload(winners: list[dict], settings_summary: dict) -> dict:
    """The bot hand-off contract: everything needed to run a strategy, nothing more."""
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "engine": "ai-trading-strategy-agents",
        "settings": settings_summary,
        "strategies": [
            {
                "id": w["id"], "symbol": w["symbol"], "name": w["name"], "agent": w["agent"],
                "rules": {k: w["spec"].get(k) for k in ("entry_long", "exit_long", "entry_short", "exit_short",
                                                         "stop_loss", "take_profit", "max_hold_bars")},
                "thesis": w["spec"].get("thesis", ""),
                "signal_now": _pos(w.get("next_position", 0)),
                "metrics": {
                    "sharpe_in_sample": round(w["in_sample"]["sharpe"], 3),
                    "sharpe_out_of_sample": round(w["out_of_sample"]["sharpe"], 3),
                    "deflated_sharpe": round(w["dsr"], 3),
                    "cagr": round(w["full"]["cagr"], 4), "max_drawdown": round(w["full"]["max_drawdown"], 4),
                    "trades": w["full"]["trades"], "win_rate": round(w["full"]["win_rate"], 3),
                    "walk_forward_positive": round(w["wf_positive"], 3),
                },
                "score": round(w["score"], 4),
                "variants": w.get("variants", 1),
            }
            for w in winners
        ],
    }


def write_reports(reports_dir: str, run_id: int, header: dict, briefs: list[dict], winners: list[dict],
                  near_misses: list[dict], stats: dict) -> tuple[Path, Path]:
    out = Path(reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    md = [
        f"# Strategy Hunt #{run_id}",
        "",
        f"**Model:** `{header['model']}` | **Data:** {header['source']} {header['interval']} | "
        f"**Tested:** {stats['tested']} strategies | **Passed the gate:** {stats.get('passed', len(winners))} "
        f"({len(winners)} distinct ideas) | **Time:** {stats['seconds']:.1f}s",
        "",
        "## Live market research",
        *[brief_markdown(b) + "\n" for b in briefs],
        "## Winners (passed every robustness test)",
    ]
    if winners:
        md += ["", "One row per independent idea (best variant shown; *variants* = how many correlated versions also passed).",
               "", "| # | Symbol | Strategy | Agent | Sharpe IS | Sharpe OOS | DSR | CAGR | Max DD | Trades | Variants "
               "| Signal now |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for i, w in enumerate(winners, 1):
            md.append(f"| {i} | {w['symbol']} | {w['name']} | {w['agent']} | {w['in_sample']['sharpe']:.2f} | "
                      f"{w['out_of_sample']['sharpe']:.2f} | {w['dsr']:.2f} | {_pct(w['full']['cagr'])} | "
                      f"{w['full']['max_drawdown']:.1%} | {w['full']['trades']} | {w.get('variants', 1)} | "
                      f"{_pos(w.get('next_position', 0))} |")
        md.append("")
        for w in winners[:10]:
            s = w["spec"]
            md += [f"### {w['name']} ({w['symbol']})", f"> {s.get('thesis', '')}", "", "```"]
            for k in ("entry_long", "exit_long", "entry_short", "exit_short"):
                if s.get(k):
                    md.append(f"{k:12s}: {s[k]}")
            for k in ("stop_loss", "take_profit", "max_hold_bars"):
                if s.get(k):
                    md.append(f"{k:12s}: {s[k]}")
            md += ["```", ""]
    else:
        md += ["", "No strategy survived the gate this cycle. That is the gate doing its job - "
               "the swarm keeps hunting. Try more generations, more symbols or a stronger model.", ""]
    if near_misses:
        md += ["## Near misses (why they were rejected)", ""]
        for w in near_misses[:8]:
            md.append(f"- **{w['name']}** ({w['symbol']}, OOS Sharpe {w['out_of_sample'].get('sharpe', 0):.2f}): "
                      + "; ".join(w["reasons"][:3]))
        md.append("")
    md += ["", DISCLAIMER, ""]
    md_path = out / f"run_{run_id:04d}.md"
    md_path.write_text("\n".join(md))
    payload = winners_payload(winners, header)
    json_path = out / "winners_latest.json"
    json_path.write_text(json.dumps(payload, indent=2))
    (out / f"winners_run_{run_id:04d}.json").write_text(json.dumps(payload, indent=2))
    return md_path, json_path
