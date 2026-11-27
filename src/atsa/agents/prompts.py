"""Prompt builders. Short, strict, JSON-only - works on frontier and small local models alike."""

from __future__ import annotations

import json

from ..dsl.expr import reference
from ..dsl.strategy import STRATEGY_JSON_SCHEMA_HINT
from .personas import Persona

SYSTEM_TEMPLATE = """You are the {title} on an autonomous AI quant research desk.
MANDATE: {mandate}

You design rule-based trading strategies for ONE instrument using a strict expression language.
Signals are evaluated at each bar's close and filled at the next bar's open. Costs are charged on every trade.

{dsl}

Each strategy is a JSON object:
{schema}

Exit semantics: if exit_long is null and max_hold_bars is set, the trade is held for max_hold_bars bars
(event trades); if both are null the long closes when entry_long stops being true (same for short).
stop_loss/take_profit are fractions of entry price (0.05 = 5%). max_hold_bars force-closes old trades.

Quality bar (a robustness gate will reject anything else):
- Fewer parameters beat more parameters. Prefer round, common values (10, 20, 50, 100, 200).
- Every strategy needs a real economic thesis, not curve fitting.
- Must trade often enough to be statistically meaningful (20+ trades over the history), but not every bar.
- Must survive 2x trading costs and work out-of-sample.
- Make the strategies in one answer genuinely different from each other.

Reply with ONLY a JSON object: {{"strategies": [ ... ]}}"""


def system_prompt(persona: Persona) -> str:
    return SYSTEM_TEMPLATE.format(title=persona.title, mandate=persona.mandate, dsl=reference(),
                                  schema=STRATEGY_JSON_SCHEMA_HINT)


def _compact(result: dict) -> dict:
    o, i, f = result.get("out_of_sample", {}), result.get("in_sample", {}), result.get("full", {})
    return {
        "spec": {k: v for k, v in result.get("spec", {}).items() if k not in ("agent", "generation", "parent_id")},
        "sharpe_in": round(i.get("sharpe", 0.0), 2), "sharpe_out": round(o.get("sharpe", 0.0), 2),
        "max_dd": round(f.get("max_drawdown", 0.0), 3), "trades": f.get("trades", 0),
        "walk_forward_positive": round(result.get("wf_positive", 0.0), 2),
        "rejected_because": result.get("reasons", [])[:3],
    }


def user_prompt(brief: dict, n: int, allow_short: bool, cost_bps: float, generation: int,
                leaders: list[dict] | None = None, failures: list[dict] | None = None) -> str:
    parts = [
        f"LIVE MARKET RESEARCH BRIEF:\n{json.dumps(brief, indent=1)}",
        f"CONSTRAINTS: interval={brief.get('interval')}, short selling {'ALLOWED' if allow_short else 'NOT allowed'}, "
        f"round-trip cost ~{2 * cost_bps:.0f} bps.",
    ]
    if leaders:
        parts.append("CURRENT LEADERBOARD (beat these - improve or find different edges):\n"
                     + json.dumps([_compact(r) for r in leaders], indent=1))
    if failures:
        parts.append("RECENT REJECTS (learn from why they failed, do not repeat them):\n"
                     + json.dumps([_compact(r) for r in failures], indent=1))
    parts.append(f"GENERATION {generation}: propose exactly {n} strategies as JSON.")
    return "\n\n".join(parts)
