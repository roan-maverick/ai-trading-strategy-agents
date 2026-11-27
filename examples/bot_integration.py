"""Minimal example: read the swarm's winners and act on their current signals.

Run `atsa run` (or `atsa demo`) first. This example only PRINTS orders. Wire `place_order` to your
exchange / broker SDK, and paper-trade first.

    python examples/bot_integration.py --db .atsa-demo/atsa.db
"""

from __future__ import annotations

import argparse

from atsa.backtest import evaluate_strategy
from atsa.config import infer_periods_per_year, load_settings
from atsa.data import load_bars
from atsa.dsl import Strategy
from atsa.store import Store


def place_order(symbol: str, target: str, strategy_name: str) -> None:
    # Replace with your execution code, e.g. ccxt: exchange.create_market_order(symbol, side, qty)
    print(f"[ORDER] {symbol:10s} -> {target:5s} | {strategy_name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None, help="path to atsa.db")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    settings = load_settings()
    store = Store(args.db or settings.runtime.db_path)
    winners = store.leaderboard(args.top)
    store.close()
    if not winners:
        print("No winners yet - run `atsa demo` or `atsa run` first.")
        return

    ppy = infer_periods_per_year(settings.data.interval, settings.data.source)
    for row in winners:
        bars = load_bars(settings.data, row["symbol"], max_age_minutes=settings.runtime.interval_minutes)
        ev = evaluate_strategy(bars, Strategy.model_validate(row["spec"]), settings.backtest, ppy)
        target = {1: "LONG", -1: "SHORT", 0: "FLAT"}[ev.next_position]
        place_order(row["symbol"], target, row["name"])


if __name__ == "__main__":
    main()
