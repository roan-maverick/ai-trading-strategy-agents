"""Use the engine from Python instead of the CLI.

    python examples/python_api.py
"""

import asyncio

from atsa.config import Settings
from atsa.pipeline import Pipeline


async def main() -> None:
    s = Settings()
    s.llm.provider = "offline"          # or "gemini" with GEMINI_API_KEY set
    s.data.source = "synthetic"         # or "binance" / "yahoo" / "csv"
    s.data.symbols = ["DEMO-A", "DEMO-C"]
    s.swarm.generations = 3
    s.runtime.db_path = ".atsa-example/atsa.db"
    s.runtime.reports_dir = ".atsa-example/reports"

    pipe = Pipeline(s)
    cycle = await pipe.run_cycle()
    pipe.close()

    print(f"\n{cycle.tested} tested, {len(cycle.winners)} passed, {len(cycle.distinct_winners)} distinct ideas")
    for w in cycle.distinct_winners:
        print(f"- {w.symbol} {w.strategy.name}: OOS Sharpe {w.out_of_sample['sharpe']:.2f}, DSR {w.dsr:.2f}")
        print(f"    entry_long = {w.strategy.entry_long}")


if __name__ == "__main__":
    asyncio.run(main())
