"""FastAPI dashboard: read-only view over the SQLite leaderboard (safe to run next to the 24/7 runner)."""

from __future__ import annotations

from pathlib import Path

from ..store import Store

STATIC = Path(__file__).parent / "static"


def create_app(db_path: str):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse

    app = FastAPI(title="AI Trading Strategy Agents", version="0.1.0")
    store = Store(db_path)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC / "index.html").read_text()

    @app.get("/api/summary")
    def summary() -> dict:
        c = store.counts()
        runs = store.runs(1)
        best = store.leaderboard(1)
        return {**c, "last_run": runs[0] if runs else None,
                "best_oos_sharpe": best[0]["sharpe_oos"] if best else None,
                "runs": len(store.runs(100000))}

    @app.get("/api/leaderboard")
    def leaderboard(limit: int = 50, symbol: str | None = None, all: bool = False) -> list[dict]:
        rows = store.leaderboard(min(limit, 500), symbol, winners_only=not all)
        return [{k: v for k, v in r.items() if k != "equity"} for r in rows]

    @app.get("/api/strategy/{key}")
    def strategy(key: str) -> dict:
        row = store.get(key)
        if not row:
            raise HTTPException(404, "strategy not found")
        return row

    @app.get("/api/runs")
    def runs(limit: int = 20) -> list[dict]:
        return store.runs(min(limit, 500))

    return app
