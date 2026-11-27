"""SQLite leaderboard + run history (stdlib only, safe to share between the runner and the dashboard)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  started_at TEXT, finished_at TEXT, provider TEXT, model TEXT, source TEXT, symbols TEXT,
  tested INTEGER DEFAULT 0, winners INTEGER DEFAULT 0, status TEXT, usage TEXT, report_path TEXT
);
CREATE TABLE IF NOT EXISTS strategies (
  key TEXT PRIMARY KEY, id TEXT, symbol TEXT, run_id INTEGER, name TEXT, agent TEXT, generation INTEGER,
  spec TEXT, result TEXT, equity TEXT, passed INTEGER, score REAL, sharpe_oos REAL, dsr REAL,
  first_seen TEXT, updated_at TEXT, times_validated INTEGER DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_strat_rank ON strategies(passed, score DESC);
CREATE INDEX IF NOT EXISTS idx_strat_symbol ON strategies(symbol);
"""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    # runs -------------------------------------------------------------------------------------
    def start_run(self, provider: str, model: str, source: str, symbols: list[str]) -> int:
        cur = self.conn.execute(
            "INSERT INTO runs(started_at, provider, model, source, symbols, status) VALUES (?,?,?,?,?,?)",
            (_now(), provider, model, source, ",".join(symbols), "running"))
        self.conn.commit()
        return int(cur.lastrowid)

    def finish_run(self, run_id: int, tested: int, winners: int, status: str, usage: dict,
                   report_path: str | None) -> None:
        self.conn.execute(
            "UPDATE runs SET finished_at=?, tested=?, winners=?, status=?, usage=?, report_path=? WHERE id=?",
            (_now(), tested, winners, status, json.dumps(usage), report_path, run_id))
        self.conn.commit()

    def runs(self, limit: int = 20) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    # strategies -------------------------------------------------------------------------------
    def upsert(self, run_id: int, result: dict, equity: list[float]) -> None:
        key = f"{result['symbol']}|{result['id']}"
        now = _now()
        self.conn.execute(
            """INSERT INTO strategies(key,id,symbol,run_id,name,agent,generation,spec,result,equity,passed,score,
                                      sharpe_oos,dsr,first_seen,updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(key) DO UPDATE SET run_id=excluded.run_id, result=excluded.result,
                 equity=excluded.equity, passed=excluded.passed, score=excluded.score,
                 sharpe_oos=excluded.sharpe_oos, dsr=excluded.dsr, updated_at=excluded.updated_at,
                 times_validated=strategies.times_validated + 1""",
            (key, result["id"], result["symbol"], run_id, result["name"], result["agent"], result["generation"],
             json.dumps(result["spec"]), json.dumps(result), json.dumps(equity), int(result["passed"]),
             float(result["score"]), float(result.get("out_of_sample", {}).get("sharpe", 0.0)),
             float(result["dsr"]), now, now))

    def commit(self) -> None:
        self.conn.commit()

    def leaderboard(self, limit: int = 20, symbol: str | None = None, winners_only: bool = True) -> list[dict]:
        q = "SELECT * FROM strategies WHERE 1=1"
        args: list = []
        if winners_only:
            q += " AND passed=1"
        if symbol:
            q += " AND symbol=?"
            args.append(symbol)
        q += " ORDER BY score DESC LIMIT ?"
        args.append(limit)
        return [self._row(r) for r in self.conn.execute(q, args).fetchall()]

    def get(self, key_or_id: str) -> dict | None:
        r = self.conn.execute("SELECT * FROM strategies WHERE key=? OR id=? ORDER BY score DESC LIMIT 1",
                              (key_or_id, key_or_id)).fetchone()
        return self._row(r) if r else None

    def counts(self) -> dict:
        r = self.conn.execute("SELECT COUNT(*) AS tested, COALESCE(SUM(passed),0) AS winners FROM strategies").fetchone()
        return {"tested": int(r["tested"]), "winners": int(r["winners"])}

    @staticmethod
    def _row(r: sqlite3.Row) -> dict:
        d = dict(r)
        for k in ("spec", "result", "equity"):
            if d.get(k):
                d[k] = json.loads(d[k])
        return d
