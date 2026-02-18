"""SQLite persistence layer."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import aiosqlite

from agenteval.models import EvalResult, Run


class Store:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def init(self) -> None:
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY, project TEXT NOT NULL, scenario TEXT NOT NULL,
                success INTEGER NOT NULL, total_tokens INTEGER DEFAULT 0,
                total_cost REAL DEFAULT 0.0, total_latency_ms REAL DEFAULT 0.0,
                checkpoints_reached TEXT DEFAULT '[]', turns_json TEXT DEFAULT '[]',
                final_state_json TEXT DEFAULT '{}', created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS results (
                result_id INTEGER PRIMARY KEY AUTOINCREMENT, project TEXT NOT NULL,
                scenario TEXT NOT NULL, k INTEGER NOT NULL, pass_k REAL DEFAULT 0.0,
                state_correctness REAL DEFAULT 0.0, checkpoint_completion REAL DEFAULT 0.0,
                tool_accuracy REAL DEFAULT 0.0, forbidden_violations INTEGER DEFAULT 0,
                avg_turns REAL DEFAULT 0.0, avg_tokens INTEGER DEFAULT 0,
                avg_cost REAL DEFAULT 0.0, avg_latency_ms REAL DEFAULT 0.0,
                created_at TEXT NOT NULL);
        """)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    async def save_run(self, run: Run, project: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (run.run_id, project, run.scenario, int(run.success),
             run.total_tokens, run.total_cost, run.total_latency_ms,
             json.dumps(run.checkpoints_reached),
             json.dumps([t.model_dump() for t in run.turns]),
             json.dumps(run.final_state), now))
        await self._db.commit()

    async def load_runs(self, project: str, scenario: str | None = None) -> list[dict]:
        q = "SELECT * FROM runs WHERE project = ?"
        p: list = [project]
        if scenario:
            q += " AND scenario = ?"
            p.append(scenario)
        cur = await self._db.execute(q, p)
        rows = await cur.fetchall()
        cols = [d[0] for d in cur.description]
        out = []
        for row in rows:
            d = dict(zip(cols, row))
            d["success"] = bool(d["success"])
            d["checkpoints_reached"] = json.loads(d["checkpoints_reached"])
            out.append(d)
        return out

    async def save_result(self, result: EvalResult) -> None:
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            "INSERT INTO results (project,scenario,k,pass_k,state_correctness,"
            "checkpoint_completion,tool_accuracy,forbidden_violations,avg_turns,"
            "avg_tokens,avg_cost,avg_latency_ms,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (result.project, result.scenario, result.k, result.pass_k,
             result.state_correctness, result.checkpoint_completion,
             result.tool_accuracy, result.forbidden_tool_violations,
             result.avg_turns, result.avg_tokens, result.avg_cost,
             result.avg_latency_ms, now))
        await self._db.commit()

    async def load_results(self, project: str, scenario: str | None = None) -> list[dict]:
        q = "SELECT * FROM results WHERE project = ?"
        p: list = [project]
        if scenario:
            q += " AND scenario = ?"
            p.append(scenario)
        q += " ORDER BY created_at DESC"
        cur = await self._db.execute(q, p)
        rows = await cur.fetchall()
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in rows]
