"""SQLite persistence layer."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import aiosqlite

from agenteval.models import EvalResult, Run


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


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

    async def _query(
        self,
        table: str,
        project: str,
        scenario: str | None = None,
        order_by: str | None = None,
    ) -> tuple[list[str], list[tuple]]:
        """Execute a filtered SELECT query and return column names and rows."""
        query = f"SELECT * FROM {table} WHERE project = ?"
        params: list[str] = [project]
        if scenario:
            query += " AND scenario = ?"
            params.append(scenario)
        if order_by:
            query += f" ORDER BY {order_by}"
        cursor = await self._db.execute(query, params)
        rows = await cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return columns, rows

    async def save_run(self, run: Run, project: str) -> None:
        await self._db.execute(
            "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (run.run_id, project, run.scenario, int(run.success),
             run.total_tokens, run.total_cost, run.total_latency_ms,
             json.dumps(run.checkpoints_reached),
             json.dumps([t.model_dump() for t in run.turns]),
             json.dumps(run.final_state), _utc_now_iso()),
        )
        await self._db.commit()

    async def load_runs(self, project: str, scenario: str | None = None) -> list[dict]:
        columns, rows = await self._query("runs", project, scenario)
        results = []
        for row in rows:
            record = dict(zip(columns, row))
            record["success"] = bool(record["success"])
            record["checkpoints_reached"] = json.loads(record["checkpoints_reached"])
            results.append(record)
        return results

    async def save_result(self, result: EvalResult) -> None:
        await self._db.execute(
            "INSERT INTO results (project,scenario,k,pass_k,state_correctness,"
            "checkpoint_completion,tool_accuracy,forbidden_violations,avg_turns,"
            "avg_tokens,avg_cost,avg_latency_ms,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (result.project, result.scenario, result.k, result.pass_k,
             result.state_correctness, result.checkpoint_completion,
             result.tool_accuracy, result.forbidden_tool_violations,
             result.avg_turns, result.avg_tokens, result.avg_cost,
             result.avg_latency_ms, _utc_now_iso()),
        )
        await self._db.commit()

    async def load_results(self, project: str, scenario: str | None = None) -> list[dict]:
        columns, rows = await self._query("results", project, scenario, order_by="created_at DESC")
        return [dict(zip(columns, row)) for row in rows]
