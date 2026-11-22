"""Transactional SQLite WAL telemetry & cost-savings ledger for RouteGuard."""
from __future__ import annotations

from pathlib import Path
import sqlite3
import time
from typing import Any
import uuid

import numpy as np


class TelemetryLedger:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS request_logs (
                    request_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    created_ts REAL NOT NULL,
                    tier_selected TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    cache_hit INTEGER NOT NULL DEFAULT 0,
                    cache_type TEXT NOT NULL DEFAULT 'none',
                    repair_attempts INTEGER NOT NULL DEFAULT 0,
                    schema_valid INTEGER NOT NULL DEFAULT 1,
                    injection_blocked INTEGER NOT NULL DEFAULT 0,
                    pii_redacted INTEGER NOT NULL DEFAULT 0,
                    latency_ms REAL NOT NULL DEFAULT 0.0,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    actual_cost_usd REAL NOT NULL DEFAULT 0.0,
                    baseline_frontier_cost_usd REAL NOT NULL DEFAULT 0.0,
                    error_code TEXT DEFAULT ''
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_tenant_ts ON request_logs(tenant_id, created_ts)")

    def record(
        self,
        *,
        tenant_id: str,
        tier_selected: str,
        model_id: str,
        cache_hit: bool = False,
        cache_type: str = "none",
        repair_attempts: int = 0,
        schema_valid: bool = True,
        injection_blocked: bool = False,
        pii_redacted: bool = False,
        latency_ms: float = 0.0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        actual_cost_usd: float = 0.0,
        baseline_frontier_cost_usd: float = 0.0,
        error_code: str = "",
    ) -> str:
        req_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO request_logs (
                    request_id, tenant_id, created_ts, tier_selected, model_id, cache_hit, cache_type,
                    repair_attempts, schema_valid, injection_blocked, pii_redacted, latency_ms,
                    prompt_tokens, completion_tokens, actual_cost_usd, baseline_frontier_cost_usd, error_code
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    req_id, tenant_id, time.time(), tier_selected, model_id, int(cache_hit), cache_type,
                    int(repair_attempts), int(schema_valid), int(injection_blocked), int(pii_redacted),
                    round(float(latency_ms), 3), int(prompt_tokens), int(completion_tokens),
                    float(actual_cost_usd), float(baseline_frontier_cost_usd), error_code,
                ),
            )
            conn.commit()
        return req_id

    def get_tenant_spend_today(self, tenant_id: str) -> float:
        cutoff = time.time() - 86400.0
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(actual_cost_usd), 0.0) AS spend FROM request_logs WHERE tenant_id = ? AND created_ts >= ?",
                (tenant_id, cutoff),
            ).fetchone()
        return float(row["spend"]) if row else 0.0

    def get_metrics(self) -> dict[str, Any]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM request_logs ORDER BY created_ts DESC").fetchall()
        if not rows:
            return {
                "total_requests": 0,
                "cache_hits": 0,
                "cache_hit_rate_pct": 0.0,
                "injections_blocked": 0,
                "pii_redacted_count": 0,
                "repairs_triggered": 0,
                "schema_validity_pct": 100.0,
                "actual_cost_usd": 0.0,
                "baseline_frontier_cost_usd": 0.0,
                "cost_savings_pct": 0.0,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "tier_distribution": {},
            }
        latencies = [float(r["latency_ms"]) for r in rows]
        actual_sum = sum(float(r["actual_cost_usd"]) for r in rows)
        base_sum = sum(float(r["baseline_frontier_cost_usd"]) for r in rows)
        non_blocked = [r for r in rows if not r["injection_blocked"]]
        valid_count = sum(1 for r in non_blocked if r["schema_valid"])
        tier_counts: dict[str, int] = {}
        for r in rows:
            t = r["tier_selected"]
            tier_counts[t] = tier_counts.get(t, 0) + 1

        savings_pct = ((base_sum - actual_sum) / base_sum * 100.0) if base_sum > 0 else 0.0
        return {
            "total_requests": len(rows),
            "cache_hits": sum(1 for r in rows if r["cache_hit"]),
            "cache_hit_rate_pct": round(100.0 * sum(1 for r in rows if r["cache_hit"]) / len(rows), 2),
            "injections_blocked": sum(1 for r in rows if r["injection_blocked"]),
            "pii_redacted_count": sum(1 for r in rows if r["pii_redacted"]),
            "repairs_triggered": sum(1 for r in rows if r["repair_attempts"] > 0),
            "schema_validity_pct": round(100.0 * valid_count / len(non_blocked), 2) if non_blocked else 100.0,
            "actual_cost_usd": round(actual_sum, 6),
            "baseline_frontier_cost_usd": round(base_sum, 6),
            "cost_savings_pct": round(savings_pct, 2),
            "p50_latency_ms": round(float(np.percentile(latencies, 50)), 2),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
            "tier_distribution": tier_counts,
        }
