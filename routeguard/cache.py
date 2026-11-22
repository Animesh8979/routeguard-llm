"""SQLite WAL exact + TF-IDF cosine semantic cache isolated by schema version."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticCache:
    """Two-tier exact hash + TF-IDF semantic response cache backed by SQLite WAL."""

    def __init__(self, db_path: str | Path, *, similarity_threshold: float = 0.80, ttl_seconds: int = 86400) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT NOT NULL DEFAULT 'default',
                    prompt_hash TEXT NOT NULL,
                    prompt_text TEXT NOT NULL,
                    schema_name TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    tier TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    response_text TEXT NOT NULL,
                    created_ts REAL NOT NULL,
                    UNIQUE(tenant_id, prompt_hash, schema_name, schema_version)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_lookup ON cache_entries(tenant_id, schema_name, schema_version, created_ts)")

    @staticmethod
    def _hash_prompt(prompt: str) -> str:
        norm = " ".join((prompt or "").lower().split())
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()

    def get(
        self,
        prompt: str,
        *,
        tenant_id: str = "default",
        schema_name: str = "chat",
        schema_version: str = "v1",
    ) -> Optional[dict[str, Any]]:
        now = time.time()
        min_ts = now - self.ttl_seconds
        p_hash = self._hash_prompt(prompt)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT response_text, tier, model_id, prompt_text
                FROM cache_entries
                WHERE tenant_id = ? AND prompt_hash = ? AND schema_name = ? AND schema_version = ? AND created_ts >= ?
                LIMIT 1
                """,
                (tenant_id, p_hash, schema_name, schema_version, min_ts),
            ).fetchone()
            if row:
                return {
                    "cache_type": "exact",
                    "similarity": 1.0,
                    "response_text": row[0],
                    "tier": row[1],
                    "model_id": row[2],
                }

            candidates = conn.execute(
                """
                SELECT prompt_text, response_text, tier, model_id
                FROM cache_entries
                WHERE tenant_id = ? AND schema_name = ? AND schema_version = ? AND created_ts >= ?
                ORDER BY id DESC LIMIT 200
                """,
                (tenant_id, schema_name, schema_version, min_ts),
            ).fetchall()

        if not candidates:
            return None

        texts = [prompt] + [c[0] for c in candidates]
        try:
            vec = TfidfVectorizer(ngram_range=(1, 1), lowercase=True).fit_transform(texts)
            sims = cosine_similarity(vec[0:1], vec[1:])[0]
        except Exception:
            return None

        best_idx = int(sims.argmax())
        best_score = float(sims[best_idx])
        if best_score >= self.similarity_threshold:
            best = candidates[best_idx]
            return {
                "cache_type": "semantic",
                "similarity": round(best_score, 4),
                "response_text": best[1],
                "tier": best[2],
                "model_id": best[3],
            }
        return None

    def put(
        self,
        prompt: str,
        response_text: str,
        *,
        tenant_id: str = "default",
        tier: str,
        model_id: str,
        schema_name: str = "chat",
        schema_version: str = "v1",
    ) -> None:
        p_hash = self._hash_prompt(prompt)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cache_entries
                (tenant_id, prompt_hash, prompt_text, schema_name, schema_version, tier, model_id, response_text, created_ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (tenant_id, p_hash, prompt, schema_name, schema_version, tier, model_id, response_text, time.time()),
            )
            conn.commit()
