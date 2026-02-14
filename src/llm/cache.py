# -*- coding: utf-8 -*-
"""
src/llm/cache.py

Local cache for LLM/VLM calls.

Goal:
- Make experiments reproducible & cheap.
- Cache by a stable "request key" (model + task + prompt/messages + images digest + params).
- Store response text + parsed json (optional) + usage metadata.

Backend:
- SQLite (recommended) with a single table.
- Safe for multi-process READ; single-writer recommended for batch runs.

Usage:
    cache = SqliteCache("runs/_cache/llm_cache.sqlite3")
    hit = cache.get(key)
    if hit is None:
        resp = call_model(...)
        cache.set(key, resp_text=..., resp_json=..., meta=...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path


# ============================================================
# Helpers
# ============================================================

def _stable_json(obj: Any) -> str:
    """
    Stable JSON stringify for hashing.
    - sort keys
    - keep unicode
    - compact separators
    """
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_hex(s: str) -> str:
    h = hashlib.sha256()
    h.update(s.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def make_cache_key(
    task: str,
    model: str,
    payload: Dict[str, Any],
    params: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build a stable cache key from:
      - task: "extract_constraints" / "judge_constraint" / "verify_pair" / "edit" etc.
      - model: model name
      - payload: request content (messages, prompt, images digests, etc.)
      - params: decoding params (temperature, max_tokens, seed, etc.)

    Returns:
      - short key string
    """
    blob = {
        "task": task,
        "model": model,
        "payload": payload,
        "params": params or {},
    }
    return sha256_hex(_stable_json(blob))


# ============================================================
# Cache Record
# ============================================================

@dataclass(frozen=True)
class CacheHit:
    key: str
    created_at_unix: int
    resp_text: str
    resp_json: Optional[Dict[str, Any]]
    meta: Dict[str, Any]


# ============================================================
# SQLite Cache
# ============================================================

class SqliteCache:
    """
    SQLite cache.

    Table schema:
      cache(
        key TEXT PRIMARY KEY,
        created_at INTEGER,
        resp_text TEXT,
        resp_json TEXT,
        meta TEXT
      )
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ----------------------------
    # Public API
    # ----------------------------

    def get(self, key: str) -> Optional[CacheHit]:
        """
        Get cached record by key.
        """
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT key, created_at, resp_text, resp_json, meta FROM cache WHERE key=?",
                (key,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            k, created_at, resp_text, resp_json_s, meta_s = row
            resp_json = None
            if resp_json_s:
                try:
                    resp_json = json.loads(resp_json_s)
                except Exception:
                    resp_json = None
            meta: Dict[str, Any] = {}
            if meta_s:
                try:
                    meta = json.loads(meta_s)
                except Exception:
                    meta = {}
            return CacheHit(
                key=str(k),
                created_at_unix=int(created_at),
                resp_text=str(resp_text or ""),
                resp_json=resp_json,
                meta=meta,
            )

    def set(
        self,
        key: str,
        resp_text: str,
        resp_json: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Insert a cache record.

        overwrite=False:
          - keep first response for determinism (recommended)
        overwrite=True:
          - update existing (use sparingly)
        """
        created_at = int(time.time())
        resp_json_s = _stable_json(resp_json) if resp_json is not None else ""
        meta_s = _stable_json(meta or {}) if meta is not None else _stable_json({})

        with self._connect() as conn:
            cur = conn.cursor()
            if overwrite:
                cur.execute(
                    """
                    INSERT INTO cache(key, created_at, resp_text, resp_json, meta)
                    VALUES(?,?,?,?,?)
                    ON CONFLICT(key) DO UPDATE SET
                        created_at=excluded.created_at,
                        resp_text=excluded.resp_text,
                        resp_json=excluded.resp_json,
                        meta=excluded.meta
                    """,
                    (key, created_at, resp_text, resp_json_s, meta_s),
                )
            else:
                # keep-first: ignore if exists
                cur.execute(
                    """
                    INSERT OR IGNORE INTO cache(key, created_at, resp_text, resp_json, meta)
                    VALUES(?,?,?,?,?)
                    """,
                    (key, created_at, resp_text, resp_json_s, meta_s),
                )
            conn.commit()

    def delete(self, key: str) -> None:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM cache WHERE key=?", (key,))
            conn.commit()

    def stats(self) -> Dict[str, Any]:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(1) FROM cache")
            n = int(cur.fetchone()[0])
            cur.execute("SELECT MIN(created_at), MAX(created_at) FROM cache")
            r = cur.fetchone()
            mn = int(r[0]) if r and r[0] is not None else 0
            mx = int(r[1]) if r and r[1] is not None else 0
            return {"entries": n, "min_created_at": mn, "max_created_at": mx, "path": str(self.path)}

    # ----------------------------
    # Internal
    # ----------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=30)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS cache(
                    key TEXT PRIMARY KEY,
                    created_at INTEGER NOT NULL,
                    resp_text TEXT NOT NULL,
                    resp_json TEXT NOT NULL,
                    meta TEXT NOT NULL
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_cache_created_at ON cache(created_at)")
            conn.commit()


# ============================================================
# Optional: Null cache (disable caching)
# ============================================================

class NullCache:
    def get(self, key: str) -> Optional[CacheHit]:  # noqa: D401
        return None

    def set(
        self,
        key: str,
        resp_text: str,
        resp_json: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> None:
        return
