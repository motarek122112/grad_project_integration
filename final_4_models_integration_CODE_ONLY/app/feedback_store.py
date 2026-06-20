from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "ishara_feedback.sqlite3"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def init_feedback_db() -> None:
    with closing(_connect()) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                page TEXT,
                rating INTEGER,
                comment TEXT,
                metadata_json TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        con.commit()


def save_feedback(
    *,
    user_id: Optional[str],
    page: Optional[str],
    rating: Optional[int],
    comment: Optional[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    init_feedback_db()
    fid = str(uuid.uuid4())
    with closing(_connect()) as con:
        con.execute(
            """
            INSERT INTO feedback (id, user_id, page, rating, comment, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fid,
                user_id,
                page,
                rating,
                comment,
                json.dumps(metadata or {}, ensure_ascii=False, default=str),
                _now_iso(),
            ),
        )
        con.commit()
    return fid


def list_feedback(*, user_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    init_feedback_db()
    clauses = []
    params: List[Any] = []
    if user_id:
        clauses.append("user_id = ?")
        params.append(user_id)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    params.append(int(limit))
    with closing(_connect()) as con:
        rows = con.execute(f"SELECT * FROM feedback{where} ORDER BY created_at DESC LIMIT ?", params).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        try:
            item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
        except Exception:
            item["metadata"] = {}
        out.append(item)
    return out
