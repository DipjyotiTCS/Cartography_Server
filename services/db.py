import os
import sqlite3
from typing import Optional, Dict, Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS repositories (
  id TEXT PRIMARY KEY,
  repo_name TEXT NOT NULL UNIQUE,
  repo_link TEXT NOT NULL,
  language TEXT NOT NULL,
  local_path TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS kv (
  k TEXT PRIMARY KEY,
  v TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ingest_errors (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  repo_id TEXT NOT NULL,
  repo_name TEXT NOT NULL,
  file_path TEXT,
  stage TEXT,
  error_type TEXT,
  message TEXT,
  traceback TEXT,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ingest_errors_repo ON ingest_errors(repo_id);
"""

def ensure_db(sqlite_path: str) -> None:
    os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
    with sqlite3.connect(sqlite_path) as conn:
        conn.executescript(SCHEMA)
        conn.commit()

def insert_repo(sqlite_path: str, repo: Dict[str, Any]) -> None:
    with sqlite3.connect(sqlite_path) as conn:
        conn.execute(
            "INSERT INTO repositories (id, repo_name, repo_link, language, local_path, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (repo["id"], repo["repo_name"], repo["repo_link"], repo["language"], repo["local_path"], repo["created_at"]),
        )
        conn.commit()

def get_repo(sqlite_path: str, repo_id: str) -> Optional[Dict[str, Any]]:
    with sqlite3.connect(sqlite_path) as conn:
        cur = conn.execute("SELECT id, repo_name, repo_link, language, local_path, created_at FROM repositories WHERE id=?", (repo_id,))
        row = cur.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "repo_name": row[1],
            "repo_link": row[2],
            "language": row[3],
            "local_path": row[4],
            "created_at": row[5],
        }

def next_vector_id(sqlite_path: str) -> int:
    with sqlite3.connect(sqlite_path) as conn:
        cur = conn.execute("SELECT v FROM kv WHERE k='vector_id'")
        row = cur.fetchone()
        current = int(row[0]) if row else 0
        nxt = current + 1
        if row:
            conn.execute("UPDATE kv SET v=? WHERE k='vector_id'", (str(nxt),))
        else:
            conn.execute("INSERT INTO kv (k, v) VALUES ('vector_id', ?)", (str(nxt),))
        conn.commit()
        return nxt


def upsert_repo(sqlite_path: str, repo: Dict[str, Any]) -> str:
    """Upsert repo by repo_name. Returns the repo_id to use."""
    with sqlite3.connect(sqlite_path) as conn:
        cur = conn.execute("SELECT id FROM repositories WHERE repo_name=?", (repo["repo_name"],))
        row = cur.fetchone()
        if row:
            repo_id = row[0]
            conn.execute(
                "UPDATE repositories SET repo_link=?, language=?, local_path=?, created_at=? WHERE repo_name=?",
                (repo["repo_link"], repo["language"], repo["local_path"], repo["created_at"], repo["repo_name"]),
            )
            conn.commit()
            return repo_id
        # insert new
        conn.execute(
            "INSERT INTO repositories (id, repo_name, repo_link, language, local_path, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (repo["id"], repo["repo_name"], repo["repo_link"], repo["language"], repo["local_path"], repo["created_at"]),
        )
        conn.commit()
        return repo["id"]

def insert_ingest_error(sqlite_path: str, err: Dict[str, Any]) -> None:
    with sqlite3.connect(sqlite_path) as conn:
        conn.execute(
            "INSERT INTO ingest_errors (repo_id, repo_name, file_path, stage, error_type, message, traceback, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                err.get("repo_id",""),
                err.get("repo_name",""),
                err.get("file_path"),
                err.get("stage"),
                err.get("error_type"),
                err.get("message"),
                err.get("traceback"),
                err.get("created_at"),
            ),
        )
        conn.commit()


def get_repo_by_name(sqlite_path: str, repo_name: str) -> Optional[Dict[str, Any]]:
    with sqlite3.connect(sqlite_path) as conn:
        cur = conn.execute("SELECT id, repo_name, repo_link, language, local_path, created_at FROM repositories WHERE repo_name=?", (repo_name,))
        row = cur.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "repo_name": row[1],
            "repo_link": row[2],
            "language": row[3],
            "local_path": row[4],
            "created_at": row[5],
        }
