"""Task 5 - request history. SQLite (local/tests) or PostgreSQL (Docker Compose) via DB-API."""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

COLUMNS = ["id", "created_at", "filename", "label", "confidence", "prob_ai", "threshold",
           "model_name", "model_version", "source", "latency_ms"]

DDL = {
    "sqlite": """CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, filename TEXT NOT NULL,
        label TEXT NOT NULL, confidence REAL NOT NULL, prob_ai REAL NOT NULL, threshold REAL NOT NULL,
        model_name TEXT NOT NULL, model_version TEXT NOT NULL, source TEXT NOT NULL,
        latency_ms REAL NOT NULL)""",
    "postgres": """CREATE TABLE IF NOT EXISTS predictions (
        id SERIAL PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL, filename TEXT NOT NULL,
        label TEXT NOT NULL, confidence DOUBLE PRECISION NOT NULL, prob_ai DOUBLE PRECISION NOT NULL,
        threshold DOUBLE PRECISION NOT NULL, model_name TEXT NOT NULL, model_version TEXT NOT NULL,
        source TEXT NOT NULL, latency_ms DOUBLE PRECISION NOT NULL)""",
}


class Database:
    """URL forms: sqlite:///relative/path.db  |  postgresql://user:pass@host:5432/dbname"""

    def __init__(self, url):
        self.url = url
        self.kind = "sqlite" if url.startswith("sqlite") else "postgres"
        if self.kind == "sqlite":
            self.path = url.split("sqlite:///", 1)[1]
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)

    def _connect(self):
        if self.kind == "sqlite":
            conn = sqlite3.connect(self.path, timeout=10)
            conn.row_factory = sqlite3.Row
            return conn
        import psycopg2
        import psycopg2.extras
        return psycopg2.connect(self.url, cursor_factory=psycopg2.extras.RealDictCursor,
                                connect_timeout=5)

    def _sql(self, sql):
        return sql.replace("?", "%s") if self.kind == "postgres" else sql

    def _execute(self, sql, params=(), fetch=False):
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(self._sql(sql), params)
            rows = [dict(r) for r in cur.fetchall()] if fetch else None
            conn.commit()
            return rows
        finally:
            conn.close()

    def init(self):
        self._execute(DDL[self.kind])

    def add(self, *, filename, label, confidence, prob_ai, threshold, model_name, model_version,
            source, latency_ms):
        created = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._execute(
            "INSERT INTO predictions (created_at, filename, label, confidence, prob_ai, threshold, "
            "model_name, model_version, source, latency_ms) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (created, filename, label, confidence, prob_ai, threshold, model_name,
             str(model_version), source, latency_ms))

    def recent(self, limit=200):
        rows = self._execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,), fetch=True)
        for r in rows:
            r["created_at"] = str(r["created_at"])[:19].replace("T", " ")
        return rows

    def count(self):
        return self._execute("SELECT COUNT(*) AS n FROM predictions", fetch=True)[0]["n"]

    def ping(self):
        try:
            self._execute("SELECT 1")
            return True
        except Exception:  # noqa: BLE001
            return False
