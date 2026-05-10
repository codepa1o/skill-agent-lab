import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "skill_agent_lab.db"

CREATE_RUNS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_url TEXT NOT NULL,
    raw_url TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    model TEXT NOT NULL,
    api_mode TEXT NOT NULL DEFAULT '',
    base_url TEXT NOT NULL DEFAULT '',
    reasoning_effort TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    error_message TEXT NOT NULL,
    latency_ms INTEGER NOT NULL,
    created_at TEXT NOT NULL
)
"""


@dataclass(frozen=True)
class RunRecord:
    id: int
    skill_url: str
    raw_url: str
    question: str
    answer: str
    model: str
    api_mode: str
    base_url: str
    reasoning_effort: str
    status: str
    error_message: str
    latency_ms: int
    created_at: str


def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with _connect() as connection:
        connection.execute(CREATE_RUNS_TABLE_SQL)
        _ensure_column(connection, "api_mode", "TEXT NOT NULL DEFAULT ''")
        _ensure_column(connection, "base_url", "TEXT NOT NULL DEFAULT ''")
        _ensure_column(connection, "reasoning_effort", "TEXT NOT NULL DEFAULT ''")


def _ensure_column(connection: sqlite3.Connection, name: str, definition: str) -> None:
    columns = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(runs)").fetchall()
    }
    if name not in columns:
        connection.execute(f"ALTER TABLE runs ADD COLUMN {name} {definition}")


def create_run(
    *,
    skill_url: str,
    raw_url: str = "",
    question: str,
    answer: str = "",
    model: str = "",
    api_mode: str = "",
    base_url: str = "",
    reasoning_effort: str = "",
    status: str,
    error_message: str = "",
    latency_ms: int = 0,
) -> int:
    init_db()
    created_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO runs (
                skill_url, raw_url, question, answer, model, api_mode,
                base_url, reasoning_effort, status, error_message,
                latency_ms, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                skill_url,
                raw_url,
                question,
                answer,
                model,
                api_mode,
                base_url,
                reasoning_effort,
                status,
                error_message,
                latency_ms,
                created_at,
            ),
        )
        return int(cursor.lastrowid)


def list_runs(limit: int = 20) -> list[RunRecord]:
    init_db()
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT id, skill_url, raw_url, question, answer, model, status,
                   api_mode, base_url, reasoning_effort, error_message,
                   latency_ms, created_at
            FROM runs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [_row_to_record(row) for row in rows]


def get_run(run_id: int) -> RunRecord | None:
    init_db()
    with _connect() as connection:
        row = connection.execute(
            """
            SELECT id, skill_url, raw_url, question, answer, model, status,
                   api_mode, base_url, reasoning_effort, error_message,
                   latency_ms, created_at
            FROM runs
            WHERE id = ?
            """,
            (run_id,),
        ).fetchone()
    return _row_to_record(row) if row else None


def _row_to_record(row: sqlite3.Row) -> RunRecord:
    return RunRecord(
        id=row["id"],
        skill_url=row["skill_url"],
        raw_url=row["raw_url"],
        question=row["question"],
        answer=row["answer"],
        model=row["model"],
        api_mode=row["api_mode"],
        base_url=row["base_url"],
        reasoning_effort=row["reasoning_effort"],
        status=row["status"],
        error_message=row["error_message"],
        latency_ms=row["latency_ms"],
        created_at=row["created_at"],
    )
