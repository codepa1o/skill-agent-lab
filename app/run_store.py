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
    search_used INTEGER NOT NULL DEFAULT 0,
    search_results TEXT NOT NULL DEFAULT '',
    rag_used INTEGER NOT NULL DEFAULT 0,
    rag_results TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    error_message TEXT NOT NULL,
    latency_ms INTEGER NOT NULL,
    created_at TEXT NOT NULL
)
"""

CREATE_RAG_DOCUMENTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS rag_documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,
    file_path TEXT NOT NULL,
    status TEXT NOT NULL,
    error_message TEXT NOT NULL DEFAULT '',
    chunk_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
)
"""

CREATE_RAG_CHUNKS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS rag_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    page_number INTEGER NOT NULL DEFAULT 0,
    content TEXT NOT NULL,
    embedding TEXT NOT NULL,
    embedding_dim INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES rag_documents(id)
)
"""

CREATE_CONVERSATIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    skill_url TEXT NOT NULL,
    raw_url TEXT NOT NULL,
    model TEXT NOT NULL,
    api_mode TEXT NOT NULL,
    base_url TEXT NOT NULL,
    reasoning_effort TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""

CREATE_MESSAGES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    latency_ms INTEGER NOT NULL DEFAULT 0,
    error_message TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
)
"""

CREATE_TEST_SUITES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS test_suites (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    skill_url TEXT NOT NULL,
    description TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""

CREATE_TEST_CASES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS test_cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    suite_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    question TEXT NOT NULL,
    expected_behavior TEXT NOT NULL,
    scoring_focus TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (suite_id) REFERENCES test_suites(id)
)
"""

CREATE_EVAL_RUNS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS eval_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    suite_id INTEGER NOT NULL,
    status TEXT NOT NULL,
    average_score REAL NOT NULL DEFAULT 0,
    error_message TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY (suite_id) REFERENCES test_suites(id)
)
"""

CREATE_EVAL_RESULTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS eval_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    eval_run_id INTEGER NOT NULL,
    test_case_id INTEGER NOT NULL,
    variant TEXT NOT NULL DEFAULT 'enhanced',
    agent_answer TEXT NOT NULL,
    role_adherence INTEGER NOT NULL DEFAULT 0,
    constraint_adherence INTEGER NOT NULL DEFAULT 0,
    task_completion INTEGER NOT NULL DEFAULT 0,
    factual_safety INTEGER NOT NULL DEFAULT 0,
    format_quality INTEGER NOT NULL DEFAULT 0,
    source_usage INTEGER NOT NULL DEFAULT 0,
    overall REAL NOT NULL DEFAULT 0,
    judge_comment TEXT NOT NULL DEFAULT '',
    search_used INTEGER NOT NULL DEFAULT 0,
    search_results TEXT NOT NULL DEFAULT '',
    rag_used INTEGER NOT NULL DEFAULT 0,
    rag_results TEXT NOT NULL DEFAULT '',
    error_message TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY (eval_run_id) REFERENCES eval_runs(id),
    FOREIGN KEY (test_case_id) REFERENCES test_cases(id)
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
    search_used: int
    search_results: str
    rag_used: int
    rag_results: str
    status: str
    error_message: str
    latency_ms: int
    created_at: str


@dataclass(frozen=True)
class ConversationRecord:
    id: int
    title: str
    skill_url: str
    raw_url: str
    model: str
    api_mode: str
    base_url: str
    reasoning_effort: str
    created_at: str
    updated_at: str
    message_count: int = 0


@dataclass(frozen=True)
class MessageRecord:
    id: int
    conversation_id: int
    role: str
    content: str
    latency_ms: int
    error_message: str
    created_at: str


@dataclass(frozen=True)
class TestSuiteRecord:
    id: int
    name: str
    skill_url: str
    description: str
    created_at: str
    case_count: int = 0


@dataclass(frozen=True)
class TestCaseRecord:
    id: int
    suite_id: int
    title: str
    question: str
    expected_behavior: str
    scoring_focus: str
    created_at: str


@dataclass(frozen=True)
class EvalRunRecord:
    id: int
    suite_id: int
    status: str
    average_score: float
    error_message: str
    created_at: str
    suite_name: str = ""


@dataclass(frozen=True)
class EvalResultRecord:
    id: int
    eval_run_id: int
    test_case_id: int
    variant: str
    agent_answer: str
    role_adherence: int
    constraint_adherence: int
    task_completion: int
    factual_safety: int
    format_quality: int
    source_usage: int
    overall: float
    judge_comment: str
    search_used: int
    search_results: str
    rag_used: int
    rag_results: str
    error_message: str
    created_at: str
    test_case_title: str = ""
    question: str = ""


@dataclass(frozen=True)
class RagDocumentRecord:
    id: int
    filename: str
    file_type: str
    file_path: str
    status: str
    error_message: str
    chunk_count: int
    created_at: str


@dataclass(frozen=True)
class RagChunkRecord:
    id: int
    document_id: int
    chunk_index: int
    page_number: int
    content: str
    embedding: str
    embedding_dim: int
    created_at: str
    document_filename: str = ""


def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with _connect() as connection:
        connection.execute(CREATE_RUNS_TABLE_SQL)
        connection.execute(CREATE_CONVERSATIONS_TABLE_SQL)
        connection.execute(CREATE_MESSAGES_TABLE_SQL)
        connection.execute(CREATE_TEST_SUITES_TABLE_SQL)
        connection.execute(CREATE_TEST_CASES_TABLE_SQL)
        connection.execute(CREATE_EVAL_RUNS_TABLE_SQL)
        connection.execute(CREATE_EVAL_RESULTS_TABLE_SQL)
        connection.execute(CREATE_RAG_DOCUMENTS_TABLE_SQL)
        connection.execute(CREATE_RAG_CHUNKS_TABLE_SQL)
        _ensure_column(connection, "runs", "api_mode", "TEXT NOT NULL DEFAULT ''")
        _ensure_column(connection, "runs", "base_url", "TEXT NOT NULL DEFAULT ''")
        _ensure_column(connection, "runs", "reasoning_effort", "TEXT NOT NULL DEFAULT ''")
        _ensure_column(connection, "runs", "search_used", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(connection, "runs", "search_results", "TEXT NOT NULL DEFAULT ''")
        _ensure_column(connection, "runs", "rag_used", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(connection, "runs", "rag_results", "TEXT NOT NULL DEFAULT ''")
        _ensure_column(connection, "eval_results", "variant", "TEXT NOT NULL DEFAULT 'enhanced'")
        _ensure_column(connection, "eval_results", "source_usage", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(connection, "eval_results", "search_used", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(connection, "eval_results", "search_results", "TEXT NOT NULL DEFAULT ''")
        _ensure_column(connection, "eval_results", "rag_used", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(connection, "eval_results", "rag_results", "TEXT NOT NULL DEFAULT ''")


def _ensure_column(
    connection: sqlite3.Connection,
    table_name: str,
    name: str,
    definition: str,
) -> None:
    columns = {
        row["name"]
        for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    if name not in columns:
        connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {name} {definition}")


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


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
    search_used: int = 0,
    search_results: str = "",
    rag_used: int = 0,
    rag_results: str = "",
    status: str,
    error_message: str = "",
    latency_ms: int = 0,
) -> int:
    init_db()
    created_at = _now()
    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO runs (
                skill_url, raw_url, question, answer, model, api_mode,
                base_url, reasoning_effort, search_used, search_results,
                rag_used, rag_results, status, error_message, latency_ms, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                search_used,
                search_results,
                rag_used,
                rag_results,
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
                   api_mode, base_url, reasoning_effort, search_used, search_results,
                   rag_used, rag_results, error_message,
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
                   api_mode, base_url, reasoning_effort, search_used, search_results,
                   rag_used, rag_results, error_message,
                   latency_ms, created_at
            FROM runs
            WHERE id = ?
            """,
            (run_id,),
        ).fetchone()
    return _row_to_record(row) if row else None


def create_conversation(
    *,
    title: str,
    skill_url: str,
    raw_url: str,
    model: str,
    api_mode: str,
    base_url: str,
    reasoning_effort: str,
) -> int:
    init_db()
    created_at = _now()
    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO conversations (
                title, skill_url, raw_url, model, api_mode, base_url,
                reasoning_effort, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                title,
                skill_url,
                raw_url,
                model,
                api_mode,
                base_url,
                reasoning_effort,
                created_at,
                created_at,
            ),
        )
        return int(cursor.lastrowid)


def list_conversations(limit: int = 50) -> list[ConversationRecord]:
    init_db()
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT c.id, c.title, c.skill_url, c.raw_url, c.model, c.api_mode,
                   c.base_url, c.reasoning_effort, c.created_at, c.updated_at,
                   COUNT(m.id) AS message_count
            FROM conversations c
            LEFT JOIN messages m ON m.conversation_id = c.id
            GROUP BY c.id
            ORDER BY c.updated_at DESC, c.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [_row_to_conversation(row) for row in rows]


def get_conversation(conversation_id: int) -> ConversationRecord | None:
    init_db()
    with _connect() as connection:
        row = connection.execute(
            """
            SELECT c.id, c.title, c.skill_url, c.raw_url, c.model, c.api_mode,
                   c.base_url, c.reasoning_effort, c.created_at, c.updated_at,
                   COUNT(m.id) AS message_count
            FROM conversations c
            LEFT JOIN messages m ON m.conversation_id = c.id
            WHERE c.id = ?
            GROUP BY c.id
            """,
            (conversation_id,),
        ).fetchone()
    return _row_to_conversation(row) if row else None


def rename_conversation(conversation_id: int, title: str) -> None:
    init_db()
    with _connect() as connection:
        connection.execute(
            """
            UPDATE conversations
            SET title = ?, updated_at = ?
            WHERE id = ?
            """,
            (title, _now(), conversation_id),
        )


def touch_conversation(conversation_id: int) -> None:
    with _connect() as connection:
        connection.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (_now(), conversation_id),
        )


def create_message(
    *,
    conversation_id: int,
    role: str,
    content: str,
    latency_ms: int = 0,
    error_message: str = "",
) -> int:
    init_db()
    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO messages (
                conversation_id, role, content, latency_ms, error_message, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (conversation_id, role, content, latency_ms, error_message, _now()),
        )
    touch_conversation(conversation_id)
    return int(cursor.lastrowid)


def list_messages(conversation_id: int) -> list[MessageRecord]:
    init_db()
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT id, conversation_id, role, content, latency_ms, error_message, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id ASC
            """,
            (conversation_id,),
        ).fetchall()
    return [_row_to_message(row) for row in rows]


def create_test_suite(*, name: str, skill_url: str, description: str = "") -> int:
    init_db()
    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO test_suites (name, skill_url, description, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (name, skill_url, description, _now()),
        )
        return int(cursor.lastrowid)


def list_test_suites() -> list[TestSuiteRecord]:
    init_db()
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT s.id, s.name, s.skill_url, s.description, s.created_at,
                   COUNT(c.id) AS case_count
            FROM test_suites s
            LEFT JOIN test_cases c ON c.suite_id = s.id
            GROUP BY s.id
            ORDER BY s.id DESC
            """
        ).fetchall()
    return [_row_to_test_suite(row) for row in rows]


def get_test_suite(suite_id: int) -> TestSuiteRecord | None:
    init_db()
    with _connect() as connection:
        row = connection.execute(
            """
            SELECT s.id, s.name, s.skill_url, s.description, s.created_at,
                   COUNT(c.id) AS case_count
            FROM test_suites s
            LEFT JOIN test_cases c ON c.suite_id = s.id
            WHERE s.id = ?
            GROUP BY s.id
            """,
            (suite_id,),
        ).fetchone()
    return _row_to_test_suite(row) if row else None


def create_test_case(
    *,
    suite_id: int,
    title: str,
    question: str,
    expected_behavior: str,
    scoring_focus: str,
) -> int:
    init_db()
    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO test_cases (
                suite_id, title, question, expected_behavior, scoring_focus, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (suite_id, title, question, expected_behavior, scoring_focus, _now()),
        )
        return int(cursor.lastrowid)


def list_test_cases(suite_id: int) -> list[TestCaseRecord]:
    init_db()
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT id, suite_id, title, question, expected_behavior,
                   scoring_focus, created_at
            FROM test_cases
            WHERE suite_id = ?
            ORDER BY id ASC
            """,
            (suite_id,),
        ).fetchall()
    return [_row_to_test_case(row) for row in rows]


def create_eval_run(*, suite_id: int, status: str = "running") -> int:
    init_db()
    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO eval_runs (suite_id, status, average_score, error_message, created_at)
            VALUES (?, ?, 0, '', ?)
            """,
            (suite_id, status, _now()),
        )
        return int(cursor.lastrowid)


def update_eval_run(
    eval_run_id: int,
    *,
    status: str,
    average_score: float = 0,
    error_message: str = "",
) -> None:
    init_db()
    with _connect() as connection:
        connection.execute(
            """
            UPDATE eval_runs
            SET status = ?, average_score = ?, error_message = ?
            WHERE id = ?
            """,
            (status, average_score, error_message, eval_run_id),
        )


def list_eval_runs(limit: int = 20) -> list[EvalRunRecord]:
    init_db()
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT r.id, r.suite_id, r.status, r.average_score, r.error_message,
                   r.created_at, s.name AS suite_name
            FROM eval_runs r
            JOIN test_suites s ON s.id = r.suite_id
            ORDER BY r.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [_row_to_eval_run(row) for row in rows]


def get_eval_run(eval_run_id: int) -> EvalRunRecord | None:
    init_db()
    with _connect() as connection:
        row = connection.execute(
            """
            SELECT r.id, r.suite_id, r.status, r.average_score, r.error_message,
                   r.created_at, s.name AS suite_name
            FROM eval_runs r
            JOIN test_suites s ON s.id = r.suite_id
            WHERE r.id = ?
            """,
            (eval_run_id,),
        ).fetchone()
    return _row_to_eval_run(row) if row else None


def create_eval_result(
    *,
    eval_run_id: int,
    test_case_id: int,
    variant: str = "enhanced",
    agent_answer: str = "",
    role_adherence: int = 0,
    constraint_adherence: int = 0,
    task_completion: int = 0,
    factual_safety: int = 0,
    format_quality: int = 0,
    source_usage: int = 0,
    overall: float = 0,
    judge_comment: str = "",
    search_used: int = 0,
    search_results: str = "",
    rag_used: int = 0,
    rag_results: str = "",
    error_message: str = "",
) -> int:
    init_db()
    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO eval_results (
                eval_run_id, test_case_id, variant, agent_answer, role_adherence,
                constraint_adherence, task_completion, factual_safety,
                format_quality, source_usage, overall, judge_comment,
                search_used, search_results, rag_used, rag_results,
                error_message, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                eval_run_id,
                test_case_id,
                variant,
                agent_answer,
                role_adherence,
                constraint_adherence,
                task_completion,
                factual_safety,
                format_quality,
                source_usage,
                overall,
                judge_comment,
                search_used,
                search_results,
                rag_used,
                rag_results,
                error_message,
                _now(),
            ),
        )
        return int(cursor.lastrowid)


def list_eval_results(eval_run_id: int) -> list[EvalResultRecord]:
    init_db()
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT r.id, r.eval_run_id, r.test_case_id, r.variant, r.agent_answer,
                   r.role_adherence, r.constraint_adherence, r.task_completion,
                   r.factual_safety, r.format_quality, r.source_usage, r.overall,
                   r.judge_comment, r.search_used, r.search_results,
                   r.rag_used, r.rag_results, r.error_message, r.created_at,
                   c.title AS test_case_title, c.question
            FROM eval_results r
            JOIN test_cases c ON c.id = r.test_case_id
            WHERE r.eval_run_id = ?
            ORDER BY r.test_case_id ASC, r.variant ASC, r.id ASC
            """,
            (eval_run_id,),
        ).fetchall()
    return [_row_to_eval_result(row) for row in rows]


def create_rag_document(
    *,
    filename: str,
    file_type: str,
    file_path: str,
    status: str = "indexing",
    error_message: str = "",
    chunk_count: int = 0,
) -> int:
    init_db()
    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO rag_documents (
                filename, file_type, file_path, status, error_message,
                chunk_count, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                filename,
                file_type,
                file_path,
                status,
                error_message,
                chunk_count,
                _now(),
            ),
        )
        return int(cursor.lastrowid)


def update_rag_document(
    document_id: int,
    *,
    status: str,
    error_message: str = "",
    chunk_count: int = 0,
) -> None:
    init_db()
    with _connect() as connection:
        connection.execute(
            """
            UPDATE rag_documents
            SET status = ?, error_message = ?, chunk_count = ?
            WHERE id = ?
            """,
            (status, error_message, chunk_count, document_id),
        )


def list_rag_documents(limit: int = 100) -> list[RagDocumentRecord]:
    init_db()
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT id, filename, file_type, file_path, status, error_message,
                   chunk_count, created_at
            FROM rag_documents
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [_row_to_rag_document(row) for row in rows]


def get_rag_document(document_id: int) -> RagDocumentRecord | None:
    init_db()
    with _connect() as connection:
        row = connection.execute(
            """
            SELECT id, filename, file_type, file_path, status, error_message,
                   chunk_count, created_at
            FROM rag_documents
            WHERE id = ?
            """,
            (document_id,),
        ).fetchone()
    return _row_to_rag_document(row) if row else None


def delete_rag_document(document_id: int) -> None:
    init_db()
    with _connect() as connection:
        connection.execute("DELETE FROM rag_chunks WHERE document_id = ?", (document_id,))
        connection.execute("DELETE FROM rag_documents WHERE id = ?", (document_id,))


def replace_rag_chunks(
    document_id: int,
    chunks: list[dict[str, object]],
) -> None:
    init_db()
    with _connect() as connection:
        connection.execute("DELETE FROM rag_chunks WHERE document_id = ?", (document_id,))
        connection.executemany(
            """
            INSERT INTO rag_chunks (
                document_id, chunk_index, page_number, content,
                embedding, embedding_dim, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    document_id,
                    int(chunk["chunk_index"]),
                    int(chunk.get("page_number") or 0),
                    str(chunk["content"]),
                    str(chunk["embedding"]),
                    int(chunk["embedding_dim"]),
                    _now(),
                )
                for chunk in chunks
            ],
        )


def list_rag_chunks(document_id: int | None = None) -> list[RagChunkRecord]:
    init_db()
    params: tuple[object, ...] = ()
    where_clause = ""
    if document_id is not None:
        where_clause = "WHERE c.document_id = ?"
        params = (document_id,)
    with _connect() as connection:
        rows = connection.execute(
            f"""
            SELECT c.id, c.document_id, c.chunk_index, c.page_number,
                   c.content, c.embedding, c.embedding_dim, c.created_at,
                   d.filename AS document_filename
            FROM rag_chunks c
            JOIN rag_documents d ON d.id = c.document_id
            {where_clause}
            ORDER BY c.document_id DESC, c.chunk_index ASC
            """,
            params,
        ).fetchall()
    return [_row_to_rag_chunk(row) for row in rows]


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
        search_used=row["search_used"],
        search_results=row["search_results"],
        rag_used=row["rag_used"],
        rag_results=row["rag_results"],
        status=row["status"],
        error_message=row["error_message"],
        latency_ms=row["latency_ms"],
        created_at=row["created_at"],
    )


def _row_to_conversation(row: sqlite3.Row) -> ConversationRecord:
    return ConversationRecord(
        id=row["id"],
        title=row["title"],
        skill_url=row["skill_url"],
        raw_url=row["raw_url"],
        model=row["model"],
        api_mode=row["api_mode"],
        base_url=row["base_url"],
        reasoning_effort=row["reasoning_effort"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        message_count=row["message_count"],
    )


def _row_to_message(row: sqlite3.Row) -> MessageRecord:
    return MessageRecord(
        id=row["id"],
        conversation_id=row["conversation_id"],
        role=row["role"],
        content=row["content"],
        latency_ms=row["latency_ms"],
        error_message=row["error_message"],
        created_at=row["created_at"],
    )


def _row_to_test_suite(row: sqlite3.Row) -> TestSuiteRecord:
    return TestSuiteRecord(
        id=row["id"],
        name=row["name"],
        skill_url=row["skill_url"],
        description=row["description"],
        created_at=row["created_at"],
        case_count=row["case_count"],
    )


def _row_to_test_case(row: sqlite3.Row) -> TestCaseRecord:
    return TestCaseRecord(
        id=row["id"],
        suite_id=row["suite_id"],
        title=row["title"],
        question=row["question"],
        expected_behavior=row["expected_behavior"],
        scoring_focus=row["scoring_focus"],
        created_at=row["created_at"],
    )


def _row_to_eval_run(row: sqlite3.Row) -> EvalRunRecord:
    return EvalRunRecord(
        id=row["id"],
        suite_id=row["suite_id"],
        status=row["status"],
        average_score=row["average_score"],
        error_message=row["error_message"],
        created_at=row["created_at"],
        suite_name=row["suite_name"],
    )


def _row_to_eval_result(row: sqlite3.Row) -> EvalResultRecord:
    return EvalResultRecord(
        id=row["id"],
        eval_run_id=row["eval_run_id"],
        test_case_id=row["test_case_id"],
        variant=row["variant"],
        agent_answer=row["agent_answer"],
        role_adherence=row["role_adherence"],
        constraint_adherence=row["constraint_adherence"],
        task_completion=row["task_completion"],
        factual_safety=row["factual_safety"],
        format_quality=row["format_quality"],
        source_usage=row["source_usage"],
        overall=row["overall"],
        judge_comment=row["judge_comment"],
        search_used=row["search_used"],
        search_results=row["search_results"],
        rag_used=row["rag_used"],
        rag_results=row["rag_results"],
        error_message=row["error_message"],
        created_at=row["created_at"],
        test_case_title=row["test_case_title"],
        question=row["question"],
    )


def _row_to_rag_document(row: sqlite3.Row) -> RagDocumentRecord:
    return RagDocumentRecord(
        id=row["id"],
        filename=row["filename"],
        file_type=row["file_type"],
        file_path=row["file_path"],
        status=row["status"],
        error_message=row["error_message"],
        chunk_count=row["chunk_count"],
        created_at=row["created_at"],
    )


def _row_to_rag_chunk(row: sqlite3.Row) -> RagChunkRecord:
    return RagChunkRecord(
        id=row["id"],
        document_id=row["document_id"],
        chunk_index=row["chunk_index"],
        page_number=row["page_number"],
        content=row["content"],
        embedding=row["embedding"],
        embedding_dim=row["embedding_dim"],
        created_at=row["created_at"],
        document_filename=row["document_filename"],
    )
