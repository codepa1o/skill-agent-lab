import sqlite3

from app import run_store


def test_create_run_saves_runtime_config(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(run_store, "DB_PATH", tmp_path / "test.db")

    run_id = run_store.create_run(
        skill_url="https://github.com/owner/repo/blob/main/SKILL.md",
        raw_url="https://raw.githubusercontent.com/owner/repo/main/SKILL.md",
        question="测试问题",
        answer="测试回答",
        model="gpt-5.5",
        api_mode="responses",
        base_url="https://api.freemodel.dev",
        reasoning_effort="medium",
        status="success",
        latency_ms=123,
    )

    run = run_store.get_run(run_id)

    assert run is not None
    assert run.api_mode == "responses"
    assert run.base_url == "https://api.freemodel.dev"
    assert run.reasoning_effort == "medium"


def test_init_db_migrates_existing_runs_table(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(run_store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(run_store, "DB_PATH", db_path)

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                skill_url TEXT NOT NULL,
                raw_url TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                model TEXT NOT NULL,
                status TEXT NOT NULL,
                error_message TEXT NOT NULL,
                latency_ms INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

    run_store.init_db()

    with sqlite3.connect(db_path) as connection:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(runs)").fetchall()
        }

    assert "api_mode" in columns
    assert "base_url" in columns
    assert "reasoning_effort" in columns
