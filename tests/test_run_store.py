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
        search_used=1,
        search_results="[1] 来源 - https://example.com",
        rag_used=1,
        rag_results="[L1] 本地资料",
        status="success",
        latency_ms=123,
    )

    run = run_store.get_run(run_id)

    assert run is not None
    assert run.api_mode == "responses"
    assert run.base_url == "https://api.freemodel.dev"
    assert run.reasoning_effort == "medium"
    assert run.search_used == 1
    assert "example.com" in run.search_results
    assert run.rag_used == 1
    assert "本地资料" in run.rag_results


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
    assert "rag_used" in columns
    assert "rag_results" in columns


def test_conversation_and_messages_are_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(run_store, "DB_PATH", tmp_path / "test.db")

    conversation_id = run_store.create_conversation(
        title="浙江计算机志愿咨询",
        skill_url="https://github.com/owner/repo/blob/main/SKILL.md",
        raw_url="https://raw.githubusercontent.com/owner/repo/main/SKILL.md",
        model="gpt-5.5",
        api_mode="responses",
        base_url="https://api.freemodel.dev",
        reasoning_effort="medium",
    )
    run_store.create_message(
        conversation_id=conversation_id,
        role="user",
        content="我是浙江考生，617 分，想学计算机。",
    )
    run_store.create_message(
        conversation_id=conversation_id,
        role="assistant",
        content="可以考虑，但要结合位次。",
        latency_ms=321,
    )

    conversation = run_store.get_conversation(conversation_id)
    messages = run_store.list_messages(conversation_id)

    assert conversation is not None
    assert conversation.message_count == 2
    assert messages[0].role == "user"
    assert messages[1].latency_ms == 321


def test_eval_suite_case_run_and_result_are_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(run_store, "DB_PATH", tmp_path / "test.db")

    suite_id = run_store.create_test_suite(
        name="基础评测集",
        skill_url="https://github.com/owner/repo/blob/main/SKILL.md",
        description="测试描述",
    )
    case_id = run_store.create_test_case(
        suite_id=suite_id,
        title="寒暄",
        question="你好",
        expected_behavior="自然回应",
        scoring_focus="角色和格式",
    )
    eval_run_id = run_store.create_eval_run(suite_id=suite_id)
    run_store.create_eval_result(
        eval_run_id=eval_run_id,
        test_case_id=case_id,
        agent_answer="你好",
        role_adherence=4,
        constraint_adherence=5,
        task_completion=4,
        factual_safety=5,
        format_quality=4,
        source_usage=3,
        overall=4.4,
        judge_comment="整体不错",
        variant="baseline",
    )
    run_store.update_eval_run(eval_run_id, status="completed", average_score=4.4)

    suite = run_store.get_test_suite(suite_id)
    eval_run = run_store.get_eval_run(eval_run_id)
    results = run_store.list_eval_results(eval_run_id)

    assert suite is not None
    assert suite.case_count == 1
    assert eval_run is not None
    assert eval_run.average_score == 4.4
    assert results[0].variant == "baseline"
    assert results[0].source_usage == 3
    assert results[0].judge_comment == "整体不错"


def test_rag_document_and_chunks_are_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(run_store, "DB_PATH", tmp_path / "test.db")

    document_id = run_store.create_rag_document(
        filename="资料.md",
        file_type="md",
        file_path="data/uploads/test.md",
    )
    run_store.replace_rag_chunks(
        document_id,
        [
            {
                "chunk_index": 0,
                "page_number": 0,
                "content": "杭州电子科技大学计算机资料",
                "embedding": "[1, 0, 0]",
                "embedding_dim": 3,
            }
        ],
    )
    run_store.update_rag_document(document_id, status="ready", chunk_count=1)

    document = run_store.get_rag_document(document_id)
    chunks = run_store.list_rag_chunks(document_id)

    assert document is not None
    assert document.status == "ready"
    assert document.chunk_count == 1
    assert chunks[0].document_filename == "资料.md"


def test_job_lifecycle_is_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(run_store, "DB_PATH", tmp_path / "test.db")

    job_id = run_store.create_job(
        job_type="index_document",
        payload={"document_id": 42},
    )
    claimed = run_store.claim_next_job()

    assert claimed is not None
    assert claimed.id == job_id
    assert claimed.status == "running"
    assert claimed.payload["document_id"] == 42

    run_store.complete_job(job_id, {"ok": True})
    completed = run_store.get_job(job_id)

    assert completed is not None
    assert completed.status == "completed"
    assert completed.result["ok"] is True


def test_job_failure_is_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(run_store, "DB_PATH", tmp_path / "test.db")

    job_id = run_store.create_job(
        job_type="run_eval_suite",
        payload={"suite_id": 1},
    )
    claimed = run_store.claim_next_job()
    assert claimed is not None

    run_store.fail_job(job_id, "失败原因")
    failed = run_store.get_job(job_id)

    assert failed is not None
    assert failed.status == "failed"
    assert failed.error_message == "失败原因"
