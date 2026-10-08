from fastapi.testclient import TestClient

from app.agent_runner import AgentResult
from app.agent_runner import AgentRunError
from app import main, run_store
from app.skill_loader import LoadedSkill


def test_empty_question_redirects_with_error(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(run_store, "DB_PATH", tmp_path / "test.db")

    with TestClient(main.app) as client:
        response = client.post(
            "/runs",
            data={
                "skill_url": "https://github.com/owner/repo/blob/main/SKILL.md",
                "question": "   ",
            },
            follow_redirects=False,
        )

    assert response.status_code == 303
    assert "error=" in response.headers["location"]


def test_invalid_skill_url_saves_failed_run(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(run_store, "DB_PATH", tmp_path / "test.db")

    with TestClient(main.app) as client:
        response = client.post(
            "/runs",
            data={
                "skill_url": "https://example.com/SKILL.md",
                "question": "测试问题",
            },
            follow_redirects=False,
        )

    run_id = int(response.headers["location"].split("run_id=")[1])
    run = run_store.get_run(run_id)

    assert run is not None
    assert run.status == "failed"
    assert "github.com" in run.error_message


def test_create_conversation_redirects_with_autostart_and_saves_user_message(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(run_store, "DB_PATH", tmp_path / "test.db")

    called = False

    async def fake_load_skill(skill_url):
        raise AssertionError("create conversation should not fetch SKILL.md")

    monkeypatch.setattr(main, "load_skill", fake_load_skill)
    async def fake_answer_conversation_message(conversation_id, skill_content, raw_url):
        nonlocal called
        called = True

    monkeypatch.setattr(main, "answer_conversation_message", fake_answer_conversation_message)
    monkeypatch.setattr(main, "ensure_default_test_suites", lambda: None)

    with TestClient(main.app) as client:
        response = client.post(
            "/conversations",
            data={
                "skill_url": "https://github.com/owner/repo/blob/main/SKILL.md",
                "title": "测试对话",
                "question": "第一条问题",
            },
            follow_redirects=False,
        )

    assert response.status_code == 303
    assert response.headers["location"].endswith("?autostart=1")

    conversation_id = int(response.headers["location"].split("/conversations/")[1].split("?")[0])
    conversation = run_store.get_conversation(conversation_id)
    messages = run_store.list_messages(conversation_id)
    runs = run_store.list_runs()

    assert conversation is not None
    assert conversation.title == "测试对话"
    assert [message.role for message in messages] == ["user"]
    assert messages[0].content == "第一条问题"
    assert runs == []
    assert called is False


def test_create_run_saves_rag_results(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(run_store, "DB_PATH", tmp_path / "test.db")

    async def fake_load_skill(skill_url):
        return LoadedSkill(
            skill_url=skill_url,
            raw_url="https://raw.githubusercontent.com/owner/repo/main/SKILL.md",
            content="测试 skill",
        )

    def fake_retrieve_local_context(question):
        from app.rag_service import RagBundle, RagResult

        return RagBundle(
            used=True,
            query=question,
            results=[
                RagResult(
                    document_id=1,
                    document_filename="local.md",
                    chunk_index=0,
                    page_number=0,
                    content="本地资料",
                    score=0.9,
                )
            ],
        )

    async def fake_search_if_needed(question):
        from app.search_service import SearchBundle

        return SearchBundle(needed=False, query="", results=[])

    def fake_run_skill_agent(skill_content, question):
        assert "本地资料库检索结果" in skill_content
        return AgentResult(answer="回答", model="fake-model", latency_ms=12)

    monkeypatch.setattr(main, "load_skill", fake_load_skill)
    monkeypatch.setattr(main, "retrieve_local_context", fake_retrieve_local_context)
    monkeypatch.setattr(main, "search_if_needed", fake_search_if_needed)
    monkeypatch.setattr(main, "run_skill_agent", fake_run_skill_agent)
    monkeypatch.setattr(main, "ensure_default_test_suites", lambda: None)

    with TestClient(main.app) as client:
        response = client.post(
            "/runs",
            data={
                "skill_url": "https://github.com/owner/repo/blob/main/SKILL.md",
                "question": "本地资料问题",
            },
            follow_redirects=False,
        )

    run_id = int(response.headers["location"].split("run_id=")[1])
    run = run_store.get_run(run_id)

    assert run is not None
    assert run.rag_used == 1
    assert "local.md" in run.rag_results
    assert "本地资料来源" in run.answer


def test_conversation_history_for_model_keeps_recent_messages(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(run_store, "DB_PATH", tmp_path / "test.db")

    conversation_id = run_store.create_conversation(
        title="测试对话",
        skill_url="https://github.com/owner/repo/blob/main/SKILL.md",
        raw_url="https://raw.githubusercontent.com/owner/repo/main/SKILL.md",
        model="fake-model",
        api_mode="responses",
        base_url="",
        reasoning_effort="medium",
    )
    run_store.create_message(
        conversation_id=conversation_id,
        role="user",
        content="第一条问题",
    )

    messages = run_store.list_messages(conversation_id)
    history = main.conversation_history_for_model(messages)

    assert history == [{"role": "user", "content": "第一条问题"}]


def test_stream_conversation_message_returns_events_and_records_messages(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(run_store, "DB_PATH", tmp_path / "test.db")

    async def fake_load_skill(skill_url):
        return LoadedSkill(
            skill_url=skill_url,
            raw_url="https://raw.githubusercontent.com/owner/repo/main/SKILL.md",
            content="fake skill",
        )

    async def fake_search_if_needed(question):
        from app.search_service import SearchBundle

        return SearchBundle(needed=False, query="", results=[])

    def fake_retrieve_local_context(question):
        from app.rag_service import RagBundle

        return RagBundle(used=False, query=question, results=[])

    async def fake_stream_skill_agent_chat(skill_content, messages):
        yield "hello"
        yield " world"

    monkeypatch.setattr(main, "load_skill", fake_load_skill)
    monkeypatch.setattr(main, "search_if_needed", fake_search_if_needed)
    monkeypatch.setattr(main, "retrieve_local_context", fake_retrieve_local_context)
    monkeypatch.setattr(main, "stream_skill_agent_chat", fake_stream_skill_agent_chat)
    monkeypatch.setattr(main, "maybe_compress_conversation_memory", lambda conversation_id: None)
    monkeypatch.setattr(main, "ensure_default_test_suites", lambda: None)

    conversation_id = run_store.create_conversation(
        title="stream test",
        skill_url="https://github.com/owner/repo/blob/main/SKILL.md",
        raw_url="https://raw.githubusercontent.com/owner/repo/main/SKILL.md",
        model="fake-model",
        api_mode="responses",
        base_url="",
        reasoning_effort="medium",
    )

    with TestClient(main.app) as client:
        with client.stream(
            "POST",
            f"/conversations/{conversation_id}/messages/stream",
            data={"question": "hi"},
        ) as response:
            body = "".join(response.iter_text())

    messages = run_store.list_messages(conversation_id)
    runs = run_store.list_runs()

    assert "event: message" in body
    assert "event: delta" in body
    assert "event: done" in body
    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[1].status == "complete"
    assert "hello world" in messages[1].content
    assert runs[0].status == "success"


def test_stream_conversation_message_can_reuse_existing_user_message(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(run_store, "DB_PATH", tmp_path / "test.db")

    async def fake_load_skill(skill_url):
        return LoadedSkill(
            skill_url=skill_url,
            raw_url="https://raw.githubusercontent.com/owner/repo/main/SKILL.md",
            content="fake skill",
        )

    async def fake_search_if_needed(question):
        from app.search_service import SearchBundle

        return SearchBundle(needed=False, query="", results=[])

    def fake_retrieve_local_context(question):
        from app.rag_service import RagBundle

        return RagBundle(used=False, query=question, results=[])

    async def fake_stream_skill_agent_chat(skill_content, messages):
        yield "hello"
        yield " world"

    monkeypatch.setattr(main, "load_skill", fake_load_skill)
    monkeypatch.setattr(main, "search_if_needed", fake_search_if_needed)
    monkeypatch.setattr(main, "retrieve_local_context", fake_retrieve_local_context)
    monkeypatch.setattr(main, "stream_skill_agent_chat", fake_stream_skill_agent_chat)
    monkeypatch.setattr(main, "maybe_compress_conversation_memory", lambda conversation_id: None)
    monkeypatch.setattr(main, "ensure_default_test_suites", lambda: None)

    conversation_id = run_store.create_conversation(
        title="stream reuse test",
        skill_url="https://github.com/owner/repo/blob/main/SKILL.md",
        raw_url="https://raw.githubusercontent.com/owner/repo/main/SKILL.md",
        model="fake-model",
        api_mode="responses",
        base_url="",
        reasoning_effort="medium",
    )
    user_message_id = run_store.create_message(
        conversation_id=conversation_id,
        role="user",
        content="hi",
    )

    with TestClient(main.app) as client:
        with client.stream(
            "POST",
            f"/conversations/{conversation_id}/messages/stream",
            data={"message_id": str(user_message_id)},
        ) as response:
            body = "".join(response.iter_text())

    messages = run_store.list_messages(conversation_id)

    assert "event: message" in body
    assert "event: delta" in body
    assert "event: done" in body
    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[0].id == user_message_id
    assert messages[1].status == "complete"


def test_stream_existing_user_message_replays_completed_assistant_without_duplicate(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(run_store, "DB_PATH", tmp_path / "test.db")

    async def fail_load_skill(skill_url):
        raise AssertionError("completed answer should be replayed without a new model call")

    monkeypatch.setattr(main, "load_skill", fail_load_skill)
    monkeypatch.setattr(main, "ensure_default_test_suites", lambda: None)

    conversation_id = run_store.create_conversation(
        title="stream replay test",
        skill_url="https://github.com/owner/repo/blob/main/SKILL.md",
        raw_url="https://raw.githubusercontent.com/owner/repo/main/SKILL.md",
        model="fake-model",
        api_mode="responses",
        base_url="",
        reasoning_effort="medium",
    )
    user_message_id = run_store.create_message(
        conversation_id=conversation_id,
        role="user",
        content="hi",
    )
    assistant_message_id = run_store.create_message(
        conversation_id=conversation_id,
        role="assistant",
        content="already answered",
        latency_ms=123,
    )

    with TestClient(main.app) as client:
        with client.stream(
            "POST",
            f"/conversations/{conversation_id}/messages/stream",
            data={"message_id": str(user_message_id)},
        ) as response:
            body = "".join(response.iter_text())

    messages = run_store.list_messages(conversation_id)
    runs = run_store.list_runs()

    assert "event: delta" in body
    assert "already answered" in body
    assert f'"message_id": {assistant_message_id}' in body
    assert [message.role for message in messages] == ["user", "assistant"]
    assert runs == []


def test_stream_conversation_error_is_returned_and_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(run_store, "DB_PATH", tmp_path / "test.db")

    async def fake_load_skill(skill_url):
        return LoadedSkill(
            skill_url=skill_url,
            raw_url="https://raw.githubusercontent.com/owner/repo/main/SKILL.md",
            content="测试 skill",
        )

    async def fake_stream_skill_agent_chat(skill_content, messages):
        if False:
            yield ""
        raise AgentRunError("模型没有返回可展示的文本结果。")

    monkeypatch.setattr(main, "load_skill", fake_load_skill)
    monkeypatch.setattr(main, "stream_skill_agent_chat", fake_stream_skill_agent_chat)
    monkeypatch.setattr(main, "ensure_default_test_suites", lambda: None)

    with TestClient(main.app) as client:
        response = client.post(
            "/conversations",
            data={
                "skill_url": "https://github.com/owner/repo/blob/main/SKILL.md",
                "title": "测试对话",
                "question": "你好",
            },
            follow_redirects=False,
        )
        location = response.headers["location"]
        conversation_id = int(location.split("/conversations/")[1].split("?")[0])
        user_message_id = run_store.list_messages(conversation_id)[0].id

        with client.stream(
            "POST",
            f"/conversations/{conversation_id}/messages/stream",
            data={"message_id": str(user_message_id)},
        ) as stream_response:
            body = "".join(stream_response.iter_text())

    conversation = run_store.get_conversation(conversation_id)
    messages = run_store.list_messages(conversation_id)
    runs = run_store.list_runs()

    assert location.endswith("?autostart=1")
    assert "event: error" in body
    assert conversation is not None
    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[1].error_message
    assert runs[0].status == "failed"


def test_conversation_detail_autostarts_unanswered_latest_user_message(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(run_store, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(main, "ensure_default_test_suites", lambda: None)

    conversation_id = run_store.create_conversation(
        title="needs answer",
        skill_url="https://github.com/owner/repo/blob/main/SKILL.md",
        raw_url="https://raw.githubusercontent.com/owner/repo/main/SKILL.md",
        model="fake-model",
        api_mode="responses",
        base_url="",
        reasoning_effort="medium",
    )
    user_message_id = run_store.create_message(
        conversation_id=conversation_id,
        role="user",
        content="hi",
    )

    with TestClient(main.app) as client:
        response = client.get(f"/conversations/{conversation_id}")

    assert response.status_code == 200
    assert 'data-autostart="1"' in response.text
    assert f'data-latest-user-message-id="{user_message_id}"' in response.text


def test_conversation_detail_does_not_autostart_after_completed_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(run_store, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(main, "ensure_default_test_suites", lambda: None)

    conversation_id = run_store.create_conversation(
        title="answered",
        skill_url="https://github.com/owner/repo/blob/main/SKILL.md",
        raw_url="https://raw.githubusercontent.com/owner/repo/main/SKILL.md",
        model="fake-model",
        api_mode="responses",
        base_url="",
        reasoning_effort="medium",
    )
    run_store.create_message(
        conversation_id=conversation_id,
        role="user",
        content="hi",
    )
    run_store.create_message(
        conversation_id=conversation_id,
        role="assistant",
        content="done",
    )

    with TestClient(main.app) as client:
        response = client.get(f"/conversations/{conversation_id}?autostart=1")

    assert response.status_code == 200
    assert 'data-autostart="0"' in response.text


def test_rename_conversation(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(run_store, "DB_PATH", tmp_path / "test.db")
    conversation_id = run_store.create_conversation(
        title="旧标题",
        skill_url="https://github.com/owner/repo/blob/main/SKILL.md",
        raw_url="https://raw.githubusercontent.com/owner/repo/main/SKILL.md",
        model="fake-model",
        api_mode="responses",
        base_url="",
        reasoning_effort="medium",
    )

    with TestClient(main.app) as client:
        response = client.post(
            f"/conversations/{conversation_id}/rename",
            data={"title": "新标题"},
            follow_redirects=False,
        )

    conversation = run_store.get_conversation(conversation_id)

    assert response.status_code == 303
    assert conversation is not None
    assert conversation.title == "新标题"


def test_upload_knowledge_document_creates_job(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(run_store, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(main, "ensure_default_test_suites", lambda: None)

    async def fake_save_uploaded_document(file):
        return run_store.create_rag_document(
            filename="note.txt",
            file_type="txt",
            file_path="data/uploads/note.txt",
            status="queued",
        )

    monkeypatch.setattr(main, "save_uploaded_document", fake_save_uploaded_document)

    with TestClient(main.app) as client:
        response = client.post(
            "/knowledge/upload",
            files={"file": ("note.txt", b"hello", "text/plain")},
            follow_redirects=False,
        )

    document_id = int(response.headers["location"].split("/knowledge/")[1])
    job = run_store.get_latest_job_for_target("index_document", "document_id", document_id)

    assert response.status_code == 303
    assert job is not None
    assert job.status == "queued"


def test_run_eval_suite_creates_job(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(run_store, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(main, "ensure_default_test_suites", lambda: None)

    suite_id = run_store.create_test_suite(
        name="测试集",
        skill_url="https://github.com/owner/repo/blob/main/SKILL.md",
    )

    with TestClient(main.app) as client:
        response = client.post(
            f"/evals/suites/{suite_id}/run",
            follow_redirects=False,
        )

    eval_run_id = int(response.headers["location"].split("/evals/runs/")[1])
    job = run_store.get_latest_job_for_target("run_eval_suite", "eval_run_id", eval_run_id)

    assert response.status_code == 303
    assert job is not None
    assert job.payload["suite_id"] == suite_id


def test_job_status_endpoint_returns_json(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(run_store, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(main, "ensure_default_test_suites", lambda: None)
    monkeypatch.setenv("JOB_WORKER_ENABLED", "false")
    job_id = run_store.create_job(
        job_type="index_document",
        payload={"document_id": 1},
    )

    with TestClient(main.app) as client:
        response = client.get(f"/jobs/{job_id}")

    assert response.status_code == 200
    assert response.json()["status"] == "queued"
