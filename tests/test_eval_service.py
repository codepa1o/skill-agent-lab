from app import eval_service, run_store
from app.agent_runner import AgentResult, JudgeResult
from app.rag_service import RagBundle
from app.skill_loader import LoadedSkill


def test_run_suite_evaluation_success(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(run_store, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(eval_service, "get_test_suite", run_store.get_test_suite)
    monkeypatch.setattr(eval_service, "list_test_cases", run_store.list_test_cases)
    monkeypatch.setattr(eval_service, "create_eval_run", run_store.create_eval_run)
    monkeypatch.setattr(eval_service, "update_eval_run", run_store.update_eval_run)
    monkeypatch.setattr(eval_service, "create_eval_result", run_store.create_eval_result)
    monkeypatch.setattr(eval_service, "create_run", run_store.create_run)

    async def fake_load_skill(skill_url):
        return LoadedSkill(skill_url=skill_url, raw_url="raw", content="skill")

    def fake_run_skill_agent(skill_content, question):
        if "本地资料库" in skill_content or "实时搜索" in skill_content:
            return AgentResult(answer="增强回答", model="fake-model", latency_ms=10)
        return AgentResult(answer="基础回答", model="fake-model", latency_ms=10)

    def fake_run_judge_agent(**kwargs):
        is_enhanced = "增强回答" in kwargs["agent_answer"]
        return JudgeResult(
            role_adherence=4,
            constraint_adherence=4,
            task_completion=5,
            factual_safety=4,
            format_quality=5,
            source_usage=5 if is_enhanced else 2,
            overall=4.6 if is_enhanced else 3.8,
            judge_comment="不错" if is_enhanced else "缺少来源",
            model="judge",
            latency_ms=20,
        )

    monkeypatch.setattr(eval_service, "load_skill", fake_load_skill)
    monkeypatch.setattr(eval_service, "run_skill_agent", fake_run_skill_agent)
    monkeypatch.setattr(eval_service, "run_judge_agent", fake_run_judge_agent)
    async def fake_search_if_needed(question):
        from app.search_service import SearchBundle, SearchResult

        return SearchBundle(
            needed=True,
            query=question,
            results=[SearchResult(title="来源", url="https://example.com", snippet="摘要")],
        )

    monkeypatch.setattr(eval_service, "search_if_needed", fake_search_if_needed)
    monkeypatch.setattr(
        eval_service,
        "retrieve_local_context",
        lambda question: RagBundle(used=False, query=question, results=[]),
    )

    suite_id = run_store.create_test_suite(
        name="基础评测集",
        skill_url="https://github.com/owner/repo/blob/main/SKILL.md",
    )
    run_store.create_test_case(
        suite_id=suite_id,
        title="寒暄",
        question="你好",
        expected_behavior="自然回应",
        scoring_focus="角色和格式",
    )

    import asyncio

    eval_run_id = asyncio.run(eval_service.run_suite_evaluation(suite_id))
    eval_run = run_store.get_eval_run(eval_run_id)
    results = run_store.list_eval_results(eval_run_id)

    assert eval_run is not None
    assert eval_run.status == "completed"
    assert eval_run.average_score == 4.6
    assert [result.variant for result in results] == ["baseline", "enhanced"]
    assert results[0].overall == 3.8
    assert results[1].overall == 4.6
    assert results[1].source_usage == 5
    assert results[1].search_used == 1
