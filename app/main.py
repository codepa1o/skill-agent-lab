from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.agent_runner import (
    AgentRunError,
    get_api_mode,
    get_base_url,
    get_model_name,
    get_reasoning_effort,
    run_skill_agent_chat,
    run_skill_agent,
)
from app.default_data import ensure_default_test_suites
from app.diagnostics import build_diagnostics
from app.eval_service import run_suite_evaluation
from app.job_worker import start_worker, stop_worker
from app.run_store import (
    create_conversation,
    create_eval_run,
    create_job,
    create_message,
    create_run,
    get_conversation,
    get_run,
    get_eval_run,
    get_job,
    get_rag_document,
    get_latest_job_for_target,
    get_test_suite,
    init_db,
    list_eval_results,
    list_eval_runs,
    list_conversations,
    list_messages,
    list_rag_chunks,
    list_rag_documents,
    list_runs,
    list_test_cases,
    list_test_suites,
    rename_conversation,
    create_test_case,
    create_test_suite,
    update_rag_document,
)
from app.rag_service import (
    RagError,
    append_rag_sources,
    augment_skill_with_rag,
    build_rag_context,
    remove_document,
    retrieve_local_context,
    serialize_rag_results,
    save_uploaded_document,
)
from app.search_service import (
    append_sources,
    augment_skill_with_search,
    build_search_context,
    search_if_needed,
    serialize_results,
)
from app.skill_loader import DEFAULT_SKILL_URL, SkillLoadError, load_skill


BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

app = FastAPI(title="Skill Agent Lab")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


@app.on_event("startup")
async def on_startup() -> None:
    init_db()
    ensure_default_test_suites()
    start_worker()


@app.on_event("shutdown")
async def on_shutdown() -> None:
    await stop_worker()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, run_id: int | None = None, error: str = ""):
    selected_run = get_run(run_id) if run_id else None
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "default_skill_url": DEFAULT_SKILL_URL,
            "conversations": list_conversations(),
            "runs": list_runs(),
            "selected_run": selected_run,
            "error": error,
            "default_model": get_model_name(),
            "api_mode": get_api_mode(),
            "base_url": get_base_url(),
            "reasoning_effort": get_reasoning_effort(),
        },
    )


@app.post("/conversations")
async def create_conversation_endpoint(
    skill_url: str = Form(...),
    question: str = Form(...),
    title: str = Form(""),
):
    cleaned_skill_url = skill_url.strip()
    cleaned_question = question.strip()
    if not cleaned_question:
        return RedirectResponse(
            url="/?error=请输入第一条对话消息后再创建。",
            status_code=303,
        )

    try:
        loaded_skill = await load_skill(cleaned_skill_url)
    except SkillLoadError as exc:
        create_run(
            skill_url=cleaned_skill_url,
            question=cleaned_question,
            status="failed",
            error_message=friendly_error_message(str(exc)),
            api_mode=get_api_mode(),
            base_url=get_base_url(),
            reasoning_effort=get_reasoning_effort(),
        )
        return RedirectResponse(
            url=f"/?error={quote(friendly_error_message(str(exc)))}",
            status_code=303,
        )

    conversation_id = create_conversation(
        title=normalize_title(title, cleaned_question),
        skill_url=loaded_skill.skill_url,
        raw_url=loaded_skill.raw_url,
        model=get_model_name(),
        api_mode=get_api_mode(),
        base_url=get_base_url(),
        reasoning_effort=get_reasoning_effort(),
    )
    create_message(
        conversation_id=conversation_id,
        role="user",
        content=cleaned_question,
    )

    try:
        await answer_conversation_message(conversation_id, loaded_skill.content, loaded_skill.raw_url)
    except AgentRunError as exc:
        return RedirectResponse(
            url=f"/conversations/{conversation_id}?error={quote(friendly_error_message(str(exc)))}",
            status_code=303,
        )
    return RedirectResponse(url=f"/conversations/{conversation_id}", status_code=303)


@app.get("/conversations/{conversation_id}", response_class=HTMLResponse)
async def conversation_detail(
    request: Request,
    conversation_id: int,
    error: str = "",
):
    conversation = get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return templates.TemplateResponse(
        request,
        "conversation.html",
        {
            "conversation": conversation,
            "conversations": list_conversations(),
            "messages": list_messages(conversation_id),
            "error": error,
        },
    )


@app.post("/conversations/{conversation_id}/messages")
async def append_conversation_message(
    conversation_id: int,
    question: str = Form(...),
):
    conversation = get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    cleaned_question = question.strip()
    if not cleaned_question:
        return RedirectResponse(
            url=f"/conversations/{conversation_id}?error={quote('请输入追问内容后再发送。')}",
            status_code=303,
        )

    create_message(
        conversation_id=conversation_id,
        role="user",
        content=cleaned_question,
    )

    try:
        loaded_skill = await load_skill(conversation.skill_url)
        await answer_conversation_message(
            conversation_id,
            loaded_skill.content,
            loaded_skill.raw_url,
        )
    except SkillLoadError as exc:
        error_message = friendly_error_message(str(exc))
        create_message(
            conversation_id=conversation_id,
            role="assistant",
            content="",
            error_message=error_message,
        )
        create_run(
            skill_url=conversation.skill_url,
            raw_url=conversation.raw_url,
            question=cleaned_question,
            status="failed",
            error_message=error_message,
            model=conversation.model,
            api_mode=conversation.api_mode,
            base_url=conversation.base_url,
            reasoning_effort=conversation.reasoning_effort,
        )
        return RedirectResponse(
            url=f"/conversations/{conversation_id}?error={quote(error_message)}",
            status_code=303,
        )
    except AgentRunError as exc:
        return RedirectResponse(
            url=f"/conversations/{conversation_id}?error={quote(friendly_error_message(str(exc)))}",
            status_code=303,
        )

    return RedirectResponse(url=f"/conversations/{conversation_id}", status_code=303)


@app.post("/conversations/{conversation_id}/rename")
async def rename_conversation_endpoint(
    conversation_id: int,
    title: str = Form(...),
):
    conversation = get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    cleaned_title = title.strip()
    if cleaned_title:
        rename_conversation(conversation_id, cleaned_title[:80])
    return RedirectResponse(url=f"/conversations/{conversation_id}", status_code=303)


@app.post("/runs")
async def create_agent_run(
    skill_url: str = Form(...),
    question: str = Form(...),
):
    cleaned_skill_url = skill_url.strip()
    cleaned_question = question.strip()

    if not cleaned_question:
        return RedirectResponse(
            url="/?error=请输入测试问题后再提交。",
            status_code=303,
        )

    try:
        loaded_skill = await load_skill(cleaned_skill_url)
        rag_bundle = retrieve_local_context(cleaned_question)
        search_bundle = await search_if_needed(cleaned_question)
        skill_content = augment_skill_with_search(
            augment_skill_with_rag(
                loaded_skill.content,
                build_rag_context(rag_bundle),
            ),
            build_search_context(search_bundle),
        )
        result = run_skill_agent(skill_content, cleaned_question)
        answer = append_sources(
            append_rag_sources(result.answer, rag_bundle),
            search_bundle,
        )
        run_id = create_run(
            skill_url=loaded_skill.skill_url,
            raw_url=loaded_skill.raw_url,
            question=cleaned_question,
            answer=answer,
            model=result.model,
            api_mode=get_api_mode(),
            base_url=get_base_url(),
            reasoning_effort=get_reasoning_effort(),
            search_used=1 if search_bundle.needed else 0,
            search_results=serialize_results(search_bundle),
            rag_used=1 if rag_bundle.used else 0,
            rag_results=serialize_rag_results(rag_bundle),
            status="success",
            latency_ms=result.latency_ms,
        )
    except SkillLoadError as exc:
        run_id = create_run(
            skill_url=cleaned_skill_url,
            question=cleaned_question,
            status="failed",
            error_message=friendly_error_message(str(exc)),
            api_mode=get_api_mode(),
            base_url=get_base_url(),
            reasoning_effort=get_reasoning_effort(),
        )
    except AgentRunError as exc:
        run_id = create_run(
            skill_url=cleaned_skill_url,
            question=cleaned_question,
            status="failed",
            error_message=friendly_error_message(str(exc)),
            api_mode=get_api_mode(),
            base_url=get_base_url(),
            reasoning_effort=get_reasoning_effort(),
        )
    except Exception as exc:
        run_id = create_run(
            skill_url=cleaned_skill_url,
            question=cleaned_question,
            status="failed",
            error_message=friendly_error_message(f"系统错误：{exc}"),
            api_mode=get_api_mode(),
            base_url=get_base_url(),
            reasoning_effort=get_reasoning_effort(),
        )

    return RedirectResponse(url=f"/?run_id={run_id}", status_code=303)


@app.get("/runs", response_class=HTMLResponse)
async def runs_history(request: Request):
    return templates.TemplateResponse(
        request,
        "history.html",
        {
            "runs": list_runs(100),
        },
    )


@app.post("/runs/{run_id}/rerun")
async def rerun_agent_run(run_id: int):
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return await create_agent_run(skill_url=run.skill_url, question=run.question)


@app.get("/runs/{run_id}", response_class=HTMLResponse)
async def run_detail(request: Request, run_id: int):
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return templates.TemplateResponse(
        request,
        "detail.html",
        {
            "run": run,
        },
    )


@app.get("/settings/diagnostics", response_class=HTMLResponse)
async def diagnostics_page(request: Request):
    return templates.TemplateResponse(
        request,
        "diagnostics.html",
        {
            "items": build_diagnostics(),
        },
    )


@app.get("/knowledge", response_class=HTMLResponse)
async def knowledge_index(request: Request, error: str = ""):
    return templates.TemplateResponse(
        request,
        "knowledge.html",
        {
            "documents": list_rag_documents(),
            "error": error,
        },
    )


@app.post("/knowledge/upload")
async def upload_knowledge_document(file: UploadFile = File(...)):
    try:
        document_id = await save_uploaded_document(file)
        create_job(job_type="index_document", payload={"document_id": document_id})
    except RagError as exc:
        return RedirectResponse(
            url=f"/knowledge?error={quote(friendly_error_message(str(exc)))}",
            status_code=303,
        )
    return RedirectResponse(url=f"/knowledge/{document_id}", status_code=303)


@app.get("/knowledge/{document_id}", response_class=HTMLResponse)
async def knowledge_detail(request: Request, document_id: int):
    document = get_rag_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return templates.TemplateResponse(
        request,
        "knowledge_detail.html",
        {
            "document": document,
            "chunks": list_rag_chunks(document_id),
            "job": get_latest_job_for_target("index_document", "document_id", document_id),
        },
    )


@app.post("/knowledge/{document_id}/reindex")
async def reindex_knowledge_document(document_id: int):
    document = get_rag_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    update_rag_document(document_id, status="queued", error_message="", chunk_count=document.chunk_count)
    create_job(job_type="index_document", payload={"document_id": document_id})
    return RedirectResponse(url=f"/knowledge/{document_id}", status_code=303)


@app.post("/knowledge/{document_id}/delete")
async def delete_knowledge_document(document_id: int):
    try:
        remove_document(document_id)
    except RagError as exc:
        return RedirectResponse(
            url=f"/knowledge?error={quote(friendly_error_message(str(exc)))}",
            status_code=303,
        )
    return RedirectResponse(url="/knowledge", status_code=303)


def friendly_error_message(message: str) -> str:
    lowered = message.lower()
    if "504" in lowered or "gateway time-out" in lowered or "timeout" in lowered:
        return (
            "中转站或上游模型超时了。复杂问题更容易触发这个错误，建议稍后重试，"
            "或把 OPENAI_REASONING_EFFORT 调成 medium/low。"
        )
    if "404" in lowered or "not found" in lowered:
        return (
            "接口地址或调用模式不匹配。请检查 OPENAI_BASE_URL 是否为中转站给出的 base_url，"
            "以及 OPENAI_API_MODE 是否和中转站 wire_api 一致。"
        )
    if "insufficient_quota" in lowered or "quota" in lowered:
        return "接口额度不足或账号不可用。请检查中转站余额、套餐和模型权限。"
    if "openai_api_key" in lowered or "api key" in lowered:
        return "API Key 未配置或不可用。请检查 .env 中的 OPENAI_API_KEY。"
    if "dashscope_api_key" in lowered:
        return "未配置 DASHSCOPE_API_KEY，无法使用阿里云百炼 Embedding。请在 .env 中填写。"
    if "embedding" in lowered or "向量" in lowered:
        return message
    if "资料" in lowered or "pdf" in lowered:
        return message
    if "skill.md" in lowered or "github" in lowered:
        return message
    return message


@app.get("/evals", response_class=HTMLResponse)
async def evals_index(request: Request, error: str = ""):
    ensure_default_test_suites()
    return templates.TemplateResponse(
        request,
        "evals.html",
        {
            "suites": list_test_suites(),
            "eval_runs": list_eval_runs(),
            "error": error,
        },
    )


@app.post("/evals/suites")
async def create_suite_endpoint(
    name: str = Form(...),
    skill_url: str = Form(...),
    description: str = Form(""),
):
    cleaned_name = name.strip()
    cleaned_skill_url = skill_url.strip()
    if not cleaned_name or not cleaned_skill_url:
        return RedirectResponse(url="/evals?error=请填写测试集名称和 Skill URL。", status_code=303)
    suite_id = create_test_suite(
        name=cleaned_name,
        skill_url=cleaned_skill_url,
        description=description.strip(),
    )
    return RedirectResponse(url=f"/evals/suites/{suite_id}", status_code=303)


@app.get("/evals/suites/{suite_id}", response_class=HTMLResponse)
async def suite_detail(request: Request, suite_id: int, error: str = ""):
    suite = get_test_suite(suite_id)
    if not suite:
        raise HTTPException(status_code=404, detail="Suite not found")
    return templates.TemplateResponse(
        request,
        "suite.html",
        {
            "suite": suite,
            "cases": list_test_cases(suite_id),
            "error": error,
        },
    )


@app.post("/evals/suites/{suite_id}/cases")
async def create_case_endpoint(
    suite_id: int,
    title: str = Form(...),
    question: str = Form(...),
    expected_behavior: str = Form(...),
    scoring_focus: str = Form(...),
):
    if not get_test_suite(suite_id):
        raise HTTPException(status_code=404, detail="Suite not found")
    if not title.strip() or not question.strip():
        return RedirectResponse(
            url=f"/evals/suites/{suite_id}?error=请填写测试用例标题和问题。",
            status_code=303,
        )
    create_test_case(
        suite_id=suite_id,
        title=title.strip(),
        question=question.strip(),
        expected_behavior=expected_behavior.strip(),
        scoring_focus=scoring_focus.strip(),
    )
    return RedirectResponse(url=f"/evals/suites/{suite_id}", status_code=303)


@app.post("/evals/suites/{suite_id}/run")
async def run_suite_endpoint(suite_id: int):
    if not get_test_suite(suite_id):
        raise HTTPException(status_code=404, detail="Suite not found")
    eval_run_id = create_eval_run(suite_id=suite_id, status="running")
    create_job(
        job_type="run_eval_suite",
        payload={"suite_id": suite_id, "eval_run_id": eval_run_id},
    )
    return RedirectResponse(url=f"/evals/runs/{eval_run_id}", status_code=303)


@app.get("/evals/runs/{eval_run_id}", response_class=HTMLResponse)
async def eval_run_detail(request: Request, eval_run_id: int):
    eval_run = get_eval_run(eval_run_id)
    if not eval_run:
        raise HTTPException(status_code=404, detail="Eval run not found")
    return templates.TemplateResponse(
        request,
        "eval_run.html",
        {
            "eval_run": eval_run,
            "results": list_eval_results(eval_run_id),
            "result_groups": group_eval_results(list_eval_results(eval_run_id)),
            "job": get_latest_job_for_target("run_eval_suite", "eval_run_id", eval_run_id),
        },
    )


@app.get("/jobs/{job_id}")
async def job_status(job_id: int):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(
        {
            "id": job.id,
            "type": job.type,
            "status": job.status,
            "payload": job.payload,
            "result": job.result,
            "error_message": job.error_message,
            "attempt_count": job.attempt_count,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
        }
    )


async def answer_conversation_message(
    conversation_id: int,
    skill_content: str,
    raw_url: str,
) -> None:
    conversation = get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = list_messages(conversation_id)
    history = conversation_history_for_model(messages)
    latest_user_message = next(
        (message.content for message in reversed(messages) if message.role == "user"),
        "",
    )

    try:
        rag_bundle = retrieve_local_context(latest_user_message)
        search_bundle = await search_if_needed(latest_user_message)
        augmented_skill = augment_skill_with_search(
            augment_skill_with_rag(
                skill_content,
                build_rag_context(rag_bundle),
            ),
            build_search_context(search_bundle),
        )
        result = run_skill_agent_chat(augmented_skill, history)
        answer = append_sources(
            append_rag_sources(result.answer, rag_bundle),
            search_bundle,
        )
        create_message(
            conversation_id=conversation_id,
            role="assistant",
            content=answer,
            latency_ms=result.latency_ms,
        )
        create_run(
            skill_url=conversation.skill_url,
            raw_url=raw_url,
            question=latest_user_message,
            answer=answer,
            model=result.model,
            api_mode=get_api_mode(),
            base_url=get_base_url(),
            reasoning_effort=get_reasoning_effort(),
            search_used=1 if search_bundle.needed else 0,
            search_results=serialize_results(search_bundle),
            rag_used=1 if rag_bundle.used else 0,
            rag_results=serialize_rag_results(rag_bundle),
            status="success",
            latency_ms=result.latency_ms,
        )
    except AgentRunError as exc:
        error_message = friendly_error_message(str(exc))
        create_message(
            conversation_id=conversation_id,
            role="assistant",
            content="",
            error_message=error_message,
        )
        create_run(
            skill_url=conversation.skill_url,
            raw_url=raw_url,
            question=latest_user_message,
            model=get_model_name(),
            api_mode=get_api_mode(),
            base_url=get_base_url(),
            reasoning_effort=get_reasoning_effort(),
            status="failed",
            error_message=error_message,
        )
        raise AgentRunError(error_message) from exc


def conversation_history_for_model(messages) -> list[dict[str, str]]:
    useful_messages = [
        message
        for message in messages
        if message.role in {"user", "assistant"} and message.content.strip()
    ]
    recent_messages = useful_messages[-20:]
    return [
        {"role": message.role, "content": message.content}
        for message in recent_messages
    ]


def normalize_title(title: str, question: str) -> str:
    cleaned_title = title.strip()
    if cleaned_title:
        return cleaned_title[:80]
    compact_question = " ".join(question.split())
    return compact_question[:28] or "新的 Skill 对话"


def group_eval_results(results):
    groups = []
    by_case: dict[int, dict[str, object]] = {}
    for result in results:
        group = by_case.setdefault(
            result.test_case_id,
            {
                "title": result.test_case_title,
                "question": result.question,
                "baseline": None,
                "enhanced": None,
                "delta": 0,
            },
        )
        group[result.variant] = result
    for group in by_case.values():
        baseline = group.get("baseline")
        enhanced = group.get("enhanced")
        if baseline and enhanced:
            group["delta"] = round(enhanced.overall - baseline.overall, 2)
        groups.append(group)
    return groups
