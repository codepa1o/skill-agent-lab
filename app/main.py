from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.agent_runner import (
    AgentRunError,
    get_api_mode,
    get_base_url,
    get_model_name,
    get_reasoning_effort,
    run_skill_agent,
)
from app.run_store import create_run, get_run, init_db, list_runs
from app.skill_loader import DEFAULT_SKILL_URL, SkillLoadError, load_skill


BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

app = FastAPI(title="Skill Agent Lab")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, run_id: int | None = None, error: str = ""):
    selected_run = get_run(run_id) if run_id else None
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "default_skill_url": DEFAULT_SKILL_URL,
            "runs": list_runs(),
            "selected_run": selected_run,
            "error": error,
            "default_model": get_model_name(),
            "api_mode": get_api_mode(),
            "base_url": get_base_url(),
            "reasoning_effort": get_reasoning_effort(),
        },
    )


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
        result = run_skill_agent(loaded_skill.content, cleaned_question)
        run_id = create_run(
            skill_url=loaded_skill.skill_url,
            raw_url=loaded_skill.raw_url,
            question=cleaned_question,
            answer=result.answer,
            model=result.model,
            api_mode=get_api_mode(),
            base_url=get_base_url(),
            reasoning_effort=get_reasoning_effort(),
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
    if "skill.md" in lowered or "github" in lowered:
        return message
    return message
