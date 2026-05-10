import asyncio
import os
from contextlib import suppress

from app.eval_service import execute_eval_run
from app.rag_service import index_document
from app.run_store import JobRecord, claim_next_job, complete_job, fail_job, update_eval_run


_worker_task: asyncio.Task | None = None


def job_worker_enabled() -> bool:
    value = os.getenv("JOB_WORKER_ENABLED", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def job_poll_interval_seconds() -> float:
    value = os.getenv("JOB_POLL_INTERVAL_SECONDS", "2").strip()
    try:
        return max(0.2, float(value))
    except ValueError:
        return 2.0


def start_worker() -> None:
    global _worker_task
    if not job_worker_enabled() or _worker_task is not None:
        return
    _worker_task = asyncio.create_task(_worker_loop())


async def stop_worker() -> None:
    global _worker_task
    if _worker_task is None:
        return
    _worker_task.cancel()
    with suppress(asyncio.CancelledError):
        await _worker_task
    _worker_task = None


async def _worker_loop() -> None:
    while True:
        job = claim_next_job()
        if not job:
            await asyncio.sleep(job_poll_interval_seconds())
            continue
        await run_job(job)


async def run_job(job: JobRecord) -> None:
    try:
        if job.type == "index_document":
            document_id = int(job.payload.get("document_id") or 0)
            if not document_id:
                raise ValueError("index_document job 缺少 document_id。")
            index_document(document_id)
            complete_job(job.id, {"document_id": document_id})
            return

        if job.type == "run_eval_suite":
            eval_run_id = int(job.payload.get("eval_run_id") or 0)
            suite_id = int(job.payload.get("suite_id") or 0)
            if not eval_run_id or not suite_id:
                raise ValueError("run_eval_suite job 缺少 suite_id 或 eval_run_id。")
            _run_eval_sync(suite_id, eval_run_id)
            complete_job(job.id, {"eval_run_id": eval_run_id, "suite_id": suite_id})
            return

        raise ValueError(f"未知任务类型：{job.type}")
    except Exception as exc:
        error_message = str(exc)
        if job.type == "run_eval_suite":
            eval_run_id = int(job.payload.get("eval_run_id") or 0)
            if eval_run_id:
                update_eval_run(eval_run_id, status="failed", error_message=error_message)
        fail_job(job.id, error_message)


def _run_eval_sync(suite_id: int, eval_run_id: int) -> None:
    asyncio.run(execute_eval_run(suite_id=suite_id, eval_run_id=eval_run_id))
