import os
from dataclasses import dataclass

from app.agent_runner import (
    get_api_mode,
    get_base_url,
    get_model_name,
    get_reasoning_effort,
)
from app.embedding_service import (
    get_dashscope_base_url,
    get_embedding_dimensions,
    get_embedding_model,
)
from app.job_worker import job_worker_enabled, job_poll_interval_seconds
from app.rag_service import (
    get_rag_chunk_overlap,
    get_rag_chunk_size,
    get_rag_min_score,
    get_rag_top_k,
    rag_enabled,
)
from app.search_service import search_enabled


@dataclass(frozen=True)
class DiagnosticItem:
    name: str
    status: str
    message: str


def build_diagnostics() -> list[DiagnosticItem]:
    return [
        _openai_diagnostic(),
        _search_diagnostic(),
        _dashscope_diagnostic(),
        _rag_diagnostic(),
        _job_worker_diagnostic(),
    ]


def _openai_diagnostic() -> DiagnosticItem:
    if not os.getenv("OPENAI_API_KEY", "").strip():
        return DiagnosticItem("OpenAI 兼容模型", "failed", "未配置 OPENAI_API_KEY。")
    return DiagnosticItem(
        "OpenAI 兼容模型",
        "success",
        f"模型 {get_model_name()}，模式 {get_api_mode()}，Base URL {get_base_url() or '官方默认'}，推理强度 {get_reasoning_effort() or '-'}。",
    )


def _search_diagnostic() -> DiagnosticItem:
    if not search_enabled():
        return DiagnosticItem("搜索增强", "warning", "SEARCH_ENABLED=false，搜索增强已关闭。")
    if not os.getenv("TAVILY_API_KEY", "").strip():
        return DiagnosticItem("搜索增强", "warning", "未配置 TAVILY_API_KEY，需要搜索时会提示未启用。")
    return DiagnosticItem("搜索增强", "success", "Tavily API Key 已配置。")


def _dashscope_diagnostic() -> DiagnosticItem:
    if not os.getenv("DASHSCOPE_API_KEY", "").strip():
        return DiagnosticItem("阿里云百炼 Embedding", "failed", "未配置 DASHSCOPE_API_KEY，本地资料无法生成向量。")
    return DiagnosticItem(
        "阿里云百炼 Embedding",
        "success",
        f"{get_embedding_model()}，维度 {get_embedding_dimensions()}，Base URL {get_dashscope_base_url()}。",
    )


def _rag_diagnostic() -> DiagnosticItem:
    if not rag_enabled():
        return DiagnosticItem("本地资料库 RAG", "warning", "RAG_ENABLED=false，本地资料检索已关闭。")
    return DiagnosticItem(
        "本地资料库 RAG",
        "success",
        (
            f"top_k={get_rag_top_k()}，min_score={get_rag_min_score()}，"
            f"chunk_size={get_rag_chunk_size()}，overlap={get_rag_chunk_overlap()}。"
        ),
    )


def _job_worker_diagnostic() -> DiagnosticItem:
    if not job_worker_enabled():
        return DiagnosticItem("后台任务 Worker", "failed", "JOB_WORKER_ENABLED=false，资料索引和评测任务不会自动执行。")
    return DiagnosticItem(
        "后台任务 Worker",
        "success",
        f"已启用，轮询间隔 {job_poll_interval_seconds()} 秒。",
    )
