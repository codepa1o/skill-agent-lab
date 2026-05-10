import os
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI, OpenAIError


DEFAULT_MODEL = "gpt-5.2"
DEFAULT_API_MODE = "chat_completions"
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_DELAY_SECONDS = 2
DEFAULT_TIMEOUT_SECONDS = 120


class AgentRunError(RuntimeError):
    """Raised when the model call fails."""


@dataclass(frozen=True)
class AgentResult:
    answer: str
    model: str
    latency_ms: int


def get_model_name() -> str:
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def get_api_mode() -> str:
    return os.getenv("OPENAI_API_MODE", DEFAULT_API_MODE).strip() or DEFAULT_API_MODE


def get_base_url() -> str:
    return os.getenv("OPENAI_BASE_URL", "").strip()


def get_reasoning_effort() -> str:
    return os.getenv("OPENAI_REASONING_EFFORT", "").strip()


def should_disable_response_storage() -> bool:
    value = os.getenv("OPENAI_DISABLE_RESPONSE_STORAGE", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def get_max_output_tokens() -> int | None:
    value = os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "").strip()
    if not value:
        return None
    try:
        return max(256, int(value))
    except ValueError:
        return None


def get_timeout_seconds() -> float:
    value = os.getenv("OPENAI_TIMEOUT_SECONDS", "").strip()
    if not value:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        return max(10.0, float(value))
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS


def _build_instructions(skill_content: str) -> str:
    return f"""
你正在运行一个用于学习和测试的 Skill Agent。

请严格遵守下面的 SKILL.md 内容，把它当作本次任务的角色说明、工作流程和行为约束。

安全边界：
- 这是基于 Skill 的角色模拟与测试，不代表真实人物、机构或官方服务。
- 不要声称自己是真实公众人物本人。
- 不要给出绝对化升学、就业、投资或人生保证。
- 如果用户问题需要最新事实，而当前系统没有提供搜索工具，请明确说明信息可能需要进一步核验。
- 输出应直接回答用户测试问题，并尽量体现 Skill 的方法论。
- 控制回答长度，优先给清晰结论、理由和下一步建议，避免无限展开。

以下是 SKILL.md：

{skill_content}
""".strip()


def run_skill_agent(skill_content: str, question: str) -> AgentResult:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise AgentRunError("未配置 OPENAI_API_KEY，请先在 .env 中填写中转站或 OpenAI API Key。")

    model = get_model_name()
    api_mode = get_api_mode()
    base_url = get_base_url()
    client = OpenAI(
        api_key=api_key,
        base_url=base_url or None,
        timeout=get_timeout_seconds(),
    )
    instructions = _build_instructions(skill_content)
    started_at = time.perf_counter()

    try:
        answer = _run_with_retries(
            lambda: _run_once(client, api_mode, model, instructions, question),
            base_url,
        )
    except AgentRunError:
        raise
    except Exception as exc:
        raise AgentRunError(f"模型调用发生未知错误：{exc}") from exc

    latency_ms = int((time.perf_counter() - started_at) * 1000)
    if not answer:
        raise AgentRunError(
            "模型没有返回可展示的文本结果。请尝试把 OPENAI_REASONING_EFFORT 改为 medium，"
            "或把 OPENAI_MAX_OUTPUT_TOKENS 调大后重试。"
        )

    return AgentResult(answer=answer, model=model, latency_ms=latency_ms)


def _run_with_retries(call_model, base_url: str) -> str:
    last_error: OpenAIError | None = None
    for attempt in range(DEFAULT_MAX_RETRIES + 1):
        try:
            return call_model()
        except OpenAIError as exc:
            last_error = exc
            if not _is_retryable_error(exc) or attempt >= DEFAULT_MAX_RETRIES:
                raise AgentRunError(_format_openai_error(exc, base_url)) from exc
            time.sleep(DEFAULT_RETRY_DELAY_SECONDS * (attempt + 1))

    raise AgentRunError(_format_openai_error(last_error, base_url))


def _run_once(
    client: OpenAI,
    api_mode: str,
    model: str,
    instructions: str,
    question: str,
) -> str:
    if api_mode == "responses":
        return _run_responses(client, model, instructions, question)
    if api_mode == "chat_completions":
        return _run_chat_completions(client, model, instructions, question)
    raise AgentRunError("OPENAI_API_MODE 只能设置为 chat_completions 或 responses。")


def _is_retryable_error(exc: OpenAIError) -> bool:
    message = str(exc).lower()
    retry_markers = (
        "timeout",
        "time-out",
        "504",
        "502",
        "503",
        "gateway",
        "temporarily unavailable",
        "connection",
    )
    return any(marker in message for marker in retry_markers)


def _extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    dumped: dict[str, Any] | None = None
    if hasattr(response, "model_dump"):
        dumped = response.model_dump()
    elif isinstance(response, dict):
        dumped = response

    if not dumped:
        return ""

    texts: list[str] = []

    def walk(value: Any, parent_key: str = "") -> None:
        if isinstance(value, dict):
            value_type = str(value.get("type", ""))
            text_value = value.get("text")
            if isinstance(text_value, str) and (
                value_type in {"output_text", "message"} or parent_key == "content"
            ):
                texts.append(text_value)
            for key, child in value.items():
                walk(child, key)
        elif isinstance(value, list):
            for child in value:
                walk(child, parent_key)

    walk(dumped)
    return "\n".join(text.strip() for text in texts if text.strip()).strip()


def _run_responses(
    client: OpenAI,
    model: str,
    instructions: str,
    question: str,
) -> str:
    payload = {
        "model": model,
        "instructions": instructions,
        "input": question.strip(),
    }
    reasoning_effort = get_reasoning_effort()
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}
    if should_disable_response_storage():
        payload["store"] = False
    max_output_tokens = get_max_output_tokens()
    if max_output_tokens:
        payload["max_output_tokens"] = max_output_tokens

    response = client.responses.create(**payload)
    return _extract_response_text(response)


def _run_chat_completions(
    client: OpenAI,
    model: str,
    instructions: str,
    question: str,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": question.strip()},
        ],
    )
    message = response.choices[0].message
    return (message.content or "").strip()


def _format_openai_error(exc: OpenAIError, base_url: str) -> str:
    if exc is None:
        return "模型调用失败，请稍后重试。"
    message = str(exc)
    lowered_message = message.lower()
    if "504" in lowered_message or "gateway time-out" in lowered_message:
        return (
            "中转站或上游模型超时（504 Gateway Time-out）。这个问题通常出现在复杂问题、"
            "高 reasoning_effort 或模型排队较久时。系统已经自动重试仍失败；建议把 "
            "OPENAI_REASONING_EFFORT 改为 medium，或稍后再试。"
        )
    if "insufficient_quota" in message or "exceeded your current quota" in message:
        if base_url:
            return (
                "模型中转站返回额度不足或账号不可用。请检查中转站后台余额、套餐、"
                "模型名称是否支持，以及 .env 中的 OPENAI_BASE_URL 是否为中转站地址。"
            )
        return (
            "OpenAI 官方接口返回额度不足。若你使用的是中转站，请在 .env 中配置 "
            "OPENAI_BASE_URL，并将 OPENAI_API_MODE 设置为 chat_completions。"
        )
    return f"OpenAI 兼容接口调用失败：{message}"
