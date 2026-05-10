import json
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
FALLBACK_MAX_OUTPUT_TOKENS = 4096


class AgentRunError(RuntimeError):
    """Raised when the model call fails."""


@dataclass(frozen=True)
class AgentResult:
    answer: str
    model: str
    latency_ms: int


@dataclass(frozen=True)
class JudgeResult:
    role_adherence: int
    constraint_adherence: int
    task_completion: int
    factual_safety: int
    format_quality: int
    source_usage: int
    overall: float
    judge_comment: str
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


def _build_judge_instructions() -> str:
    return """
你是 Agent Skill 评测员。你需要根据 Skill 内容、测试问题、期望行为、评分重点和 Agent 回答进行评分。

评分规则：
- role_adherence：是否遵守角色，1-5 分。
- constraint_adherence：是否遵守 Skill 约束，1-5 分。
- task_completion：是否完成用户任务，1-5 分。
- factual_safety：是否避免无依据事实、幻觉、绝对化承诺，1-5 分。
- format_quality：输出结构、清晰度和格式质量，1-5 分。
- source_usage：在需要事实依据的问题中，是否合理引用搜索或本地资料来源，1-5 分；如果问题不需要来源，则评价其是否恰当地不滥用来源。
- overall：总分，1-5 分，可以是小数。

只评价 Agent 回答，不评价用户问题本身。
必须只输出 JSON，不要输出 Markdown，不要包裹代码块。
JSON 字段必须包含：
role_adherence, constraint_adherence, task_completion, factual_safety,
format_quality, source_usage, overall, judge_comment
""".strip()


def _build_judge_input(
    *,
    skill_content: str,
    question: str,
    expected_behavior: str,
    scoring_focus: str,
    agent_answer: str,
) -> str:
    return f"""
【SKILL.md】
{skill_content}

【测试问题】
{question}

【期望行为】
{expected_behavior}

【评分重点】
{scoring_focus}

【Agent 回答】
{agent_answer}
""".strip()


def run_skill_agent(skill_content: str, question: str) -> AgentResult:
    return run_skill_agent_chat(
        skill_content,
        [{"role": "user", "content": question.strip()}],
    )


def run_skill_agent_chat(
    skill_content: str,
    messages: list[dict[str, str]],
) -> AgentResult:
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
            lambda: _run_once(client, api_mode, model, instructions, messages),
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


def run_judge_agent(
    *,
    skill_content: str,
    question: str,
    expected_behavior: str,
    scoring_focus: str,
    agent_answer: str,
) -> JudgeResult:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise AgentRunError("未配置 OPENAI_API_KEY，请先在 .env 中填写中转站或 OpenAI API Key。")

    model = os.getenv("OPENAI_REVIEW_MODEL", "").strip() or get_model_name()
    api_mode = get_api_mode()
    base_url = get_base_url()
    client = OpenAI(
        api_key=api_key,
        base_url=base_url or None,
        timeout=get_timeout_seconds(),
    )
    judge_input = _build_judge_input(
        skill_content=skill_content,
        question=question,
        expected_behavior=expected_behavior,
        scoring_focus=scoring_focus,
        agent_answer=agent_answer,
    )
    started_at = time.perf_counter()

    try:
        answer = _run_with_retries(
            lambda: _run_once(
                client,
                api_mode,
                model,
                _build_judge_instructions(),
                [{"role": "user", "content": judge_input}],
            ),
            base_url,
        )
    except AgentRunError:
        raise
    except Exception as exc:
        raise AgentRunError(f"Judge 模型调用发生未知错误：{exc}") from exc

    latency_ms = int((time.perf_counter() - started_at) * 1000)
    data = _parse_judge_json(answer)
    return JudgeResult(
        role_adherence=_score(data.get("role_adherence")),
        constraint_adherence=_score(data.get("constraint_adherence")),
        task_completion=_score(data.get("task_completion")),
        factual_safety=_score(data.get("factual_safety")),
        format_quality=_score(data.get("format_quality")),
        source_usage=_score(data.get("source_usage")),
        overall=_overall_score(data),
        judge_comment=str(data.get("judge_comment") or data.get("comment") or "").strip(),
        model=model,
        latency_ms=latency_ms,
    )


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
    messages: list[dict[str, str]],
) -> str:
    if api_mode == "responses":
        return _run_responses(client, model, instructions, messages)
    if api_mode == "chat_completions":
        return _run_chat_completions(client, model, instructions, messages)
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

    choice_text = _extract_choice_text(dumped)
    if choice_text:
        return choice_text

    def walk(value: Any, parent_key: str = "") -> None:
        if isinstance(value, dict):
            value_type = str(value.get("type", ""))
            text_value = value.get("text")
            normalized_text = _string_from_text_like_value(text_value)
            if normalized_text and _looks_like_answer_text(
                normalized_text,
                value_type,
                parent_key,
            ):
                texts.append(normalized_text)
            content_value = value.get("content")
            if isinstance(content_value, str) and content_value.strip():
                texts.append(content_value)
            elif isinstance(content_value, list) and value_type == "message":
                content_text = _string_from_text_like_value(content_value)
                if content_text:
                    texts.append(content_text)
            for key, child in value.items():
                walk(child, key)
        elif isinstance(value, list):
            for child in value:
                walk(child, parent_key)

    walk(dumped)
    return "\n".join(_unique_nonempty_texts(texts)).strip()


def _extract_choice_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return ""
    texts = []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message") or choice.get("delta")
        if isinstance(message, dict):
            content = _string_from_text_like_value(message.get("content"))
            if content:
                texts.append(content)
        text = _string_from_text_like_value(choice.get("text"))
        if text:
            texts.append(text)
    return "\n".join(_unique_nonempty_texts(texts)).strip()


def _unique_nonempty_texts(texts: list[str]) -> list[str]:
    unique_texts = []
    seen = set()
    for text in texts:
        cleaned = text.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        unique_texts.append(cleaned)
    return unique_texts


def _string_from_text_like_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("value", "text", "content", "output_text"):
            nested = _string_from_text_like_value(value.get(key))
            if nested:
                return nested
    if isinstance(value, list):
        parts = [_string_from_text_like_value(item) for item in value]
        return "\n".join(part for part in parts if part).strip()
    return ""


def _looks_like_answer_text(text: str, value_type: str, parent_key: str) -> bool:
    if not text.strip():
        return False
    if value_type in {"output_text", "message"}:
        return True
    if parent_key in {"content", "message", "output"}:
        return True
    return False


def _run_responses(
    client: OpenAI,
    model: str,
    instructions: str,
    messages: list[dict[str, str]],
) -> str:
    payload = {
        "model": model,
        "instructions": instructions,
        "input": _format_response_transcript(messages),
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
    answer = _extract_response_text(response)
    if answer:
        return answer

    fallback_payload = dict(payload)
    fallback_payload["reasoning"] = {"effort": "low"}
    fallback_payload["max_output_tokens"] = max(
        get_max_output_tokens() or 0,
        FALLBACK_MAX_OUTPUT_TOKENS,
    )
    fallback_response = client.responses.create(**fallback_payload)
    fallback_answer = _extract_response_text(fallback_response)
    if fallback_answer:
        return fallback_answer

    status_message = _response_status_message(fallback_response) or _response_status_message(response)
    if status_message:
        raise AgentRunError(status_message)
    return ""


def _response_status_message(response: Any) -> str:
    dumped = response.model_dump() if hasattr(response, "model_dump") else response
    if not isinstance(dumped, dict):
        return ""
    status = dumped.get("status")
    incomplete_details = dumped.get("incomplete_details") or {}
    reason = ""
    if isinstance(incomplete_details, dict):
        reason = str(incomplete_details.get("reason") or "")
    if status == "incomplete":
        return (
            "模型返回不完整，可能是输出 token 不足或推理消耗过多。"
            f"原因：{reason or '未知'}。请增大 OPENAI_MAX_OUTPUT_TOKENS 或降低 OPENAI_REASONING_EFFORT。"
        )
    if status == "failed":
        error = dumped.get("error") or {}
        return f"模型返回失败状态：{error or '未知错误'}"
    return ""


def _run_chat_completions(
    client: OpenAI,
    model: str,
    instructions: str,
    messages: list[dict[str, str]],
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": instructions}, *messages],
    )
    message = response.choices[0].message
    return (message.content or "").strip()


def _format_response_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    formatted_messages = []
    for message in messages:
        role = message.get("role", "").strip()
        content = message.get("content", "").strip()
        if role in {"user", "assistant"} and content:
            formatted_messages.append({"role": role, "content": content})
    return formatted_messages


def _format_response_transcript(messages: list[dict[str, str]]) -> str:
    lines = [
        "下面是当前对话窗口的历史消息。请结合这些上下文回答最后一个用户问题。",
        "",
    ]
    for message in messages:
        role = message.get("role", "").strip()
        content = message.get("content", "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        display_role = "用户" if role == "user" else "助手"
        lines.append(f"{display_role}：{content}")
    lines.append("")
    lines.append("请只输出本轮助手回答。")
    return "\n".join(lines).strip()


def _parse_judge_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise AgentRunError(f"Judge 未返回合法 JSON：{text}") from exc
    if not isinstance(data, dict):
        raise AgentRunError("Judge JSON 顶层必须是对象。")
    return data


def _score(value: Any) -> int:
    try:
        return min(5, max(1, int(round(float(value)))))
    except (TypeError, ValueError):
        return 1


def _overall_score(data: dict[str, Any]) -> float:
    try:
        return min(5.0, max(1.0, round(float(data.get("overall")), 2)))
    except (TypeError, ValueError):
        scores = [
            _score(data.get("role_adherence")),
            _score(data.get("constraint_adherence")),
            _score(data.get("task_completion")),
            _score(data.get("factual_safety")),
            _score(data.get("format_quality")),
            _score(data.get("source_usage")),
        ]
        return round(sum(scores) / len(scores), 2)


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
