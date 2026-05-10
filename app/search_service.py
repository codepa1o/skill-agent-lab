import os
from dataclasses import dataclass

import httpx


SEARCH_KEYWORDS = (
    "学校",
    "大学",
    "学院",
    "院校",
    "专业",
    "就业",
    "分数线",
    "录取",
    "位次",
    "志愿",
    "报考",
    "考生",
    "薪资",
    "行业",
    "城市",
    "计算机",
    "人工智能",
    "电气",
    "医学",
    "法学",
    "金融",
)


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


@dataclass(frozen=True)
class SearchBundle:
    needed: bool
    query: str
    results: list[SearchResult]
    error_message: str = ""


def search_enabled() -> bool:
    value = os.getenv("SEARCH_ENABLED", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def should_search(question: str) -> bool:
    normalized = question.strip().lower()
    if not normalized:
        return False
    return any(keyword.lower() in normalized for keyword in SEARCH_KEYWORDS)


async def search_if_needed(question: str) -> SearchBundle:
    if not should_search(question):
        return SearchBundle(needed=False, query="", results=[])
    if not search_enabled():
        return SearchBundle(
            needed=True,
            query=question,
            results=[],
            error_message="搜索增强已关闭。",
        )

    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        return SearchBundle(
            needed=True,
            query=question,
            results=[],
            error_message="未配置 TAVILY_API_KEY，无法执行实时搜索。",
        )

    max_results = _max_results()
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": question,
                    "search_depth": "basic",
                    "max_results": max_results,
                    "include_answer": False,
                },
            )
            response.raise_for_status()
    except httpx.HTTPError as exc:
        return SearchBundle(
            needed=True,
            query=question,
            results=[],
            error_message=f"搜索 API 调用失败：{exc}",
        )

    payload = response.json()
    results = []
    for item in payload.get("results", [])[:max_results]:
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        snippet = str(item.get("content") or item.get("snippet") or "").strip()
        if title and url:
            results.append(SearchResult(title=title, url=url, snippet=snippet))

    if not results:
        return SearchBundle(
            needed=True,
            query=question,
            results=[],
            error_message="搜索 API 没有返回可用结果。",
        )

    return SearchBundle(needed=True, query=question, results=results)


def build_search_context(bundle: SearchBundle) -> str:
    if not bundle.needed:
        return ""
    if bundle.error_message:
        return f"【搜索增强状态】{bundle.error_message}"

    lines = [
        "【实时搜索资料】",
        "请优先基于以下搜索结果回答；如果资料不足，请明确说明仍需核验。",
    ]
    for index, result in enumerate(bundle.results, start=1):
        lines.append(f"[{index}] {result.title}")
        lines.append(f"URL: {result.url}")
        if result.snippet:
            lines.append(f"摘要: {result.snippet}")
    return "\n".join(lines)


def append_sources(answer: str, bundle: SearchBundle) -> str:
    if not bundle.needed:
        return answer
    if bundle.error_message:
        return f"{answer}\n\n【搜索增强】{bundle.error_message}"
    if not bundle.results:
        return answer

    lines = [answer.rstrip(), "", "【来源】"]
    for index, result in enumerate(bundle.results, start=1):
        lines.append(f"[{index}] {result.title} - {result.url}")
    return "\n".join(lines).strip()


def serialize_results(bundle: SearchBundle) -> str:
    if not bundle.needed:
        return ""
    if bundle.error_message:
        return bundle.error_message
    return "\n".join(
        f"[{index}] {result.title} - {result.url}\n{result.snippet}".strip()
        for index, result in enumerate(bundle.results, start=1)
    )


def augment_skill_with_search(skill_content: str, search_context: str) -> str:
    if not search_context:
        return skill_content
    return f"{skill_content}\n\n{search_context}"


def _max_results() -> int:
    value = os.getenv("SEARCH_MAX_RESULTS", "5").strip()
    try:
        return min(8, max(1, int(value)))
    except ValueError:
        return 5
