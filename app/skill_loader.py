from dataclasses import dataclass
from typing import Final
from pathlib import Path
from urllib.parse import quote, urlparse

import httpx


DEFAULT_SKILL_URL: Final[str] = (
    "https://github.com/alchaincyf/zhangxuefeng-skill/blob/main/SKILL.md"
)
SKILL_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "skill_cache"
SKILL_REQUEST_TIMEOUT = httpx.Timeout(20.0, connect=10.0, read=20.0, write=10.0, pool=10.0)


class SkillLoadError(ValueError):
    """Raised when a GitHub Skill URL cannot be converted or fetched."""


@dataclass(frozen=True)
class LoadedSkill:
    skill_url: str
    raw_url: str
    content: str


def github_url_to_raw_url(skill_url: str) -> str:
    cleaned_url = skill_url.strip()
    if not cleaned_url:
        raise SkillLoadError("请输入 GitHub Skill URL。")

    parsed = urlparse(cleaned_url)
    if parsed.scheme != "https":
        raise SkillLoadError("请使用 https 开头的 GitHub 文件链接。")

    if parsed.netloc == "raw.githubusercontent.com":
        if not parsed.path.endswith("/SKILL.md"):
            raise SkillLoadError("只支持指向 SKILL.md 的链接。")
        return cleaned_url

    if parsed.netloc != "github.com":
        raise SkillLoadError("只支持 github.com 或 raw.githubusercontent.com 链接。")

    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 5 or parts[2] != "blob":
        raise SkillLoadError("请输入 GitHub 文件页链接，例如 /owner/repo/blob/main/SKILL.md。")

    owner, repo, _, branch = parts[:4]
    file_path = "/".join(parts[4:])
    if not file_path.endswith("SKILL.md"):
        raise SkillLoadError("只支持指向 SKILL.md 的链接。")

    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{file_path}"


async def load_skill(skill_url: str) -> LoadedSkill:
    raw_url = github_url_to_raw_url(skill_url)
    cached_content = _read_cached_skill(raw_url)
    if cached_content:
        return LoadedSkill(skill_url=skill_url.strip(), raw_url=raw_url, content=cached_content)

    try:
        content = await _fetch_skill_text(raw_url)
    except SkillLoadError as primary_error:
        fallback_url = _github_api_contents_url(skill_url)
        if not fallback_url:
            raise primary_error from None
        try:
            content = await _fetch_skill_text(fallback_url, accept="application/vnd.github.raw")
        except SkillLoadError:
            raise primary_error from None

    if not content:
        raise SkillLoadError("SKILL.md 内容为空，请检查链接是否正确。")

    _write_cached_skill(raw_url, content)
    return LoadedSkill(skill_url=skill_url.strip(), raw_url=raw_url, content=content)


async def _fetch_skill_text(url: str, accept: str = "text/plain") -> str:
    headers = {"Accept": accept, "User-Agent": "Skill-Agent-Lab/1.0"}
    try:
        async with httpx.AsyncClient(
            timeout=SKILL_REQUEST_TIMEOUT,
            follow_redirects=True,
            trust_env=False,
        ) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response.text.strip()
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code
        raise SkillLoadError(f"拉取 SKILL.md 失败，远程服务返回状态码 {status_code}。") from exc
    except httpx.HTTPError as exc:
        detail = str(exc).strip() or exc.__class__.__name__
        raise SkillLoadError(f"拉取 SKILL.md 失败：{exc.__class__.__name__}: {detail}") from exc


def _github_api_contents_url(skill_url: str) -> str:
    parsed = urlparse(skill_url.strip())
    if parsed.netloc != "github.com":
        return ""

    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 5 or parts[2] != "blob":
        return ""

    owner, repo, _, branch = parts[:4]
    file_path = "/".join(parts[4:])
    encoded_path = quote(file_path, safe="/")
    encoded_ref = quote(branch, safe="")
    return f"https://api.github.com/repos/{owner}/{repo}/contents/{encoded_path}?ref={encoded_ref}"


def _cache_path(raw_url: str) -> Path:
    safe_name = quote(raw_url, safe="")
    return SKILL_CACHE_DIR / f"{safe_name}.txt"


def _read_cached_skill(raw_url: str) -> str:
    cache_file = _cache_path(raw_url)
    if not cache_file.exists():
        return ""
    try:
        return cache_file.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _write_cached_skill(raw_url: str, content: str) -> None:
    try:
        SKILL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path(raw_url).write_text(content, encoding="utf-8")
    except OSError:
        pass
