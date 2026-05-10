from dataclasses import dataclass
from typing import Final
from urllib.parse import urlparse

import httpx


DEFAULT_SKILL_URL: Final[str] = (
    "https://github.com/alchaincyf/zhangxuefeng-skill/blob/main/SKILL.md"
)


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
            raise SkillLoadError("第一版只支持指向 SKILL.md 的链接。")
        return cleaned_url

    if parsed.netloc != "github.com":
        raise SkillLoadError("第一版只支持 github.com 或 raw.githubusercontent.com 链接。")

    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 5 or parts[2] != "blob":
        raise SkillLoadError("请输入 GitHub 文件页链接，例如 /owner/repo/blob/main/SKILL.md。")

    owner, repo, _, branch = parts[:4]
    file_path = "/".join(parts[4:])
    if not file_path.endswith("SKILL.md"):
        raise SkillLoadError("第一版只支持指向 SKILL.md 的链接。")

    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{file_path}"


async def load_skill(skill_url: str) -> LoadedSkill:
    raw_url = github_url_to_raw_url(skill_url)

    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            response = await client.get(raw_url)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code
        raise SkillLoadError(f"拉取 SKILL.md 失败，GitHub 返回状态码 {status_code}。") from exc
    except httpx.HTTPError as exc:
        raise SkillLoadError(f"拉取 SKILL.md 失败：{exc}") from exc

    content = response.text.strip()
    if not content:
        raise SkillLoadError("SKILL.md 内容为空，请检查链接是否正确。")

    return LoadedSkill(skill_url=skill_url.strip(), raw_url=raw_url, content=content)
