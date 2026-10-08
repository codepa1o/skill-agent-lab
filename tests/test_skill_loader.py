import asyncio

import pytest

from app import skill_loader
from app.skill_loader import SkillLoadError, github_url_to_raw_url, load_skill


def test_github_blob_url_to_raw_url():
    raw_url = github_url_to_raw_url(
        "https://github.com/alchaincyf/zhangxuefeng-skill/blob/main/SKILL.md"
    )

    assert raw_url == (
        "https://raw.githubusercontent.com/"
        "alchaincyf/zhangxuefeng-skill/main/SKILL.md"
    )


def test_non_github_url_is_rejected():
    with pytest.raises(SkillLoadError):
        github_url_to_raw_url("https://example.com/SKILL.md")


def test_load_skill_uses_github_api_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(skill_loader, "SKILL_CACHE_DIR", tmp_path)
    calls = []

    async def fake_fetch(url, accept="text/plain"):
        calls.append((url, accept))
        if url.startswith("https://raw.githubusercontent.com/"):
            raise SkillLoadError("raw failed")
        return "fallback skill"

    monkeypatch.setattr(skill_loader, "_fetch_skill_text", fake_fetch)

    loaded = asyncio.run(load_skill("https://github.com/owner/repo/blob/main/SKILL.md"))

    assert loaded.content == "fallback skill"
    assert calls == [
        ("https://raw.githubusercontent.com/owner/repo/main/SKILL.md", "text/plain"),
        (
            "https://api.github.com/repos/owner/repo/contents/SKILL.md?ref=main",
            "application/vnd.github.raw",
        ),
    ]


def test_load_skill_reads_cache_before_network(monkeypatch, tmp_path):
    monkeypatch.setattr(skill_loader, "SKILL_CACHE_DIR", tmp_path)
    raw_url = "https://raw.githubusercontent.com/owner/repo/main/SKILL.md"
    skill_loader._write_cached_skill(raw_url, "cached skill")

    async def fail_fetch(url, accept="text/plain"):
        raise AssertionError("network should not be called")

    monkeypatch.setattr(skill_loader, "_fetch_skill_text", fail_fetch)

    loaded = asyncio.run(load_skill("https://github.com/owner/repo/blob/main/SKILL.md"))

    assert loaded.content == "cached skill"
