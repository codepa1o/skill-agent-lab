import pytest

from app.skill_loader import SkillLoadError, github_url_to_raw_url


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
