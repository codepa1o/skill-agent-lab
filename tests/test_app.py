from fastapi.testclient import TestClient

from app import main, run_store


def test_empty_question_redirects_with_error(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(run_store, "DB_PATH", tmp_path / "test.db")

    with TestClient(main.app) as client:
        response = client.post(
            "/runs",
            data={
                "skill_url": "https://github.com/owner/repo/blob/main/SKILL.md",
                "question": "   ",
            },
            follow_redirects=False,
        )

    assert response.status_code == 303
    assert "error=" in response.headers["location"]


def test_invalid_skill_url_saves_failed_run(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(run_store, "DB_PATH", tmp_path / "test.db")

    with TestClient(main.app) as client:
        response = client.post(
            "/runs",
            data={
                "skill_url": "https://example.com/SKILL.md",
                "question": "测试问题",
            },
            follow_redirects=False,
        )

    run_id = int(response.headers["location"].split("run_id=")[1])
    run = run_store.get_run(run_id)

    assert run is not None
    assert run.status == "failed"
    assert "github.com" in run.error_message
