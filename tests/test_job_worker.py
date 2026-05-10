import pytest

from app import job_worker, run_store


@pytest.mark.anyio
async def test_worker_runs_index_document_job(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(run_store, "DB_PATH", tmp_path / "test.db")

    calls = []

    def fake_index_document(document_id):
        calls.append(document_id)

    monkeypatch.setattr(job_worker, "index_document", fake_index_document)
    job_id = run_store.create_job(
        job_type="index_document",
        payload={"document_id": 7},
    )
    job = run_store.claim_next_job()

    assert job is not None
    await job_worker.run_job(job)
    completed = run_store.get_job(job_id)

    assert calls == [7]
    assert completed is not None
    assert completed.status == "completed"
    assert completed.result["document_id"] == 7


@pytest.mark.anyio
async def test_worker_marks_unknown_job_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(run_store, "DB_PATH", tmp_path / "test.db")

    job_id = run_store.create_job(job_type="missing", payload={})
    job = run_store.claim_next_job()

    assert job is not None
    await job_worker.run_job(job)
    failed = run_store.get_job(job_id)

    assert failed is not None
    assert failed.status == "failed"
    assert "未知任务类型" in failed.error_message
