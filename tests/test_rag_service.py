from app import rag_service, run_store
from app.rag_service import (
    RagBundle,
    RagResult,
    append_rag_sources,
    build_rag_context,
    retrieve_local_context,
)


def test_retrieve_local_context_finds_similar_chunk(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(run_store, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(rag_service, "create_embeddings", lambda texts: [[1.0, 0.0, 0.0]])
    monkeypatch.setenv("RAG_MIN_SCORE", "0.5")

    document_id = run_store.create_rag_document(
        filename="local.md",
        file_type="md",
        file_path="data/uploads/local.md",
        status="ready",
        chunk_count=2,
    )
    run_store.replace_rag_chunks(
        document_id,
        [
            {
                "chunk_index": 0,
                "page_number": 0,
                "content": "杭州电子科技大学计算机资料",
                "embedding": "[1, 0, 0]",
                "embedding_dim": 3,
            },
            {
                "chunk_index": 1,
                "page_number": 0,
                "content": "无关资料",
                "embedding": "[0, 1, 0]",
                "embedding_dim": 3,
            },
        ],
    )

    bundle = retrieve_local_context("杭电计算机怎么样")

    assert bundle.used
    assert len(bundle.results) == 1
    assert bundle.results[0].document_filename == "local.md"


def test_rag_context_and_sources_are_rendered():
    bundle = RagBundle(
        used=True,
        query="计算机",
        results=[
            RagResult(
                document_id=1,
                document_filename="local.md",
                chunk_index=0,
                page_number=2,
                content="本地资料正文",
                score=0.88,
            )
        ],
    )

    context = build_rag_context(bundle)
    answer = append_rag_sources("回答正文", bundle)

    assert "本地资料库检索结果" in context
    assert "local.md 第 2 页" in answer


def test_upload_with_missing_embedding_key_saves_failed_document(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(run_store, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(rag_service, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    async def run_upload():
        from starlette.datastructures import UploadFile
        from io import BytesIO

        file = UploadFile(filename="note.txt", file=BytesIO("本地资料正文".encode("utf-8")))
        return await rag_service.upload_and_index_document(file)

    import asyncio

    document_id = asyncio.run(run_upload())
    document = run_store.get_rag_document(document_id)

    assert document is not None
    assert document.status == "failed"
    assert "DASHSCOPE_API_KEY" in document.error_message
