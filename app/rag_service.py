import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.document_loader import (
    DocumentChunk,
    DocumentLoadError,
    chunk_document,
    load_document_sections,
    validate_document_file,
)
from app.embedding_service import EmbeddingError, create_embeddings
from app.run_store import (
    BASE_DIR,
    create_rag_document,
    delete_rag_document,
    get_rag_document,
    list_rag_chunks,
    replace_rag_chunks,
    update_rag_document,
)


UPLOAD_DIR = BASE_DIR / "data" / "uploads"
EMBEDDING_BATCH_SIZE = 10


class RagError(RuntimeError):
    """Raised when local RAG upload, indexing, or retrieval fails."""


@dataclass(frozen=True)
class RagResult:
    document_id: int
    document_filename: str
    chunk_index: int
    page_number: int
    content: str
    score: float


@dataclass(frozen=True)
class RagBundle:
    used: bool
    query: str
    results: list[RagResult]
    error_message: str = ""


def rag_enabled() -> bool:
    value = os.getenv("RAG_ENABLED", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def get_rag_top_k() -> int:
    value = os.getenv("RAG_TOP_K", "5").strip()
    try:
        return min(10, max(1, int(value)))
    except ValueError:
        return 5


def get_rag_min_score() -> float:
    value = os.getenv("RAG_MIN_SCORE", "0.2").strip()
    try:
        return min(1.0, max(-1.0, float(value)))
    except ValueError:
        return 0.2


def get_rag_chunk_size() -> int:
    value = os.getenv("RAG_CHUNK_SIZE", "800").strip()
    try:
        return min(2000, max(200, int(value)))
    except ValueError:
        return 800


def get_rag_chunk_overlap() -> int:
    value = os.getenv("RAG_CHUNK_OVERLAP", "120").strip()
    try:
        return min(500, max(0, int(value)))
    except ValueError:
        return 120


async def upload_and_index_document(file: UploadFile) -> int:
    document_id = await save_uploaded_document(file)
    index_document(document_id)
    return document_id


async def save_uploaded_document(file: UploadFile) -> int:
    original_filename = Path(file.filename or "").name
    if not original_filename:
        raise RagError("请选择要上传的资料文件。")

    data = await file.read()
    try:
        file_type = validate_document_file(original_filename, len(data))
    except DocumentLoadError as exc:
        raise RagError(str(exc)) from exc
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    stored_name = f"{uuid4().hex}{file_type}"
    stored_path = UPLOAD_DIR / stored_name
    stored_path.write_bytes(data)

    document_id = create_rag_document(
        filename=original_filename,
        file_type=file_type.lstrip("."),
        file_path=_stored_file_path(stored_path),
        status="queued",
    )
    return document_id


def index_document(document_id: int) -> None:
    document = get_rag_document(document_id)
    if not document:
        raise RagError("资料不存在或已被删除。")
    path = Path(document.file_path)
    if not path.is_absolute():
        path = BASE_DIR / document.file_path
    update_rag_document(document_id, status="indexing", error_message="", chunk_count=0)
    try:
        sections = load_document_sections(path)
        chunks = chunk_document(
            sections,
            chunk_size=get_rag_chunk_size(),
            chunk_overlap=get_rag_chunk_overlap(),
        )
        indexed_chunks = _embed_chunks(chunks)
        replace_rag_chunks(document_id, indexed_chunks)
        update_rag_document(
            document_id,
            status="ready",
            chunk_count=len(indexed_chunks),
        )
    except (DocumentLoadError, EmbeddingError, RagError) as exc:
        update_rag_document(
            document_id,
            status="failed",
            error_message=str(exc),
        )
        raise RagError(str(exc)) from exc
    except Exception as exc:
        update_rag_document(
            document_id,
            status="failed",
            error_message=f"资料索引失败：{exc}",
        )
        raise RagError(f"资料索引失败：{exc}") from exc


def remove_document(document_id: int) -> None:
    document = get_rag_document(document_id)
    if not document:
        raise RagError("资料不存在或已被删除。")
    path = Path(document.file_path)
    if not path.is_absolute():
        path = BASE_DIR / document.file_path
    delete_rag_document(document_id)
    if path.exists() and path.is_file():
        path.unlink()


def retrieve_local_context(query: str) -> RagBundle:
    cleaned_query = query.strip()
    if not cleaned_query:
        return RagBundle(used=False, query="", results=[])
    if not rag_enabled():
        return RagBundle(
            used=False,
            query=cleaned_query,
            results=[],
            error_message="本地资料库 RAG 已关闭。",
        )

    chunks = list_rag_chunks()
    if not chunks:
        return RagBundle(used=False, query=cleaned_query, results=[])

    try:
        query_embedding = create_embeddings([cleaned_query])[0]
    except (EmbeddingError, IndexError) as exc:
        return RagBundle(
            used=False,
            query=cleaned_query,
            results=[],
            error_message=str(exc),
        )

    scored_results = []
    min_score = get_rag_min_score()
    for chunk in chunks:
        try:
            chunk_embedding = json.loads(chunk.embedding)
        except json.JSONDecodeError:
            continue
        score = cosine_similarity(query_embedding, chunk_embedding)
        if score >= min_score:
            scored_results.append(
                RagResult(
                    document_id=chunk.document_id,
                    document_filename=chunk.document_filename,
                    chunk_index=chunk.chunk_index,
                    page_number=chunk.page_number,
                    content=chunk.content,
                    score=score,
                )
            )

    scored_results.sort(key=lambda result: result.score, reverse=True)
    results = scored_results[: get_rag_top_k()]
    return RagBundle(used=bool(results), query=cleaned_query, results=results)


def build_rag_context(bundle: RagBundle) -> str:
    if not bundle.results:
        if bundle.error_message:
            return f"【本地资料库状态】{bundle.error_message}"
        return ""

    lines = [
        "【本地资料库检索结果】",
        "请优先基于以下本地资料回答；如果资料不足，请明确说明需要进一步核验。",
    ]
    for index, result in enumerate(bundle.results, start=1):
        lines.append(f"[L{index}] {result.document_filename}{_page_label(result)}")
        lines.append(f"相似度: {result.score:.3f}")
        lines.append(result.content)
    return "\n".join(lines)


def append_rag_sources(answer: str, bundle: RagBundle) -> str:
    if bundle.error_message and not bundle.results:
        return f"{answer.rstrip()}\n\n【本地资料库】{bundle.error_message}".strip()
    if not bundle.results:
        return answer

    lines = [answer.rstrip(), "", "【本地资料来源】"]
    for index, result in enumerate(bundle.results, start=1):
        lines.append(
            f"[L{index}] {result.document_filename}{_page_label(result)} "
            f"chunk#{result.chunk_index + 1} score={result.score:.3f}"
        )
    return "\n".join(lines).strip()


def serialize_rag_results(bundle: RagBundle) -> str:
    if bundle.error_message and not bundle.results:
        return bundle.error_message
    return "\n".join(
        (
            f"[L{index}] {result.document_filename}{_page_label(result)} "
            f"chunk#{result.chunk_index + 1} score={result.score:.3f}\n"
            f"{result.content}"
        ).strip()
        for index, result in enumerate(bundle.results, start=1)
    )


def augment_skill_with_rag(skill_content: str, rag_context: str) -> str:
    if not rag_context:
        return skill_content
    return f"{skill_content}\n\n{rag_context}"


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _embed_chunks(chunks: list[DocumentChunk]) -> list[dict[str, object]]:
    indexed_chunks = []
    for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch = chunks[start : start + EMBEDDING_BATCH_SIZE]
        embeddings = create_embeddings([chunk.content for chunk in batch])
        for chunk, embedding in zip(batch, embeddings):
            indexed_chunks.append(
                {
                    "chunk_index": chunk.chunk_index,
                    "page_number": chunk.page_number,
                    "content": chunk.content,
                    "embedding": json.dumps(embedding, ensure_ascii=False),
                    "embedding_dim": len(embedding),
                }
            )
    if len(indexed_chunks) != len(chunks):
        raise RagError("资料切片向量化数量不完整。")
    return indexed_chunks


def _stored_file_path(path: Path) -> str:
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def _page_label(result: RagResult) -> str:
    return f" 第 {result.page_number} 页" if result.page_number else ""
