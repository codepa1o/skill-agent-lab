import re
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".pdf"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


class DocumentLoadError(RuntimeError):
    """Raised when an uploaded local document cannot be parsed."""


@dataclass(frozen=True)
class DocumentSection:
    text: str
    page_number: int = 0


@dataclass(frozen=True)
class DocumentChunk:
    content: str
    chunk_index: int
    page_number: int = 0


def validate_document_file(filename: str, file_size: int) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise DocumentLoadError("只支持上传 Markdown、TXT 或 PDF 文件。")
    if file_size <= 0:
        raise DocumentLoadError("上传文件为空，请换一个有内容的资料。")
    if file_size > MAX_UPLOAD_BYTES:
        raise DocumentLoadError("上传文件超过 10MB，第一版请先上传更小的资料。")
    return suffix


def load_document_sections(path: Path) -> list[DocumentSection]:
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown", ".txt"}:
        return _load_text_sections(path)
    if suffix == ".pdf":
        return _load_pdf_sections(path)
    raise DocumentLoadError("只支持上传 Markdown、TXT 或 PDF 文件。")


def chunk_document(
    sections: list[DocumentSection],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for section in sections:
        for text in _chunk_text(section.text, chunk_size, chunk_overlap):
            chunks.append(
                DocumentChunk(
                    content=text,
                    chunk_index=len(chunks),
                    page_number=section.page_number,
                )
            )
    if not chunks:
        raise DocumentLoadError("文档没有解析出可索引的正文内容。")
    return chunks


def _load_text_sections(path: Path) -> list[DocumentSection]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8-sig")
    cleaned = _clean_text(text)
    if not cleaned:
        raise DocumentLoadError("文档没有解析出可索引的正文内容。")
    return [DocumentSection(text=cleaned)]


def _load_pdf_sections(path: Path) -> list[DocumentSection]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise DocumentLoadError("缺少 pypdf 依赖，请先执行 pip install -r requirements.txt。") from exc

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise DocumentLoadError(f"PDF 解析失败：{exc}") from exc

    sections = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        cleaned = _clean_text(text)
        if cleaned:
            sections.append(DocumentSection(text=cleaned, page_number=index))
    if not sections:
        raise DocumentLoadError("PDF 没有解析出可索引的正文内容。")
    return sections


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    chunk_size = max(200, chunk_size)
    chunk_overlap = min(max(0, chunk_overlap), chunk_size // 2)
    blocks = _split_blocks(text)
    chunks = []
    current = ""

    for block in blocks:
        if not block:
            continue
        if len(block) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(_split_long_block(block, chunk_size, chunk_overlap))
            continue
        candidate = f"{current}\n\n{block}".strip() if current else block
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            current = _with_overlap(chunks[-1], chunk_overlap, block) if chunks else block

    if current:
        chunks.append(current.strip())
    return [chunk for chunk in chunks if chunk]


def _split_blocks(text: str) -> list[str]:
    normalized = _clean_text(text)
    heading_split = re.sub(r"\n(#{1,6}\s+)", r"\n\n\1", normalized)
    return [block.strip() for block in re.split(r"\n\s*\n+", heading_split) if block.strip()]


def _split_long_block(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(0, end - chunk_overlap)
    return [chunk for chunk in chunks if chunk]


def _with_overlap(previous: str, chunk_overlap: int, block: str) -> str:
    if chunk_overlap <= 0:
        return block
    overlap = previous[-chunk_overlap:].strip()
    return f"{overlap}\n\n{block}".strip() if overlap else block


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
