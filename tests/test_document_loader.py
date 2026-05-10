import pytest

from app.document_loader import (
    DocumentLoadError,
    chunk_document,
    load_document_sections,
    validate_document_file,
)


def test_markdown_document_is_loaded_and_chunked(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("# 杭电计算机\n\n计算机专业实力较强。\n\n就业要看城市和个人能力。", encoding="utf-8")

    sections = load_document_sections(path)
    chunks = chunk_document(sections, chunk_size=20, chunk_overlap=4)

    assert sections[0].page_number == 0
    assert chunks
    assert "杭电计算机" in chunks[0].content


def test_invalid_document_suffix_is_rejected():
    with pytest.raises(DocumentLoadError):
        validate_document_file("data.docx", 100)


def test_empty_document_is_rejected():
    with pytest.raises(DocumentLoadError):
        validate_document_file("data.txt", 0)
