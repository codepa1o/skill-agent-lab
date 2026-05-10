import os

from openai import OpenAI, OpenAIError


DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_EMBEDDING_MODEL = "text-embedding-v4"
DEFAULT_EMBEDDING_DIMENSIONS = 1024


class EmbeddingError(RuntimeError):
    """Raised when the embedding API cannot return usable vectors."""


def get_dashscope_base_url() -> str:
    return os.getenv("DASHSCOPE_BASE_URL", DEFAULT_DASHSCOPE_BASE_URL).strip() or DEFAULT_DASHSCOPE_BASE_URL


def get_embedding_model() -> str:
    return os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip() or DEFAULT_EMBEDDING_MODEL


def get_embedding_dimensions() -> int:
    value = os.getenv("EMBEDDING_DIMENSIONS", "").strip()
    if not value:
        return DEFAULT_EMBEDDING_DIMENSIONS
    try:
        return max(64, int(value))
    except ValueError:
        return DEFAULT_EMBEDDING_DIMENSIONS


def create_embeddings(texts: list[str]) -> list[list[float]]:
    cleaned_texts = [text.strip() for text in texts if text.strip()]
    if not cleaned_texts:
        return []

    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise EmbeddingError("未配置 DASHSCOPE_API_KEY，无法为本地资料生成向量。")

    client = OpenAI(
        api_key=api_key,
        base_url=get_dashscope_base_url(),
        timeout=60,
    )
    try:
        response = client.embeddings.create(
            model=get_embedding_model(),
            input=cleaned_texts,
            dimensions=get_embedding_dimensions(),
        )
    except OpenAIError as exc:
        raise EmbeddingError(f"阿里云百炼 Embedding API 调用失败：{exc}") from exc

    embeddings = [item.embedding for item in response.data]
    if len(embeddings) != len(cleaned_texts):
        raise EmbeddingError("Embedding API 返回数量和输入文本数量不一致。")
    if not all(embedding for embedding in embeddings):
        raise EmbeddingError("Embedding API 返回了空向量。")
    return embeddings
