from app.search_service import (
    SearchBundle,
    SearchResult,
    append_sources,
    build_search_context,
    should_search,
)


def test_should_search_for_school_major_questions():
    assert should_search("杭州电子科技大学怎么样，可以报计算机吗")
    assert should_search("计算机专业就业怎么样")
    assert not should_search("你好")


def test_search_context_and_sources_are_rendered():
    bundle = SearchBundle(
        needed=True,
        query="计算机专业就业",
        results=[
            SearchResult(
                title="计算机专业就业分析",
                url="https://example.com/cs",
                snippet="就业竞争加剧，需要关注学校层次和城市。",
            )
        ],
    )

    context = build_search_context(bundle)
    answer = append_sources("回答正文", bundle)

    assert "实时搜索资料" in context
    assert "https://example.com/cs" in answer
