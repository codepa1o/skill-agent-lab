from app.agent_runner import (
    _extract_response_text,
    _format_response_transcript,
    _overall_score,
    _parse_judge_json,
    _score,
    _response_status_message,
)


def test_response_transcript_contains_latest_user_message():
    transcript = _format_response_transcript(
        [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好，有什么想咨询的？"},
            {"role": "user", "content": "继续测试"},
        ]
    )

    assert "用户：你好" in transcript
    assert "助手：你好，有什么想咨询的？" in transcript
    assert "用户：继续测试" in transcript
    assert transcript.endswith("请只输出本轮助手回答。")


def test_extract_response_text_from_content_string():
    response = {
        "output": [
            {
                "type": "message",
                "content": "你好，我在。",
            }
        ]
    }

    assert _extract_response_text(response) == "你好，我在。"


def test_extract_response_text_from_chat_choices_shape():
    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "你好，这是兼容格式回答。",
                }
            }
        ]
    }

    assert _extract_response_text(response) == "你好，这是兼容格式回答。"


def test_extract_response_text_from_nested_content_list():
    response = {
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": {"value": "你好，这是嵌套回答。"},
                    }
                ],
            }
        ]
    }

    assert _extract_response_text(response) == "你好，这是嵌套回答。"


def test_response_status_message_for_incomplete_response():
    response = {
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
    }

    assert "max_output_tokens" in _response_status_message(response)


def test_parse_judge_json_from_markdown_block():
    data = _parse_judge_json(
        '```json\n{"role_adherence": 4, "overall": 4.5, "judge_comment": "不错"}\n```'
    )

    assert data["role_adherence"] == 4
    assert data["overall"] == 4.5


def test_judge_source_usage_score_is_supported():
    data = _parse_judge_json(
        '{"source_usage": 5, "role_adherence": 4, "constraint_adherence": 4, '
        '"task_completion": 4, "factual_safety": 4, "format_quality": 4, '
        '"judge_comment": "来源引用清晰"}'
    )

    assert _score(data["source_usage"]) == 5
    assert _overall_score(data) == 4.17
