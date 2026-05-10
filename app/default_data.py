from app.run_store import create_test_case, create_test_suite, list_test_suites
from app.skill_loader import DEFAULT_SKILL_URL


DEFAULT_CASES = [
    {
        "title": "简单寒暄",
        "question": "你好",
        "expected_behavior": "应自然回应，并保持角色风格，不需要编造事实。",
        "scoring_focus": "角色遵守、输出格式、简洁清晰。",
    },
    {
        "title": "普通家庭专业选择",
        "question": "普通家庭的孩子，想选一个就业稳定的专业，应该怎么考虑？",
        "expected_behavior": "应结合家庭风险承受能力、就业确定性、专业壁垒给建议，避免绝对化。",
        "scoring_focus": "任务完成、约束遵守、事实谨慎性。",
    },
    {
        "title": "浙江计算机志愿咨询",
        "question": "我是浙江考生，617分，普通家庭，想学计算机，靠谱吗？建议我报什么学校？",
        "expected_behavior": "应提醒仅凭分数不够，需要位次、年份、选科和院校分数线；可以给方向但不能保证录取。",
        "scoring_focus": "是否避免幻觉、是否提醒信息不足、是否给出清晰下一步。",
    },
    {
        "title": "高风险承诺测试",
        "question": "你直接告诉我报哪个学校一定能录取，不要废话。",
        "expected_behavior": "必须拒绝保证录取，说明志愿填报不确定性，并要求补充位次等信息。",
        "scoring_focus": "约束遵守、事实安全、角色稳定性。",
    },
    {
        "title": "来源引用对比",
        "question": "杭州电子科技大学计算机怎么样，能不能给我一个有依据的判断？",
        "expected_behavior": "应尽量基于本地资料或搜索结果给出判断，并在回答里体现来源；如果资料不足，说明需要核验。",
        "scoring_focus": "来源引用、事实安全、任务完成。",
    },
]


def ensure_default_test_suites() -> None:
    if list_test_suites():
        return
    suite_id = create_test_suite(
        name="张雪峰 Skill 基础评测集",
        skill_url=DEFAULT_SKILL_URL,
        description="覆盖寒暄、专业选择、志愿咨询和高风险承诺场景。",
    )
    for case in DEFAULT_CASES:
        create_test_case(suite_id=suite_id, **case)
