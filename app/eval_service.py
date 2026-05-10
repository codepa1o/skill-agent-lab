from dataclasses import dataclass

from app.agent_runner import AgentRunError, AgentResult, run_judge_agent, run_skill_agent
from app.run_store import (
    create_eval_result,
    create_eval_run,
    create_run,
    get_test_suite,
    list_test_cases,
    update_eval_run,
)
from app.rag_service import (
    append_rag_sources,
    augment_skill_with_rag,
    build_rag_context,
    retrieve_local_context,
    serialize_rag_results,
)
from app.search_service import (
    append_sources,
    augment_skill_with_search,
    build_search_context,
    search_if_needed,
    serialize_results,
)
from app.skill_loader import SkillLoadError, load_skill


@dataclass(frozen=True)
class EvalVariant:
    name: str
    skill_content: str
    answer: str
    agent_result: AgentResult | None
    search_used: int = 0
    search_results: str = ""
    rag_used: int = 0
    rag_results: str = ""


def average_score(scores: list[float]) -> float:
    if not scores:
        return 0
    return round(sum(scores) / len(scores), 2)


async def run_suite_evaluation(suite_id: int) -> int:
    eval_run_id = create_eval_run(suite_id=suite_id)
    await execute_eval_run(suite_id=suite_id, eval_run_id=eval_run_id)
    return eval_run_id


async def execute_eval_run(*, suite_id: int, eval_run_id: int) -> None:
    suite = get_test_suite(suite_id)
    if not suite:
        raise ValueError("测试集不存在。")

    scores: list[float] = []

    try:
        loaded_skill = await load_skill(suite.skill_url)
        cases = list_test_cases(suite_id)
        if not cases:
            raise ValueError("测试集里还没有测试用例。")

        for test_case in cases:
            variants = await build_eval_variants(loaded_skill.content, test_case.question)
            for variant in variants:
                if variant.agent_result is None:
                    create_eval_result(
                        eval_run_id=eval_run_id,
                        test_case_id=test_case.id,
                        variant=variant.name,
                        error_message=variant.answer,
                    )
                    continue

                try:
                    judge_result = run_judge_agent(
                        skill_content=variant.skill_content,
                        question=test_case.question,
                        expected_behavior=test_case.expected_behavior,
                        scoring_focus=test_case.scoring_focus,
                        agent_answer=variant.answer,
                    )
                    if variant.name == "enhanced":
                        scores.append(judge_result.overall)
                    create_eval_result(
                        eval_run_id=eval_run_id,
                        test_case_id=test_case.id,
                        variant=variant.name,
                        agent_answer=variant.answer,
                        role_adherence=judge_result.role_adherence,
                        constraint_adherence=judge_result.constraint_adherence,
                        task_completion=judge_result.task_completion,
                        factual_safety=judge_result.factual_safety,
                        format_quality=judge_result.format_quality,
                        source_usage=judge_result.source_usage,
                        overall=judge_result.overall,
                        judge_comment=judge_result.judge_comment,
                        search_used=variant.search_used,
                        search_results=variant.search_results,
                        rag_used=variant.rag_used,
                        rag_results=variant.rag_results,
                    )
                    create_run(
                        skill_url=loaded_skill.skill_url,
                        raw_url=loaded_skill.raw_url,
                        question=f"[{variant.name}] {test_case.question}",
                        answer=variant.answer,
                        model=variant.agent_result.model,
                        search_used=variant.search_used,
                        search_results=variant.search_results,
                        rag_used=variant.rag_used,
                        rag_results=variant.rag_results,
                        status="success",
                        latency_ms=variant.agent_result.latency_ms,
                    )
                except AgentRunError as exc:
                    create_eval_result(
                        eval_run_id=eval_run_id,
                        test_case_id=test_case.id,
                        variant=variant.name,
                        agent_answer=variant.answer,
                        search_used=variant.search_used,
                        search_results=variant.search_results,
                        rag_used=variant.rag_used,
                        rag_results=variant.rag_results,
                        error_message=str(exc),
                    )

        update_eval_run(
            eval_run_id,
            status="completed",
            average_score=average_score(scores),
        )
    except (SkillLoadError, ValueError) as exc:
        update_eval_run(eval_run_id, status="failed", error_message=str(exc))
        raise


async def build_eval_variants(skill_content: str, question: str) -> list[EvalVariant]:
    variants: list[EvalVariant] = []
    try:
        baseline_result = run_skill_agent(skill_content, question)
        variants.append(
            EvalVariant(
                name="baseline",
                skill_content=skill_content,
                answer=baseline_result.answer,
                agent_result=baseline_result,
            )
        )
    except AgentRunError as exc:
        variants.append(
            EvalVariant(
                name="baseline",
                skill_content=skill_content,
                answer=str(exc),
                agent_result=None,
            )
        )

    rag_bundle = retrieve_local_context(question)
    search_bundle = await search_if_needed(question)
    enhanced_skill_content = augment_skill_with_search(
        augment_skill_with_rag(
            skill_content,
            build_rag_context(rag_bundle),
        ),
        build_search_context(search_bundle),
    )
    try:
        enhanced_result = run_skill_agent(enhanced_skill_content, question)
        enhanced_answer = append_sources(
            append_rag_sources(enhanced_result.answer, rag_bundle),
            search_bundle,
        )
        variants.append(
            EvalVariant(
                name="enhanced",
                skill_content=enhanced_skill_content,
                answer=enhanced_answer,
                agent_result=enhanced_result,
                search_used=1 if search_bundle.needed else 0,
                search_results=serialize_results(search_bundle),
                rag_used=1 if rag_bundle.used else 0,
                rag_results=serialize_rag_results(rag_bundle),
            )
        )
    except AgentRunError as exc:
        variants.append(
            EvalVariant(
                name="enhanced",
                skill_content=enhanced_skill_content,
                answer=str(exc),
                agent_result=None,
                search_used=1 if search_bundle.needed else 0,
                search_results=serialize_results(search_bundle),
                rag_used=1 if rag_bundle.used else 0,
                rag_results=serialize_rag_results(rag_bundle),
            )
        )
    return variants
