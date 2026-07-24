from __future__ import annotations

from ondamm_models import Dossier, RecommendationEntry


def _join_or_default(values: list[str], fallback: str) -> str:
    return ", ".join(values[:3]) if values else fallback


def build_baseline_recommendation(
    dossier: Dossier,
    *,
    goal: str,
    caregiver_input: str,
    drafted_by: str,
    approved_by: str | None = None,
) -> RecommendationEntry:
    preference_text = _join_or_default(dossier.confirmed_preferences, "시각적으로 부담이 적은 익숙한 자극")
    avoidance_text = _join_or_default(dossier.confirmed_avoidances, "과도한 자극")
    strategy_text = _join_or_default(dossier.effective_strategies, "짧은 단계 제시와 즉시 피드백")
    support_text = _join_or_default(
        dossier.triggers_and_calming_supports,
        "익숙한 전환 문구와 휴식 신호",
    )

    suggested_activities = [
        f"선호 자극({preference_text})을 먼저 제시한 뒤 목표 활동 `{goal}`로 짧게 전환한다.",
        f"한 번에 한 단계씩 안내하고, 전략({strategy_text})에 맞춰 성공 기준을 작게 나눈다.",
        f"회피 자극({avoidance_text})이 나타나면 지원 수단({support_text})으로 강도를 낮춘 대체 과제를 제시한다.",
    ]

    rationale_lines = [
        f"보호자/교사 입력: {caregiver_input.strip() or '추가 메모 없음'}",
        f"확인된 선호 자극: {preference_text}",
        f"효과적 전략: {strategy_text}",
        f"주의할 회피 자극: {avoidance_text}",
    ]

    summary = (
        f"{dossier.display_name}의 현재 목표는 `{goal}`이며, 선호 자극을 활용해 진입 부담을 낮추고 "
        f"확인된 전략을 유지하는 보수적 학습 초안을 권장한다."
    )

    return RecommendationEntry.create(
        goal=goal,
        summary=summary,
        suggested_activities=suggested_activities,
        rationale_lines=rationale_lines,
        drafted_by=drafted_by,
        approved_by=approved_by,
    )
