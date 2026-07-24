from __future__ import annotations

from dataclasses import dataclass

from ondamm_models import Dossier, unique_preserving_order, utc_now

__all__ = [
    "LearningStep",
    "LearningProgramPlan",
    "LearningRunSummary",
    "build_learning_program_plan",
    "build_learning_run_summary",
    "render_learning_program_markdown",
    "render_learning_run_summary_markdown",
]

SUPPORT_BOUNDARY_NOTICE = (
    "이 학습 프로그램은 지원 계획 초안이며 진단, 점수화, 순응도 판정에 사용하지 않습니다."
)
LOCAL_RECORD_GUIDANCE = (
    "로컬 기록은 raw media 자동 승격 없이 교사/보호자 검토용 메모와 session summary 작성에만 사용합니다."
)


@dataclass
class LearningStep:
    title: str
    activity_focus: str
    prompt_hint: str
    reinforcement_hint: str
    transition_hint: str
    duration_seconds: int


@dataclass
class LearningProgramPlan:
    child_id: str
    child_name: str
    goal: str
    created_at: str
    steps: list[LearningStep]
    caregiver_input: str | None = None
    support_boundary_notice: str = SUPPORT_BOUNDARY_NOTICE
    local_record_guidance: str = LOCAL_RECORD_GUIDANCE

    @property
    def total_duration_seconds(self) -> int:
        return sum(step.duration_seconds for step in self.steps)


@dataclass
class LearningRunSummary:
    child_id: str
    child_name: str
    goal: str
    started_at: str
    finished_at: str
    completed_step_titles: list[str]
    educator_notes: list[str]
    reinforcement_observations: list[str]
    transition_observations: list[str]
    caregiver_note: str | None = None
    support_boundary_notice: str = SUPPORT_BOUNDARY_NOTICE
    local_record_guidance: str = LOCAL_RECORD_GUIDANCE


def _join_or_default(values: list[str], fallback: str) -> str:
    cleaned = unique_preserving_order(values)
    return ", ".join(cleaned[:3]) if cleaned else fallback



def _clean_optional_text(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.strip()
    return cleaned or None



def build_learning_program_plan(
    dossier: Dossier,
    *,
    goal: str,
    caregiver_input: str | None = None,
    created_at: str | None = None,
) -> LearningProgramPlan:
    goal_text = goal.strip()
    caregiver_note = _clean_optional_text(caregiver_input)
    preference_text = _join_or_default(dossier.confirmed_preferences, "시각적으로 부담이 적은 익숙한 선호 자극")
    strategy_text = _join_or_default(dossier.effective_strategies, "짧은 단계 제시와 즉시 피드백")
    support_text = _join_or_default(
        dossier.triggers_and_calming_supports,
        "전환 전에 예고하고 짧은 휴식 신호를 제공하기",
    )
    avoidance_text = _join_or_default(dossier.confirmed_avoidances, "과도한 자극")
    communication_text = dossier.communication_modality.strip() or "시각 단서와 짧은 구두 지시"
    caregiver_text = caregiver_note or "추가 보호자 메모 없음"

    steps = [
        LearningStep(
            title="시각 도입",
            activity_focus=(
                f"오늘의 목표 `{goal_text}` 와 활동 순서, 끝나는 조건을 시각 카드로 먼저 보여 주고 "
                f"{dossier.display_name}이 시작 신호를 예측할 수 있게 한다."
            ),
            prompt_hint=(
                f"{communication_text} 방식으로 first-then 또는 순서표를 함께 제시하고, "
                "한 번에 한 문장만 말한다."
            ),
            reinforcement_hint=(
                f"도입 직후 선호 자극({preference_text})을 짧게 연결해 진입 부담을 낮춘다."
            ),
            transition_hint=(
                f"회피 자극({avoidance_text})을 줄이고, 다음 단계로 넘어가기 전에 {support_text}를 다시 예고한다."
            ),
            duration_seconds=120,
        ),
        LearningStep(
            title="짧은 과제 블록 1",
            activity_focus=(
                f"목표 `{goal_text}` 의 첫 과제를 3~5분 길이의 작은 블록으로 제시하고 성공 기준을 작게 나눈다."
            ),
            prompt_hint=(
                f"확인된 전략({strategy_text})에 맞춰 시각 단서, 제스처, 짧은 구두 prompt를 낮은 강도부터 사용한다."
            ),
            reinforcement_hint=(
                "성공 직후 짧은 칭찬이나 선호 자극 접근을 제공하고, 무엇을 잘했는지 즉시 구체적으로 말한다."
            ),
            transition_hint=(
                "블록 종료 전 10~20초 정도 남았음을 예고하고, 다음에 할 한 가지 행동만 다시 알려 준다."
            ),
            duration_seconds=300,
        ),
        LearningStep(
            title="짧은 과제 블록 2",
            activity_focus=(
                "두 번째 짧은 과제 또는 일반화 연습을 제공하되, 요구량은 첫 블록과 같거나 더 낮게 유지한다."
            ),
            prompt_hint=(
                "반응이 안정되면 prompt를 점진적으로 줄이고, 어려워지면 바로 더 쉬운 단계로 낮춘다."
            ),
            reinforcement_hint=(
                f"강화는 짧고 예측 가능하게 유지하며, 선호 자극({preference_text}) 또는 짧은 휴식을 연결한다."
            ),
            transition_hint=(
                f"거부나 피로 신호가 보이면 {support_text}를 사용해 과제를 축소하거나 마무리 단계로 전환한다."
            ),
            duration_seconds=300,
        ),
        LearningStep(
            title="강화와 전환 지원",
            activity_focus=(
                "완료 즉시 강화, 짧은 회복 시간, next-step 예고를 연결해 세션 종료 또는 후속 활동 전환을 돕는다."
            ),
            prompt_hint=(
                "마무리 문구를 일정하게 유지하고, 필요하면 완료 카드나 finished 표식을 다시 보여 준다."
            ),
            reinforcement_hint=(
                "완료된 행동을 짧게 요약해 칭찬하고, 다음 활동 전에 과도한 요구를 추가하지 않는다."
            ),
            transition_hint=(
                f"전환 지원은 {support_text}를 우선 사용하고, 어려우면 더 쉬운 종료 절차나 짧은 휴식을 선택한다."
            ),
            duration_seconds=180,
        ),
        LearningStep(
            title="로컬 기록 가이드",
            activity_focus=(
                "정답률만이 아니라 어떤 prompt가 먹혔는지, 어떤 강화가 안정화에 도움 되었는지, 어떤 전환에서 저항이 있었는지 기록한다."
            ),
            prompt_hint=(
                f"보호자/교사 추가 입력은 `{caregiver_text}` 로 반영하고, 다음 회기에서 유지하거나 조정할 지원만 메모한다."
            ),
            reinforcement_hint=(
                "기록은 성공 경험과 유지할 지원을 함께 남기고, 행동을 점수화하거나 진단 해석으로 확장하지 않는다."
            ),
            transition_hint=(
                "세션 종료 후에는 local note와 session summary 후보만 남기고, dossier 반영은 사람 검토 후 수동으로 결정한다."
            ),
            duration_seconds=120,
        ),
    ]

    return LearningProgramPlan(
        child_id=dossier.child_id,
        child_name=dossier.display_name,
        goal=goal_text,
        created_at=created_at or utc_now(),
        steps=steps,
        caregiver_input=caregiver_note,
    )



def build_learning_run_summary(
    plan: LearningProgramPlan,
    *,
    started_at: str,
    finished_at: str,
    completed_step_titles: list[str],
    educator_notes: list[str],
    reinforcement_observations: list[str] | None = None,
    transition_observations: list[str] | None = None,
    caregiver_note: str | None = None,
) -> LearningRunSummary:
    return LearningRunSummary(
        child_id=plan.child_id,
        child_name=plan.child_name,
        goal=plan.goal,
        started_at=started_at.strip(),
        finished_at=finished_at.strip(),
        completed_step_titles=unique_preserving_order(completed_step_titles),
        educator_notes=unique_preserving_order(educator_notes),
        reinforcement_observations=unique_preserving_order(reinforcement_observations or []),
        transition_observations=unique_preserving_order(transition_observations or []),
        caregiver_note=_clean_optional_text(caregiver_note),
    )



def render_learning_program_markdown(plan: LearningProgramPlan) -> str:
    lines = [
        f"# ON DAMM Learning Program — {plan.child_name}",
        "",
        "## Safety boundary",
        f"- {plan.support_boundary_notice}",
        f"- {plan.local_record_guidance}",
        "- raw media 저장/승격은 별도 명시적 선택이 있을 때만 다룬다.",
        "",
        "## Plan summary",
        f"- child_id: `{plan.child_id}`",
        f"- goal: {plan.goal}",
        f"- created_at: {plan.created_at}",
        f"- caregiver_input: {plan.caregiver_input or '추가 메모 없음'}",
        f"- planned_steps: {len(plan.steps)}",
        f"- total_duration_seconds: {plan.total_duration_seconds}",
        "",
        "## Ordered steps",
    ]
    for index, step in enumerate(plan.steps, start=1):
        lines.extend(
            [
                f"### {index}. {step.title}",
                f"- focus/activity: {step.activity_focus}",
                f"- prompt_hint: {step.prompt_hint}",
                f"- reinforcement_hint: {step.reinforcement_hint}",
                f"- transition_hint: {step.transition_hint}",
                f"- duration_seconds: {step.duration_seconds}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"



def render_learning_run_summary_markdown(summary: LearningRunSummary) -> str:
    lines = [
        f"# ON DAMM Learning Run Summary — {summary.child_name}",
        "",
        "## Safety boundary",
        f"- {summary.support_boundary_notice}",
        f"- {summary.local_record_guidance}",
        "- 이 요약은 raw media 없이 검토 가능한 실행 요약이다.",
        "",
        "## Run summary",
        f"- child_id: `{summary.child_id}`",
        f"- goal: {summary.goal}",
        f"- started_at: {summary.started_at}",
        f"- finished_at: {summary.finished_at}",
        f"- caregiver_note: {summary.caregiver_note or '추가 메모 없음'}",
        "",
        "## Completed steps",
    ]
    if summary.completed_step_titles:
        lines.extend([f"- {title}" for title in summary.completed_step_titles])
    else:
        lines.append("- 없음")

    lines.extend(["", "## Educator notes"])
    if summary.educator_notes:
        lines.extend([f"- {note}" for note in summary.educator_notes])
    else:
        lines.append("- 없음")

    lines.extend(["", "## Reinforcement observations"])
    if summary.reinforcement_observations:
        lines.extend([f"- {note}" for note in summary.reinforcement_observations])
    else:
        lines.append("- 없음")

    lines.extend(["", "## Transition observations"])
    if summary.transition_observations:
        lines.extend([f"- {note}" for note in summary.transition_observations])
    else:
        lines.append("- 없음")

    return "\n".join(lines) + "\n"
