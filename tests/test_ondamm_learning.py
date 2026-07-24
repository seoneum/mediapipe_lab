from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from ondamm_learning import (  # noqa: E402
    build_learning_program_plan,
    build_learning_run_summary,
    render_learning_program_markdown,
    render_learning_run_summary_markdown,
)
from ondamm_models import Dossier  # noqa: E402


class OnDammLearningTests(unittest.TestCase):
    def test_build_learning_program_plan_creates_literature_grounded_sequence(self) -> None:
        dossier = Dossier.create(
            child_id="child-learning-a",
            display_name="Demo Child",
            age_band="초등 저학년",
            communication_modality="시각 단서 + 짧은 구두 지시",
            confirmed_preferences=["동물 카드", "짧은 칭찬"],
            confirmed_avoidances=["갑작스러운 큰 소리"],
            effective_strategies=["한 번에 한 단계 제시"],
            triggers_and_calming_supports=["전환 전에 10초 예고하기"],
        )

        plan = build_learning_program_plan(
            dossier,
            goal="분류 활동 5분 유지",
            caregiver_input="짧은 피드백이 있을 때 더 잘 참여함",
            created_at="2026-07-11T15:30:00+00:00",
        )

        self.assertEqual(plan.child_id, "child-learning-a")
        self.assertEqual(plan.child_name, "Demo Child")
        self.assertEqual(plan.goal, "분류 활동 5분 유지")
        self.assertEqual(plan.created_at, "2026-07-11T15:30:00+00:00")
        self.assertEqual(plan.caregiver_input, "짧은 피드백이 있을 때 더 잘 참여함")
        self.assertEqual([step.title for step in plan.steps], [
            "시각 도입",
            "짧은 과제 블록 1",
            "짧은 과제 블록 2",
            "강화와 전환 지원",
            "로컬 기록 가이드",
        ])
        self.assertEqual(plan.total_duration_seconds, 1020)
        self.assertIn("시각 카드", plan.steps[0].activity_focus)
        self.assertIn("한 번에 한 단계 제시", plan.steps[1].prompt_hint)
        self.assertIn("동물 카드", plan.steps[2].reinforcement_hint)
        self.assertIn("전환 전에 10초 예고하기", plan.steps[3].transition_hint)
        self.assertIn("짧은 피드백이 있을 때 더 잘 참여함", plan.steps[4].prompt_hint)

    def test_render_learning_program_markdown_includes_boundary_and_steps(self) -> None:
        dossier = Dossier.create(
            child_id="child-learning-b",
            display_name="Demo Child",
            age_band="초등 저학년",
            communication_modality="시각 단서",
        )
        plan = build_learning_program_plan(
            dossier,
            goal="matching cards for 3 minutes",
            caregiver_input=None,
            created_at="2026-07-11T16:00:00+00:00",
        )

        markdown = render_learning_program_markdown(plan)

        self.assertIn("ON DAMM Learning Program — Demo Child", markdown)
        self.assertIn("지원 계획 초안이며 진단", markdown)
        self.assertIn("raw media 저장/승격은 별도 명시적 선택", markdown)
        self.assertIn("caregiver_input: 추가 메모 없음", markdown)
        self.assertIn("planned_steps: 5", markdown)
        self.assertIn("### 1. 시각 도입", markdown)
        self.assertIn("### 5. 로컬 기록 가이드", markdown)
        self.assertIn("focus/activity:", markdown)

    def test_render_learning_run_summary_markdown_uses_local_non_media_summary(self) -> None:
        dossier = Dossier.create(
            child_id="child-learning-c",
            display_name="Demo Child",
            age_band="초등 저학년",
            communication_modality="시각 단서",
        )
        plan = build_learning_program_plan(
            dossier,
            goal="퍼즐 조각 맞추기 2회",
            created_at="2026-07-11T16:30:00+00:00",
        )
        summary = build_learning_run_summary(
            plan,
            started_at="2026-07-11T16:31:00+00:00",
            finished_at="2026-07-11T16:46:00+00:00",
            completed_step_titles=["시각 도입", "짧은 과제 블록 1", "시각 도입"],
            educator_notes=["첫 블록에서 시각 단서를 보고 바로 착석함", "첫 블록에서 시각 단서를 보고 바로 착석함"],
            reinforcement_observations=["짧은 칭찬 뒤 다음 카드로 부드럽게 전환함"],
            transition_observations=["마무리 예고 뒤 거부 없이 종료함"],
            caregiver_note="집에서도 같은 first-then 표현 사용 중",
        )

        markdown = render_learning_run_summary_markdown(summary)

        self.assertEqual(summary.completed_step_titles, ["시각 도입", "짧은 과제 블록 1"])
        self.assertEqual(summary.educator_notes, ["첫 블록에서 시각 단서를 보고 바로 착석함"])
        self.assertIn("raw media 없이 검토 가능한 실행 요약", markdown)
        self.assertIn("caregiver_note: 집에서도 같은 first-then 표현 사용 중", markdown)
        self.assertIn("## Completed steps", markdown)
        self.assertIn("- 시각 도입", markdown)
        self.assertIn("## Reinforcement observations", markdown)
        self.assertIn("짧은 칭찬 뒤 다음 카드로 부드럽게 전환함", markdown)
        self.assertIn("## Transition observations", markdown)
        self.assertIn("마무리 예고 뒤 거부 없이 종료함", markdown)


if __name__ == "__main__":
    unittest.main()
