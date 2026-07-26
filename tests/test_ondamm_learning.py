from __future__ import annotations

import sys
from dataclasses import asdict
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from ondamm_learning import (  # noqa: E402
    build_learning_program_plan,
    build_personalized_learning_program_plan,
    build_learning_run_summary,
    render_learning_program_markdown,
    render_learning_run_summary_markdown,
)
from ondamm_personalization import TeacherLabel  # noqa: E402
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

    def _personalization_dossier(self) -> Dossier:
        return Dossier.create(
            child_id="child-personalized",
            display_name="Demo Child",
            age_band="초등 저학년",
            communication_modality="시각 단서",
        )

    def _event_summaries(self, person_id: str = "child-personalized") -> list[dict[str, object]]:
        return [
            {
                "person_id": person_id,
                "event_id": f"event-{index}",
                "event_type": "task_start",
                "duration_seconds": 10,
                "quality_flags": {
                    "quality_ok": True,
                    "quality_complete": True,
                    "teacher_reviewed": True,
                },
                "teacher_context": "visual_schedule",
            }
            for index in range(3)
        ]

    def test_personalized_plan_keeps_baseline_defaults_and_asdict_fields(self) -> None:
        plan = build_learning_program_plan(
            self._personalization_dossier(),
            goal="matching cards",
            created_at="2026-07-11T16:00:00+00:00",
        )
        self.assertEqual(plan.personalization_provenance, "baseline")
        self.assertEqual(plan.personalization_sample_count, 0)
        self.assertEqual(plan.personalization_evidence_ids, ())
        self.assertIn("personalization_notice", asdict(plan))
        self.assertIsNone(plan.personalization_applied_manifest_digest)

    def test_approved_personalization_changes_only_support_hint(self) -> None:
        dossier = self._personalization_dossier()
        baseline = build_learning_program_plan(dossier, goal="matching cards", created_at="x")
        plan = build_personalized_learning_program_plan(
            dossier,
            goal="matching cards",
            event_summaries=self._event_summaries(),
            teacher_labels=[
                TeacherLabel(f"event-{index}", "visual_schedule", "helpful", teacher_approved=True)
                for index in range(3)
            ],
            target_person_id=dossier.child_id,
            teacher_approved=True,
            created_at="x",
        )
        self.assertEqual(
            [step.activity_focus for step in plan.steps],
            [step.activity_focus for step in baseline.steps],
        )
        self.assertEqual(
            [step.duration_seconds for step in plan.steps],
            [step.duration_seconds for step in baseline.steps],
        )
        self.assertNotEqual(plan.steps[0].prompt_hint, baseline.steps[0].prompt_hint)
        self.assertEqual(plan.personalization_sample_count, 3)
        self.assertEqual(plan.personalization_evidence_ids, ("event-0", "event-1", "event-2"))
        self.assertEqual(plan.personalization_recommendations, ("visual_schedule",))
        self.assertEqual(plan.personalization_manifest_digest and len(plan.personalization_manifest_digest), 64)
    def test_personalized_plan_caps_recommendations_and_keeps_provenance_consistent(self) -> None:
        dossier = self._personalization_dossier()
        support_labels = ("visual_schedule", "short_prompt", "transition_preview", "brief_break")
        events = []
        labels = []
        for index, support_label in enumerate(support_labels):
            for offset in range(3):
                event_id = f"{support_label}-{offset}"
                events.append(dict(self._event_summaries()[0], event_id=event_id, teacher_context=support_label))
                labels.append(TeacherLabel(event_id, support_label, "helpful", teacher_approved=True))
        for index in range(6):
            event_id = f"unlabelled-{index}"
            events.append(
                dict(
                    self._event_summaries()[0],
                    event_id=event_id,
                    teacher_context="visual_schedule",
                )
            )
        for index in range(6):
            event_id = f"negative-{index}"
            events.append(
                dict(
                    self._event_summaries()[0],
                    event_id=event_id,
                    teacher_context="visual_schedule",
                )
            )
            labels.append(
                TeacherLabel(event_id, "visual_schedule", "not_helpful", teacher_approved=True)
            )
        for index in range(6):
            event_id = f"unauthorized-{index}"
            events.append(
                dict(
                    self._event_summaries()[0],
                    event_id=event_id,
                    teacher_context="visual_schedule",
                )
            )
            labels.append(
                TeacherLabel(event_id, "visual_schedule", "helpful", teacher_approved=False)
            )
        plan = build_personalized_learning_program_plan(
            dossier,
            goal="matching cards",
            event_summaries=events,
            teacher_labels=labels,
            target_person_id=dossier.child_id,
            teacher_approved=True,
            created_at="x",
        )
        repeat_plan = build_personalized_learning_program_plan(
            dossier,
            goal="matching cards",
            event_summaries=events,
            teacher_labels=labels,
            target_person_id=dossier.child_id,
            teacher_approved=True,
            created_at="x",
        )
        self.assertEqual(plan.personalization_recommendations, ("brief_break", "short_prompt", "transition_preview"))
        self.assertEqual(plan.personalization_sample_count, len(plan.personalization_evidence_ids))
        self.assertEqual(plan.personalization_sample_count, 9)
        self.assertEqual(plan.personalization_confidence, 0.833333)
        self.assertEqual(len(plan.personalization_manifest_digest or ""), 64)
        self.assertEqual(len(plan.personalization_applied_manifest_digest or ""), 64)
        self.assertEqual(
            plan.personalization_manifest_digest,
            repeat_plan.personalization_manifest_digest,
        )
        self.assertEqual(
            plan.personalization_applied_manifest_digest,
            repeat_plan.personalization_applied_manifest_digest,
        )
        self.assertNotEqual(
            plan.personalization_manifest_digest,
            plan.personalization_applied_manifest_digest,
        )
        self.assertNotIn("visual_schedule", plan.personalization_recommendations)
        self.assertTrue(
            all(
                not evidence_id.startswith(("unlabelled-", "negative-"))
                and not evidence_id.startswith("unauthorized-")
                for evidence_id in plan.personalization_evidence_ids
            )
        )
    def test_personalized_plan_rejects_duplicate_normalized_event_ids(self) -> None:
        dossier = self._personalization_dossier()
        events = self._event_summaries()
        events[1] = dict(events[1], event_id=" event-0 ")
        labels = [
            TeacherLabel(f"event-{index}", "visual_schedule", "helpful", teacher_approved=True)
            for index in range(3)
        ]
        with self.assertRaisesRegex(ValueError, "duplicate event_id"):
            build_personalized_learning_program_plan(
                dossier,
                goal="matching cards",
                event_summaries=events,
                teacher_labels=labels,
                target_person_id=dossier.child_id,
                teacher_approved=True,
            )
    def test_cyclic_summary_abstains_to_baseline(self) -> None:
        dossier = self._personalization_dossier()
        cyclic = {"person_id": dossier.child_id, "event_id": "cycle", "event_type": "task_start"}
        cyclic["trigger_values"] = cyclic
        plan = build_personalized_learning_program_plan(
            dossier,
            goal="matching cards",
            event_summaries=[cyclic],
            teacher_labels=[],
            target_person_id=dossier.child_id,
            teacher_approved=True,
        )
        self.assertEqual(plan.personalization_provenance, "baseline")
        self.assertIn("Sensor data did not directly alter the plan.", plan.personalization_notice)

    def test_personalization_abstains_for_sparse_low_quality_mismatched_or_unapproved(self) -> None:
        dossier = self._personalization_dossier()
        events = self._event_summaries()
        labels = [TeacherLabel(f"event-{index}", "visual_schedule", "helpful", teacher_approved=True) for index in range(3)]
        sparse = build_personalized_learning_program_plan(
            dossier,
            goal="matching cards",
            event_summaries=events[:2],
            teacher_labels=labels[:2],
            target_person_id=dossier.child_id,
            teacher_approved=True,
        )
        low_quality = [dict(event, quality_flags={"quality_ok": False}) for event in events]
        low_quality_plan = build_personalized_learning_program_plan(
            dossier,
            goal="matching cards",
            event_summaries=low_quality,
            teacher_labels=labels,
            target_person_id=dossier.child_id,
            teacher_approved=True,
        )
        mismatched = build_personalized_learning_program_plan(
            dossier,
            goal="matching cards",
            event_summaries=events,
            teacher_labels=labels,
            target_person_id="other-child",
            teacher_approved=True,
        )
        unapproved = build_personalized_learning_program_plan(
            dossier,
            goal="matching cards",
            event_summaries=events,
            teacher_labels=[TeacherLabel("event-0", "visual_schedule", "helpful", False)] + labels[1:],
            target_person_id=dossier.child_id,
            teacher_approved=True,
        )
        not_approved = build_personalized_learning_program_plan(
            dossier,
            goal="matching cards",
            event_summaries=events,
            teacher_labels=labels,
            target_person_id=dossier.child_id,
            teacher_approved=False,
        )
        for plan in (sparse, low_quality_plan, mismatched, unapproved, not_approved):
            self.assertEqual(plan.personalization_provenance, "baseline")
            self.assertEqual(plan.personalization_sample_count, 0)
            self.assertTrue(all("시각 순서표" not in step.prompt_hint for step in plan.steps))

    def test_personalization_rendering_has_provenance_and_safe_fields(self) -> None:
        dossier = self._personalization_dossier()
        plan = build_personalized_learning_program_plan(
            dossier,
            goal="matching cards",
            event_summaries=self._event_summaries(),
            teacher_labels=[
                TeacherLabel(f"event-{index}", "visual_schedule", "helpful", teacher_approved=True)
                for index in range(3)
            ],
            target_person_id=dossier.child_id,
            teacher_approved=True,
        )
        markdown = render_learning_program_markdown(plan)
        self.assertIn("Teacher-approved personalization evidence", markdown)
        self.assertIn("model_config_version: centroid-baseline-v1", markdown)
        self.assertIn("manifest_digest:", markdown)
        self.assertIn("Sensor data did not directly alter the plan.", markdown)
        serialized = repr(asdict(plan)).lower()
        for forbidden in ("landmark", "frame_score", "emotion", "attention", "compliance", "asd"):
            self.assertNotIn(forbidden, serialized)
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
