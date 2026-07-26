import json
import math
import unittest
from dataclasses import replace

from app.ondamm_personalization import (
    EventFeatureRow,
    PersonalizationModel,
    PersonalizationPrediction,
    Recommendation,
    TeacherLabel,
    deserialize_model,
    generate_recommendations,
    grouped_person_split,
    split_person_events,
)


class PersonalizationTests(unittest.TestCase):
    def row(self, number, person="p1", *, quality_ok=True, duration=10, context="visual_schedule"):
        return EventFeatureRow.from_summary(
            {
                "person_id": person,
                "event_id": f"event-{number}",
                "event_type": "task_start",
                "duration_seconds": duration,
                "quality_flags": {
                    "quality_ok": quality_ok,
                    "quality_complete": True,
                    "teacher_reviewed": True,
                },
                "zone_transition": "none",
                "teacher_context": context,
            }
        )

    def test_strict_categorical_feature_slots(self):
        row = self.row(0)
        fractional = list(row.features)
        fractional[0] = 0.5
        fractional[1] = 0.5
        with self.assertRaises(ValueError):
            replace(row, features=tuple(fractional))
        zone_start = 10 + 4
        fractional = list(row.features)
        fractional[zone_start] = 0.5
        fractional[zone_start + 1] = 0.5
        with self.assertRaises(ValueError):
            replace(row, features=tuple(fractional))

    def test_direct_prediction_and_recommendation_fail_closed(self):
        with self.assertRaises(ValueError):
            PersonalizationPrediction("visual_schedule", float("nan"), 1, ("e",))
        with self.assertRaises(ValueError):
            PersonalizationPrediction("visual_schedule", 0.5, 0, ("e",))
        with self.assertRaises(ValueError):
            PersonalizationPrediction("visual_schedule", 0.5, 1, ())
        with self.assertRaises(ValueError):
            Recommendation("visual_schedule", "hint", False, ("e",))
        with self.assertRaises(ValueError):
            Recommendation("visual_schedule", "hint", True, ())
        with self.assertRaises(ValueError):
            Recommendation("visual_schedule", "hint", True, ("e",))

    def test_nested_cycles_and_invalid_centroids_are_value_errors(self):
        summary = {"person_id": "p1", "event_id": "cycle", "event_type": "task_start"}
        summary["trigger_values"] = summary
        with self.assertRaises(ValueError):
            EventFeatureRow.from_summary(summary)
        rows = [self.row(i) for i in range(3)]
        labels = [TeacherLabel(f"event-{i}", "visual_schedule", "helpful", teacher_approved=True) for i in range(3)]
        model = PersonalizationModel.fit(rows, labels, target_person_id="p1")
        with self.assertRaises(ValueError):
            replace(model, centroids=(("visual_schedule", (0.0,) * len(model.feature_names)),))
    def test_bounded_features_and_exact_duration_boundaries(self):
        for duration in (0, 300):
            row = self.row(duration, duration=duration)
            self.assertTrue(all(math.isfinite(value) and 0 <= value <= 1 for value in row.features))
        with self.assertRaises(ValueError):
            self.row(1, duration=301)
        with self.assertRaises(ValueError):
            self.row(2, duration=-1)
        with self.assertRaises(ValueError):
            self.row(3, duration=float("nan"))

    def test_forbidden_and_unknown_fields(self):
        with self.assertRaises(ValueError):
            EventFeatureRow.from_summary({"person_id": "p1", "event_id": "x", "event_type": "unknown"})
        with self.assertRaises(ValueError):
            EventFeatureRow.from_summary({"person_id": "p1", "event_id": "x", "event_type": "task_start", "raw_landmarks": []})
        with self.assertRaises(ValueError):
            EventFeatureRow.from_summary({"person_id": "p1", "event_id": "x", "event_type": "task_start", "emotion": "happy"})

    def test_label_allowlist_and_approval(self):
        label = TeacherLabel("event-1", "visual_schedule", "helpful", teacher_approved=True)
        self.assertTrue(label.teacher_approved)
        with self.assertRaises(ValueError):
            TeacherLabel("event-1", "diagnosis", "helpful")
        with self.assertRaises(ValueError):
            TeacherLabel("event-1", "visual_schedule", "unknown")

    def test_group_split_has_no_person_leakage(self):
        rows = [self.row(i, person) for i, person in enumerate(("a", "a", "b", "b", "c"))]
        train, test = grouped_person_split(rows, test_fraction=0.4)
        self.assertFalse({r.person_id for r in train} & {r.person_id for r in test})
    def test_group_split_is_input_order_invariant(self):
        rows = [self.row(1, "b"), self.row(0, "a"), self.row(2, "c"), self.row(3, "a")]
        expected = grouped_person_split(rows, test_fraction=0.4)
        self.assertEqual(expected, grouped_person_split(list(reversed(rows)), test_fraction=0.4))
    def test_group_split_edge_cases(self):
        self.assertEqual(grouped_person_split([], test_fraction=0.2), ((), ()))
        with self.assertRaises(ValueError):
            grouped_person_split([self.row(0)], test_fraction=0.2)
        train, test = grouped_person_split(
            [self.row(0, "a"), self.row(1, "b")],
            test_fraction=1.0,
        )
        self.assertEqual({row.person_id for row in train}, {"a"})
        self.assertEqual({row.person_id for row in test}, {"b"})
    def test_group_split_rejects_cross_person_duplicate_event_ids(self):
        rows = [self.row(0, "a"), self.row(0, "b")]
        with self.assertRaisesRegex(ValueError, "duplicate event_id"):
            grouped_person_split(rows, test_fraction=0.5)

    def test_within_person_split_is_deterministic_and_non_leaky(self):
        rows = [self.row(i) for i in range(4)]
        train, validation = split_person_events(rows, validation_fraction=0.5)
        self.assertEqual({row.event_id for row in train} & {row.event_id for row in validation}, set())
        self.assertEqual([row.event_id for row in validation], ["event-2", "event-3"])
        with self.assertRaises(ValueError):
            split_person_events(rows[:1])
    def test_fit_requires_target_approved_minimum_and_is_deterministic(self):
        rows = [self.row(i) for i in range(3)] + [self.row(9, "other")]
        labels = [TeacherLabel(f"event-{i}", "visual_schedule", "helpful", teacher_approved=True) for i in range(3)]
        first = PersonalizationModel.fit(rows, labels, target_person_id="p1")
        second = PersonalizationModel.fit(list(reversed(rows)), list(reversed(labels)), target_person_id="p1")
        self.assertEqual(first.to_json(), second.to_json())
        with self.assertRaises(ValueError):
            PersonalizationModel.fit(rows[:2], labels[:2], target_person_id="p1")
        labels[0] = TeacherLabel("event-0", "visual_schedule", "helpful", teacher_approved=False)
        with self.assertRaises(ValueError):
            PersonalizationModel.fit(rows, labels, target_person_id="p1")
    def test_model_constructor_enforces_invariants(self):
        rows = [self.row(i) for i in range(3)]
        labels = [TeacherLabel(f"event-{i}", "visual_schedule", "helpful", teacher_approved=True) for i in range(3)]
        model = PersonalizationModel.fit(rows, labels, target_person_id="p1")
        normalized = replace(model, target_person_id=" p1 ")
        self.assertEqual(normalized.target_person_id, "p1")
        for field, value in (
            ("sample_count", 0),
            ("min_samples", 4),
            ("min_confidence", float("nan")),
            ("centroids", ()),
            ("label_counts", ()),
            ("feature_names", ("wrong",)),
            ("centroids", (("unknown", model.centroids[0][1]),)),
            ("centroids", (("visual_schedule", (float("nan"),) * len(model.feature_names)),)),
            ("label_counts", (("visual_schedule", 2),)),
        ):
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    replace(model, **{field: value})
        tampered = model
        object.__setattr__(tampered, "label_counts", (("visual_schedule", 2),))
        self.assertIsNone(tampered.predict(rows[0]))
    def test_fit_rejects_duplicate_normalized_row_event_ids(self):
        duplicate = EventFeatureRow.from_summary(
            {
                "person_id": "p1",
                "event_id": " event-0 ",
                "event_type": "task_start",
                "quality_flags": {
                    "quality_ok": True,
                    "quality_complete": True,
                    "teacher_reviewed": True,
                },
            }
        )
        rows = [self.row(0), duplicate, self.row(1), self.row(2)]
        labels = [
            TeacherLabel(f"event-{index}", "visual_schedule", "helpful", teacher_approved=True)
            for index in range(3)
        ]
        with self.assertRaisesRegex(ValueError, "duplicate event_id"):
            PersonalizationModel.fit(rows, labels, target_person_id="p1")

    def test_prediction_confidence_and_abstention(self):
        rows = [self.row(i) for i in range(3)]
        labels = [TeacherLabel(f"event-{i}", "visual_schedule", "helpful", teacher_approved=True) for i in range(3)]
        model = PersonalizationModel.fit(rows, labels, target_person_id="p1")
        prediction = model.predict(rows[0])
        self.assertIsNotNone(prediction)
        self.assertEqual(prediction.support_label, "visual_schedule")
        self.assertIsNone(model.predict(self.row(4, quality_ok=False)))
        self.assertIsNone(model.predict(self.row(4, "other")))
        unlabelled = model.predict(self.row(4))
        self.assertIsNotNone(unlabelled)
        altered = self.row(0, duration=11)
        self.assertIsNone(model.predict(altered))
        self.assertEqual(
            generate_recommendations(
                unlabelled, teacher_approved=True, evidence_ids=("event-4",)
            ),
            (),
        )
        self.assertIsNone(model.predict({"event_type": "not-safe"}))

    def test_serialization_round_trip_is_strict(self):
        rows = [self.row(i) for i in range(3)]
        labels = [TeacherLabel(f"event-{i}", "visual_schedule", "helpful", teacher_approved=True) for i in range(3)]
        model = PersonalizationModel.fit(rows, labels, target_person_id="p1")
        encoded = model.to_json()
        self.assertEqual(encoded, model.to_json())
        self.assertEqual(deserialize_model(encoded).to_json(), encoded)
        self.assertEqual(deserialize_model(encoded).approved_event_ids, ("event-0", "event-1", "event-2"))
        payload = json.loads(encoded)
        payload["model_config_version"] = "wrong"
        with self.assertRaises(ValueError):
            deserialize_model(json.dumps(payload))
        payload["model_config_version"] = "centroid-baseline-v1"
        payload["approved_event_ids"] = ["forged"]
        with self.assertRaises(ValueError):
            deserialize_model(json.dumps(payload))

    def test_recommendation_requires_approval_and_evidence(self):
        self.assertEqual(generate_recommendations("visual_schedule", teacher_approved=False, evidence_ids=("e",)), ())
        self.assertEqual(generate_recommendations("visual_schedule", teacher_approved=True), ())
        self.assertEqual(
            generate_recommendations("visual_schedule", teacher_approved=True, evidence_ids=("e2", "e1")),
            (),
        )
        prediction = PersonalizationPrediction("visual_schedule", 0.8, 1, ("e1",))
        self.assertEqual(
            generate_recommendations(prediction, teacher_approved=True, evidence_ids=("e2",)),
            (),
        )


if __name__ == "__main__":
    unittest.main()
