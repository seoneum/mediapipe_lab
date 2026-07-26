import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from app.ondamm_personalization import EventFeatureRow, TeacherLabel
from app.ondamm_personalization_cli import main, run


class PersonalizationCliTests(unittest.TestCase):
    def _rows(self, *, count=3, complete=True, reviewed=True):
        return [
            EventFeatureRow.from_summary(
                {
                    "person_id": "p1",
                    "event_id": f"e{index}",
                    "event_type": "task_start",
                    "quality_flags": {
                        "quality_ok": True,
                        "quality_complete": complete,
                        "teacher_reviewed": reviewed,
                    },
                }
            )
            for index in range(count)
        ]

    def _row(self, event_id, *, complete=True, reviewed=True):
        return EventFeatureRow.from_summary(
            {
                "person_id": "p1",
                "event_id": event_id,
                "event_type": "task_start",
                "quality_flags": {
                    "quality_ok": True,
                    "quality_complete": complete,
                    "teacher_reviewed": reviewed,
                },
            }
        )
    def test_demo_subprocess_is_synthetic_and_deterministic(self):
        command = [sys.executable, "-m", "app.ondamm_personalization_cli", "--demo"]
        first = subprocess.run(command, capture_output=True, text=True, check=True)
        second = subprocess.run(command, capture_output=True, text=True, check=True)
        self.assertEqual(first.stdout, second.stdout)
        payload = json.loads(first.stdout)
        self.assertTrue(payload["synthetic"])
        self.assertFalse(payload["abstained"])

    def test_main_writes_only_requested_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(main(["--demo", "--output", str(output)]), 0)
            self.assertEqual(json.loads(stdout.getvalue()), json.loads(output.read_text()))
            files = {path.relative_to(directory).as_posix() for path in Path(directory).rglob("*") if path.is_file()}
            self.assertEqual(files, {"result.json"})

    def test_strict_invalid_input_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = Path(directory) / "rows.json"
            labels = Path(directory) / "labels.json"
            rows.write_text(json.dumps({"not": "an array"}))
            labels.write_text("[]")
            self.assertEqual(main(["--rows-json", str(rows), "--labels-json", str(labels)]), 2)

    def test_fractional_serialized_categorical_row_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            rows_path = Path(directory) / "rows.json"
            labels_path = Path(directory) / "labels.json"
            payload = self._row("e0").to_dict()
            payload["features"][0] = 0.5
            payload["features"][1] = 0.5
            rows_path.write_text(json.dumps([payload]))
            labels_path.write_text(json.dumps([TeacherLabel("e0", "visual_schedule", "helpful", True).to_dict()]))
            self.assertEqual(main(["--rows-json", str(rows_path), "--labels-json", str(labels_path)]), 2)
    def test_sparse_evidence_abstains_structurally(self):
        row = EventFeatureRow.from_summary(
            {
                "person_id": "p1",
                "event_id": "e1",
                "event_type": "task_start",
                "quality_flags": {
                    "quality_ok": True,
                    "quality_complete": True,
                    "teacher_reviewed": True,
                },
            }
        )
        label = TeacherLabel("e1", "visual_schedule", "helpful", teacher_approved=True)
        result = run(
            rows=[row],
            labels=[label],
            person_id="p1",
            min_samples=3,
            min_confidence=0.55,
            recommend=False,
        )
        self.assertTrue(result["abstained"])
        self.assertEqual(result["abstention_reason"], "sparse")
        self.assertEqual(result["sample_count"], 1)
    def test_sample_count_and_reason_use_only_positive_evidence(self):
        rows = self._rows()
        labels = [
            TeacherLabel("e0", "visual_schedule", "not_helpful", teacher_approved=True),
            TeacherLabel("e1", "visual_schedule", "helpful", teacher_approved=True),
            TeacherLabel("e2", "visual_schedule", "effective", teacher_approved=True),
        ]
        result = run(
            rows=rows,
            labels=labels,
            person_id="p1",
            min_samples=2,
            min_confidence=0.55,
            recommend=False,
        )
        self.assertFalse(result["abstained"])
        self.assertEqual(result["sample_count"], 2)

        negative_only = run(
            rows=rows,
            labels=[
                TeacherLabel(f"e{i}", "visual_schedule", "not_helpful", teacher_approved=True)
                for i in range(3)
            ],
            person_id="p1",
            min_samples=1,
            min_confidence=0.55,
            recommend=False,
        )
        self.assertTrue(negative_only["abstained"])
        self.assertEqual(negative_only["sample_count"], 0)
        self.assertEqual(negative_only["abstention_reason"], "insufficient_positive_evidence")

    def test_complete_and_reviewed_quality_gates(self):
        rows = [
            self._row("incomplete", complete=False),
            self._row("unreviewed", reviewed=False),
            self._row("eligible"),
        ]
        labels = [
            TeacherLabel(event_id, "visual_schedule", "helpful", teacher_approved=True)
            for event_id in ("incomplete", "unreviewed", "eligible")
        ]
        result = run(
            rows=rows,
            labels=labels,
            person_id="p1",
            min_samples=1,
            min_confidence=0.55,
            recommend=False,
        )
        self.assertFalse(result["abstained"])
        self.assertEqual(result["sample_count"], 1)
        self.assertEqual(result["evidence_ids"], ["eligible"])

    def test_multi_sample_provenance_includes_only_approved_positive_rows(self):
        rows = [
            self._row("approved-a"),
            self._row("approved-b"),
            self._row("approved-c"),
            self._row("negative"),
            EventFeatureRow.from_summary(
                {
                    "person_id": "p1",
                    "event_id": "low-quality",
                    "event_type": "task_start",
                    "quality_flags": {
                        "quality_ok": False,
                        "quality_complete": True,
                        "teacher_reviewed": True,
                    },
                }
            ),
            self._row("unapproved"),
        ]
        labels = [
            TeacherLabel("approved-a", "visual_schedule", "helpful", teacher_approved=True),
            TeacherLabel("approved-b", "visual_schedule", "effective", teacher_approved=True),
            TeacherLabel("approved-c", "visual_schedule", "helpful", teacher_approved=True),
            TeacherLabel("negative", "visual_schedule", "not_helpful", teacher_approved=True),
            TeacherLabel("low-quality", "visual_schedule", "helpful", teacher_approved=True),
            TeacherLabel("unapproved", "visual_schedule", "helpful", teacher_approved=False),
        ]
        result = run(
            rows=rows,
            labels=labels,
            person_id="p1",
            min_samples=3,
            min_confidence=0.55,
            recommend=True,
        )
        self.assertFalse(result["abstained"])
        self.assertEqual(result["sample_count"], 3)
        self.assertEqual(result["support_label"], "visual_schedule")
        self.assertEqual(result["evidence_ids"], ["approved-a", "approved-b", "approved-c"])
        self.assertEqual(
            result["recommendations"][0]["evidence_ids"],
            ["approved-a", "approved-b", "approved-c"],
        )

    def test_recommendation_requires_explicit_approval_flag(self):
        rows = [
            EventFeatureRow.from_summary(
                {
                    "person_id": "p1",
                    "event_id": f"e{i}",
                    "event_type": "task_start",
                    "quality_flags": {
                        "quality_ok": True,
                        "quality_complete": True,
                        "teacher_reviewed": True,
                    },
                }
            )
            for i in range(3)
        ]
        labels = [TeacherLabel(f"e{i}", "visual_schedule", "helpful", teacher_approved=True) for i in range(3)]
        without = run(rows=rows, labels=labels, person_id="p1", min_samples=3, min_confidence=0.55, recommend=False)
        with_approval = run(rows=rows, labels=labels, person_id="p1", min_samples=3, min_confidence=0.55, recommend=True)
        self.assertEqual(without["recommendations"], [])
        self.assertEqual(len(with_approval["recommendations"]), 1)
        self.assertTrue(with_approval["recommendations"][0]["teacher_approved"])


if __name__ == "__main__":
    unittest.main()
