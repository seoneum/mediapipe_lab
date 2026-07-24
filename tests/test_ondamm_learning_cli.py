from __future__ import annotations

import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from ondamm_event_recording import EventMetadata  # noqa: E402
from ondamm_learning_cli import (  # noqa: E402
    ObservationAccumulator,
    RunCapture,
    build_educator_notes,
    deterministic_demo_finished_at,
    normalize_demo_event,
    prepare_output_dirs,
    resolve_clips_dir,
)


class OnDammLearningCliTests(unittest.TestCase):
    def test_deterministic_demo_finished_at_handles_minute_rollover(self) -> None:
        self.assertEqual(
            deterministic_demo_finished_at(61.0),
            "2026-01-01T00:01:01+00:00",
        )
        self.assertEqual(
            deterministic_demo_finished_at(12.0),
            "2026-01-01T00:00:12+00:00",
        )

    def test_normalize_demo_event_rewrites_event_identity_without_changing_timing(self) -> None:
        event = EventMetadata.create(
            event_type="gaze_diverted",
            start_timestamp=1.6,
            end_timestamp=3.6,
            trigger_values={"gaze_zone": "left", "duration_seconds": 2.0},
        )
        normalized = normalize_demo_event(event, index=1)
        self.assertEqual(normalized.event_id, "demo-01-gaze_diverted")
        self.assertEqual(normalized.start_timestamp, 1.6)
        self.assertEqual(normalized.end_timestamp, 3.6)
        self.assertEqual(normalized.created_at, "2026-01-01T00:00:00+00:00")

    def test_prepare_output_dirs_clears_stale_recording_artifacts_for_no_record_run(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ondamm-learning-cli-") as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "run"
            clips_dir = root / "clips"
            output_dir.mkdir(parents=True, exist_ok=True)
            clips_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "event_recording.json").write_text("{}", encoding="utf-8")
            (clips_dir / "old.mp4").write_bytes(b"old")

            prepare_output_dirs(output_dir, clips_dir, record_events=False)

            self.assertFalse((output_dir / "event_recording.json").exists())
            self.assertEqual(list(clips_dir.glob("*")), [])

    def test_resolve_clips_dir_keeps_recordings_inside_output_dir(self) -> None:
        output_dir = Path("/tmp/ondamm-run-root")
        self.assertEqual(
            resolve_clips_dir(output_dir),
            (output_dir / "event-clips").resolve(),
        )

    def test_build_educator_notes_reports_detected_events_even_without_recording(self) -> None:
        capture = RunCapture(
            mode="demo",
            started_at="2026-01-01T00:00:00+00:00",
            finished_at="2026-01-01T00:00:12+00:00",
            duration_seconds=12.0,
            observations=ObservationAccumulator(
                frame_count=12,
                face_present_frames=9,
                gaze_zone_counts=Counter({"center": 8, "left": 4}),
                posture_proxy_counts=Counter({"centered": 12}),
            ),
            detected_events=[
                EventMetadata.create(
                    event_type="gaze_diverted",
                    start_timestamp=1.6,
                    end_timestamp=3.6,
                    trigger_values={"gaze_zone": "left", "duration_seconds": 2.0},
                )
            ],
            recorded_events=[],
            clip_directory=None,
        )

        notes = build_educator_notes(capture)

        joined = " ".join(notes)
        self.assertIn("지속 이벤트 1건을 관찰했습니다", joined)
        self.assertIn("--record-events 없이 실행되어", joined)
        self.assertIn("완료 step 표시는 자동 추정하지 않았으며", joined)


if __name__ == "__main__":
    unittest.main()
