from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from ondamm_event_recording import (  # noqa: E402
    EventMetadata,
    EventObservation,
    EventRecordingPolicy,
    LocalEventClipRecorder,
    SustainedEventDetector,
)


class OnDammEventRecordingTests(unittest.TestCase):
    def test_detector_emits_only_after_sustained_duration(self) -> None:
        policy = EventRecordingPolicy(face_missing_min_seconds=2.0)
        detector = SustainedEventDetector(policy)

        events = detector.add_observation(
            EventObservation(timestamp=0.0, face_present=False, gaze_zone="unknown", posture_proxy="centered")
        )
        self.assertEqual(events, [])

        events = detector.add_observation(
            EventObservation(timestamp=1.9, face_present=False, gaze_zone="unknown", posture_proxy="centered")
        )
        self.assertEqual(events, [])

        events = detector.add_observation(
            EventObservation(timestamp=2.0, face_present=False, gaze_zone="unknown", posture_proxy="centered")
        )
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.event_type, "face_missing")
        self.assertEqual(event.start_timestamp, 0.0)
        self.assertEqual(event.end_timestamp, 2.0)
        self.assertFalse(event.trigger_values["face_present"])
        self.assertEqual(event.trigger_values["duration_seconds"], 2.0)

        events = detector.add_observation(
            EventObservation(timestamp=2.5, face_present=False, gaze_zone="unknown", posture_proxy="centered")
        )
        self.assertEqual(events, [])

        detector.add_observation(
            EventObservation(timestamp=3.0, face_present=True, gaze_zone="center", posture_proxy="centered")
        )
        events = detector.add_observation(
            EventObservation(timestamp=4.0, face_present=False, gaze_zone="unknown", posture_proxy="centered")
        )
        self.assertEqual(events, [])
        events = detector.add_observation(
            EventObservation(timestamp=6.0, face_present=False, gaze_zone="unknown", posture_proxy="centered")
        )
        self.assertEqual(len(events), 1)

    def test_detector_uses_gaze_and_posture_labels_without_scoring(self) -> None:
        policy = EventRecordingPolicy(gaze_diverted_min_seconds=1.0, posture_shifted_min_seconds=1.5)
        detector = SustainedEventDetector(policy)

        gaze_events = detector.add_observation(
            EventObservation(timestamp=0.0, face_present=True, gaze_zone="left", posture_proxy="centered")
        )
        self.assertEqual(gaze_events, [])
        gaze_events = detector.add_observation(
            EventObservation(timestamp=1.0, face_present=True, gaze_zone="left", posture_proxy="centered")
        )
        self.assertEqual(len(gaze_events), 1)
        self.assertEqual(gaze_events[0].event_type, "gaze_diverted")
        self.assertEqual(gaze_events[0].trigger_values["gaze_zone"], "left")
        self.assertNotIn("score", gaze_events[0].trigger_values)

        detector.add_observation(
            EventObservation(timestamp=2.0, face_present=True, gaze_zone="center", posture_proxy="centered")
        )
        posture_events = detector.add_observation(
            EventObservation(timestamp=3.0, face_present=True, gaze_zone="center", posture_proxy="left_shifted")
        )
        self.assertEqual(posture_events, [])
        posture_events = detector.add_observation(
            EventObservation(timestamp=4.5, face_present=True, gaze_zone="center", posture_proxy="left_shifted")
        )
        self.assertEqual(len(posture_events), 1)
        self.assertEqual(posture_events[0].event_type, "posture_shifted")
        self.assertEqual(posture_events[0].trigger_values["posture_proxy"], "left_shifted")

    def test_detector_emits_only_configured_sustained_facial_movement(self) -> None:
        policy = EventRecordingPolicy(
            facial_movement_min_seconds=0.4,
            target_facial_movement_labels=("lip_corner_pull",),
        )
        detector = SustainedEventDetector(policy)

        self.assertEqual(
            detector.add_observation(
                EventObservation(0.0, True, "center", "centered", ("brow_raise",))
            ),
            [],
        )
        self.assertEqual(
            detector.add_observation(
                EventObservation(1.0, True, "center", "centered", ("lip_corner_pull",))
            ),
            [],
        )
        events = detector.add_observation(
            EventObservation(1.4, True, "center", "centered", ("lip_corner_pull",))
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "facial_movement_detected")
        self.assertEqual(events[0].trigger_values["facial_movement_labels"], ["lip_corner_pull"])
        self.assertNotIn("emotion", events[0].trigger_values)

    def test_local_clip_recorder_writes_npz_clip_with_metadata(self) -> None:
        policy = EventRecordingPolicy(pre_event_buffer_seconds=1.0, face_missing_min_seconds=2.0)
        with tempfile.TemporaryDirectory(prefix="ondamm-event-clips-") as temp_dir:
            recorder = LocalEventClipRecorder(
                policy=policy,
                output_dir=Path(temp_dir),
                recording_enabled=True,
            )

            for timestamp, value in enumerate([10, 20, 30, 40]):
                recorder.add_frame(frame=np.full((2, 2, 3), value, dtype=np.uint8), timestamp=float(timestamp))

            event = EventMetadata.create(
                event_type="face_missing",
                start_timestamp=1.0,
                end_timestamp=3.0,
                trigger_values={"face_present": False, "duration_seconds": 2.0},
            )
            recorded_event = recorder.record_event(event)

            self.assertIsNotNone(recorded_event.clip_path)
            clip_path = Path(recorded_event.clip_path or "")
            self.assertTrue(clip_path.exists())

            clip = np.load(clip_path, allow_pickle=False)
            frames = clip["frames"]
            timestamps = clip["timestamps"]
            metadata = clip["event_metadata"].item()

            self.assertEqual(frames.shape, (4, 2, 2, 3))
            self.assertEqual(timestamps.tolist(), [0.0, 1.0, 2.0, 3.0])
            self.assertIn(recorded_event.event_id, metadata)
            self.assertIn('"event_type": "face_missing"', metadata)

    def test_local_clip_recorder_can_be_disabled(self) -> None:
        recorder = LocalEventClipRecorder(recording_enabled=False)
        recorder.add_frame(frame=np.zeros((1, 1, 3), dtype=np.uint8), timestamp=0.0)
        event = EventMetadata.create(
            event_type="gaze_diverted",
            start_timestamp=0.0,
            end_timestamp=2.0,
            trigger_values={"gaze_zone": "right", "duration_seconds": 2.0},
        )

        recorded_event = recorder.record_event(event)
        self.assertIsNone(recorded_event.clip_path)

    def test_review_buffer_is_downscaled_and_throttled(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ondamm-review-buffer-") as temp_dir:
            recorder = LocalEventClipRecorder(
                policy=EventRecordingPolicy(pre_event_buffer_seconds=1.0),
                output_dir=Path(temp_dir),
                buffer_enabled=True,
                persist_enabled=True,
                output_format="npz",
                buffer_frame_size=(64, 36),
                buffer_fps=10.0,
            )
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            recorder.add_frame(frame=frame, timestamp=0.0)
            recorder.add_frame(frame=frame, timestamp=0.05)
            recorder.add_frame(frame=frame, timestamp=0.1)
            event = EventMetadata.create(
                event_type="facial_movement_detected",
                start_timestamp=0.0,
                end_timestamp=0.1,
                trigger_values={"duration_seconds": 0.1},
            )
            recorded = recorder.record_event(event)
            with np.load(recorded.clip_path, allow_pickle=False) as archive:
                self.assertEqual(archive["frames"].shape, (2, 36, 64, 3))
                self.assertEqual(archive["timestamps"].tolist(), [0.0, 0.1])

    def test_child_stop_discards_ram_frames_and_pending_clip(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ondamm-stop-") as temp_dir:
            recorder = LocalEventClipRecorder(
                policy=EventRecordingPolicy(clip_tail_seconds=1.0),
                output_dir=Path(temp_dir),
                recording_enabled=True,
            )
            recorder.add_frame(frame=np.zeros((2, 2, 3), dtype=np.uint8), timestamp=0.0)
            recorder.record_event(
                EventMetadata.create(
                    event_type="facial_movement_detected",
                    start_timestamp=0.0,
                    end_timestamp=0.2,
                    trigger_values={"duration_seconds": 0.2},
                )
            )
            self.assertEqual(recorder.pending_event_count, 1)

            recorder.discard_buffered()

            self.assertEqual(recorder.pending_event_count, 0)
            self.assertEqual(recorder.buffered_frame_count, 0)
            self.assertEqual(list(Path(temp_dir).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
