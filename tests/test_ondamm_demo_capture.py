from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from ondamm_demo_overlay import render_demo_overlay  # noqa: E402
from ondamm_live_temporal_demo import LiveTemporalDemo  # noqa: E402
from ondamm_temporal_encoder import (  # noqa: E402
    TemporalEncoderSpec,
    build_torch_encoder,
    export_temporal_encoder_checkpoint,
)
from ondamm_temporal_features import (  # noqa: E402
    PersonalMotionCalibrator,
    TemporalFeatureAdapter,
)


def demo_signal(*, motion: float = 0.0, face_detected: bool = True) -> dict:
    points = np.zeros((478, 3), dtype=np.float32)
    points[:, 0] = np.linspace(0.3, 0.7, 478)
    points[:, 1] = np.linspace(0.25, 0.75, 478)
    return {
        "face_detected": face_detected,
        "landmarks": points,
        "canonical_landmarks": points,
        "blendshapes": {"a": 0.4},
        "motion_mean": motion / 3,
        "motion_max": motion,
        "motion_mouth": motion,
        "motion_left_eye": 0.0,
        "motion_right_eye": 0.0,
        "motion_left_brow": 0.0,
        "motion_right_brow": 0.0,
    }


class OndammDemoCaptureTests(unittest.TestCase):
    def test_live_signal_adapter_matches_checkpoint_order(self) -> None:
        names = ("motion_mouth", "bs_a", "geom_abs_mouth_width", "motion_eyes")
        values = TemporalFeatureAdapter(names).from_signal(demo_signal(motion=0.02))

        self.assertEqual(tuple(values), names)
        self.assertEqual(values["bs_a"], 0.4)
        self.assertEqual(values["motion_mouth"], 0.02)
        self.assertEqual(values["motion_eyes"], 0.0)
        self.assertGreater(values["geom_abs_mouth_width"], 0.0)

    def test_personal_motion_calibration_separates_neutral_and_motion(self) -> None:
        calibrator = PersonalMotionCalibrator(calibration_seconds=0.2, scale_floor=0.001)
        self.assertEqual(calibrator.add(timestamp=0.0, raw_motion=0.001, face_detected=True), 0.0)
        self.assertEqual(calibrator.add(timestamp=0.2, raw_motion=0.001, face_detected=True), 0.0)
        self.assertTrue(calibrator.ready)
        self.assertGreater(calibrator.add(timestamp=0.3, raw_motion=0.02, face_detected=True), 4.0)

    def test_overlay_draws_landmarks_and_demo_state_without_resizing(self) -> None:
        frame = np.zeros((480, 720, 3), dtype=np.uint8)
        rendered = render_demo_overlay(
            frame,
            demo_signal(motion=0.02),
            {
                "temporal_enabled": True,
                "motion_score": 8.0,
                "motion_active": True,
                "lifecycle": "REPEATING_CANDIDATE",
                "candidate_id": "candidate-demo",
                "occurrence_count": 3,
                "occurrence_threshold": 3,
                "event_saved": True,
            },
        )

        self.assertEqual(rendered.shape, frame.shape)
        self.assertGreater(int(rendered.sum()), 0)
        self.assertEqual(int(frame.sum()), 0)

    def test_live_demo_saves_only_third_independent_episode_with_overlay_clip(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ondamm-live-demo-") as temp_dir:
            root = Path(temp_dir)
            spec = TemporalEncoderSpec(
                feature_names=("bs_a", "geom_abs_mouth_width", "motion_mouth"),
                sequence_length=3,
                stride_frames=1,
                channels=(4,),
                embedding_dim=4,
            )
            model = build_torch_encoder(spec)
            for value in model.state_dict().values():
                value.fill_(0.1)
            checkpoint = root / "encoder.pt"
            export_temporal_encoder_checkpoint(
                checkpoint,
                spec=spec,
                model_state_dict=model.state_dict(),
                normalization_mean=[0.0, 0.0, 0.0],
                normalization_std=[1.0, 1.0, 1.0],
            )
            demo = LiveTemporalDemo(
                child_id="demo-child",
                checkpoint_path=checkpoint,
                pattern_memory_root=root / "pattern-memory",
                clips_dir=root / "clips",
                event_metadata_path=root / "event_recording.json",
                record_events=True,
                clip_fps=5.0,
                calibration_seconds=0.0,
                onset_z=4.0,
                offset_z=2.0,
                min_episode_seconds=0.2,
                refractory_seconds=0.5,
                min_occurrences_for_clip=3,
                strong_candidate_occurrences=5,
                pre_seconds=0.2,
                post_seconds=0.2,
            )
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            schedule = [
                (0.0, 0.0),
                (0.1, 0.0),
                (0.2, 0.02),
                (0.4, 0.02),
                (0.6, 0.0),
                (1.2, 0.02),
                (1.4, 0.02),
                (1.6, 0.0),
                (2.2, 0.02),
                (2.4, 0.02),
                (2.6, 0.0),
                (2.8, 0.0),
            ]
            requested = []
            finalized = []
            for timestamp, motion in schedule:
                signal = demo_signal(motion=motion)
                overlay = render_demo_overlay(frame, signal, demo.overlay_status(timestamp=timestamp))
                result = demo.process(timestamp=timestamp, signal=signal, frame_for_record=overlay)
                requested.extend(result.requested_events)
                finalized.extend(result.finalized_events)

            self.assertEqual(len(requested), 1)
            self.assertEqual(len(finalized), 1)
            self.assertEqual(finalized[0].trigger_values["occurrence_count"], 3)
            self.assertTrue(Path(finalized[0].clip_path).is_file())
            metadata = json.loads((root / "event_recording.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["recorded_event_count"], 1)
            self.assertEqual(metadata["events"][0]["event_type"], "temporal_movement_candidate")
            state = demo.overlay_status(timestamp=2.8)
            self.assertTrue(state["event_saved"])
            self.assertEqual(state["occurrence_count"], 3)


if __name__ == "__main__":
    unittest.main()
