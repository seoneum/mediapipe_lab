from __future__ import annotations

import csv
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

from ondamm_demo_overlay import (  # noqa: E402
    GuidanceTiming,
    GuidedDemoCountdown,
    render_demo_overlay,
)
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
        calibrator = PersonalMotionCalibrator(
            calibration_seconds=0.2,
            minimum_valid_samples=2,
            minimum_face_coverage=1.0,
            minimum_effective_duration=0.2,
            scale_floor=0.001,
        )
        self.assertEqual(calibrator.add(timestamp=0.0, raw_motion=0.001, face_detected=True), 0.0)
        self.assertEqual(calibrator.add(timestamp=0.2, raw_motion=0.001, face_detected=True), 0.0)
        self.assertTrue(calibrator.ready)
        self.assertGreater(calibrator.add(timestamp=0.3, raw_motion=0.02, face_detected=True), 4.0)

    def test_calibration_requires_minimum_valid_samples(self) -> None:
        calibrator = PersonalMotionCalibrator(
            calibration_seconds=0.2,
            minimum_valid_samples=3,
            minimum_face_coverage=0.5,
            minimum_effective_duration=0.2,
        )
        calibrator.add(timestamp=0.0, raw_motion=0.001, face_detected=True)
        calibrator.add(timestamp=0.2, raw_motion=0.001, face_detected=True)
        self.assertFalse(calibrator.ready)
        self.assertIn("valid face samples", calibrator.status(0.2)["missing_requirements"])
        calibrator.add(timestamp=0.3, raw_motion=0.001, face_detected=True)
        self.assertTrue(calibrator.ready)

    def test_calibration_requires_face_coverage(self) -> None:
        calibrator = PersonalMotionCalibrator(
            calibration_seconds=0.2,
            minimum_valid_samples=2,
            minimum_face_coverage=0.75,
            minimum_effective_duration=0.1,
        )
        calibrator.add(timestamp=0.0, raw_motion=0.0, face_detected=False)
        calibrator.add(timestamp=0.1, raw_motion=0.001, face_detected=True)
        calibrator.add(timestamp=0.2, raw_motion=0.001, face_detected=True)
        self.assertFalse(calibrator.ready)
        self.assertIn("face coverage", calibrator.status(0.2)["missing_requirements"])
        calibrator.add(timestamp=0.3, raw_motion=0.001, face_detected=True)
        self.assertTrue(calibrator.ready)

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

    def test_guided_countdown_shows_large_three_two_one_move_and_neutral_cues(self) -> None:
        guidance = GuidedDemoCountdown(
            GuidanceTiming(
                stable_seconds=1.0,
                announcement_seconds=0.8,
                countdown_seconds=3,
                move_seconds=1.0,
                neutral_seconds=2.0,
                repeats=3,
            )
        )
        ready = {
            "calibration_ready": True,
            "face_lost": False,
            "warming_up": False,
            "motion_active": False,
        }

        self.assertEqual(guidance.decorate(ready, timestamp=0.0)["guidance_phase"], "settling")
        self.assertEqual(guidance.decorate(ready, timestamp=1.0)["guidance_phase"], "announcement")
        self.assertEqual(guidance.decorate(ready, timestamp=1.81)["guidance_title"], "3")
        self.assertEqual(guidance.decorate(ready, timestamp=2.81)["guidance_title"], "2")
        self.assertEqual(guidance.decorate(ready, timestamp=3.81)["guidance_title"], "1")
        self.assertEqual(guidance.decorate(ready, timestamp=4.81)["guidance_phase"], "move")
        self.assertEqual(guidance.decorate(ready, timestamp=5.81)["guidance_phase"], "neutral")
        self.assertEqual(guidance.decorate(ready, timestamp=7.81)["guidance_cycle"], 2)

    def test_guided_countdown_resets_when_face_is_lost(self) -> None:
        guidance = GuidedDemoCountdown(GuidanceTiming(stable_seconds=0.1))
        ready = {
            "calibration_ready": True,
            "face_lost": False,
            "warming_up": False,
            "motion_active": False,
        }
        guidance.decorate(ready, timestamp=0.0)
        self.assertEqual(guidance.decorate(ready, timestamp=0.2)["guidance_phase"], "announcement")
        lost = dict(ready, face_lost=True)
        self.assertEqual(guidance.decorate(lost, timestamp=0.3)["guidance_phase"], "face_lost")
        self.assertEqual(guidance.decorate(ready, timestamp=0.4)["guidance_phase"], "settling")

    def test_large_guidance_is_visualization_only(self) -> None:
        frame = np.zeros((480, 720, 3), dtype=np.uint8)
        rendered = render_demo_overlay(
            frame,
            demo_signal(),
            {
                "temporal_enabled": True,
                "calibration_ready": True,
                "guidance_phase": "countdown",
                "guidance_title": "3",
                "guidance_fallback": "3",
                "guidance_detail": "첫 번째 동작 준비",
            },
        )

        self.assertEqual(int(frame.sum()), 0)
        self.assertGreater(int(rendered[-180:].sum()), 0)

    def test_live_demo_saves_only_third_independent_episode_with_overlay_clip(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ondamm-live-demo-") as temp_dir:
            root = Path(temp_dir)
            feature_names = tuple(
                ["bs_a"]
                + [f"bs_fixture_{index:02d}" for index in range(51)]
                + ["geom_abs_mouth_width"]
                + [f"geom_abs_fixture_{index:02d}" for index in range(17)]
                + ["motion_mouth"]
                + [f"motion_fixture_{index:02d}" for index in range(8)]
            )
            spec = TemporalEncoderSpec(
                feature_names=feature_names,
                sequence_length=60,
                stride_frames=5,
                channels=(64, 64, 64),
                kernel_size=3,
                dropout=0.2,
                embedding_dim=64,
            )
            model = build_torch_encoder(spec)
            for value in model.state_dict().values():
                value.fill_(0.1)
            checkpoint = root / "encoder.pt"
            digest = export_temporal_encoder_checkpoint(
                checkpoint,
                spec=spec,
                model_state_dict=model.state_dict(),
                normalization_mean=np.zeros(79, dtype=np.float32),
                normalization_std=np.ones(79, dtype=np.float32),
                metadata={
                    "held_out_participant": "fixture-held-out",
                    "train_participants": ["fixture-train-a", "fixture-train-b"],
                    "best_epoch": 1,
                    "normalization": "fixture robust center/scale",
                },
            )
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "features": list(feature_names),
                        "feature_counts": {"blendshape": 52, "geometry": 18, "motion": 9, "total": 79},
                        "sequence": {"causal": True, "seq_len_frames": 60, "stride_frames": 5},
                        "model": {"channels": "64,64,64", "kernel_size": 3},
                        "encoder_checkpoints": {
                            "fixture": {"path": str(checkpoint), "sha256": digest}
                        },
                    }
                ),
                encoding="utf-8",
            )
            demo = LiveTemporalDemo(
                child_id="demo-child",
                session_id="future-session-03",
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
            schedule = []
            for index in range(126):
                timestamp = round(index * 0.1, 3)
                moving = any(start <= timestamp <= start + 1.0 for start in (6.0, 8.0, 10.0))
                schedule.append((timestamp, 0.02 if moving else 0.0))
            requested = []
            finalized = []
            tail_ready_overlay_seen = False
            for timestamp, motion in schedule:
                signal = demo_signal(motion=motion)
                status = demo.overlay_status(timestamp=timestamp)
                tail_ready_overlay_seen = tail_ready_overlay_seen or bool(status["event_saved"])
                overlay = render_demo_overlay(frame, signal, status)
                result = demo.process(timestamp=timestamp, signal=signal, frame_for_record=overlay)
                requested.extend(result.requested_events)
                finalized.extend(result.finalized_events)

            self.assertEqual(len(requested), 1)
            self.assertEqual(len(finalized), 1)
            self.assertTrue(tail_ready_overlay_seen)
            self.assertEqual(finalized[0].trigger_values["occurrence_count"], 3)
            self.assertTrue(Path(finalized[0].clip_path).is_file())
            metadata = json.loads((root / "event_recording.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["recorded_event_count"], 1)
            self.assertEqual(metadata["events"][0]["event_type"], "repeating_micro_motion")
            state = demo.overlay_status(timestamp=12.5)
            self.assertTrue(state["event_saved"])
            self.assertEqual(state["occurrence_count"], 3)
            with demo.detection_log_path.open("r", encoding="utf-8", newline="") as handle:
                detection_rows = list(csv.DictReader(handle))
            self.assertEqual(len(detection_rows), 3)
            self.assertTrue(all(row["child_id"] == "demo-child" for row in detection_rows))
            self.assertTrue(
                all(row["session_id"] == "future-session-03" for row in detection_rows)
            )
            self.assertEqual(detection_rows[-1]["lifecycle"], "REPEATING_CANDIDATE")


if __name__ == "__main__":
    unittest.main()
