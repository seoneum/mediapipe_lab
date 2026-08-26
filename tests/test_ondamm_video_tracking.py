"""Hermetic tests for ON DAMM person tracking with persistent global IDs.

No network, no real model loads in the default run: the detector is mocked for
wiring tests, the resolver uses deterministic stub embeddings, videos are
synthetic mp4s written into temp dirs (auto-cleaned). The real-model end-to-end
check is opt-in behind ``@pytest.mark.smoke`` plus an env gate.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from ondamm_video_tracking import (  # noqa: E402
    FaceEmbedder,
    FrameReader,
    GlobalIdResolver,
    PersonTracker,
    TrackObservation,
    VideoInputError,
    cosine,
    l2_normalize,
    padded_crop,
)

DIM = 512


# ---------------------------------------------------------------------------
# Deterministic fixtures
# ---------------------------------------------------------------------------


def _unit_vec(index: int, dim: int = DIM) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[index % dim] = 1.0
    return v


def _noisy(vec: np.ndarray, seed: int = 0, scale: float = 0.01) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return l2_normalize(vec + scale * rng.standard_normal(vec.shape[0]).astype(np.float32))


def _write_synthetic_video(
    path: Path,
    frames: int = 120,
    width: int = 320,
    height: int = 240,
    fps: int = 30,
) -> Path:
    """Two colored rectangles moving along crossing diagonals, then separating."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open {path}")
    for i in range(frames):
        t = i / max(frames - 1, 1)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        # red: top-left -> bottom-right ; blue: bottom-left(top start) opposite diagonal
        red_cx, red_cy = int(40 + 240 * t), int(60 + 120 * t)
        blue_cx, blue_cy = int(280 - 240 * t), int(180 - 120 * t)
        cv2.rectangle(canvas, (red_cx - 18, red_cy - 18), (red_cx + 18, red_cy + 18), (0, 0, 255), -1)
        cv2.rectangle(canvas, (blue_cx - 18, blue_cy - 18), (blue_cx + 18, blue_cy + 18), (255, 0, 0), -1)
        writer.write(canvas)
    writer.release()
    probe = cv2.VideoCapture(str(path))
    opened = probe.isOpened()
    probe.release()
    if not opened:
        raise RuntimeError(f"synthetic video {path} is not readable after writing")
    return path


class _StubBoxes:
    def __init__(self, rows: list[tuple[int, float, list[float]]]) -> None:
        self.id = [r[0] for r in rows]
        self.conf = [r[1] for r in rows]
        self.xyxy = [r[2] for r in rows]

    def __len__(self) -> int:
        return len(self.id)


class _StubResult:
    def __init__(self, boxes: _StubBoxes | None) -> None:
        self.boxes = boxes


class _ScriptedDetector:
    """Mock detector: no model load. Returns scripted boxes by call count."""

    def __init__(self, phases: list[list[tuple[int, float, list[float]]]], frames_per_phase: int) -> None:
        self.phases = phases
        self.frames_per_phase = frames_per_phase
        self.calls: list[dict] = []

    def track(self, frame: np.ndarray, **kwargs):
        self.calls.append(kwargs)
        raw_phase = (len(self.calls) - 1) // self.frames_per_phase
        phase = min(raw_phase, len(self.phases) - 1)
        rows = self.phases[phase]
        return [_StubResult(_StubBoxes(rows) if rows else None)]


class _PositionStubEmbedder:
    """Deterministic stub: embedding chosen by rounded bbox center (no insightface)."""

    def __init__(self, mapping: dict[tuple[int, int], np.ndarray]) -> None:
        self.mapping = mapping
        self.requested: list[tuple[int, int]] = []

    def embed_crop(self, frame: np.ndarray, bbox_xyxy: list[float]) -> np.ndarray | None:
        x1, y1, x2, y2 = bbox_xyxy
        key = (int((x1 + x2) // 2), int((y1 + y2) // 2))
        self.requested.append(key)
        return self.mapping.get(key)


class OndammVideoTrackingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp_path = Path(self._tmp.name)

    # -- FrameReader --------------------------------------------------------

    def test_videoreader_missing_file_raises(self) -> None:
        with self.assertRaises(VideoInputError) as ctx:
            FrameReader("/nonexistent.mp4")
        self.assertIn("cannot open", str(ctx.exception))
        self.assertIn("/nonexistent.mp4", str(ctx.exception))

    def test_videoreader_corrupt_file_raises(self) -> None:
        bad = self.tmp_path / "corrupt.mp4"
        bad.write_bytes(b"this is not a video container" * 8)
        with self.assertRaises(VideoInputError) as ctx:
            FrameReader(bad)
        self.assertIn("cannot open", str(ctx.exception))

    def test_videoreader_yields_frames_with_timestamps(self) -> None:
        video = _write_synthetic_video(self.tmp_path / "synth.mp4", frames=120, fps=30)
        reader = FrameReader(video)
        self.assertEqual(reader.fps, 30.0)
        seen = []
        for frame_idx, ts_sec, frame in reader:
            seen.append((frame_idx, ts_sec))
            self.assertEqual(frame.shape, (240, 320, 3))
        reader.close()
        self.assertEqual(len(seen), 120)
        self.assertEqual([f for f, _ in seen], list(range(120)))
        self.assertTrue(all(b > a for (_, a), (_, b) in zip(seen, seen[1:])))
        self.assertAlmostEqual(seen[59][1], 59 / 30.0, places=6)

    # -- TrackObservation schema -------------------------------------------

    def test_trackobservation_schema_frozen_and_typed(self) -> None:
        obs = TrackObservation(
            frame_idx=3,
            ts_sec=0.1,
            track_id=7,
            global_id="unknown_0",
            bbox_xyxy=[1.0, 2.0, 3.0, 4.0],
            det_conf=0.9,
            low_confidence=True,
        )
        self.assertIsInstance(obs.frame_idx, int)
        self.assertIsInstance(obs.ts_sec, float)
        self.assertIsInstance(obs.track_id, int)
        self.assertIsInstance(obs.global_id, str)
        self.assertEqual(len(obs.bbox_xyxy), 4)
        self.assertTrue(all(isinstance(v, float) for v in obs.bbox_xyxy))
        self.assertIsInstance(obs.det_conf, float)
        self.assertIsInstance(obs.low_confidence, bool)
        with self.assertRaises(Exception):
            obs.det_conf = 0.0  # frozen schema: mutation must fail

    # -- GlobalIdResolver ----------------------------------------------------

    def test_resolver_binds_after_occlusion_gap(self) -> None:
        """Track vanishes >=30 synthetic frames, returns -> same global_id."""
        resolver = GlobalIdResolver(bind_threshold=0.45, margin=0.05)
        vec_a, vec_b = _unit_vec(0), _unit_vec(1)

        gids_first = [resolver.update(1, vec_a)[0] for _ in range(5)]
        gid_a = gids_first[0]
        # occlusion window: another person occupies >=30 frames while track 1 gone
        for _ in range(35):
            resolver.update(2, vec_b)
        # track 1 re-appears as a NEW tracker id with slightly noisy embedding
        gid_returned, low = resolver.update(3, _noisy(vec_a, seed=42))
        self.assertEqual(gid_returned, gid_a)
        self.assertFalse(low)

    def test_resolver_margin_rejects_ambiguous(self) -> None:
        """Two centroids within margin -> provisional unknown_N, low_confidence."""
        resolver = GlobalIdResolver(bind_threshold=0.45, margin=0.05)
        vec_a, vec_b = _unit_vec(0), _unit_vec(1)  # orthogonal: cos=0 -> separate clusters
        gid_a, _ = resolver.update(1, vec_a)
        gid_b, _ = resolver.update(2, vec_b)
        self.assertNotEqual(gid_a, gid_b)
        query = l2_normalize((vec_a + vec_b) / np.float32(2.0))  # ~0.707 to both
        gid_q, low_q = resolver.update(3, query)
        self.assertTrue(low_q)
        self.assertNotEqual(gid_q, gid_a)
        self.assertNotEqual(gid_q, gid_b)
        self.assertTrue(gid_q.startswith("unknown_"))
        # sanity: query really clears bind_threshold against both centroids
        self.assertGreaterEqual(cosine(query, resolver.centroid_of(gid_a)), 0.45)
        self.assertLess(cosine(query, resolver.centroid_of(gid_a))
                        - cosine(query, resolver.centroid_of(gid_b)), 0.05)

    def test_resolver_ema_alpha_is_0_9(self) -> None:
        resolver = GlobalIdResolver()
        e1, e2 = _unit_vec(0), _unit_vec(1)
        resolver.update(1, e1)
        gid, _ = resolver.update(1, e2)
        expected = l2_normalize(0.9 * e1 + 0.1 * e2)
        got = resolver.centroid_of(gid)
        self.assertIsNotNone(got)
        self.assertLess(float(np.abs(got - expected).max()), 1e-6)

    def test_resolver_distinct_people_get_distinct_provisional_ids(self) -> None:
        resolver = GlobalIdResolver()
        gid_a, low_a = resolver.update(1, _unit_vec(0))
        gid_b, low_b = resolver.update(2, _unit_vec(1))
        self.assertNotEqual(gid_a, gid_b)
        self.assertTrue(low_a and low_b)
        self.assertTrue(gid_a.startswith("unknown_") and gid_b.startswith("unknown_"))

    def test_resolver_wrong_first_embedding_converges_without_flapping(self) -> None:
        resolver = GlobalIdResolver()
        vec_a, vec_b = _unit_vec(0), _unit_vec(1)
        gid_a, _ = resolver.update(1, vec_a)
        gid_b, _ = resolver.update(2, vec_b)
        # track 2's first embedding was wrong (person B); consistent evidence
        # must accumulate in the EMA without ever flapping between clusters,
        # and the assigned cluster must converge to the true person A
        gids = [
            resolver.update(2, _noisy(vec_a, seed=7))[0] for _ in range(30)
        ]
        self.assertEqual(len(set(gids)), 1)
        final_centroid = resolver.centroid_of(gids[-1])
        self.assertIsNotNone(final_centroid)
        self.assertGreater(cosine(final_centroid, vec_a), 0.95)
        self.assertGreater(cosine(final_centroid, vec_a), cosine(final_centroid, vec_b))

    # -- PersonTracker wiring (mocked detector, no model load) ---------------

    def test_tracker_wiring_mocked_detector_recovers_identity(self) -> None:
        bbox_a = [40.0, 40.0, 100.0, 140.0]
        detector = _ScriptedDetector(
            phases=[
                [(1, 0.91, bbox_a)],   # frames 0..39: tracker id 1
                [(2, 0.88, bbox_a)],   # frames 40..79: SAME person, new tracker id
                [],                    # frames 80+: gone
            ],
            frames_per_phase=40,
        )
        embedder = _PositionStubEmbedder({(70, 90): _unit_vec(0)})
        tracker = PersonTracker(
            device="cpu",
            embedder=embedder,
            resolver=GlobalIdResolver(),
            sample_every=15,
            fps=30.0,
            model=detector,
        )
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        observations = []
        for idx in range(81):
            observations.extend(tracker.process_frame(frame, frame_idx=idx))

        self.assertTrue(detector.calls)
        call = detector.calls[0]
        self.assertTrue(call["persist"])
        self.assertEqual(call["classes"], [0])
        self.assertEqual(call["tracker"], "configs/botsort_static.yaml")

        early = [o for o in observations if o.frame_idx < 40]
        late = [o for o in observations if 40 <= o.frame_idx < 80]
        self.assertEqual(len(early), 40)
        self.assertEqual(len(late), 40)
        # identity survives the detector ID switch: same global_id throughout
        self.assertEqual({o.global_id for o in early}, {early[0].global_id})
        self.assertEqual({o.global_id for o in late}, {early[0].global_id})
        self.assertEqual({o.track_id for o in early}, {1})
        self.assertEqual({o.track_id for o in late}, {2})
        for o in observations[:1] + observations[40:41]:
            self.assertEqual(o.bbox_xyxy, bbox_a)
            self.assertAlmostEqual(o.ts_sec, o.frame_idx / 30.0, places=6)
            self.assertIsInstance(o.low_confidence, bool)

    def test_tracker_two_simultaneous_tracks_keep_distinct_ids(self) -> None:
        bbox_a = [40.0, 40.0, 100.0, 140.0]
        bbox_b = [180.0, 60.0, 240.0, 160.0]
        detector = _ScriptedDetector(
            phases=[[(1, 0.9, bbox_a), (2, 0.85, bbox_b)]],
            frames_per_phase=1000,
        )
        embedder = _PositionStubEmbedder(
            {(70, 90): _unit_vec(0), (210, 110): _unit_vec(1)}
        )
        tracker = PersonTracker(
            embedder=embedder, resolver=GlobalIdResolver(), sample_every=15, model=detector
        )
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        observations = tracker.process_frame(frame, frame_idx=0)
        self.assertEqual(len(observations), 2)
        self.assertNotEqual(observations[0].global_id, observations[1].global_id)

    def test_tracker_no_detections_returns_empty(self) -> None:
        detector = _ScriptedDetector(phases=[[]], frames_per_phase=10)
        tracker = PersonTracker(model=detector)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        self.assertEqual(tracker.process_frame(frame, frame_idx=0), [])

    def test_tracker_sample_every_gates_embedding_calls(self) -> None:
        bbox_a = [40.0, 40.0, 100.0, 140.0]
        detector = _ScriptedDetector(phases=[[(1, 0.9, bbox_a)]], frames_per_phase=1000)
        embedder = _PositionStubEmbedder({(70, 90): _unit_vec(0)})
        tracker = PersonTracker(
            embedder=embedder, resolver=GlobalIdResolver(), sample_every=15, model=detector
        )
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        for idx in range(31):
            tracker.process_frame(frame, frame_idx=idx)
        # embedded at frames 0, 15, 30 -> exactly 3 calls
        self.assertEqual(len(embedder.requested), 3)

    # -- geometry helper ------------------------------------------------------

    def test_padded_crop_pads_10_percent_and_clamps(self) -> None:
        frame = np.zeros((100, 80, 3), dtype=np.uint8)
        crop = padded_crop(frame, [10.0, 20.0, 50.0, 60.0], pad_ratio=0.10)
        self.assertEqual(crop.shape, (48, 48, 3))  # 40px box + 4px pad each side
        edge = padded_crop(frame, [0.0, 0.0, 10.0, 10.0], pad_ratio=0.10)
        self.assertEqual(edge.shape[:2], (11, 11))  # clamped at 0
        overflow = padded_crop(frame, [70.0, 90.0, 200.0, 200.0], pad_ratio=0.10)
        self.assertEqual(overflow.shape[:2], (21, 23))  # clamped at h,w: (100-79, 80-57)
        self.assertIsNone(padded_crop(frame, [-50.0, -50.0, -10.0, -10.0]))

    # -- static-camera tracker config -----------------------------------------

    def test_botsort_static_config_disables_gmc_and_reid(self) -> None:
        import yaml

        cfg = yaml.safe_load((ROOT / "configs" / "botsort_static.yaml").read_text())
        self.assertEqual(cfg["tracker_type"], "botsort")
        self.assertEqual(cfg["gmc_method"], "none")
        self.assertFalse(cfg["with_reid"])  # identity never from tracker ReID

    def test_faceembedder_exists_with_expected_surface(self) -> None:
        # hermetic surface check only: heavy insightface init is covered by smoke
        self.assertTrue(hasattr(FaceEmbedder, "embed_crop"))


# ---------------------------------------------------------------------------
# Opt-in real-model smoke (excluded from default run)
# ---------------------------------------------------------------------------


def _smoke_requested() -> bool:
    return bool(os.environ.get("ONDAMM_SMOKE_CLIP") or os.environ.get("ONDAMM_SMOKE_TRACKING"))


@pytest.mark.smoke
@unittest.skipUnless(_smoke_requested(), "opt-in: set ONDAMM_SMOKE_TRACKING=1 or ONDAMM_SMOKE_CLIP")
@unittest.skipUnless((ROOT / "models" / "yolo26s.pt").exists(), "models/yolo26s.pt absent")
class RealModelSmokeTests(unittest.TestCase):
    def test_real_detector_end_to_end_schema_and_persistence(self) -> None:
        clip_env = os.environ.get("ONDAMM_SMOKE_CLIP")
        with tempfile.TemporaryDirectory() as tmp:
            source = clip_env or str(_write_synthetic_video(Path(tmp) / "synth.mp4"))
            tracker = PersonTracker(device="cpu", sample_every=15, fps=30.0)
            per_track_global: dict[int, dict[str, int]] = {}
            total_obs = 0
            for frame_idx, ts_sec, frame in FrameReader(source):
                for obs in tracker.process_frame(frame, frame_idx=frame_idx, ts_sec=ts_sec):
                    total_obs += 1
                    self.assertIsInstance(obs, TrackObservation)
                    hist = per_track_global.setdefault(obs.track_id, {})
                    hist[obs.global_id] = hist.get(obs.global_id, 0) + 1
            # persistence: wherever detections exist, each tracker id maps to a
            # single dominant global id for >=90% of its observations
            for tid, hist in per_track_global.items():
                dominant = max(hist.values())
                self.assertGreaterEqual(
                    dominant / sum(hist.values()),
                    0.9,
                    f"track {tid} flapped global ids: {hist}",
                )


if __name__ == "__main__":
    unittest.main()
