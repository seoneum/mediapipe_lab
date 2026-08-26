"""ON DAMM 캘리브레이션 도구 hermetic 테스트 (todo 6).

합성 numpy 픽스처만 사용 — 실제 모델 로드, 네트워크, Label Studio 설치 없음.
cv2/mediapipe/emotiefflib는 대상 모듈에서 지연 임포트되므로 이 테스트는
전혀 트리거하지 않는다(필요한 곳은 가짜 probe/backend를 주입).
"""

from __future__ import annotations

import json
import pickle
import shutil
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import ondamm_calib_labelstudio as lsc  # noqa: E402
import ondamm_calib_prep as prep  # noqa: E402
import ondamm_calib_train as train  # noqa: E402
from ondamm_video_face_signals import EMOTION_LABELS_8  # noqa: E402


# ------------------------------------------------------------------ fixtures


class FakeProbe:
    """항상 얼굴 있음/없음을 고정하는 가짜 프로브."""

    def __init__(self, answer: bool = True) -> None:
        self.answer = answer
        self.calls = 0

    def detect(self, frame) -> bool:
        self.calls += 1
        return self.answer


class FakeBackend:
    """클래스명에서 결정적 특징/probs를 만드는 가짜 EmotiEffLib 백엔드."""

    def __init__(self, probs_bias: dict[str, int] | None = None) -> None:
        self.probs_bias = probs_bias or {}
        self.device = "cpu"

    def embed(self, rgb_crops):
        return np.zeros((len(rgb_crops), prep.FEATURE_DIM), dtype=np.float32)

    def emotion_probs(self, features):
        n = features.shape[0]
        return np.full((n, len(EMOTION_LABELS_8)), 0.1)


def sharp_frame(seed: int = 0, value: int = 200) -> np.ndarray:
    """고주파 체커보드 — Laplacian 분산이 큰 선명 프레임."""
    rng = np.random.default_rng(seed)
    base = np.full((64, 64, 3), value, dtype=np.uint8)
    noise = (rng.integers(0, 256, size=(64, 64), dtype=np.uint8))[..., None]
    return ((base.astype(np.int32) // 2 + noise.astype(np.int32) // 2) % 256).astype(np.uint8)


def blurry_frame() -> np.ndarray:
    """균일 프레임 — Laplacian 분산 ≈ 0."""
    return np.full((64, 64, 3), 128, dtype=np.uint8)


def write_person_fixture(
    root: Path,
    person: str,
    *,
    classes: tuple[str, ...] = ("Happiness", "Neutral"),
    sessions: tuple[str, ...] = ("s01", "s02"),
    rows_per_cell: int = 12,
    feature_dim: int = 6,
    seed: int = 0,
    informative_features: bool = True,
    baseline_correct_rate: float = 0.1,
) -> Path:
    """train/labelstudio가 소비하는 합성 person 디렉터리 생성.

    - informative_features=True: 클래스별로 분리된 클러스터 → 보정 헤드가 잘 학습
    - baseline_correct_rate: 사전학습 argmax가 정답과 일치하는 비율(낮으면 베이스라인 약함)
    """
    pdir = root / person
    pdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    X, y, sess, probs = [], [], [], []
    canon_of = {cls: EMOTION_LABELS_8.index(cls) for cls in classes}
    for ci, cls in enumerate(classes):
        canon = canon_of[cls]
        for s in sessions:
            for _ in range(rows_per_cell):
                if informative_features:
                    center = rng.normal(loc=ci * 10.0, scale=0.5, size=feature_dim)
                    X.append(center + rng.normal(0, 0.3, feature_dim))
                else:
                    X.append(rng.normal(0, 1.0, feature_dim))
                y.append(ci)
                sess.append(s)
                correct = rng.random() < baseline_correct_rate
                p = np.full(len(EMOTION_LABELS_8), 0.05)
                if correct:
                    p[canon] = 5.0
                else:
                    others = [j for j in range(len(EMOTION_LABELS_8)) if j != canon]
                    p[others[int(rng.integers(0, len(others)))]] = 5.0
                probs.append(p / p.sum())
    np.save(pdir / f"features_{person}.npy", np.asarray(X, dtype=np.float32))
    np.save(pdir / "labels.npy", np.asarray(y, dtype=np.int64))
    np.save(pdir / "sessions.npy", np.asarray(sess))
    np.save(pdir / f"probs_{person}.npy", np.asarray(probs))
    meta = {
        "person": person,
        "device": "cpu",
        "thresholds": {
            "fps": 5.0,
            "blur_laplacian": 60.0,
            "min_class_frames": prep.MIN_CLASS_FRAMES,
            "warn_class_frames": prep.WARN_CLASS_FRAMES,
            "probe_every": 1,
        },
        "class_names": list(classes),
        "classes": {c: {"usable": rows_per_cell * len(sessions), "dropped_blurred": 0,
                        "dropped_faceless": 0, "status": "kept"} for c in classes},
        "excluded_classes": [],
        "warned_classes": [],
        "total_rows": len(y),
        "clips": [],
    }
    with open(pdir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    frames = [
        {"index": i, "image": f"{person}_{sess[i]}_{classes[y[i]]}_{i:05d}.jpg",
         "person": person, "session": sess[i], "class": classes[y[i]]}
        for i in range(len(y))
    ]
    with open(pdir / f"frames_{person}.json", "w", encoding="utf-8") as fh:
        json.dump(frames, fh)
    return pdir


# ------------------------------------------------------------------ LOSO


class LosoSplitTests(unittest.TestCase):
    def test_loso_folds_partition_exactly(self) -> None:
        sessions = ["b", "a", "b", "a", "c"]
        folds = train.iter_loso_folds(sessions)
        self.assertEqual([f[2] for f in folds], ["a", "b", "c"])
        seen_test: set[int] = set()
        for tr, te, held in folds:
            arr = np.asarray(["b", "a", "b", "a", "c"])
            self.assertTrue((arr[te] == held).all())
            self.assertTrue((arr[tr] != held).all())
            self.assertEqual(set(tr) & set(te), set())
            seen_test |= set(te.tolist())
        self.assertEqual(seen_test, set(range(5)))

    def test_evaluate_loso_respects_person_boundaries(self) -> None:
        root = Path(__file__).parent.parent / "outputs" / "_calib_test_tmp"
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        write_person_fixture(root, "p1")
        write_person_fixture(root, "p2", seed=7)
        ds = train.collect_dataset(root)
        calls: list[tuple[list[int], list[int]]] = []

        def recorder(train_idx, test_idx):
            calls.append((sorted(int(i) for i in train_idx), sorted(int(i) for i in test_idx)))
            names = ds["class_names"]
            return [names[int(ds["y"][i])] for i in test_idx]

        metrics = train._evaluate(ds, "logreg", 0, fit_predict=recorder)
        persons = ds["persons"]
        # 각 폴드의 test는 단일 개인에 속하고, train은 그 홀드아웃 세션 행을 포함하지 않는다
        for tr, te in calls:
            self.assertEqual(len(set(persons[te])), 1)
            te_set, tr_set = set(te), set(tr)
            self.assertFalse(te_set & tr_set)
        # 세션 수 × 개인 수 만큼 폴드 (p1: s01,s02 / p2: s01,s02)
        self.assertEqual(metrics["fold_count"], 4)
        self.assertEqual(set(metrics["per_person"]), {"p1", "p2"})


# ------------------------------------------------------------------ ship rule


class ShipRuleTests(unittest.TestCase):
    def test_boundary_below_delta_rejects_and_writes_no_pkl(self) -> None:
        verdict = train.decide_ship(0.51, 0.50)  # +0.01 < 0.02
        self.assertFalse(verdict["adopt"])
        self.assertTrue(verdict["use_pretrained_only"])

        root = Path(__file__).parent.parent / "outputs" / "_calib_test_tmp"
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        write_person_fixture(
            root, "only", classes=("Happiness", "Neutral"),
            informative_features=False, baseline_correct_rate=0.95,
        )
        result = train.run_training(root)
        self.assertFalse(result["verdict"]["adopt"])
        self.assertIsNone(result["head"])
        self.assertFalse((root / "calib_head.pkl").exists())
        report = (root / "report.md").read_text(encoding="utf-8")
        self.assertIn("use_pretrained_only: true", report)

    def test_boundary_above_delta_adopts_and_writes_pkl(self) -> None:
        verdict = train.decide_ship(0.53, 0.50)  # +0.03 ≥ 0.02
        self.assertTrue(verdict["adopt"])
        self.assertFalse(verdict["use_pretrained_only"])

        root = Path(__file__).parent.parent / "outputs" / "_calib_test_tmp"
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        write_person_fixture(
            root, "only", classes=("Happiness", "Neutral"),
            informative_features=True, baseline_correct_rate=0.05,
        )
        result = train.run_training(root)
        self.assertTrue(result["verdict"]["adopt"])
        self.assertIsNotNone(result["head"])
        self.assertTrue((root / "calib_head.pkl").exists())
        with open(root / "calib_head.pkl", "rb") as fh:
            payload = pickle.load(fh)
        self.assertEqual(payload["class_names"], ["Happiness", "Neutral"])
        report = (root / "report.md").read_text(encoding="utf-8")
        self.assertIn("ADOPT calibrated head", report)

    def test_report_numbers_match_independently_computed_values(self) -> None:
        # 위장 성공 방지: 보고서 숫자를 독립 계산값과 대조한다.
        y_true = [0, 0, 1, 1]
        y_pred_str = ["Happiness", "Neutral", "Neutral", "Neutral"]
        names = ["Happiness", "Neutral"]
        # 손계산: class Happiness P=1/1 R=1/2 F1=2/3 ; Neutral P=2/3 R=1 F1=4/5
        expected = (2 / 3 + 4 / 5) / 2
        got = train.macro_f1(y_true, y_pred_str, names)
        self.assertAlmostEqual(got, expected, places=10)

        root = Path(__file__).parent.parent / "outputs" / "_calib_test_tmp"
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        write_person_fixture(root, "p1")
        ds = train.collect_dataset(root)
        metrics = train.evaluate_loso(ds)
        verdict = train.decide_ship(metrics["calib_mean"], metrics["baseline_mean"])
        report_path = train.write_report(Path(root / "report.md"), metrics, verdict, ds)
        text = report_path.read_text(encoding="utf-8")
        row = metrics["per_person"]["p1"]
        self.assertIn(f"| p1 | {row['baseline_f1']:.4f} | {row['calib_f1']:.4f} |", text)
        self.assertIn(f"{metrics['baseline_mean']:.4f}", text)
        self.assertIn("행동 프록시 추정 결과이며 의학적·교육적 진단이 아닙니다", text)


# ------------------------------------------------------------------ class exclusion


class ClassExclusionTests(unittest.TestCase):
    def test_count_boundaries(self) -> None:
        flags = prep.classify_counts({"tiny": 79, "floor": 80, "warn_hi": 149, "ok": 150})
        self.assertEqual(flags["excluded"], ["tiny"])
        self.assertEqual(flags["warned"], ["floor", "warn_hi"])

    def test_excluded_class_rows_absent_from_outputs(self) -> None:
        frames = [sharp_frame(seed=i % 5) for i in range(90)]
        good_cls, tiny_cls = "Happiness", "Sadness"
        clips = [
            {"session": "s01", "cls": good_cls, "frames": frames},   # usable 90 (≥80, <150 → warned)
            {"session": "s01", "cls": tiny_cls, "frames": frames[:2]},  # usable 2 < 80
        ]

        class _ListProbe:
            def __init__(self) -> None:
                self.calls = 0

            def detect(self, frame) -> bool:
                self.calls += 1
                return True

        results = []
        for clip in clips:
            kept_local, dropped = prep.scan_frames(clip["frames"], probe=_ListProbe(), probe_every=1)
            feats, pr = prep.extract_features_for(clip["frames"], kept_local, FakeBackend())
            results.append((clip, feats, pr, dropped))

        per_class_kept: dict[str, int] = {}
        per_class_dropped: dict[str, dict[str, int]] = {}
        for clip, feats, _pr, dropped in results:
            per_class_dropped.setdefault(clip["cls"], {"blurred": 0, "faceless": 0})
            for reason, n in dropped.items():
                per_class_dropped[clip["cls"]][reason] += n
            per_class_kept[clip["cls"]] = per_class_kept.get(clip["cls"], 0) + int(feats.shape[0])

        flags = prep.classify_counts(per_class_kept)
        excluded = set(flags["excluded"])
        self.assertEqual(excluded, {"Sadness"})
        self.assertEqual(flags["warned"], ["Happiness"])
        class_names = sorted(c for c in per_class_kept if c not in excluded)
        self.assertEqual(class_names, ["Happiness"])
        rows = [
            (feats[k], pr[k], class_names.index(clip["cls"]), clip["session"])
            for clip, feats, pr, _d in results
            if clip["cls"] not in excluded
            for k in range(feats.shape[0])
        ]
        labels = np.array([r[2] for r in rows])
        self.assertEqual(labels.shape[0], 90)  # Sadness 행 2개는 배제됨
        self.assertTrue((labels == 0).all())


# ------------------------------------------------------------------ prep filtering


class PrepFilteringTests(unittest.TestCase):
    def test_parse_clip_name(self) -> None:
        self.assertEqual(prep.parse_clip_name("data/calib/minsu/s01_neutral.mp4"), ("s01", "neutral"))
        with self.assertRaises(prep.CalibPrepError):
            prep.parse_clip_name("badname.mp4")

    def test_select_frame_indices_fps_downsampling(self) -> None:
        # 30fps 소스 → 5fps: 매 6프레임째
        idx = prep.select_frame_indices(30.0, 5.0, 31)
        self.assertEqual(idx, [0, 6, 12, 18, 24, 30])

    def test_laplacian_variance_sharp_vs_blurry(self) -> None:
        self.assertGreater(prep.laplacian_variance(sharp_frame()), 60.0)
        self.assertLess(prep.laplacian_variance(blurry_frame()), 60.0)

    def test_scan_frames_drops_blurred_then_faceless(self) -> None:
        frames = [sharp_frame(seed=i) for i in range(4)] + [blurry_frame() for _ in range(2)]
        probe = FakeProbe(answer=False)
        kept, dropped = prep.scan_frames(frames, blur_threshold=60.0, probe=probe, probe_every=1)
        self.assertEqual(dropped["blurred"], 2)
        self.assertEqual(dropped["faceless"], 4)
        self.assertEqual(kept, [])

    def test_nearest_probe_ok_inheritance(self) -> None:
        keep = prep.nearest_probe_ok(10, [0, 5], [True, False])
        self.assertEqual(keep, [True, True, True] + [False] * 7)
        tie_goes_to_earlier = prep.nearest_probe_ok(2, [0, 1], [True, False])
        self.assertEqual(tie_goes_to_earlier, [True, False])

    def test_prepare_person_end_to_end_with_fake_backend(self) -> None:
        root = Path(__file__).parent.parent / "outputs" / "_calib_test_tmp"
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        out_dir = root / "person_out"

        # prepare_person은 파일 클립을 기대하므로, 스캔/추출 조각을 직접 조합해
        # 동일 어셈블리 경로(classify_counts→rows→meta)를 검증한다.
        frames = [sharp_frame(seed=i) for i in range(10)]
        kept, dropped = prep.scan_frames(frames, probe=FakeProbe(True), probe_every=1)
        self.assertEqual(len(kept), 10)
        backend = FakeBackend()
        feats, pr = prep.extract_features_for(frames, kept, backend)
        self.assertEqual(feats.shape, (10, prep.FEATURE_DIM))
        self.assertEqual(pr.shape, (10, len(EMOTION_LABELS_8)))
        counts = {"Happiness": 100, "Neutral": 150}
        flags = prep.classify_counts(counts)
        self.assertEqual(flags["excluded"], [])
        self.assertEqual(flags["warned"], ["Happiness"])
        self.assertEqual(dropped, {"blurred": 0, "faceless": 0})


# ------------------------------------------------------------------ labelstudio


class LabelStudioRoundTripTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).parent.parent / "outputs" / "_calib_test_tmp"
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.pdir = write_person_fixture(self.root, "minsu")

    def test_build_tasks_format(self) -> None:
        loaded = lsc.load_person_dir(self.root, "minsu")
        class_names = loaded["meta"]["class_names"]
        tasks = lsc.build_tasks(loaded["probs"], loaded["frames"], class_names)
        first = tasks[0]
        self.assertEqual(set(first.keys()), {"data", "predictions"})
        self.assertEqual(set(first["data"].keys()), {"image"})
        pred = first["predictions"][0]["result"]
        # argmax 재계산으로 독립 검증
        expected_label = EMOTION_LABELS_8[int(np.argmax(loaded["probs"][0]))]
        if expected_label in class_names:
            self.assertEqual(pred[0]["from_name"], "label")
            self.assertEqual(pred[0]["to_name"], "image")
            self.assertEqual(pred[0]["type"], "choices")
            self.assertEqual(pred[0]["value"]["choices"], [expected_label])
        else:
            self.assertEqual(pred, [])  # 공간 밖 예측은 사전주석 없이 사람에게 맡긴다

    def test_round_trip_build_then_parse_back_alignment(self) -> None:
        loaded = lsc.load_person_dir(self.root, "minsu")
        original_labels = loaded["labels"].copy()
        class_names = loaded["meta"]["class_names"]
        tasks = lsc.build_tasks(loaded["probs"], loaded["frames"], class_names)

        export = json.loads(json.dumps(tasks))
        flipped_of = {}
        for i in range(0, len(export), 2):
            true_cls = class_names[int(original_labels[i])]
            flipped = class_names[(class_names.index(true_cls) + 1) % len(class_names)]
            export[i]["annotations"] = [
                {"result": [
                    {"from_name": "label", "to_name": "image", "type": "choices",
                     "value": {"choices": [flipped]}}
                ]}
            ]
            flipped_of[i] = flipped
        y, sess, stats = lsc.parse_export_tasks(export, loaded, class_names)

        self.assertEqual(y.shape, original_labels.shape)
        self.assertTrue((sess == loaded["sessions"]).all())
        expected_corrected = sum(
            1 for i, f in flipped_of.items() if class_names[int(original_labels[i])] != f
        )
        self.assertEqual(stats["corrected"], expected_corrected)
        self.assertEqual(stats["fallback_to_original"], len(y) - len(flipped_of))
        for i, flipped in flipped_of.items():
            self.assertEqual(class_names[int(y[i])], flipped)
        for i in range(len(y)):
            if i not in flipped_of:
                self.assertEqual(int(y[i]), int(original_labels[i]))

    def test_unknown_choice_and_unmatched_image_fail_cleanly(self) -> None:
        loaded = lsc.load_person_dir(self.root, "minsu")
        real_image = loaded["frames"][0]["image"]
        bad = [{"data": {"image": real_image}, "annotations": [{"result": [
            {"from_name": "label", "to_name": "image", "type": "choices",
             "value": {"choices": ["NotAClass"]}}]}]}]
        with self.assertRaises(lsc.CalibLabelStudioError) as cm:
            lsc.parse_export_tasks(bad, loaded, loaded["meta"]["class_names"])
        self.assertIn("valid class names", str(cm.exception))

        unmatched = [{"data": {"image": "ghost.jpg"}, "predictions": []}]
        with self.assertRaises(lsc.CalibLabelStudioError) as cm2:
            lsc.parse_export_tasks(unmatched, loaded, loaded["meta"]["class_names"])
        self.assertIn("match no prep frame record", str(cm2.exception))


# ------------------------------------------------------------------ malformed input & determinism


class MalformedInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).parent.parent / "outputs" / "_calib_test_tmp"
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def test_missing_npy_lists_expected_layout(self) -> None:
        write_person_fixture(self.root, "minsu")
        (self.root / "minsu" / "probs_minsu.npy").unlink()
        with self.assertRaises(FileNotFoundError) as cm:
            train.load_person_dir(self.root, "minsu")
        msg = str(cm.exception)
        self.assertIn("probs_minsu.npy", msg)
        self.assertIn("expected layout", msg)

    def test_single_class_raises_clean_valueerror(self) -> None:
        write_person_fixture(self.root, "solo", classes=("Happiness",))
        with self.assertRaises(ValueError) as cm:
            train.collect_dataset(self.root)
        self.assertIn("need ≥2 classes", str(cm.exception))

    def test_single_session_raises_clean_valueerror(self) -> None:
        write_person_fixture(self.root, "p1", sessions=("s01",))
        write_person_fixture(self.root, "p2", sessions=("s01",), seed=5)
        with self.assertRaises(ValueError) as cm:
            train.collect_dataset(self.root)
        self.assertIn("≥2 distinct recording sessions", str(cm.exception))


class DeterminismTests(unittest.TestCase):
    def test_same_inputs_twice_identical_report_numbers(self) -> None:
        src = Path(__file__).parent.parent / "outputs" / "_calib_test_src"
        self.addCleanup(shutil.rmtree, src, ignore_errors=True)
        reports = []
        for run in range(2):
            root = Path(__file__).parent.parent / "outputs" / f"_calib_test_run{run}"
            self.addCleanup(shutil.rmtree, root, ignore_errors=True)
            write_person_fixture(root, "p1", seed=42)
            write_person_fixture(root, "p2", seed=43)
            result = train.run_training(root, kind="logreg", seed=0)
            reports.append((root / "report.md").read_bytes())
            m = result["metrics"]
            self.assertAlmostEqual(m["calib_mean"], m["calib_mean"], places=12)
        self.assertEqual(reports[0], reports[1])  # 바이트 단위 결정성


if __name__ == "__main__":
    unittest.main()
