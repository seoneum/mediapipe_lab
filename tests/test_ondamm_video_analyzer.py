"""Hermetic tests for ON DAMM video analyzer end-to-end (todo 5).

원칙(형제 테스트와 동일):
- 기본 실행은 네트워크/실제 모델 로드 없이 스텁 tracker/extractor를 주입해 검증.
- 영상은 tmp 디렉토리에 합성 mp4로 생성(자동 정리).
- 실모델 e2e는 @pytest.mark.smoke + ONDAMM_SMOKE_CLIP/ONDAMM_SMOKE_ANALYZER 게이트.
- 렌더 검증은 파일 존재가 아니라 ffprobe/cv2 재생 길이와 픽셀 차이로 한다
  (misleading-success 방지). 임시 파일 잔존(stale state)도 매 실행 검사한다.
"""

from __future__ import annotations

import json
import os
import subprocess
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

import ondamm_video_analyzer as analyzer  # noqa: E402
import ondamm_video_analyzer_cli as cli  # noqa: E402
import ondamm_video_render as render_mod  # noqa: E402
from ondamm_video_face_signals import BLENDSHAPE_NAMES, EMOTION_LABELS_8  # noqa: E402
from ondamm_video_metrics import PersonMetrics, summarize_person  # noqa: E402
from ondamm_video_render import (  # noqa: E402
    FFMPEG_NOT_FOUND_MSG,
    RenderError,
    argmax_expression,
    draw_overlays,
    encode,
    gid_color_bgr,
    label_text,
    resolve_kr_font,
)
from ondamm_video_tracking import TrackObservation, VideoInputError  # noqa: E402

CSV_FIELDS = list(PersonMetrics.__dataclass_fields__.keys())


# ---------------------------------------------------------------------------
# Deterministic fixtures & stubs
# ---------------------------------------------------------------------------


def _write_synthetic_video(
    path: Path,
    frames: int = 120,
    width: int = 320,
    height: int = 240,
    fps: int = 30,
) -> Path:
    """Two colored rectangles moving along crossing diagonals (tracking-test pattern)."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open {path}")
    for i in range(frames):
        t = i / max(frames - 1, 1)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        red_cx, red_cy = int(40 + 240 * t), int(60 + 120 * t)
        blue_cx, blue_cy = int(280 - 240 * t), int(180 - 120 * t)
        cv2.rectangle(canvas, (red_cx - 18, red_cy - 18), (red_cx + 18, red_cy + 18), (0, 0, 255), -1)
        cv2.rectangle(canvas, (blue_cx - 18, blue_cy - 18), (blue_cx + 18, blue_cy + 18), (255, 0, 0), -1)
        writer.write(canvas)
    writer.release()
    return path


def _observation(frame_idx: int, gid: str, bbox: list[float], fps: float = 30.0) -> TrackObservation:
    return TrackObservation(
        frame_idx=frame_idx,
        ts_sec=frame_idx / fps,
        track_id=1 if gid == GID_A else 2,
        global_id=gid,
        bbox_xyxy=list(bbox),
        det_conf=0.9,
        low_confidence=False,
    )


GID_A = "unknown_0"
GID_B = "unknown_1"
BBOX_A = [30.0, 30.0, 90.0, 130.0]
BBOX_B = [200.0, 80.0, 260.0, 180.0]


class StubTracker:
    """process_frame 계약을 지키는 결정론적 스텁(모델 로드 없음)."""

    def __init__(self, rows: list[tuple[str, list[float]]], fps: float = 30.0) -> None:
        self.rows = rows
        self.fps = fps
        self.calls = 0

    def process_frame(self, frame: np.ndarray, frame_idx: int = 0, ts_sec: float | None = None):
        self.calls += 1
        assert ts_sec is not None and ts_sec == pytest.approx(frame_idx / self.fps, abs=1e-6)
        return [_observation(frame_idx, gid, bbox, self.fps) for gid, bbox in self.rows]


class StubExtractor:
    """FaceSignalExtractor.extract 계약(전 필드 포함 SignalSample dict) 스텁.

    KNOWN CONSTRAINT 방어: summarize_person은 emotion_probs 누락 시 ValueError를
    내므로, 스텁도 실제 extractor처럼 모든 필드를 채운다(파이프라인 불변식).
    """

    def __init__(self, sample_every: int = 3) -> None:
        self.sample_every = sample_every
        self.ts_seen: list[int] = []
        self.closed = False

    def extract(self, frame_idx, global_id, bbox_xyxy, rgb_frame, ts_ms_monotonic):
        if frame_idx % self.sample_every != 0:
            return None
        self.ts_seen.append(int(ts_ms_monotonic))
        probs = [0.02] * 8
        probs[EMOTION_LABELS_8.index("Happiness")] = 0.86  # argmax 고정
        return {
            "frame_idx": int(frame_idx),
            "global_id": str(global_id),
            "blendshapes": {name: 0.01 for name in BLENDSHAPE_NAMES},  # 정확히 52키
            "yaw_deg": 3.0,
            "pitch_deg": -2.0,
            "roll_deg": 0.5,
            "blink": 0.05,
            "emotion_labels": list(EMOTION_LABELS_8),
            "emotion_probs": probs,
            "valence": 0.4,
            "arousal": 0.1,
        }

    def close(self) -> None:
        self.closed = True


def _person_metrics(gid: str = GID_A) -> PersonMetrics:
    samples = [StubExtractor(sample_every=1).extract(i * 3, gid, BBOX_A, None, i * 1000)
               for i in range(12)]
    return summarize_person(gid, samples, total_frames=36, fps=10.0)


# ---------------------------------------------------------------------------
# Renderer unit tests
# ---------------------------------------------------------------------------


def test_resolve_kr_font_returns_existing_path_on_this_machine():
    try:
        resolved = resolve_kr_font()
    except RenderError:
        pytest.skip("no Korean font installed on this machine (recorded honestly)")
    assert Path(resolved).is_file() and Path(resolved).stat().st_size > 0


def test_resolve_kr_font_error_lists_tried_paths(monkeypatch):
    monkeypatch.delenv(render_mod.FONT_ENV_VAR, raising=False)
    monkeypatch.setattr(render_mod, "_FONT_CANDIDATES", ("/no/such/a.ttf", "/no/such/b.ttc"))
    with pytest.raises(RenderError) as ctx:
        resolve_kr_font()
    assert "/no/such/a.ttf" in str(ctx.value)
    assert "/no/such/b.ttc" in str(ctx.value)


def test_resolve_kr_font_env_var_takes_priority(monkeypatch, tmp_path):
    fake = tmp_path / "my.ttf"
    fake.write_bytes(b"x")
    monkeypatch.setenv(render_mod.FONT_ENV_VAR, str(fake))
    assert resolve_kr_font() == str(fake)


def test_gid_color_is_deterministic_and_distinct():
    assert gid_color_bgr(GID_A) == gid_color_bgr(GID_A)
    assert gid_color_bgr(GID_A) != gid_color_bgr(GID_B)


def test_label_text_matches_readme_contract():
    metrics = _person_metrics(GID_A)
    text = label_text(GID_A, metrics)
    # "{gid} · 집중 {attention_pct}% · 흥미 {interest} · {dominant expression}"
    assert text.startswith(f"{GID_A} · 집중 ")
    assert "% · 흥미 " in text
    assert f"흥미 {metrics.interest} · Happiness" in text  # timeline last label


def test_argmax_expression_fallback_when_no_timeline():
    labels = list(EMOTION_LABELS_8)
    probs = [0.1] * 8
    probs[labels.index("Sadness")] = 0.7
    assert argmax_expression(labels, probs) == "Sadness"
    assert argmax_expression([], []) == "-"


def test_draw_overlays_caption_and_box_pixels_differ_from_plain_frame():
    plain = np.zeros((240, 320, 3), dtype=np.uint8)
    metrics = _person_metrics(GID_A)
    obs = [_observation(0, GID_A, BBOX_A), _observation(0, GID_B, BBOX_B)]
    drawn = draw_overlays(plain, obs, {GID_A: metrics, GID_B: metrics})
    assert drawn.shape == plain.shape and drawn.dtype == np.uint8
    # caption band (bottom strip) must differ — burned-in non-diagnostic caption
    caption_band_a = drawn[222:, :, :].astype(int)
    caption_band_p = plain[222:, :, :].astype(int)
    assert np.abs(caption_band_a - caption_band_p).sum() > 1000
    # box border region must differ too
    box_region_a = drawn[28:34, 28:96, :].astype(int)
    box_region_p = plain[28:34, 28:96, :].astype(int)
    assert np.abs(box_region_a - box_region_p).sum() > 0


def test_encode_duration_and_frame_count_match(tmp_path):
    out = tmp_path / "nested" / "clip.mp4"
    n, fps = 45, 15
    encode((np.full((120, 160, 3), 128, np.uint8) for _ in range(n)), out, fps)
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened()
    got_fps = cap.get(cv2.CAP_PROP_FPS)
    count = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        count += 1
    cap.release()
    assert got_fps == pytest.approx(fps, abs=0.01)
    assert count == n
    duration = count / got_fps
    assert duration == pytest.approx(n / fps, abs=0.1)


def test_encode_without_ffmpeg_raises_rendererror_and_leaves_no_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(render_mod, "ffmpeg_on_path", lambda: None)
    out = tmp_path / "v.mp4"
    with pytest.raises(RenderError) as ctx:
        encode([np.zeros((60, 80, 3), np.uint8)], out, 30.0)
    assert "brew install ffmpeg" in str(ctx.value)
    assert "sudo apt install ffmpeg" in str(ctx.value)
    assert not out.exists()
    assert list(tmp_path.glob("*.tmp")) == []
    assert list(tmp_path.glob(".ondamm_render_*")) == []


def test_encode_ffmpeg_nonzero_rc_raises_rendererror(tmp_path, monkeypatch):
    class FakeProc:
        returncode = 3
        stderr = "boom"

    def fake_run(cmd, capture_output, text):
        assert any("ffmpeg" == part for part in cmd[:1])
        return FakeProc()

    monkeypatch.setattr(render_mod.subprocess, "run", fake_run)
    with pytest.raises(RenderError) as ctx:
        encode([np.zeros((60, 80, 3), np.uint8)], tmp_path / "v.mp4", 30.0)
    # nonzero rc는 "not found" 접두 없이 remux 실패로 정확히 진단해야 한다
    assert "ffmpeg remux failed" in str(ctx.value)
    assert "(rc=3)" in str(ctx.value)
    assert "boom" in str(ctx.value)
    assert "brew install ffmpeg" not in str(ctx.value)


def test_encode_ffmpeg_failure_removes_partial_output(tmp_path, monkeypatch):
    class FakeProc:
        returncode = 1
        stderr = "muxing error near frame 12"

    def fake_run(cmd, capture_output, text):
        Path(cmd[-1]).write_bytes(b"partial-mp4-bytes")  # ffmpeg이 부분 출력 남긴 상황 재현
        return FakeProc()

    monkeypatch.setattr(render_mod.subprocess, "run", fake_run)
    out = tmp_path / "v.mp4"
    with pytest.raises(RenderError) as ctx:
        encode([np.zeros((60, 80, 3), np.uint8)], out, 30.0)
    assert "ffmpeg remux failed" in str(ctx.value)
    assert not out.exists()  # stale partial artifact 제거
    assert list(tmp_path.glob(".ondamm_render_*")) == []


# ---------------------------------------------------------------------------
# End-to-end (hermetic stubs injected into analyzer.run)
# ---------------------------------------------------------------------------


@pytest.fixture()
def e2e_env(tmp_path):
    input_path = _write_synthetic_video(tmp_path / "in.mp4", frames=120, width=320, height=240, fps=30)
    tracker = StubTracker([(GID_A, BBOX_A), (GID_B, BBOX_B)], fps=30.0)
    extractor = StubExtractor(sample_every=3)
    args = argparse_namespace(
        input=str(input_path),
        output=str(tmp_path / "out" / "result.mp4"),
        device="cpu",
        sample_every=3,
        metrics_json=str(tmp_path / "out" / "metrics.json"),
        metrics_csv=str(tmp_path / "out" / "metrics.csv"),
    )
    return tmp_path, input_path, args, tracker, extractor


def argparse_namespace(**kwargs):
    from types import SimpleNamespace

    return SimpleNamespace(**kwargs)


def test_end_to_end_synthetic_rects(e2e_env):
    tmp_path, input_path, args, tracker, extractor = e2e_env
    code = analyzer.run(args, tracker=tracker, extractor=extractor, require_models=False)

    assert code == 0
    out_mp4 = Path(args.output)
    assert out_mp4.is_file() and out_mp4.stat().st_size > 0

    # duration within ±0.1s of input (cv2 probe; ffprobe cross-check below)
    def probe_duration(path: Path) -> float:
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return n / fps

    assert probe_duration(out_mp4) == pytest.approx(probe_duration(input_path), abs=0.1)

    # metrics JSON parses with required keys per gid
    payload = json.loads(Path(args.metrics_json).read_text(encoding="utf-8"))
    gids = {entry["global_id"] for entry in payload}
    assert gids == {GID_A, GID_B}
    for entry in payload:
        for key in ("attention_pct", "focus_seconds", "interest", "expression_timeline",
                    "frames_covered", "total_frames", "low_confidence"):
            assert key in entry, key
        assert 0.0 <= entry["attention_pct"] <= 100.0
        assert entry["interest"] in ("낮음", "중간", "높음")
        assert entry["total_frames"] == 120

    # CSV header == field names && row count == gid count
    csv_lines = Path(args.metrics_csv).read_text(encoding="utf-8").strip().splitlines()
    assert csv_lines[0].split(",") == CSV_FIELDS
    assert len(csv_lines) - 1 == len(gids) == 2

    # caption pixels differ on LAST frame vs raw input last frame (bottom band)
    last_out = _last_frame(out_mp4)
    last_in = _last_frame(input_path)
    band_out = last_out[210:, :, :].astype(int)
    band_in = last_in[210:, :, :].astype(int)
    assert np.abs(band_out - band_in).sum() > 1000

    # landmarker timestamps strictly increasing across the whole run
    assert extractor.ts_seen == sorted(extractor.ts_seen)
    assert len(set(extractor.ts_seen)) == len(extractor.ts_seen)

    # stale-state guard: no temp artifacts left behind
    assert list(Path(args.output).parent.glob("*.tmp")) == []
    assert list(Path(args.output).parent.glob(".ondamm_render_*")) == []
    assert extractor.closed  # extractor resources released


def test_end_to_end_ffprobe_duration_cross_check(e2e_env):
    """misleading-success 방지: ffprobe가 있으면 컨테이너 duration도 교차검증."""
    tmp_path, input_path, args, tracker, extractor = e2e_env
    import shutil

    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe not available")
    analyzer.run(args, tracker=tracker, extractor=extractor, require_models=False)

    def ffprobe_duration(path: Path) -> float:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, check=True,
        )
        return float(proc.stdout.strip())

    assert ffprobe_duration(Path(args.output)) == pytest.approx(ffprobe_duration(input_path), abs=0.1)


def _last_frame(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frame = None
    while True:
        ok, current = cap.read()
        if not ok:
            break
        frame = current
    cap.release()
    assert frame is not None
    return frame


def test_corrupt_input_raises_videinputerror_not_crash(tmp_path):
    bad = tmp_path / "corrupt.mp4"
    bad.write_bytes(b"not a video container" * 16)
    args = argparse_namespace(
        input=str(bad), output=str(tmp_path / "o.mp4"), device="auto",
        sample_every=3, metrics_json=None, metrics_csv=None,
    )
    with pytest.raises(VideoInputError) as ctx:
        analyzer.run(args, tracker=StubTracker([]), extractor=StubExtractor(),
                     require_models=False)
    assert "cannot open" in str(ctx.value)


def test_missing_model_files_lists_assets(monkeypatch, tmp_path):
    monkeypatch.setattr(analyzer, "REPO_ROOT", tmp_path)  # models dir empty here
    monkeypatch.setattr(Path, "home", lambda: tmp_path)   # no insightface dir
    missing = analyzer.missing_model_files()
    assert any("yolo26s.pt" in m for m in missing)
    assert any("face_landmarker.task" in m for m in missing)
    assert any("buffalo_l" in m for m in missing)


# ---------------------------------------------------------------------------
# Real-pipeline builder wiring (B1 regression: --sample-every must reach it)
# ---------------------------------------------------------------------------


def test_build_real_pipeline_forwards_sample_every_to_extractor(monkeypatch):
    """B1 회귀: build_real_pipeline의 sample_every가 FaceSignalExtractor 생성자까지 전달."""
    recorded = {}

    class FakeTracker:
        def __init__(self, device=None, embedder=None) -> None:
            recorded["tracker_device"] = device

    class FakeExtractor:
        def __init__(self, **kwargs) -> None:
            recorded.update(kwargs)
            self.sample_every = kwargs.get("sample_every")

    monkeypatch.setattr(analyzer, "PersonTracker", FakeTracker)
    monkeypatch.setattr(analyzer, "FaceEmbedder", lambda: object())
    monkeypatch.setattr(analyzer, "MediaPipeSignals", lambda path: ("mediapipe", path))
    monkeypatch.setattr(analyzer, "EmotionSignals", lambda device=None: ("emotions", device))
    monkeypatch.setattr(analyzer, "FaceSignalExtractor", FakeExtractor)

    _, extractor = analyzer.build_real_pipeline("cpu", sample_every=7)
    assert isinstance(extractor, FakeExtractor)
    assert recorded["sample_every"] == 7
    assert extractor.sample_every == 7
    assert recorded["device"] == "cpu"
    assert recorded["tracker_device"] == "cpu"

    # 기본값 계약 유지: 플래그 생략 시 종전과 동일하게 3
    recorded.clear()
    _, default_extractor = analyzer.build_real_pipeline("cpu")
    assert recorded["sample_every"] == 3
    assert default_extractor.sample_every == 3


def test_run_forwards_parsed_sample_every_to_real_builder(tmp_path, monkeypatch):
    """B1 회귀: run()이 파싱한 args.sample_every를 실파이프라인 빌더에 그대로 넘긴다."""
    input_path = _write_synthetic_video(tmp_path / "in.mp4", frames=6, fps=30)
    seen = {}

    def fake_builder(device, sample_every=3):
        seen["device"] = device
        seen["sample_every"] = sample_every
        return StubTracker([(GID_A, BBOX_A)]), StubExtractor(sample_every=sample_every)

    monkeypatch.setattr(analyzer, "build_real_pipeline", fake_builder)
    args = argparse_namespace(
        input=str(input_path),
        output=str(tmp_path / "out.mp4"),
        device="cpu",
        sample_every=7,
        metrics_json=None,
        metrics_csv=None,
    )
    code = analyzer.run(args, require_models=False)
    assert code == 0
    assert seen == {"device": "cpu", "sample_every": 7}


def test_live_label_recompute_is_throttled(e2e_env, monkeypatch):
    """M1 완화: 라이브 라벨 summarize_person 재계산이 gid당 스로틀 안쪽으로 제한."""
    tmp_path, input_path, args, tracker, extractor = e2e_env
    calls = {"n": 0}
    real_summarize = analyzer.summarize_person

    def counting_summarize(*a, **kw):
        calls["n"] += 1
        return real_summarize(*a, **kw)

    monkeypatch.setattr(analyzer, "summarize_person", counting_summarize)
    code = analyzer.run(args, tracker=tracker, extractor=extractor, require_models=False)
    assert code == 0

    # finalize_metrics는 gid별로 정확히 1회씩(=2) 전체 윈도우 재계산하므로 제외
    final_calls = 2
    live_calls = calls["n"] - final_calls
    per_gid_samples = len(extractor.ts_seen) // 2  # 120프레임/sample_every=3 → gid당 40
    bound = 2 * (per_gid_samples // analyzer.LIVE_RECOMPUTE_EVERY + 1)
    assert live_calls <= bound
    assert live_calls < 2 * per_gid_samples  # 스로틀이 없었다면 gid당 샘플 수만큼이었을 것

    # 최종 JSON은 여전히 전체 윈도우 값(스로틀 영향 없음)
    payload = json.loads(Path(args.metrics_json).read_text(encoding="utf-8"))
    assert {entry["global_id"] for entry in payload} == {GID_A, GID_B}
    for entry in payload:
        assert 0.0 <= entry["attention_pct"] <= 100.0


# ---------------------------------------------------------------------------
# CLI wiring (exit-code contract)
# ---------------------------------------------------------------------------


def test_cli_missing_input_exits_2_with_cannot_open(tmp_path, capsys):
    argv = [
        "--input", str(tmp_path / "missing.mp4"),
        "--output", str(tmp_path / "out.mp4"),
        "--child-id", "test-child",
    ]
    code = cli.main(argv, rights_check_fn=lambda _: None)
    assert code == 2
    err = capsys.readouterr().err
    assert "입력 영상 오류" in err


def test_cli_corrupt_input_exits_2_no_traceback(tmp_path, capsys):
    bad = tmp_path / "corrupt.mp4"
    bad.write_bytes(b"garbage" * 64)
    code = cli.main(["--input", str(bad), "--output", str(tmp_path / "out.mp4"), "--child-id", "test-child"], rights_check_fn=lambda _: None)
    assert code == 2
    assert "입력 영상 오류" in capsys.readouterr().err


def test_cli_missing_models_exits_3_mentions_download_script(tmp_path, capsys, monkeypatch):
    input_path = _write_synthetic_video(tmp_path / "in.mp4", frames=6, fps=30)
    monkeypatch.setattr(
        analyzer, "missing_model_files", lambda: ["models/yolo26s.pt"]
    )
    code = cli.main([
        "--input", str(input_path),
        "--output", str(tmp_path / "out.mp4"),
        "--child-id", "test-child",
    ], rights_check_fn=lambda _: None)
    assert code == 3
    err = capsys.readouterr().err
    assert "scripts/download_video_models.sh" in err
    assert "models/yolo26s.pt" in err


def test_cli_render_failure_exits_4(tmp_path, capsys, monkeypatch):
    input_path = _write_synthetic_video(tmp_path / "in.mp4", frames=6, fps=30)
    monkeypatch.setattr(render_mod, "ffmpeg_on_path", lambda: None)
    tracker = StubTracker([(GID_A, BBOX_A)])
    extractor = StubExtractor()

    def run_with_stubs(args):
        return analyzer.run(args, tracker=tracker, extractor=extractor, require_models=False)

    code = cli.main([
        "--input", str(input_path),
        "--output", str(tmp_path / "out.mp4"),
        "--child-id", "test-child",
    ], run_fn=run_with_stubs, rights_check_fn=lambda _: None)
    assert code == 4
    err = capsys.readouterr().err
    assert "brew install ffmpeg" in err


def test_cli_happy_wiring_passes_flags_through(tmp_path, capsys):
    input_path = _write_synthetic_video(tmp_path / "in.mp4", frames=6, fps=30)
    seen = {}

    def fake_run(args):
        seen.update(vars(args))
        return 0

    code = cli.main([
        "--input", str(input_path),
        "--output", str(tmp_path / "out.mp4"),
        "--child-id", "test-child",
        "--device", "mps",
        "--sample-every", "5",
        "--metrics-json", str(tmp_path / "m.json"),
        "--metrics-csv", str(tmp_path / "m.csv"),
    ], run_fn=fake_run, rights_check_fn=lambda _: None)
    assert code == 0
    assert seen["device"] == "mps"
    assert seen["sample_every"] == 5
    assert seen["metrics_json"] == str(tmp_path / "m.json")
    assert seen["metrics_csv"] == str(tmp_path / "m.csv")


def test_cli_rejects_unknown_device(tmp_path, capsys):
    with pytest.raises(SystemExit) as ctx:
        cli.main([
            "--input", str(tmp_path / "x.mp4"),
            "--output", str(tmp_path / "o.mp4"),
            "--device", "tpu",
        ])
    assert ctx.value.code == 2  # argparse usage error


# ---------------------------------------------------------------------------
# Opt-in real-model smoke (excluded from default run)
# ---------------------------------------------------------------------------


def _smoke_requested() -> bool:
    return bool(os.environ.get("ONDAMM_SMOKE_ANALYZER") or os.environ.get("ONDAMM_SMOKE_CLIP"))


@pytest.mark.smoke
@unittest.skipUnless(_smoke_requested(), "opt-in: set ONDAMM_SMOKE_ANALYZER=1 or ONDAMM_SMOKE_CLIP")
@unittest.skipUnless((ROOT / "models" / "yolo26s.pt").exists(), "models/yolo26s.pt absent")
class RealModelSmokeTests(unittest.TestCase):
    def test_real_pipeline_runs_on_synthetic_clip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = os.environ.get("ONDAMM_SMOKE_CLIP") or str(
                _write_synthetic_video(Path(tmp) / "synth.mp4", frames=90, fps=30)
            )
            args = argparse_namespace(
                input=source,
                output=str(Path(tmp) / "result.mp4"),
                device="cpu",
                sample_every=3,
                metrics_json=str(Path(tmp) / "m.json"),
                metrics_csv=str(Path(tmp) / "m.csv"),
            )
            code = analyzer.run(args)
            self.assertEqual(code, 0)
            self.assertTrue(Path(args.output).is_file())


if __name__ == "__main__":
    unittest.main()
