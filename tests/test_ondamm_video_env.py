"""ON DAMM 영상 분석기 환경 검증 모듈(app/ondamm_video_env.py) hermetic 테스트.

원칙:
- 네트워크 사용 금지, 실제 모델 로드 금지(기본 실행).
- shutil.which / pathlib 경로 / 스파이크 함수를 monkeypatch로 대체한다.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

import app.ondamm_video_env as env

REQUIRED_KEYS = ("mps_available", "emotiefflib_mps_ok", "ffmpeg_found", "models_present")


@pytest.fixture()
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """HOME을 tmp로 돌려 ~/.insightface 판정을 격리한다."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home


def _make_model_assets(model_dir: Path, landmarker_bytes: int = 2_000_000) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "face_landmarker.task").write_bytes(b"x" * landmarker_bytes)
    (model_dir / "yolo26s.pt").write_bytes(b"y" * 1024)


# ---------------------------------------------------------------- ffmpeg


def test_ffmpeg_found_when_which_hits(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(env.shutil, "which", lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None)
    assert env.find_ffmpeg() is True


def test_ffmpeg_missing_when_which_none(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(env.shutil, "which", lambda name: None)
    assert env.find_ffmpeg() is False


# ---------------------------------------------------------------- models_present


def test_models_present_true_with_all_assets(tmp_path: Path, fake_home: Path):
    models = tmp_path / "models"
    _make_model_assets(models)
    buffalo = fake_home / ".insightface" / "models" / "buffalo_l"
    buffalo.mkdir(parents=True)
    (buffalo / "det_10g.onnx").write_bytes(b"z" * 512)
    assert env.models_present(model_dir=models, insightface_home=fake_home) is True


def test_models_present_false_when_landmarker_too_small(tmp_path: Path, fake_home: Path):
    models = tmp_path / "models"
    _make_model_assets(models, landmarker_bytes=1024)  # <1MB
    (fake_home / ".insightface" / "models" / "buffalo_l").mkdir(parents=True)
    assert env.models_present(model_dir=models, insightface_home=fake_home) is False


def test_models_present_false_when_buffalo_l_empty(tmp_path: Path, fake_home: Path):
    models = tmp_path / "models"
    _make_model_assets(models)
    (fake_home / ".insightface" / "models" / "buffalo_l").mkdir(parents=True)  # 빈 디렉토리
    assert env.models_present(model_dir=models, insightface_home=fake_home) is False


def test_models_present_false_when_yolo_missing(tmp_path: Path, fake_home: Path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "face_landmarker.task").write_bytes(b"x" * 2_000_000)
    (fake_home / ".insightface" / "models" / "buffalo_l").mkdir(parents=True)
    assert env.models_present(model_dir=models, insightface_home=fake_home) is False


# ---------------------------------------------------------------- report build


def _ok_spike() -> tuple[bool, str]:
    return True, "max_abs_logit_diff=1.234e-05 tolerance=0.001"


def _fail_spike(reason: str = "mps_unavailable") -> tuple[bool, str]:
    return False, reason


def test_build_report_has_four_boolean_keys():
    report = env.build_report(
        ffmpeg_found=True,
        models_present_flag=True,
        spike_fn=_ok_spike,
        imports={"torch": True},
    )
    for key in REQUIRED_KEYS:
        assert key in report
        assert isinstance(report[key], bool)
    assert report["emotiefflib_mps_ok"] is True
    assert report["default_device"] == "mps"


def test_build_report_degrades_without_emotiefflib():
    report = env.build_report(
        ffmpeg_found=False,
        models_present_flag=False,
        spike_fn=lambda: (False, "emotiefflib unavailable: ImportError: no module"),
        imports={"emotiefflib": False},
    )
    assert report["emotiefflib_mps_ok"] is False
    assert "reason" in report
    assert "emotiefflib" in report["reason"]
    # 강등 시에도 4개 키는 모두 boolean으로 존재한다.
    for key in REQUIRED_KEYS:
        assert isinstance(report[key], bool)
    assert report["default_device"] == "cpu"


def test_build_report_survives_spiking_crash():
    def boom() -> tuple[bool, str]:
        raise RuntimeError("mps exploded")

    report = env.build_report(ffmpeg_found=True, models_present_flag=True, spike_fn=boom)
    assert report["emotiefflib_mps_ok"] is False
    assert "spike crashed" in report["reason"]


def test_default_device_cpu_when_mps_unavailable():
    assert env.compute_default_device(mps_available=False, emotiefflib_mps_ok=True) == "cpu"


def test_default_device_cpu_when_spike_fails():
    assert env.compute_default_device(mps_available=True, emotiefflib_mps_ok=False) == "cpu"


# ---------------------------------------------------------------- run_check + file IO


def test_run_check_writes_report_hermetically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    out = tmp_path / "outputs" / "ondamm" / "video" / "env_report.json"
    monkeypatch.setattr(env, "find_ffmpeg", lambda: True)
    monkeypatch.setattr(env, "models_present", lambda: True)
    monkeypatch.setattr(env, "torch_mps_available", lambda: True)

    report, code = env.run_check(report_path=out)

    assert code == 0
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    for key in REQUIRED_KEYS:
        assert key in loaded and isinstance(loaded[key], bool)
    assert loaded["ffmpeg_found"] is True
    assert loaded["models_present"] is True
    assert loaded["emotiefflib_mps_ok"] is True


def test_rerun_overwrites_cleanly_no_partial_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """stale-state 방어: 두 번 연속 실행해도 부분 JSON/잔여 tmp 없이 깨끗히 덮어쓴다."""
    out = tmp_path / "env_report.json"
    monkeypatch.setattr(env, "find_ffmpeg", lambda: True)
    monkeypatch.setattr(env, "models_present", lambda: True)
    monkeypatch.setattr(env, "torch_mps_available", lambda: True)

    first, code1 = env.run_check(report_path=out)
    assert code1 == 0

    # 두 번째 실행에서는 스파이크가 실패하는 시나리오로 바뀐다.
    monkeypatch.setattr(env, "mps_spike", lambda seed=1234: (False, "mps_unavailable"))
    second, code2 = env.run_check(report_path=out)
    assert code2 == 0

    on_disk = json.loads(out.read_text(encoding="utf-8"))
    assert on_disk == second
    assert on_disk["emotiefflib_mps_ok"] is False
    assert first["emotiefflib_mps_ok"] is True  # 첫 보고서와 독립적
    leftovers = [p.name for p in out.parent.iterdir() if p.name != out.name]
    assert leftovers == [], f"임시파일 잔존: {leftovers}"


# ---------------------------------------------------------------- CLI surface


def test_cli_failure_channel_mentions_brew_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """ffmpeg가 없을 때: 보고서 ffmpeg_found=false + stdout에 brew install ffmpeg."""
    out = tmp_path / "env_report.json"
    monkeypatch.setattr(env, "REPORT_PATH", out)
    monkeypatch.setattr(env, "find_ffmpeg", lambda: False)
    monkeypatch.setattr(env, "models_present", lambda: True)
    monkeypatch.setattr(env, "torch_mps_available", lambda: False)

    code = env.main(["--check"])

    captured = capsys.readouterr()
    assert code == 0
    assert "brew install ffmpeg" in captured.out
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["ffmpeg_found"] is False


def test_main_without_check_prints_help(capsys: pytest.CaptureFixture[str]):
    assert env.main([]) == 2
    assert "usage" in capsys.readouterr().out.lower()


# ---------------------------------------------------------------- guardrails


def test_module_never_calls_half_precision():
    """fp16 금지 계약: 모듈 소스에 .half() 호출이 없어야 한다(MPS NaN 회귀 방지)."""
    source = Path(env.__file__).read_text(encoding="utf-8")
    assert ".half()" not in source


def test_spike_reports_mps_unavailable_without_model_load(monkeypatch: pytest.MonkeyPatch):
    """MPS가 없으면 emotiefflib를 실제 로드하기 전에 (False, mps_unavailable)."""
    monkeypatch.setattr(env, "torch_mps_available", lambda: False)
    ok, detail = env.mps_spike()
    assert ok is False
    assert detail == "mps_unavailable"


def test_spike_degrades_when_emotiefflib_missing(monkeypatch: pytest.MonkeyPatch):
    """emotiefflib이 아예 없어도 예외 대신 (False, 사유) — 우아한 강등 계약."""
    monkeypatch.setattr(env, "torch_mps_available", lambda: True)
    monkeypatch.setitem(sys.modules, "emotiefflib.facial_analysis", None)
    fresh = importlib.reload(env)
    try:
        ok, detail = fresh.mps_spike()
        assert ok is False
        assert "emotiefflib unavailable" in detail
    finally:
        monkeypatch.undo()
        importlib.reload(env)
