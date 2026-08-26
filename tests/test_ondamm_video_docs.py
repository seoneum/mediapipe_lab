"""Hermetic documentation tests for the ON DAMM video analyzer docs (todo 7).

These tests only read files inside the repository. No network access, no model
loading, no imports of heavy app modules — with one documented exception:
``UbuntuDocDeviceTruthTests`` imports ``app.ondamm_video_env`` solely to call
its pure function ``compute_default_device`` (no model loads, no --check run)
so the device-selection doc claims are checked against code truth (review M2).

Parallel-wave design note: ``app/ondamm_video_analyzer_cli.py`` and
``scripts/ondamm_video_analyzer.sh`` are being built by todo 5 in parallel,
so their existence is checked SOFTLY (documented as pending-todo-5). The docs
must already NAME them; the files themselves may land later without breaking
this suite. Everything else checked here is current truth (todo 1 landed):
requirements.txt marker section, scripts/download_video_models.sh,
models/face_landmarker.task.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
README_PATH = REPO_ROOT / "README.md"
UBUNTU_DOC_PATH = REPO_ROOT / "docs" / "ondamm-video-ubuntu22.md"
ANALYZER_CLI_PATH = REPO_ROOT / "app" / "ondamm_video_analyzer_cli.py"
ANALYZER_WRAPPER_PATH = REPO_ROOT / "scripts" / "ondamm_video_analyzer.sh"

NON_DIAGNOSTIC_CAPTION = "행동 프록시 추정 결과이며 의학적·교육적 진단이 아닙니다"


def read_readme() -> str:
    return README_PATH.read_text(encoding="utf-8")


def read_ubuntu_doc() -> str:
    return UBUNTU_DOC_PATH.read_text(encoding="utf-8")


class ReadmeLicenseNoticeTests(unittest.TestCase):
    def test_readme_contains_required_license_keywords(self) -> None:
        text = read_readme()
        for keyword in ("AGPL", "buffalo_l", "Apache-2.0", "진단이 아닙니다"):
            with self.subTest(keyword=keyword):
                self.assertIn(keyword, text)

    def test_readme_notice_covers_excluded_projects(self) -> None:
        text = read_readme()
        # LibreFace and sixdrepnet must be explicitly documented as NOT included.
        self.assertIn("LibreFace", text)
        self.assertIn("sixdrepnet", text)
        self.assertIn("포함되지 않은 프로젝트", text)

    def test_readme_states_insightface_weights_restriction(self) -> None:
        text = read_readme()
        self.assertIn("MIT", text)
        self.assertIn("non-commercial research purposes only", text)


class ReadmeAnalyzerContractTests(unittest.TestCase):
    def test_readme_documents_all_cli_flags(self) -> None:
        text = read_readme()
        for flag in (
            "--input",
            "--output",
            "--device",
            "--sample-every",
            "--metrics-json",
            "--metrics-csv",
        ):
            with self.subTest(flag=flag):
                self.assertIn(flag, text)

    def test_readme_documents_device_choices(self) -> None:
        text = read_readme()
        self.assertIn("{auto,cpu,mps,cuda}", text)

    def test_readme_documents_exit_codes(self) -> None:
        text = read_readme()
        for code in ("`0`: 성공", "`2`: 입력 영상 열기 실패", "`3`: 모델 파일 없음", "`4`: 렌더 실패"):
            with self.subTest(code=code):
                self.assertIn(code, text)

    def test_readme_names_analyzer_entrypoints(self) -> None:
        text = read_readme()
        self.assertIn("app/ondamm_video_analyzer_cli.py", text)
        self.assertIn("scripts/ondamm_video_analyzer.sh", text)

    def test_readme_documents_person_metrics_fields(self) -> None:
        text = read_readme()
        for field in (
            "global_id",
            "attention_pct",
            "focus_seconds",
            "expression_timeline",
            "frames_covered",
            "total_frames",
            "low_confidence",
        ):
            with self.subTest(field=field):
                self.assertIn(field, text)
        for level in ("낮음", "중간", "높음"):
            with self.subTest(interest_level=level):
                self.assertIn(level, text)

    def test_readme_documents_unknown_n_degradation(self) -> None:
        text = read_readme()
        self.assertIn("unknown_N 표기의 의미", text)
        self.assertIn("low_confidence=true", text)

    def test_readme_burned_in_caption_is_verbatim(self) -> None:
        self.assertIn(NON_DIAGNOSTIC_CAPTION, read_readme())

    def test_readme_documents_throughput_expectation(self) -> None:
        text = read_readme()
        self.assertIn("60초 1080p/30fps", text)
        self.assertIn("10분", text)


class ReferencedPathsExistTests(unittest.TestCase):
    """File-existence checks limited to CURRENT-TRUTH paths (todo 1 landed)."""

    def test_setup_script_exists(self) -> None:
        self.assertTrue((REPO_ROOT / "scripts" / "setup_env.sh").is_file())

    def test_download_video_models_script_exists(self) -> None:
        self.assertTrue((REPO_ROOT / "scripts" / "download_video_models.sh").is_file())

    def test_face_landmarker_model_exists(self) -> None:
        model = REPO_ROOT / "models" / "face_landmarker.task"
        self.assertTrue(model.is_file())
        self.assertGreater(model.stat().st_size, 1024 * 1024)

    def test_requirements_has_video_analyzer_marker_section(self) -> None:
        requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("# --- ON DAMM video analyzer ---", requirements)

    def test_ubuntu_doc_exists(self) -> None:
        self.assertTrue(UBUNTU_DOC_PATH.is_file())


class UbuntuDocContentTests(unittest.TestCase):
    def test_ubuntu_doc_covers_environment_basics(self) -> None:
        text = read_ubuntu_doc()
        for needle in (
            "python3.10",
            "python3.10 -m venv .venv",
            "sudo apt install",
            "ffmpeg",
            "libgl1",
            "libglib2.0-0",
            "download.pytorch.org/whl",
        ):
            with self.subTest(needle=needle):
                self.assertIn(needle, text)

    def test_ubuntu_doc_reuses_same_cli_commands(self) -> None:
        text = read_ubuntu_doc()
        for flag in ("--input", "--output", "--device auto", "--sample-every 3"):
            with self.subTest(flag=flag):
                self.assertIn(flag, text)

    def test_ubuntu_doc_notes_mps_fallback(self) -> None:
        text = read_ubuntu_doc()
        self.assertIn("mps_available", text)
        self.assertIn("폴백(fallback)", text)


class UbuntuDocDeviceTruthTests(unittest.TestCase):
    """n6 보완: 문서 문자열 '존재'가 아니라 코드 '진실'과 대조(M2 회귀 방지)."""

    def test_ubuntu_doc_guides_explicit_cuda_without_false_auto_cuda_claim(self) -> None:
        text = read_ubuntu_doc()
        self.assertIn("--device cuda", text)  # Ubuntu+CUDA는 명시 지정 안내 필수
        # 거짓 주장 패턴(수정 전 원문): auto가 cuda를 고른다는 주장
        self.assertNotIn("Ubuntu + CUDA GPU라면 cuda", text)
        # 거짓 주장 패턴: mps가 CPU/CUDA로 자동 폴백한다는 주장
        self.assertNotIn("자동으로 CPU 또는 CUDA로 폴백", text)

    def test_readme_device_note_matches_auto_contract(self) -> None:
        text = read_readme()
        self.assertIn("절대 cuda를 고르지 않습니다", text)
        self.assertIn("--device cuda", text)

    def test_compute_default_device_is_binary_mps_or_cpu(self) -> None:
        import ondamm_video_env  # noqa: E402  (순수 함수만 호출; 모델 로드 없음)

        self.assertEqual(ondamm_video_env.compute_default_device(True, True), "mps")
        self.assertEqual(ondamm_video_env.compute_default_device(False, True), "cpu")
        self.assertEqual(ondamm_video_env.compute_default_device(False, False), "cpu")


class AnalyzerEntrypointSoftCheckTests(unittest.TestCase):
    """Soft existence checks for todo-5 deliverables.

    HARD part: the docs must name both entrypoints regardless of todo 5 status.
    SOFT part: if the files exist (todo 5 landed), assert they are real files;
    otherwise skip with a pending-todo-5 reason. Either way this suite stays
    green before AND after todo 5 merges.
    """

    def test_analyzer_cli_soft_existence_pending_todo5(self) -> None:
        self.assertIn("app/ondamm_video_analyzer_cli.py", read_readme())
        if not ANALYZER_CLI_PATH.exists():
            self.skipTest(
                "app/ondamm_video_analyzer_cli.py not present yet "
                "(pending todo 5); doc-level soft check only"
            )
        self.assertTrue(ANALYZER_CLI_PATH.is_file())

    def test_analyzer_wrapper_soft_existence_pending_todo5(self) -> None:
        self.assertIn("scripts/ondamm_video_analyzer.sh", read_readme())
        if not ANALYZER_WRAPPER_PATH.exists():
            self.skipTest(
                "scripts/ondamm_video_analyzer.sh not present yet "
                "(pending todo 5); doc-level soft check only"
            )
        self.assertTrue(ANALYZER_WRAPPER_PATH.is_file())


if __name__ == "__main__":
    unittest.main()
