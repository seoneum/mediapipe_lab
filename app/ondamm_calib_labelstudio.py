"""ON DAMM 캘리브레이션 ↔ Label Studio 양방향 변환기 (todo 6, option B).

로컬 전용 워크플로우 (네트워크/클라우드 불용):
    pip install label-studio
    label-studio start          # → http://localhost:8080
    # 프로젝트 생성 후 Settings → Cloud Storage → Add Source Storage(Local storage)
    # 로 prep이 기록한 프레임 이미지 디렉터리를 마운트하면 파일명 기준으로
    # 임포트한 사전주석(pre-annotations)이 각 작업에 붙는다.

두 방향:
  (a) build-labels : prep 캐시(probs_{person}.npy)의 사전학습 argmax를 클래스명으로
      매핑해 Label Studio 임포트용 pre-annotation JSON을 만든다. 태스크 형식:
        {"data": {"image": path},
         "predictions": [{"result": [{"from_name": "label", "to_name": "image",
                                      "type": "choices",
                                      "value": {"choices": [label]}}]}]}
  (b) parse-export : 사람이 수정한 LS export JSON을 다시 읽어 features 행 순서와
      정렬된 y.npy / session.npy로 기록한다. 매칭 키는 prep이 frames_{person}.json에
      기록한 image 파일명(베이스네임). train 모듈은 y.npy가 있으면 labels.npy 대신
      사용한다(원본 labels.npy는 기계 라벨로 보존).

hermetic: LS 설치/실행 없이 순수 JSON 변환만 테스트한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "app") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "app"))

from ondamm_video_face_signals import EMOTION_LABELS_8  # noqa: E402


class CalibLabelStudioError(RuntimeError):
    """LS 변환 단계의 명확한 실패."""


# ------------------------------------------------------------------ loading


def load_person_dir(features_dir: str | Path, person: str) -> dict:
    """prep 산출물 로드. 누락 시 기대 레이아웃을 나열하며 실패."""
    pdir = Path(features_dir) / person
    expected = [
        f"probs_{person}.npy",
        f"frames_{person}.json",
        "labels.npy",
        "sessions.npy",
        "meta.json",
    ]
    missing = [name for name in expected if not (pdir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing calibration artifacts for person '{person}' under '{pdir}': "
            f"{missing}; expected layout {pdir}/{{{', '.join(expected)}}} — "
            f"run `python -m app.ondamm_calib_prep` first"
        )
    with open(pdir / f"frames_{person}.json", encoding="utf-8") as fh:
        frame_records = json.load(fh)
    with open(pdir / "meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    probs = np.load(pdir / f"probs_{person}.npy")
    labels = np.load(pdir / "labels.npy")
    sessions = np.load(pdir / "sessions.npy")
    return {
        "dir": pdir,
        "frames": frame_records,
        "meta": meta,
        "probs": probs,
        "labels": labels,
        "sessions": sessions,
    }


# ------------------------------------------------------------------ build


def build_tasks(
    probs: np.ndarray,
    frame_records: list[dict],
    class_names: list[str],
    class_names_8: tuple[str, ...] = EMOTION_LABELS_8,
) -> list[dict]:
    """행별 사전학습 argmax → LS 임포트 태스크 목록(순수 함수).

    선택지 공간은 그 개인의 캘리브레이션 클래스(class_names)다. argmax의
    canonical 라벨이 이 공간에 있으면 사전주석으로 채우고, 없으면(사용자가
    녹화하지 않은 표정을 모델이 예측한 경우) predictions를 비워 사람이 직접
    판단하도록 남긴다.
    """
    if probs.ndim != 2 or probs.shape[1] != len(class_names_8):
        raise CalibLabelStudioError(
            f"expected probs (N,{len(class_names_8)}), got shape {probs.shape}"
        )
    if len(frame_records) != probs.shape[0]:
        raise CalibLabelStudioError(
            f"frame records ({len(frame_records)}) and probs rows ({probs.shape[0]}) misaligned"
        )
    tasks: list[dict] = []
    for i, rec in enumerate(frame_records):
        canonical_label = class_names_8[int(np.argmax(probs[i]))]
        result: list[dict] = []
        if canonical_label in class_names:
            result = [
                {
                    "from_name": "label",
                    "to_name": "image",
                    "type": "choices",
                    "value": {"choices": [canonical_label]},
                }
            ]
        tasks.append({"data": {"image": str(rec["image"])}, "predictions": [{"result": result}]})
    return tasks


def cmd_build_labels(features_dir: Path, person: str, out: Path | None) -> int:
    loaded = load_person_dir(features_dir, person)
    tasks = build_tasks(loaded["probs"], loaded["frames"], list(loaded["meta"]["class_names"]))
    out_path = out or (loaded["dir"] / f"preannotations_{person}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(tasks, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"wrote {len(tasks)} LS tasks → {out_path}")
    print("workflow: pip install label-studio && label-studio start  # http://localhost:8080")
    return 0


# ------------------------------------------------------------------ parse


def parse_export_tasks(export: list, loaded: dict, class_names: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    """LS export 태스크 목록 → (y, sessions, stats). features 행 순서와 정렬.

    매칭: 태스크의 data.image 베이스네임 ↔ frames_{person}.json의 image.
    사람 수정은 태스크의 ``annotations`` 아래에서만 읽는다. ``predictions``
    (사전주석 힌트)는 절대 라벨을 바꾸지 않는다 — 사람이 손대지 않은 행은
    녹화 관례가 보증하는 원본 기계 라벨(labels.npy)을 그대로 유지한다.
    - 알 수 없는 choice 값 → 유효 목록을 제시하며 실패
    - frames에 없는 export 이미지 → 매칭 실패 나열 후 실패
    """
    if not isinstance(export, list):
        raise CalibLabelStudioError(f"export JSON must be a list of tasks, got {type(export).__name__}")

    records = loaded["frames"]
    original_labels = loaded["labels"]

    by_image: dict[str, int] = {}
    for rec in records:
        key = Path(str(rec["image"])).name
        if key in by_image:
            raise CalibLabelStudioError(f"duplicate image filename in prep records: '{key}'")
        by_image[key] = int(rec["index"])

    def _first_choice(task: dict) -> str | None:
        for ann in task.get("annotations") or []:
            for r in ann.get("result") or []:
                choices = (r.get("value") or {}).get("choices") or []
                if choices:
                    return str(choices[0])
        return None

    corrections: dict[int, str] = {}
    unmatched: list[str] = []
    for task in export:
        image = str(((task.get("data") or {}).get("image")) or "")
        key = Path(image).name
        if key not in by_image:
            unmatched.append(key or "<empty>")
            continue
        choice = _first_choice(task)
        if choice is None:
            continue
        if choice not in class_names:
            raise CalibLabelStudioError(
                f"export contains unknown choice '{choice}' — valid class names: {class_names}"
            )
        corrections[by_image[key]] = choice

    if unmatched:
        raise CalibLabelStudioError(
            f"{len(unmatched)} export task(s) match no prep frame record "
            f"(checked basename of data.image against frames json): {sorted(unmatched)[:10]}"
        )

    n = len(records)
    y = np.array([int(original_labels[i]) for i in range(n)], dtype=np.int64)
    corrected = 0
    for idx, choice in corrections.items():
        if y[idx] != class_names.index(choice):
            corrected += 1
        y[idx] = class_names.index(choice)

    session_of_row = {int(rec["index"]): str(rec["session"]) for rec in records}
    sess = np.array([session_of_row[i] for i in range(n)])

    stats = {
        "rows": n,
        "tasks_in_export": len(export),
        "corrected": corrected,
        "fallback_to_original": n - len(corrections),
    }
    return y, sess, stats


def parse_export(export_path: str | Path, loaded: dict, class_names: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    """export JSON 파일을 읽어 parse_export_tasks로 위임한다."""
    with open(export_path, encoding="utf-8") as fh:
        export = json.load(fh)
    return parse_export_tasks(export, loaded, class_names)


def cmd_parse_export(features_dir: Path, person: str, export: Path) -> int:
    loaded = load_person_dir(features_dir, person)
    class_names = list(loaded["meta"]["class_names"])
    y, sess, stats = parse_export(export, loaded, class_names)
    np.save(loaded["dir"] / "y.npy", y)
    np.save(loaded["dir"] / "session.npy", sess)
    print(
        f"parsed {stats['tasks_in_export']} LS tasks for '{person}': "
        f"{stats['corrected']} label(s) changed, "
        f"{stats['fallback_to_original']} row(s) kept original machine label"
    )
    print(f"wrote {loaded['dir'] / 'y.npy'} and {loaded['dir'] / 'session.npy'} (aligned with features order)")
    return 0


# ------------------------------------------------------------------ cli


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.ondamm_calib_labelstudio",
        description="Two-way converter between cached calibration predictions and Label Studio.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build-labels", help="build LS pre-annotations JSON from cached argmax predictions")
    p_build.add_argument("--features-dir", default="outputs/ondamm/calib")
    p_build.add_argument("--person", required=True)
    p_build.add_argument("--out", default=None)

    p_parse = sub.add_parser("parse-export", help="parse corrected LS export into y.npy/session.npy")
    p_parse.add_argument("--features-dir", default="outputs/ondamm/calib")
    p_parse.add_argument("--person", required=True)
    p_parse.add_argument("--export", required=True)

    args = parser.parse_args(argv)
    try:
        if args.cmd == "build-labels":
            return cmd_build_labels(Path(args.features_dir), args.person, Path(args.out) if args.out else None)
        return cmd_parse_export(Path(args.features_dir), args.person, Path(args.export))
    except (CalibLabelStudioError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
