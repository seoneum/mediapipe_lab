"""ON DAMM 캘리브레이션 헤드 학습/평가 (todo 6, option B).

frozen EmotiEffLib 특징 위의 소형 헤드만 학습한다. 백본 재학습/가중치 수정은
일절 없다(sklearn 헤드가 전부). sklearn 임포트는 이 모듈로 국한된다.

입력(features-dir/{person}/): features_{p}.npy, labels.npy(또는 사람이 수정한
y.npy — 있으면 우선), sessions.npy, probs_{p}.npy, meta.json.
  - y.npy / session.npy 는 `python -m app.ondamm_calib_labelstudio parse-export`
    가 LS 수정 라벨로 기록하는 파일이다.

평가: 개인별 leave-one-SESSION-out(LOSO). 각 개인별로 세션 하나를 홀드아웃하고
그 개인의 나머지 세션으로 학습→홀드아웃 예측, 개인 단위로 집계해
macro-F1(라벨 순서 = 정렬된 클래스명, zero_division=0)을 낸다.
베이스라인: 저장된 사전학습 probs argmax를 EMOTION_LABELS_8을 통해 identity
매핑한 예측(같은 행 집합에서 계산).

SHIP RULE(승인된 δ=0.02): mean LOSO macro-F1 ≥ baseline + 0.02 일 때만 보정 헤드를
채택하고 calib_head.pkl 을 기록한다. 미달 시 보고서에 "use_pretrained_only": true
(헤드 pkl은 절대 기록하지 않음 — fallback에서도).

보고서 outputs/ondamm/calib/report.md: person×{baseline_f1, calib_f1} 표,
confusion matrix 코드블록, 채택 판정 행, 비진단 고지 문구로 끝난다.
결정성: --seed 고정 + 타임스탬프 없는 보고서 → 동일 입력이면 동일 바이트.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "app") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "app"))

from ondamm_video_face_signals import EMOTION_LABELS_8  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402

SHIP_DELTA = 0.02
NON_DIAGNOSTIC_FOOTER = "비고: 행동 프록시 추정 결과이며 의학적·교육적 진단이 아닙니다."
EXPECTED_PERSON_FILES = (
    "features_{person}.npy",
    "labels.npy",
    "sessions.npy",
    "probs_{person}.npy",
    "meta.json",
)


class CalibTrainError(RuntimeError):
    """캘리브레이션 학습 단계의 명확한 실패."""


# ------------------------------------------------------------------ loading


def load_person_dir(features_dir: str | Path, person: str) -> dict:
    """개인 디렉터리 로드. 누락 파일은 기대 레이아웃과 함께 나열(actionable)."""
    pdir = Path(features_dir) / person
    expected = [name.format(person=person) for name in EXPECTED_PERSON_FILES]
    missing = [name for name in expected if not (pdir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing calibration artifacts for person '{person}' under '{pdir}': {missing}. "
            f"expected layout: {pdir}/{{{', '.join(expected)}}} "
            f"(produce them with `python -m app.ondamm_calib_prep`)"
        )
    with open(pdir / "meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    X = np.load(pdir / f"features_{person}.npy")
    y_path = pdir / "y.npy"  # LS 수정 라벨(있으면 labels.npy 대신 사용)
    y = np.load(y_path) if y_path.is_file() else np.load(pdir / "labels.npy")
    sess_name = "session.npy" if (pdir / "session.npy").is_file() else "sessions.npy"
    sessions = np.load(pdir / sess_name)
    probs = np.load(pdir / f"probs_{person}.npy")
    if not (X.shape[0] == y.shape[0] == sessions.shape[0] == probs.shape[0]):
        raise CalibTrainError(
            f"row mismatch for person '{person}': features {X.shape[0]}, labels {y.shape[0]}, "
            f"sessions {sessions.shape[0]}, probs {probs.shape[0]}"
        )
    return {"person": person, "X": X, "y": y.astype(np.int64), "sessions": sessions, "probs": probs, "meta": meta}


def collect_dataset(features_dir: str | Path) -> dict:
    """모든 개인 디렉터리 수집 → 결합 데이터셋. 클래스/세션 사전조건 검사."""
    root = Path(features_dir)
    if not root.is_dir():
        raise FileNotFoundError(
            f"features dir not found: '{root}' — expected per-person subdirectories like "
            f"{root}/{{person}}/features_{{person}}.npy (run `python -m app.ondamm_calib_prep` first)"
        )
    persons = sorted(d.name for d in root.iterdir() if d.is_dir())
    if not persons:
        raise FileNotFoundError(
            f"no per-person directories under '{root}' — expected layout "
            f"{root}/{{person}}/features_{{person}}.npy"
        )

    loaded = [load_person_dir(root, p) for p in persons]
    class_names = sorted({c for item in loaded for c in item["meta"]["class_names"]})
    name_to_idx = {c: i for i, c in enumerate(class_names)}
    for item in loaded:
        local_names = item["meta"]["class_names"]
        if item["y"].size and (item["y"].min() < 0 or item["y"].max() >= len(local_names)):
            raise CalibTrainError(
                f"label index out of range for person '{item['person']}' "
                f"(meta lists {len(local_names)} classes)"
            )
        if item["probs"].shape[1] != len(EMOTION_LABELS_8):
            raise CalibTrainError(
                f"probs for person '{item['person']}' must have {len(EMOTION_LABELS_8)} "
                f"pretrained columns, got {item['probs'].shape[1]}"
            )

    X_parts, y_parts, s_parts, p_parts, pr_parts = [], [], [], [], []
    for item in loaded:
        local_names = item["meta"]["class_names"]
        y_parts.append(np.array([name_to_idx[local_names[j]] for j in item["y"]], dtype=np.int64))
        X_parts.append(item["X"])
        s_parts.append(item["sessions"])
        p_parts.append(np.full(item["y"].shape[0], item["person"]))
        pr_parts.append(item["probs"])

    X = np.concatenate(X_parts).astype(np.float64)
    y = np.concatenate(y_parts)
    sessions = np.concatenate(s_parts)
    persons_arr = np.concatenate(p_parts)
    probs = np.concatenate(pr_parts)

    if len(class_names) < 2:
        raise ValueError(f"need ≥2 classes to train a calibration head (found: {class_names})")
    unique_sessions = sorted(set(str(s) for s in sessions))
    if len(unique_sessions) < 2:
        raise ValueError(
            f"need ≥2 distinct recording sessions for leave-one-session-out CV "
            f"(found: {unique_sessions})"
        )
    return {
        "X": X,
        "y": y,
        "sessions": sessions,
        "persons": persons_arr,
        "probs": probs,
        "class_names": class_names,
        "persons_list": persons,
    }


# ------------------------------------------------------------------ folds & metrics


def iter_loso_folds(sessions: np.ndarray | list[str]) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """세션 배열 → [(train_idx, test_idx, held_session)]. 순수 함수(테스트 대상).

    각 고유 세션이 정확히 한 번 test가 되고, 그 외 모든 행이 train에 속한다.
    """
    arr = np.asarray([str(s) for s in sessions])
    folds: list[tuple[np.ndarray, np.ndarray, str]] = []
    for held in sorted(set(arr.tolist())):
        test_idx = np.flatnonzero(arr == held)
        train_idx = np.flatnonzero(arr != held)
        folds.append((train_idx, test_idx, held))
    return folds


def make_head(kind: str, seed: int):
    """스펙 고정 하이퍼파라미터의 헤드 팩토리."""
    if kind == "logreg":
        return LogisticRegression(
            class_weight="balanced", C=1.0, max_iter=2000, random_state=int(seed)
        )
    if kind == "mlp":
        return MLPClassifier(hidden_layer_sizes=(64,), random_state=int(seed))
    raise CalibTrainError(f"unknown head kind '{kind}' (expected 'logreg' or 'mlp')")


def baseline_predictions(probs: np.ndarray) -> list[str]:
    """저장된 사전학습 probs argmax → identity 매핑으로 클래스명 문자열 예측."""
    return [EMOTION_LABELS_8[int(np.argmax(row))] for row in np.asarray(probs)]


def macro_f1(y_true_idx: Sequence[int], y_pred_str: Sequence[str], class_names: list[str]) -> float:
    """문자열 예측 vs 인덱스 정답의 macro-F1. truth 밖 예측은 그냥 오답으로 처리."""
    names = np.asarray([class_names[i] for i in y_true_idx])
    preds = np.asarray([str(p) for p in y_pred_str])
    return float(
        f1_score(names, preds, labels=list(class_names), average="macro", zero_division=0)
    )


def evaluate_loso(dataset: dict, kind: str = "logreg", seed: int = 0) -> dict:
    """개인별 LOSO 집계 → 메트릭 dict(순수 — fit_predict 주입 가능)."""
    return _evaluate(dataset, kind, seed, fit_predict=None)


def _evaluate(
    dataset: dict,
    kind: str,
    seed: int,
    fit_predict: callable | None,
) -> dict:
    X, y = dataset["X"], dataset["y"]
    sessions, persons, class_names = dataset["sessions"], dataset["persons"], dataset["class_names"]
    base_str = baseline_predictions(dataset["probs"])
    if fit_predict is None:
        def fit_predict(train_idx, test_idx):
            head = make_head(kind, seed)
            head.fit(X[train_idx], y[train_idx])
            idx = head.predict(X[test_idx])
            return [class_names[int(i)] for i in idx]

    per_person: dict[str, dict[str, list]] = {}
    fold_log: list[dict] = []
    for p in sorted(set(str(x) for x in persons)):
        mask_p = np.flatnonzero(persons == p)
        sub_sessions = sessions[mask_p]
        for train_sub, test_sub, held in iter_loso_folds(sub_sessions):
            train_idx = mask_p[train_sub]
            test_idx = mask_p[test_sub]
            pred_str = fit_predict(train_idx, test_idx)
            bucket = per_person.setdefault(p, {"true": [], "calib": [], "base": []})
            bucket["true"].extend(int(v) for v in y[test_idx])
            bucket["calib"].extend(pred_str)
            bucket["base"].extend(base_str[i] for i in test_idx)
            fold_log.append({"person": p, "held_session": held, "n_test": int(len(test_idx))})

    metrics: dict = {
        "kind": kind,
        "seed": int(seed),
        "ship_delta": SHIP_DELTA,
        "per_person": {},
        "fold_count": len(fold_log),
    }
    cal_means, base_means = [], []
    for p in sorted(per_person):
        b = macro_f1(per_person[p]["true"], per_person[p]["base"], class_names)
        c = macro_f1(per_person[p]["true"], per_person[p]["calib"], class_names)
        metrics["per_person"][p] = {"baseline_f1": b, "calib_f1": c}
        cal_means.append(c)
        base_means.append(b)
    metrics["baseline_mean"] = float(np.mean(base_means)) if base_means else 0.0
    metrics["calib_mean"] = float(np.mean(cal_means)) if cal_means else 0.0
    # overall = pooled rows across persons (same aggregated folds)
    pooled_true: list[int] = []
    pooled_calib: list[str] = []
    pooled_base: list[str] = []
    for p in sorted(per_person):
        pooled_true.extend(per_person[p]["true"])
        pooled_calib.extend(per_person[p]["calib"])
        pooled_base.extend(per_person[p]["base"])
    metrics["overall_calib_f1"] = macro_f1(pooled_true, pooled_calib, class_names)
    metrics["overall_baseline_f1"] = macro_f1(pooled_true, pooled_base, class_names)
    metrics["confusion_matrix"] = confusion_matrix(
        [class_names[i] for i in pooled_true],
        pooled_calib,
        labels=class_names,
    ).tolist()
    metrics["rows"] = int(len(pooled_true))
    metrics["unique_sessions"] = sorted(set(str(s) for s in sessions))
    return metrics


# ------------------------------------------------------------------ ship rule


def decide_ship(calib_mean: float, baseline_mean: float, delta: float = SHIP_DELTA) -> dict:
    """채택 규칙(순수 함수): adopt iff calib_mean ≥ baseline_mean + delta."""
    margin = float(calib_mean) - float(baseline_mean)
    return {
        "adopt": bool(margin >= float(delta)),
        "margin": margin,
        "delta": float(delta),
        "use_pretrained_only": bool(margin < float(delta)),
    }


def fit_final_head(dataset: dict, kind: str, seed: int):
    """채택 시에만 호출: 전체 행으로 최종 헤드 적합."""
    head = make_head(kind, seed)
    head.fit(dataset["X"], dataset["y"])
    return head


def write_report(path: Path, metrics: dict, verdict: dict, dataset: dict) -> Path:
    """결정적 마크다운 보고서(타임스탬프 없음 → 동일 입력 ⇒ 동일 바이트)."""
    class_names = dataset["class_names"]
    lines: list[str] = []
    lines.append("# ON DAMM calibration report")
    lines.append("")
    lines.append(
        f"- head: `{metrics['kind']}` (seed={metrics['seed']}), "
        f"eval: leave-one-session-out macro-F1, rows={metrics['rows']}, "
        f"persons={len(metrics['per_person'])}, sessions={len(metrics['unique_sessions'])}"
    )
    lines.append(
        f"- classes ({len(class_names)}): {', '.join(class_names)}"
    )
    lines.append(f"- ship rule: adopt iff mean LOSO macro-F1 ≥ baseline + {metrics['ship_delta']:.2f}")
    lines.append("")
    lines.append("| person | baseline_f1 | calib_f1 |")
    lines.append("| --- | --- | --- |")
    for p in sorted(metrics["per_person"]):
        row = metrics["per_person"][p]
        lines.append(f"| {p} | {row['baseline_f1']:.4f} | {row['calib_f1']:.4f} |")
    lines.append(
        f"| **mean** | **{metrics['baseline_mean']:.4f}** | **{metrics['calib_mean']:.4f}** |"
    )
    lines.append(
        f"| overall(pooled) | {metrics['overall_baseline_f1']:.4f} | {metrics['overall_calib_f1']:.4f} |"
    )
    lines.append("")
    if verdict["adopt"]:
        lines.append(
            f"verdict: ADOPT calibrated head (margin +{verdict['margin']:.4f} ≥ "
            f"{verdict['delta']:.2f}) — head saved to calib_head.pkl"
        )
    else:
        lines.append(
            f"verdict: FALLBACK — use_pretrained_only: true (margin +{verdict['margin']:.4f} < "
            f"{verdict['delta']:.2f}); no calibrated head was saved"
        )
    lines.append("")
    lines.append(
        "confusion matrix (calibrated predictions; rows=true, cols=pred; "
        f"labels order = {class_names}):"
    )
    lines.append("```")
    header = "true\\pred " + " ".join(f"{c[:10]:>10}" for c in class_names)
    lines.append(header)
    for i, row in enumerate(metrics["confusion_matrix"]):
        lines.append(f"{class_names[i][:9]:>9} " + " ".join(f"{v:>11}" for v in row))
    lines.append("```")
    lines.append("")
    lines.append(NON_DIAGNOSTIC_FOOTER)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


# ------------------------------------------------------------------ pipeline


def run_training(features_dir: str | Path, *, kind: str = "logreg", seed: int = 0) -> dict:
    """전체 파이프라인. 채택 시에만 calib_head.pkl 기록. 결과 dict 반환."""
    dataset = collect_dataset(features_dir)
    metrics = evaluate_loso(dataset, kind=kind, seed=seed)
    verdict = decide_ship(metrics["calib_mean"], metrics["baseline_mean"], SHIP_DELTA)
    out_root = Path(features_dir)
    report_path = write_report(out_root / "report.md", metrics, verdict, dataset)
    head_path: Path | None = None
    if verdict["adopt"]:
        head = fit_final_head(dataset, kind, seed)
        head_path = out_root / "calib_head.pkl"
        with open(head_path, "wb") as fh:
            pickle.dump(
                {
                    "estimator": head,
                    "class_names": dataset["class_names"],
                    "kind": kind,
                    "seed": int(seed),
                    "trained_on_rows": metrics["rows"],
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
    return {"metrics": metrics, "verdict": verdict, "report": report_path, "head": head_path}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.ondamm_calib_train",
        description=(
            "Train/evaluate the tiny calibration head on frozen EmotiEffLib features "
            "(leave-one-session-out vs pretrained baseline; ship rule δ=0.02)."
        ),
    )
    parser.add_argument("--features-dir", default="outputs/ondamm/calib")
    parser.add_argument("--mlp", action="store_true", help="use MLPClassifier(64,) instead of LogisticRegression")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    kind = "mlp" if args.mlp else "logreg"
    try:
        result = run_training(args.features_dir, kind=kind, seed=args.seed)
    except (ValueError, FileNotFoundError, CalibTrainError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    m = result["metrics"]
    print(f"mean LOSO macro-F1  baseline={m['baseline_mean']:.4f}  calibrated={m['calib_mean']:.4f}")
    if result["verdict"]["adopt"]:
        print(f"verdict: ADOPT calibrated head (margin +{result['verdict']['margin']:.4f})")
        print(f"head pickled → {result['head']}")
    else:
        print("verdict: FALLBACK — use_pretrained_only: true (no head saved)")
    print(f"report → {result['report']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
