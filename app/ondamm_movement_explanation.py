"""Human-readable, non-diagnostic facts for personal movement events.

The temporal embedding remains the matching signal.  This module records a
small, local-only explanation layer so a reviewer can understand which facial
regions changed and why a recurrence crossed the event threshold.  Scores are
descriptive movement features, not emotions, diagnoses, or probabilities.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


REGION_LABELS = {
    "mouth": "입·턱 주변",
    "left_eye": "왼쪽 눈 주변",
    "right_eye": "오른쪽 눈 주변",
    "left_brow": "왼쪽 눈썹 주변",
    "right_brow": "오른쪽 눈썹 주변",
}

REGION_FEATURES = {
    "mouth": "motion_mouth",
    "left_eye": "motion_left_eye",
    "right_eye": "motion_right_eye",
    "left_brow": "motion_left_brow",
    "right_brow": "motion_right_brow",
}

BLENDSHAPE_LABELS = {
    "jawOpen": "턱 벌리기",
    "jawForward": "턱 내밀기",
    "mouthClose": "입 다물기",
    "mouthFunnel": "입술 모으기",
    "mouthPucker": "입술 오므리기",
    "mouthSmileLeft": "왼쪽 입꼬리 당기기",
    "mouthSmileRight": "오른쪽 입꼬리 당기기",
    "mouthFrownLeft": "왼쪽 입꼬리 내리기",
    "mouthFrownRight": "오른쪽 입꼬리 내리기",
    "mouthPressLeft": "왼쪽 입술 누르기",
    "mouthPressRight": "오른쪽 입술 누르기",
    "eyeBlinkLeft": "왼쪽 눈 감기",
    "eyeBlinkRight": "오른쪽 눈 감기",
    "eyeWideLeft": "왼쪽 눈 크게 뜨기",
    "eyeWideRight": "오른쪽 눈 크게 뜨기",
    "eyeSquintLeft": "왼쪽 눈 좁히기",
    "eyeSquintRight": "오른쪽 눈 좁히기",
    "browInnerUp": "안쪽 눈썹 올리기",
    "browOuterUpLeft": "왼쪽 바깥 눈썹 올리기",
    "browOuterUpRight": "오른쪽 바깥 눈썹 올리기",
    "browDownLeft": "왼쪽 눈썹 내리기",
    "browDownRight": "오른쪽 눈썹 내리기",
    "cheekPuff": "볼 부풀리기",
}


def _finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if np.isfinite(number) else 0.0


def _blendshape_region(name: str) -> str | None:
    if name.startswith(("mouth", "jaw", "cheek")):
        return "mouth"
    if name.startswith("eye"):
        if name.endswith("Left"):
            return "left_eye"
        if name.endswith("Right"):
            return "right_eye"
    if name.startswith("brow"):
        if name.endswith("Left"):
            return "left_brow"
        if name.endswith("Right"):
            return "right_brow"
        return "left_brow"
    return None


def summarize_temporal_features(samples: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    """Summarize active episode endpoints without retaining raw feature rows."""
    clean = [dict(sample) for sample in samples if sample]
    if not clean:
        return {}

    region_scores = {
        region: max(abs(_finite(sample.get(feature))) for sample in clean)
        for region, feature in REGION_FEATURES.items()
    }
    total = sum(region_scores.values())
    region_distribution = {
        region: round(score / total, 6) if total > 1e-12 else 0.0
        for region, score in region_scores.items()
    }
    dominant_region = max(region_scores, key=region_scores.get)

    changes: list[dict[str, Any]] = []
    feature_names = sorted({name for sample in clean for name in sample if name.startswith("bs_")})
    for feature_name in feature_names:
        public_name = feature_name.removeprefix("bs_")
        if public_name == "_neutral":
            continue
        values = np.asarray([_finite(sample.get(feature_name)) for sample in clean], dtype=float)
        change = float(np.max(values) - np.min(values))
        if change < 0.01:
            continue
        trend = float(values[-1] - values[0])
        changes.append(
            {
                "feature": public_name,
                "label": BLENDSHAPE_LABELS.get(public_name, public_name),
                "region": _blendshape_region(public_name),
                "change_points": round(change * 100.0, 1),
                "start_to_end_points": round(trend * 100.0, 1),
            }
        )
    changes.sort(key=lambda item: (-float(item["change_points"]), str(item["feature"])))
    top_changes = changes[:5]
    dominant_share = region_distribution[dominant_region]
    secondary = sorted(region_scores, key=region_scores.get, reverse=True)[1]
    plain_summary = (
        f"{REGION_LABELS[dominant_region]}의 상대 움직임이 가장 컸고, "
        f"전체 다섯 부위 움직임 중 약 {dominant_share * 100:.0f}%를 차지했습니다. "
        f"다음으로 큰 부위는 {REGION_LABELS[secondary]}였습니다."
    )
    if top_changes:
        lead = top_changes[0]
        plain_summary += (
            f" 영상 특징값에서는 {lead['label']} 활성도가 구간 안에서 "
            f"최대 {lead['change_points']:.1f}%p 변했습니다."
        )
    return {
        "dominant_region": dominant_region,
        "dominant_region_label": REGION_LABELS[dominant_region],
        "region_scores": {key: round(value, 8) for key, value in region_scores.items()},
        "region_distribution": region_distribution,
        "top_changes": top_changes,
        "plain_summary": plain_summary,
        "sample_count": len(clean),
        "non_diagnostic_notice": "움직임 특징의 상대 변화이며 감정·집중도·의도·진단을 뜻하지 않습니다.",
    }


def summarize_blendshape_samples(samples: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    """Describe visible blendshape ranges for existing clips lacking runtime facts."""
    clean = [dict(sample) for sample in samples if sample]
    if len(clean) < 2:
        return {}
    feature_names = sorted({name for sample in clean for name in sample if name and name != "_neutral"})
    changes: list[dict[str, Any]] = []
    region_scores = {region: 0.0 for region in REGION_LABELS}
    for feature_name in feature_names:
        values = np.asarray([_finite(sample.get(feature_name)) for sample in clean], dtype=float)
        change = float(np.max(values) - np.min(values))
        if change < 0.01:
            continue
        region = _blendshape_region(feature_name)
        if region:
            region_scores[region] = max(region_scores[region], change)
        changes.append(
            {
                "feature": feature_name,
                "label": BLENDSHAPE_LABELS.get(feature_name, feature_name),
                "region": region,
                "change_points": round(change * 100.0, 1),
                "start_to_end_points": round(float(values[-1] - values[0]) * 100.0, 1),
            }
        )
    if not changes or sum(region_scores.values()) <= 1e-12:
        return {}
    changes.sort(key=lambda item: (-float(item["change_points"]), str(item["feature"])))
    total = sum(region_scores.values())
    distribution = {key: round(value / total, 6) for key, value in region_scores.items()}
    dominant = max(region_scores, key=region_scores.get)
    lead = changes[0]
    return {
        "dominant_region": dominant,
        "dominant_region_label": REGION_LABELS[dominant],
        "region_scores": {key: round(value, 6) for key, value in region_scores.items()},
        "region_distribution": distribution,
        "top_changes": changes[:5],
        "plain_summary": (
            f"샘플 프레임 사이에서는 {REGION_LABELS[dominant]} 변화가 가장 컸습니다. "
            f"가장 큰 세부 특징값은 {lead['label']} 항목이며, 측정 구간에서 {lead['change_points']:.1f}%p 범위였습니다."
        ),
        "sample_count": len(clean),
        "source": "local_mediapipe_sampled_frames",
        "non_diagnostic_notice": "샘플 프레임의 blendshape 변화이며 감정·집중도·의도·진단을 뜻하지 않습니다.",
    }


def compare_region_profiles(
    current: Mapping[str, Any] | None,
    previous_distribution: Mapping[str, float] | None,
) -> dict[str, Any] | None:
    """Return an explainable histogram similarity against earlier occurrences."""
    if not current or not previous_distribution:
        return None
    current_distribution = current.get("region_distribution")
    if not isinstance(current_distribution, Mapping):
        return None
    regions = tuple(REGION_LABELS)
    current_values = np.asarray([_finite(current_distribution.get(name)) for name in regions])
    previous_values = np.asarray([_finite(previous_distribution.get(name)) for name in regions])
    if current_values.sum() <= 1e-12 or previous_values.sum() <= 1e-12:
        return None
    current_values /= current_values.sum()
    previous_values /= previous_values.sum()
    similarity = float(np.clip(1.0 - 0.5 * np.abs(current_values - previous_values).sum(), 0.0, 1.0))
    comparisons = []
    for index, region in enumerate(regions):
        comparisons.append(
            {
                "region": region,
                "label": REGION_LABELS[region],
                "current_percent": round(float(current_values[index]) * 100.0, 1),
                "previous_percent": round(float(previous_values[index]) * 100.0, 1),
                "difference_points": round(abs(float(current_values[index] - previous_values[index])) * 100.0, 1),
            }
        )
    comparisons.sort(key=lambda item: (-float(item["current_percent"]), str(item["region"])))
    previous_dominant = regions[int(np.argmax(previous_values))]
    current_dominant = regions[int(np.argmax(current_values))]
    return {
        "similarity_percent": round(similarity * 100.0, 1),
        "dominant_region_matches": current_dominant == previous_dominant,
        "current_dominant_region": current_dominant,
        "previous_dominant_region": previous_dominant,
        "region_comparison": comparisons,
    }


def build_selection_explanation(
    *,
    occurrence_count: int,
    occurrence_threshold: int,
    embedding_distance: float | None,
    movement_summary: Mapping[str, Any] | None,
    regional_comparison: Mapping[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    """Build deterministic facts that an LLM may rewrite but must not alter."""
    previous_count = max(0, int(occurrence_count) - 1)
    embedding_similarity = None
    if embedding_distance is not None:
        embedding_similarity = round(float(np.clip(1.0 - float(embedding_distance), 0.0, 1.0)) * 100.0, 1)
    dominant_label = None
    if movement_summary:
        dominant_label = str(movement_summary.get("dominant_region_label") or "").strip() or None
    parts = []
    if dominant_label:
        parts.append(f"이번 구간에서는 {dominant_label}의 상대 움직임이 가장 컸습니다.")
    if embedding_similarity is not None:
        parts.append(
            f"시간 흐름 임베딩이 앞선 {previous_count}회로 만든 개인 후보와 {embedding_similarity:.1f}% 유사했습니다."
        )
    if regional_comparison:
        parts.append(
            f"움직인 부위의 비율도 이전 발생 평균과 {float(regional_comparison['similarity_percent']):.1f}% 비슷했습니다."
        )
    if int(occurrence_count) >= int(occurrence_threshold):
        parts.append(
            f"서로 떨어진 발생이 {int(occurrence_count)}회 확인되어 검토 기준 {int(occurrence_threshold)}회를 충족했기 때문에 이벤트로 선정했습니다."
        )
    else:
        parts.append(
            f"현재 {int(occurrence_count)}회 관찰되어 검토 기준 {int(occurrence_threshold)}회에는 아직 이르지 않았습니다."
        )
    parts.append("이 유사도는 같은 움직임일 가능성을 돕는 수치이지 감정이나 의미를 확정하는 값은 아닙니다.")
    facts = {
        "previous_occurrence_count": previous_count,
        "occurrence_count": int(occurrence_count),
        "occurrence_threshold": int(occurrence_threshold),
        "embedding_distance": round(float(embedding_distance), 6) if embedding_distance is not None else None,
        "embedding_similarity_percent": embedding_similarity,
        "regional_comparison": dict(regional_comparison) if regional_comparison else None,
        "dominant_region_label": dominant_label,
    }
    return " ".join(parts), facts
