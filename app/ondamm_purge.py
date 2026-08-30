from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import ondamm_paths
from ondamm_store import dossier_path, load_dossier


@dataclass(frozen=True)
class PurgeTarget:
    path: Path
    category: str


CATEGORY_LABELS = {
    "dossier": "지원 기록철 원본",
    "pattern_memory": "개인별 반복 패턴 메모리",
    "event_reviews": "이벤트 교차 검토 기록",
    "learning_run": "학습 실행 기록과 짧은 영상",
    "sensing_export": "관찰 보조 내보내기",
    "handoff_export": "인수인계 내보내기",
    "owned_analysis": "연구 분석 결과",
    "browser_cache": "브라우저 재생용 임시 영상",
}


def _inside(path: Path, root: Path) -> bool:
    resolved, resolved_root = path.resolve(), root.resolve()
    return resolved == resolved_root or resolved_root in resolved.parents


def _json_child_id(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload.get("child_id") if isinstance(payload, dict) else None


def _dedupe(targets: Iterable[PurgeTarget]) -> list[PurgeTarget]:
    ordered: list[PurgeTarget] = []
    seen: set[Path] = set()
    for target in sorted(targets, key=lambda item: (len(item.path.parts), str(item.path))):
        resolved = target.path.resolve()
        if resolved in seen or any(parent in seen for parent in resolved.parents):
            continue
        seen.add(resolved)
        ordered.append(PurgeTarget(resolved, target.category))
    return ordered


def build_purge_plan(child_id: str) -> list[PurgeTarget]:
    dossier = load_dossier(child_id)
    if dossier.canonical_status != "withdrawn_locked":
        raise RuntimeError("먼저 동의를 철회하고 기록철을 잠가야 실제 삭제 대상을 확인할 수 있습니다.")
    export_root = Path(ondamm_paths.ONDAMM_EXPORTS).resolve()
    targets: list[PurgeTarget] = []
    exact = [
        (dossier_path(child_id), "dossier"),
        (export_root / "pattern-memory" / child_id, "pattern_memory"),
        (export_root / "event-reviews" / child_id, "event_reviews"),
        (export_root / ".web-cache" / child_id, "browser_cache"),
        (export_root / f"sensing-{child_id}.json", "sensing_export"),
        (export_root / f"sensing-{child_id}.md", "sensing_export"),
    ]
    targets.extend(PurgeTarget(path, category) for path, category in exact if path.exists())

    learning_root = export_root / "learning"
    if learning_root.is_dir():
        for manifest in learning_root.glob("*/manifest.json"):
            if _json_child_id(manifest) == child_id:
                targets.append(PurgeTarget(manifest.parent, "learning_run"))

    for manifest in export_root.glob(f"handoff-{child_id}-*.manifest.json"):
        targets.append(PurgeTarget(manifest, "handoff_export"))
        markdown = manifest.with_name(manifest.name.removesuffix(".manifest.json") + ".md")
        if markdown.exists():
            targets.append(PurgeTarget(markdown, "handoff_export"))

    for owner in export_root.rglob("*.ondamm-owner.json"):
        if _json_child_id(owner) != child_id:
            continue
        targets.append(PurgeTarget(owner, "owned_analysis"))
        try:
            payload = json.loads(owner.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for raw in payload.get("artifacts", []):
            candidate = Path(raw).expanduser().resolve()
            if candidate.exists() and _inside(candidate, export_root):
                targets.append(PurgeTarget(candidate, "owned_analysis"))

    allowed_roots = [Path(ondamm_paths.ONDAMM_DOSSIERS).resolve(), export_root]
    return _dedupe(
        target for target in targets if any(_inside(target.path, root) for root in allowed_roots)
    )


def preview_purge(child_id: str) -> dict[str, object]:
    targets = build_purge_plan(child_id)
    return {
        "title": "철회 후 실제 삭제 미리보기",
        "warning": "실행하면 아래 로컬 자료는 복구할 수 없습니다. 다른 아동의 자료는 포함하지 않습니다.",
        "confirmation_phrase": f"삭제 {child_id}",
        "target_count": len(targets),
        "targets": [
            {"category": item.category, "category_label": CATEGORY_LABELS[item.category], "path": str(item.path)}
            for item in targets
        ],
    }


def execute_purge(child_id: str, *, confirmation: str, actor_id: str) -> dict[str, object]:
    expected = f"삭제 {child_id}"
    if confirmation.strip() != expected:
        raise ValueError(f"확인 문구가 다릅니다. ‘{expected}’를 정확히 입력해 주세요.")
    cleaned_actor = actor_id.strip()
    if not cleaned_actor:
        raise ValueError("삭제를 실행하는 담당자를 입력해 주세요.")
    targets = build_purge_plan(child_id)
    counts: dict[str, int] = {}
    for target in targets:
        if target.path.is_dir():
            shutil.rmtree(target.path)
        elif target.path.exists():
            target.path.unlink()
        counts[target.category] = counts.get(target.category, 0) + 1

    receipt_root = Path(ondamm_paths.ONDAMM_EXPORTS) / "purge-receipts"
    receipt_root.mkdir(parents=True, exist_ok=True)
    subject_digest = hashlib.sha256(child_id.encode("utf-8")).hexdigest()
    completed_at = datetime.now(timezone.utc).isoformat()
    receipt = {
        "schema_version": 1,
        "처리_결과": "철회 후 로컬 자료 삭제 완료",
        "대상_식별자_해시": subject_digest,
        "완료_시각": completed_at,
        "처리자": cleaned_actor,
        "삭제_항목_수": len(targets),
        "유형별_삭제_수": {CATEGORY_LABELS[key]: value for key, value in counts.items()},
        "안내": "이 확인서에는 아동의 이름, 로컬 ID, 영상 경로를 남기지 않습니다.",
    }
    receipt_path = receipt_root / f"삭제확인-{completed_at[:10]}-{subject_digest[:12]}.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"message": "철회된 아동의 로컬 자료를 삭제했습니다.", "receipt_path": str(receipt_path), "receipt": receipt}
