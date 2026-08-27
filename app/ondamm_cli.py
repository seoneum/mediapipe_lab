from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from ondamm_models import Dossier, FacialMovementProfile, SessionSummary, unique_preserving_order
from ondamm_recommendations import build_baseline_recommendation
from ondamm_security import build_export_manifest, build_reestablishment_template
from ondamm_store import create_dossier, export_path, list_dossiers, load_dossier, save_dossier


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_repeatable(values: list[str] | None) -> list[str]:
    return unique_preserving_order(values or [])


def ensure_active(dossier: Dossier, action: str) -> None:
    if dossier.canonical_status != "active":
        raise RuntimeError(f"Cannot {action}: dossier status is {dossier.canonical_status}")

def resolve_export_target(path_text: str | None, default_name: str) -> Path:
    if not path_text:
        return export_path(default_name)
    candidate = Path(path_text)
    if candidate.is_absolute():
        return candidate
    return export_path(path_text) if candidate.parent == Path(".") else candidate


def render_recommendation_markdown(child_name: str, recommendation) -> str:
    lines = [
        f"# 추천 초안 — {child_name}",
        "",
        f"- recommendation_id: `{recommendation.recommendation_id}`",
        f"- status: **{recommendation.status}**",
        f"- goal: {recommendation.goal}",
        f"- drafted_by: {recommendation.drafted_by}",
        f"- approved_by: {recommendation.approved_by or '미승인'}",
        f"- created_at: {recommendation.created_at}",
        "",
        "## Summary",
        recommendation.summary,
        "",
        "## Suggested activities",
    ]
    for item in recommendation.suggested_activities:
        lines.append(f"- {item}")
    lines.extend(["", "## Rationale"])
    for item in recommendation.rationale_lines:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def render_handoff_markdown(dossier: Dossier) -> str:
    lines = [
        f"# Handoff Brief — {dossier.display_name}",
        "",
        "## Important notice",
        "- 이 문서는 human-readable handoff artifact다.",
        "- recipient-side import/promotion을 위한 데이터가 아니다.",
        "- 새 환경에서는 이 문서를 참고해 수동으로 continuity dossier를 다시 작성해야 한다.",
        "",
        "## Dossier summary",
        f"- child_id: `{dossier.child_id}`",
        f"- local_canonical_id: `{dossier.local_canonical_id}`",
        f"- canonical_status: `{dossier.canonical_status}`",
        f"- age_band: {dossier.age_band}",
        f"- communication_modality: {dossier.communication_modality}",
        f"- confirmed_preferences: {', '.join(dossier.confirmed_preferences) or '없음'}",
        f"- confirmed_avoidances: {', '.join(dossier.confirmed_avoidances) or '없음'}",
        f"- effective_strategies: {', '.join(dossier.effective_strategies) or '없음'}",
        f"- triggers_and_calming_supports: {', '.join(dossier.triggers_and_calming_supports) or '없음'}",
        "",
        "## Handoff notes",
    ]
    if dossier.handoff_notes:
        lines.extend([f"- {note}" for note in dossier.handoff_notes])
    else:
        lines.append("- 없음")

    lines.extend(["", "## Approved recent session summaries"])
    if dossier.approved_session_summaries:
        for summary in dossier.approved_session_summaries[-5:]:
            lines.extend(
                [
                    f"### {summary.title}",
                    f"- activity: {summary.activity_name}",
                    f"- observed_response: {summary.observed_response}",
                    f"- educator_interpretation: {summary.educator_interpretation}",
                    f"- approved_by: {summary.approved_by}",
                    f"- tags: {', '.join(summary.tags) or '없음'}",
                    "",
                ]
            )
    else:
        lines.append("- 없음")

    lines.extend(["", "## Approved plan history"])
    if dossier.approved_plan_history:
        for plan in dossier.approved_plan_history[-5:]:
            lines.extend(
                [
                    f"### {plan.goal}",
                    f"- summary: {plan.summary}",
                    f"- approved_by: {plan.approved_by or '미승인'}",
                    f"- suggested_activities: {' | '.join(plan.suggested_activities)}",
                    "",
                ]
            )
    else:
        lines.append("- 없음")

    lines.extend(["", "## Approved facial movement profiles"])
    if dossier.approved_facial_movement_profiles:
        for profile in dossier.approved_facial_movement_profiles:
            lines.extend(
                [
                    f"### {profile.display_name}",
                    f"- label: `{profile.label}`",
                    f"- blendshapes: {', '.join(profile.blendshape_names)}",
                    f"- aggregation/threshold: {profile.aggregation} / {profile.activation_threshold}",
                    f"- approved_by: {profile.approved_by}",
                    f"- source_session_ids: {', '.join(profile.source_session_ids)}",
                    "- interpretation_boundary: 관찰 가능한 움직임 proxy이며 감정·집중도·진단으로 해석하지 않음",
                    "",
                ]
            )
    else:
        lines.append("- 없음")

    return "\n".join(lines) + "\n"


def command_create_dossier(args: argparse.Namespace) -> None:
    dossier = Dossier.create(
        child_id=args.child_id,
        display_name=args.name,
        age_band=args.age_band,
        communication_modality=args.communication_modality,
        confirmed_preferences=parse_repeatable(args.preference),
        confirmed_avoidances=parse_repeatable(args.avoidance),
        effective_strategies=parse_repeatable(args.strategy),
        triggers_and_calming_supports=parse_repeatable(args.support),
        handoff_notes=parse_repeatable(args.handoff_note),
    )
    path = create_dossier(dossier)
    print(f"created: {path}")


def command_list_dossiers(_: argparse.Namespace) -> None:
    dossiers = list_dossiers()
    if not dossiers:
        print("No dossiers yet.")
        return
    for dossier in dossiers:
        print(
            f"{dossier.child_id}\t{dossier.display_name}\t{dossier.age_band}\t"
            f"status={dossier.canonical_status}\tplans={len(dossier.approved_plan_history)}\tsessions={len(dossier.approved_session_summaries)}"
        )


def command_show_dossier(args: argparse.Namespace) -> None:
    dossier = load_dossier(args.child_id)
    ensure_active(dossier, "show dossier")
    if args.json:
        print(json.dumps(dossier.to_dict(), ensure_ascii=False, indent=2))
        return
    print(render_handoff_markdown(dossier))


def command_update_dossier(args: argparse.Namespace) -> None:
    dossier = load_dossier(args.child_id)
    ensure_active(dossier, "update dossier")
    if args.name:
        dossier.display_name = args.name.strip()
    if args.age_band:
        dossier.age_band = args.age_band.strip()
    if args.communication_modality:
        dossier.communication_modality = args.communication_modality.strip()
    dossier.confirmed_preferences = unique_preserving_order(dossier.confirmed_preferences + parse_repeatable(args.add_preference))
    dossier.confirmed_avoidances = unique_preserving_order(dossier.confirmed_avoidances + parse_repeatable(args.add_avoidance))
    dossier.effective_strategies = unique_preserving_order(dossier.effective_strategies + parse_repeatable(args.add_strategy))
    dossier.triggers_and_calming_supports = unique_preserving_order(
        dossier.triggers_and_calming_supports + parse_repeatable(args.add_support)
    )
    dossier.handoff_notes = unique_preserving_order(dossier.handoff_notes + parse_repeatable(args.add_handoff_note))
    dossier.touch()
    path = save_dossier(dossier)
    print(f"updated: {path}")


def command_add_session_summary(args: argparse.Namespace) -> None:
    dossier = load_dossier(args.child_id)
    ensure_active(dossier, "add session summary")
    summary = SessionSummary.create(
        title=args.title,
        activity_name=args.activity,
        observed_response=args.response,
        educator_interpretation=args.interpretation,
        approved_by=args.approved_by,
        tags=parse_repeatable(args.tag),
    )
    dossier.add_session_summary(summary)
    path = save_dossier(dossier)
    print(f"session_saved: {path}")
    print(summary.session_id)


def command_approve_facial_movement_profile(args: argparse.Namespace) -> None:
    dossier = load_dossier(args.child_id)
    ensure_active(dossier, "approve facial movement profile")
    source_session_ids = parse_repeatable(args.source_session_id)
    approved_session_ids = {item.session_id for item in dossier.approved_session_summaries}
    if not source_session_ids or not set(source_session_ids).issubset(approved_session_ids):
        raise ValueError("source-session-id must reference approved dossier session summaries")
    profile = FacialMovementProfile.create(
        label=args.label,
        display_name=args.display_name,
        blendshape_names=parse_repeatable(args.blendshape),
        aggregation=args.aggregation,
        activation_threshold=args.threshold,
        approved_by=args.approved_by,
        source_session_ids=source_session_ids,
        priority=args.priority,
    )
    dossier.add_facial_movement_profile(profile)
    path = save_dossier(dossier)
    print(f"facial_profile_saved: {path}")
    print(profile.profile_id)


def command_recommend_baseline(args: argparse.Namespace) -> None:
    dossier = load_dossier(args.child_id)
    ensure_active(dossier, "recommend baseline")
    recommendation = build_baseline_recommendation(
        dossier,
        goal=args.goal,
        caregiver_input=args.caregiver_input,
        drafted_by=args.drafted_by,
        approved_by=args.approved_by,
    )
    markdown = render_recommendation_markdown(dossier.display_name, recommendation)
    if args.output:
        output = resolve_export_target(args.output, f"recommendation-{args.child_id}.md")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown, encoding="utf-8")
        print(f"recommendation_saved: {output}")
    else:
        print(markdown)
    if args.approved_by:
        dossier.add_recommendation(recommendation)
        path = save_dossier(dossier)
        print(f"dossier_updated: {path}")


def command_handoff_brief(args: argparse.Namespace) -> None:
    dossier = load_dossier(args.child_id)
    ensure_active(dossier, "generate handoff brief")
    markdown = render_handoff_markdown(dossier)
    output = resolve_export_target(args.output, f"handoff-{args.child_id}.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    dossier.add_audit_event(
        event_type="handoff_brief_generated",
        actor_id=args.actor_id,
        details={"output": str(output)},
    )
    save_dossier(dossier)
    print(f"saved: {output}")


def command_export_handoff(args: argparse.Namespace) -> None:
    dossier = load_dossier(args.child_id)
    ensure_active(dossier, "export handoff")
    markdown = render_handoff_markdown(dossier)
    issuance_time = utc_now()
    export_output = resolve_export_target(args.output, f"export-{args.child_id}.md")
    manifest_output = (
        Path(args.manifest_output)
        if args.manifest_output and Path(args.manifest_output).is_absolute()
        else resolve_export_target(args.manifest_output, export_output.name + ".manifest.json")
    )
    export_output.parent.mkdir(parents=True, exist_ok=True)
    export_output.write_text(markdown, encoding="utf-8")
    manifest = build_export_manifest(
        child_id=dossier.child_id,
        issuance_time=issuance_time,
        markdown=markdown,
        export_path=str(export_output),
    )
    manifest_output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dossier.add_audit_event(
        event_type="signed_export_generated",
        actor_id=args.actor_id,
        details={"output": str(export_output), "manifest_output": str(manifest_output), "artifact_id": manifest["artifact_id"]},
    )
    save_dossier(dossier)
    print(f"saved_export: {export_output}")
    print(f"saved_manifest: {manifest_output}")


def command_prepare_reestablishment(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = resolve_export_target(args.output, f"reestablish-{manifest['child_id']}.json")
    template = build_reestablishment_template(manifest=manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(template, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved: {output}")


def command_withdraw_dossier(args: argparse.Namespace) -> None:
    dossier = load_dossier(args.child_id)
    dossier.canonical_status = "withdrawn_locked"
    dossier.add_audit_event(
        event_type="authoritative_withdrawal",
        actor_id=args.actor_id,
        details={"reason_code": args.reason_code, "reason": args.reason},
    )
    path = save_dossier(dossier)
    print(f"withdrawn: {path}")


def command_restore_dossier(args: argparse.Namespace) -> None:
    dossier = load_dossier(args.child_id)
    dossier.canonical_status = "active"
    dossier.add_audit_event(
        event_type="restoration_approved",
        actor_id=args.actor_id,
        details={"reason_code": args.reason_code, "reason": args.reason},
    )
    path = save_dossier(dossier)
    print(f"restored: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create-dossier")
    create.add_argument("--child-id", required=True)
    create.add_argument("--name", required=True)
    create.add_argument("--age-band", required=True)
    create.add_argument("--communication-modality", required=True)
    create.add_argument("--preference", action="append")
    create.add_argument("--avoidance", action="append")
    create.add_argument("--strategy", action="append")
    create.add_argument("--support", action="append")
    create.add_argument("--handoff-note", action="append")
    create.set_defaults(func=command_create_dossier)

    listing = subparsers.add_parser("list-dossiers")
    listing.set_defaults(func=command_list_dossiers)

    show = subparsers.add_parser("show-dossier")
    show.add_argument("--child-id", required=True)
    show.add_argument("--json", action="store_true")
    show.set_defaults(func=command_show_dossier)

    update = subparsers.add_parser("update-dossier")
    update.add_argument("--child-id", required=True)
    update.add_argument("--name")
    update.add_argument("--age-band")
    update.add_argument("--communication-modality")
    update.add_argument("--add-preference", action="append")
    update.add_argument("--add-avoidance", action="append")
    update.add_argument("--add-strategy", action="append")
    update.add_argument("--add-support", action="append")
    update.add_argument("--add-handoff-note", action="append")
    update.set_defaults(func=command_update_dossier)

    session = subparsers.add_parser("add-session-summary")
    session.add_argument("--child-id", required=True)
    session.add_argument("--title", required=True)
    session.add_argument("--activity", required=True)
    session.add_argument("--response", required=True)
    session.add_argument("--interpretation", required=True)
    session.add_argument("--approved-by", required=True)
    session.add_argument("--tag", action="append")
    session.set_defaults(func=command_add_session_summary)

    facial_profile = subparsers.add_parser("approve-facial-movement-profile")
    facial_profile.add_argument("--child-id", required=True)
    facial_profile.add_argument("--label", required=True)
    facial_profile.add_argument("--display-name", required=True)
    facial_profile.add_argument("--blendshape", action="append", required=True)
    facial_profile.add_argument("--aggregation", choices=["mean", "max", "min"], required=True)
    facial_profile.add_argument("--threshold", type=float, required=True)
    facial_profile.add_argument("--priority", type=int, default=80)
    facial_profile.add_argument("--approved-by", required=True)
    facial_profile.add_argument("--source-session-id", action="append", required=True)
    facial_profile.set_defaults(func=command_approve_facial_movement_profile)

    recommend = subparsers.add_parser("recommend-baseline")
    recommend.add_argument("--child-id", required=True)
    recommend.add_argument("--goal", required=True)
    recommend.add_argument("--caregiver-input", default="")
    recommend.add_argument("--drafted-by", default="local-operator")
    recommend.add_argument("--approved-by")
    recommend.add_argument("--output")
    recommend.set_defaults(func=command_recommend_baseline)

    handoff = subparsers.add_parser("handoff-brief")
    handoff.add_argument("--child-id", required=True)
    handoff.add_argument("--output")
    handoff.add_argument("--actor-id", default="local-operator")
    handoff.set_defaults(func=command_handoff_brief)

    export_cmd = subparsers.add_parser("export-handoff")
    export_cmd.add_argument("--child-id", required=True)
    export_cmd.add_argument("--output")
    export_cmd.add_argument("--manifest-output")
    export_cmd.add_argument("--actor-id", default="local-operator")
    export_cmd.set_defaults(func=command_export_handoff)

    reestablish = subparsers.add_parser("prepare-reestablishment")
    reestablish.add_argument("--manifest", required=True)
    reestablish.add_argument("--output")
    reestablish.set_defaults(func=command_prepare_reestablishment)

    withdraw = subparsers.add_parser("withdraw-dossier")
    withdraw.add_argument("--child-id", required=True)
    withdraw.add_argument("--actor-id", default="guardian-admin")
    withdraw.add_argument("--reason-code", required=True)
    withdraw.add_argument("--reason", required=True)
    withdraw.set_defaults(func=command_withdraw_dossier)

    restore = subparsers.add_parser("restore-dossier")
    restore.add_argument("--child-id", required=True)
    restore.add_argument("--actor-id", default="guardian-admin")
    restore.add_argument("--reason-code", required=True)
    restore.add_argument("--reason", required=True)
    restore.set_defaults(func=command_restore_dossier)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
