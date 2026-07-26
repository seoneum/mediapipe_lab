from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from uuid import uuid4

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import holistic_landmarker

from holistic_camera import (
    HAND_CONNECTIONS,
    POSE_CONNECTIONS,
    draw_connections,
    draw_iris,
    draw_points,
    estimate_expression,
    estimate_gaze,
    put_lines,
)
from ondamm_paths import ONDAMM_EXPORTS
from ondamm_sensing import ObservationTally, build_sensing_draft
from paths import HOLISTIC_MODEL, base_options

POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_LEFT_HIP = 23
POSE_RIGHT_HIP = 24


def posture_proxy_from_landmarks(pose_landmarks) -> str:
    anchors = []
    for index in [POSE_LEFT_SHOULDER, POSE_RIGHT_SHOULDER, POSE_LEFT_HIP, POSE_RIGHT_HIP]:
        if index < len(pose_landmarks):
            anchors.append(pose_landmarks[index].x)
    if not anchors:
        return "unavailable"
    center_x = sum(anchors) / len(anchors)
    if center_x < 0.35:
        return "left_shifted"
    if center_x > 0.65:
        return "right_shifted"
    return "centered"


def build_preview_lines(
    *,
    face_present: bool,
    pose_present: bool,
    gaze_zone: str,
    posture_proxy: str,
    expression_label: str | None,
    blendshape_pairs: list[tuple[str, float]] | None,
) -> list[str]:
    lines = [
        f"face={face_present}",
        f"pose={pose_present}",
        f"gaze={gaze_zone}",
        f"posture={posture_proxy}",
    ]
    if expression_label:
        shown = ", ".join(f"{name}:{score:.2f}" for name, score in (blendshape_pairs or [])[:3])
        lines.append(f"expression={expression_label}  {shown}".rstrip())
    return lines


def save_outputs(output: Path, markdown_output: Path, draft: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(draft, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# Sensing Draft — {draft['child_id']}",
        "",
        f"- local_session_id: `{draft['local_session_id']}`",
        f"- duration_seconds: {draft['duration_seconds']}",
        f"- frame_count: {draft['frame_count']}",
        f"- face_present_ratio: {draft['face_present_ratio']}",
        f"- pose_present_ratio: {draft['pose_present_ratio']}",
        f"- optional_audio_presence_note: {draft['optional_audio_presence_note'] or '없음'}",
        "",
        "## Reviewed note draft",
    ]
    for line in draft["reviewed_note_draft"]:
        lines.append(f"- {line}")
    lines.extend([
        "",
        "## Gaze zone counts",
        json.dumps(draft["gaze_zone_counts"], ensure_ascii=False, indent=2),
        "",
        "## Posture proxy counts",
        json.dumps(draft["posture_proxy_counts"], ensure_ascii=False, indent=2),
        "",
        "## Storage policy",
        json.dumps(draft["storage_policy"], ensure_ascii=False, indent=2),
        "",
        f"- notice: {draft['non_authoritative_notice']}",
    ])
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_demo_mode(child_id: str, duration_seconds: float, audio_note: str | None) -> dict:
    tally = ObservationTally()
    for _ in range(24):
        tally.add_frame(face_present=True, pose_present=True, gaze_zone="center", posture_proxy="centered")
    for _ in range(6):
        tally.add_frame(face_present=True, pose_present=False, gaze_zone="left", posture_proxy="unavailable")
    draft = build_sensing_draft(
        child_id=child_id,
        local_session_id=f"sensing-{uuid4().hex[:8]}",
        duration_seconds=duration_seconds,
        tally=tally,
        optional_audio_presence_note=audio_note,
    )
    return draft.to_dict()


def run_camera_mode(args: argparse.Namespace) -> dict:
    options = holistic_landmarker.HolisticLandmarkerOptions(
        base_options=base_options(HOLISTIC_MODEL),
        running_mode=vision.RunningMode.VIDEO,
        output_face_blendshapes=True,
        min_face_detection_confidence=0.5,
        min_face_landmarks_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_pose_landmarks_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )
    tally = ObservationTally()
    cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    started = time.time()
    with holistic_landmarker.HolisticLandmarker.create_from_options(options) as detector:
        while time.time() - started < args.duration_seconds:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.time() - started) * 1000)
            result = detector.detect_for_video(mp_image, timestamp_ms)
            face_landmarks = result.face_landmarks or None
            pose_landmarks = result.pose_landmarks or None
            left_hand_landmarks = result.left_hand_landmarks or None
            right_hand_landmarks = result.right_hand_landmarks or None
            gaze = estimate_gaze(face_landmarks) if face_landmarks is not None else None
            gaze_zone = gaze["direction"] if gaze else "unknown"
            posture_proxy = posture_proxy_from_landmarks(pose_landmarks) if pose_landmarks is not None else "unavailable"
            expression_label = None
            blendshape_pairs = None
            if result.face_blendshapes:
                expression_label, blendshape_pairs = estimate_expression(result.face_blendshapes, 3)
            tally.add_frame(
                face_present=face_landmarks is not None,
                pose_present=pose_landmarks is not None,
                gaze_zone=gaze_zone,
                posture_proxy=posture_proxy,
                expression_label=expression_label,
            )
            if not args.headless:
                preview = frame.copy()
                if pose_landmarks is not None:
                    draw_connections(preview, pose_landmarks, POSE_CONNECTIONS, (0, 180, 255), 2)
                    draw_points(preview, pose_landmarks, (0, 220, 255), 3)
                if left_hand_landmarks is not None:
                    draw_connections(preview, left_hand_landmarks, HAND_CONNECTIONS, (0, 180, 0), 2)
                    draw_points(preview, left_hand_landmarks, (0, 255, 0), 3)
                if right_hand_landmarks is not None:
                    draw_connections(preview, right_hand_landmarks, HAND_CONNECTIONS, (180, 0, 180), 2)
                    draw_points(preview, right_hand_landmarks, (255, 0, 255), 3)
                if face_landmarks is not None:
                    draw_points(preview, face_landmarks, (255, 180, 0), 1)
                if gaze:
                    draw_iris(preview, gaze, 4)
                put_lines(
                    preview,
                    build_preview_lines(
                        face_present=face_landmarks is not None,
                        pose_present=pose_landmarks is not None,
                        gaze_zone=gaze_zone,
                        posture_proxy=posture_proxy,
                        expression_label=expression_label,
                        blendshape_pairs=blendshape_pairs,
                    ),
                )
                cv2.imshow("ON DAMM sensing draft", preview)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    cap.release()
    cv2.destroyAllWindows()
    draft = build_sensing_draft(
        child_id=args.child_id,
        local_session_id=f"sensing-{uuid4().hex[:8]}",
        duration_seconds=time.time() - started,
        tally=tally,
        optional_audio_presence_note=args.audio_presence_note,
    )
    return draft.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child-id", required=True)
    parser.add_argument("--duration-seconds", type=float, default=8.0)
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--audio-presence-note")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--markdown-output")
    args = parser.parse_args()

    ONDAMM_EXPORTS.mkdir(parents=True, exist_ok=True)
    output = Path(args.output) if args.output else ONDAMM_EXPORTS / f"sensing-{args.child_id}.json"
    markdown_output = (
        Path(args.markdown_output)
        if args.markdown_output
        else ONDAMM_EXPORTS / f"sensing-{args.child_id}.md"
    )

    draft = run_demo_mode(args.child_id, args.duration_seconds, args.audio_presence_note) if args.demo else run_camera_mode(args)
    save_outputs(output, markdown_output, draft)
    print(f"saved_json: {output}")
    print(f"saved_markdown: {markdown_output}")


if __name__ == "__main__":
    main()
