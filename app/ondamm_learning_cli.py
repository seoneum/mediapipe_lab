from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from ondamm_event_recording import EventMetadata, EventObservation, EventRecordingPolicy, LocalEventClipRecorder, SustainedEventDetector
from ondamm_learning import build_learning_program_plan, build_learning_run_summary, render_learning_program_markdown, render_learning_run_summary_markdown
from ondamm_models import Dossier, unique_preserving_order, utc_now
from ondamm_paths import ONDAMM_LEARNING_EXPORTS
from ondamm_store import load_dossier


RAW_MEDIA_NOTICE = "raw media 저장은 명시적으로 --record-events 를 선택한 경우에만 로컬 출력 디렉터리에 남습니다."
AUTO_WRITEBACK_NOTICE = "센싱/이벤트 결과는 dossier에 자동 반영되지 않으며, 사람이 검토해 수동으로만 요약 전환할 수 있습니다."


@dataclass
class ObservationAccumulator:
    frame_count: int = 0
    face_present_frames: int = 0
    gaze_zone_counts: Counter[str] = field(default_factory=Counter)
    posture_proxy_counts: Counter[str] = field(default_factory=Counter)

    def add(self, *, face_present: bool, gaze_zone: str, posture_proxy: str) -> None:
        self.frame_count += 1
        if face_present:
            self.face_present_frames += 1
        self.gaze_zone_counts[gaze_zone] += 1
        self.posture_proxy_counts[posture_proxy] += 1

    @property
    def face_present_ratio(self) -> float:
        if self.frame_count <= 0:
            return 0.0
        return round(self.face_present_frames / self.frame_count, 4)


@dataclass
class RunCapture:
    mode: str
    started_at: str
    finished_at: str
    duration_seconds: float
    observations: ObservationAccumulator
    detected_events: list[EventMetadata]
    recorded_events: list[EventMetadata]
    clip_directory: str | None


def slugify(value: str) -> str:
    lowered = value.strip().lower()
    return re.sub(r"[^a-z0-9_-]+", "-", lowered).strip("-") or "run"


def dominant_key(counts: Counter[str], fallback: str) -> str:
    if not counts:
        return fallback
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def deterministic_demo_started_at() -> str:
    return "2026-01-01T00:00:00+00:00"


def deterministic_demo_finished_at(duration_seconds: float) -> str:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return (base + timedelta(seconds=duration_seconds)).isoformat()


def normalize_demo_event(event: EventMetadata, *, index: int) -> EventMetadata:
    clip_path = event.clip_path
    if clip_path:
        source = Path(clip_path)
        target = source.with_name(f"demo-{index:02d}-{event.event_type}{source.suffix}")
        if source != target and source.exists():
            source.replace(target)
        clip_path = str(target)
    return EventMetadata(
        event_id=f"demo-{index:02d}-{event.event_type}",
        event_type=event.event_type,
        start_timestamp=event.start_timestamp,
        end_timestamp=event.end_timestamp,
        trigger_values=event.trigger_values,
        clip_path=clip_path,
        created_at=deterministic_demo_started_at(),
    )




def json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def save_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def serialize_learning_plan(plan) -> dict:
    payload = asdict(plan)
    payload["total_duration_seconds"] = plan.total_duration_seconds
    payload["raw_media_notice"] = RAW_MEDIA_NOTICE
    payload["auto_writeback_notice"] = AUTO_WRITEBACK_NOTICE
    return payload


def serialize_run_summary(summary, capture: RunCapture, output_dir: Path) -> dict:
    payload = asdict(summary)
    payload["mode"] = capture.mode
    payload["duration_seconds"] = round(capture.duration_seconds, 3)
    payload["frame_count"] = capture.observations.frame_count
    payload["face_present_ratio"] = capture.observations.face_present_ratio
    payload["gaze_zone_counts"] = dict(sorted(capture.observations.gaze_zone_counts.items()))
    payload["posture_proxy_counts"] = dict(sorted(capture.observations.posture_proxy_counts.items()))
    payload["event_count"] = len(capture.detected_events)
    payload["recorded_event_count"] = len(capture.recorded_events)
    payload["event_clip_directory"] = capture.clip_directory
    payload["raw_media_notice"] = RAW_MEDIA_NOTICE
    payload["auto_writeback_notice"] = AUTO_WRITEBACK_NOTICE
    payload["output_dir"] = str(output_dir)
    return payload


def render_clip_to_mp4(npz_path: Path, *, fps: float) -> Path:
    import cv2

    with np.load(npz_path, allow_pickle=False) as archive:
        frames = archive["frames"]
    if len(frames) == 0:
        raise RuntimeError(f"No frames available for clip: {npz_path}")

    first_frame = frames[0]
    height, width = first_frame.shape[:2]
    output_path = npz_path.with_suffix(".mp4")
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    try:
        for frame in frames:
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            writer.write(frame)
    finally:
        writer.release()

    npz_path.unlink()
    return output_path


def maybe_promote_clip(event: EventMetadata, *, fps: float) -> EventMetadata:
    if not event.clip_path:
        return event
    clip_path = Path(event.clip_path)
    if clip_path.suffix != ".npz":
        return event
    mp4_path = render_clip_to_mp4(clip_path, fps=fps)
    return event.with_clip_path(str(mp4_path))


def make_demo_frame(*, width: int, height: int, index: int, label: str) -> np.ndarray:
    import cv2

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    color = (40 + (index * 17) % 180, 70 + (index * 11) % 120, 120 + (index * 7) % 100)
    frame[:] = color
    cv2.putText(frame, "ON DAMM demo", (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, label, (24, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"frame={index}", (24, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame


def demo_observation_for_timestamp(timestamp: float) -> tuple[bool, str, str, str]:
    if timestamp < 1.6:
        return True, "center", "centered", "stable_center"
    if timestamp < 4.2:
        return True, "left", "centered", "gaze_diverted"
    if timestamp < 6.8:
        return False, "unknown", "unavailable", "face_missing"
    if timestamp < 10.2:
        return True, "center", "right_shifted", "posture_shifted"
    return True, "center", "centered", "recovered_center"


def record_emitted_events(
    *,
    detector_events: list[EventMetadata],
    recorder: LocalEventClipRecorder,
    clip_fps: float,
) -> list[EventMetadata]:
    if not recorder.recording_enabled:
        return []
    recorded: list[EventMetadata] = []
    for event in detector_events:
        recorded_event = recorder.record_event(event)
        recorded.append(maybe_promote_clip(recorded_event, fps=clip_fps))
    return recorded


def run_demo_mode(args: argparse.Namespace, *, recorder: LocalEventClipRecorder, detector: SustainedEventDetector) -> RunCapture:
    started_at = deterministic_demo_started_at()
    observations = ObservationAccumulator()
    detected_events: list[EventMetadata] = []
    recorded_events: list[EventMetadata] = []
    frame_total = max(1, int(round(args.duration_seconds * args.clip_fps)))

    for index in range(frame_total):
        timestamp = round(index / args.clip_fps, 3)
        face_present, gaze_zone, posture_proxy, label = demo_observation_for_timestamp(timestamp)
        frame = make_demo_frame(width=args.width, height=args.height, index=index, label=label)
        recorder.add_frame(frame=frame, timestamp=timestamp)
        observations.add(face_present=face_present, gaze_zone=gaze_zone, posture_proxy=posture_proxy)
        detector_events = detector.add_observation(
            EventObservation(
                timestamp=timestamp,
                face_present=face_present,
                gaze_zone=gaze_zone,
                posture_proxy=posture_proxy,
            )
        )
        detected_events.extend(detector_events)
        recorded_events.extend(record_emitted_events(detector_events=detector_events, recorder=recorder, clip_fps=args.clip_fps))

    normalized_detected_events = [normalize_demo_event(event, index=index) for index, event in enumerate(detected_events, start=1)]
    normalized_recorded_events = [normalize_demo_event(event, index=index) for index, event in enumerate(recorded_events, start=1)]
    return RunCapture(
        mode="demo",
        started_at=started_at,
        finished_at=deterministic_demo_finished_at(args.duration_seconds),
        duration_seconds=round(args.duration_seconds, 3),
        observations=observations,
        detected_events=normalized_detected_events,
        recorded_events=normalized_recorded_events,
        clip_directory=str(recorder.output_dir) if args.record_events and recorder.output_dir else None,
    )


def run_camera_mode(args: argparse.Namespace, *, recorder: LocalEventClipRecorder, detector: SustainedEventDetector) -> RunCapture:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.vision import holistic_landmarker

    from holistic_camera import estimate_gaze
    from ondamm_sensing_cli import posture_proxy_from_landmarks
    from paths import HOLISTIC_MODEL, base_options

    started_at = utc_now()
    wall_started = time.time()
    observations = ObservationAccumulator()
    detected_events: list[EventMetadata] = []
    recorded_events: list[EventMetadata] = []
    options = holistic_landmarker.HolisticLandmarkerOptions(
        base_options=base_options(HOLISTIC_MODEL),
        running_mode=vision.RunningMode.VIDEO,
        output_face_blendshapes=False,
        min_face_detection_confidence=0.5,
        min_face_landmarks_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_pose_landmarks_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )
    cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    with holistic_landmarker.HolisticLandmarker.create_from_options(options) as landmarker:
        while time.time() - wall_started < args.duration_seconds:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            elapsed = round(time.time() - wall_started, 3)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                int(elapsed * 1000),
            )
            face_landmarks = result.face_landmarks or None
            pose_landmarks = result.pose_landmarks or None
            gaze_zone = "unknown"
            if face_landmarks is not None:
                gaze = estimate_gaze(face_landmarks)
                gaze_zone = gaze["direction"] if gaze else "unknown"
            posture_proxy = posture_proxy_from_landmarks(pose_landmarks) if pose_landmarks is not None else "unavailable"
            face_present = face_landmarks is not None
            observations.add(face_present=face_present, gaze_zone=gaze_zone, posture_proxy=posture_proxy)
            recorder.add_frame(frame=frame, timestamp=elapsed)
            detector_events = detector.add_observation(
                EventObservation(
                    timestamp=elapsed,
                    face_present=face_present,
                    gaze_zone=gaze_zone,
                    posture_proxy=posture_proxy,
                )
            )
            detected_events.extend(detector_events)
            recorded_events.extend(record_emitted_events(detector_events=detector_events, recorder=recorder, clip_fps=args.clip_fps))

            if not args.headless:
                preview = frame.copy()
                cv2.putText(preview, f"face={face_present}", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
                cv2.putText(preview, f"gaze={gaze_zone}", (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
                cv2.putText(preview, f"posture={posture_proxy}", (16, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
                cv2.imshow("ON DAMM learning", preview)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    cap.release()
    if not args.headless:
        cv2.destroyAllWindows()
    return RunCapture(
        mode="camera",
        started_at=started_at,
        finished_at=utc_now(),
        duration_seconds=round(time.time() - wall_started, 3),
        observations=observations,
        detected_events=detected_events,
        recorded_events=recorded_events,
        clip_directory=str(recorder.output_dir) if args.record_events and recorder.output_dir else None,
    )


def build_educator_notes(capture: RunCapture) -> list[str]:
    dominant_gaze = dominant_key(capture.observations.gaze_zone_counts, "unknown")
    dominant_posture = dominant_key(capture.observations.posture_proxy_counts, "unknown")
    notes = [
        f"실행 모드: {capture.mode}",
        f"얼굴 존재 비율은 약 {capture.observations.face_present_ratio:.0%} 였습니다.",
        f"가장 자주 관찰된 시선 구역은 `{dominant_gaze}` 이었습니다.",
        f"가장 자주 관찰된 자세 proxy는 `{dominant_posture}` 이었습니다.",
        RAW_MEDIA_NOTICE,
        AUTO_WRITEBACK_NOTICE,
    ]
    if capture.detected_events:
        event_types = ", ".join(event.event_type for event in capture.detected_events)
        notes.append(f"지속 이벤트 {len(capture.detected_events)}건을 관찰했습니다: {event_types}")
        if capture.recorded_events:
            notes.append(f"이 중 {len(capture.recorded_events)}건을 로컬 메타데이터/영상 클립으로 남겼습니다.")
        else:
            notes.append("지속 이벤트는 관찰됐지만 --record-events 없이 실행되어 로컬 메타데이터/영상 클립은 저장하지 않았습니다.")
    else:
        notes.append("지속 이벤트는 관찰되지 않았습니다.")
    notes.append("완료 step 표시는 자동 추정하지 않았으며, 실제 완료 판정은 교사/보호자 검토 후 수동으로 남겨야 합니다.")
    return unique_preserving_order(notes)


def build_reinforcement_observations(plan, capture: RunCapture) -> list[str]:
    observations = [
        "강화는 짧고 예측 가능하게 제공하고 성공 직후 바로 연결합니다.",
        plan.steps[1].reinforcement_hint if len(plan.steps) > 1 else "짧은 칭찬과 선호 자극 접근을 유지합니다.",
    ]
    if capture.observations.face_present_ratio < 0.7:
        observations.append("얼굴 이탈 구간이 있었으므로 요구량을 낮추고 진입 강화부터 다시 제시합니다.")
    return unique_preserving_order(observations)


def build_transition_observations(plan, capture: RunCapture) -> list[str]:
    observations = [
        plan.steps[-2].transition_hint if len(plan.steps) > 1 else "전환 전 짧은 예고와 회복 시간을 제공합니다.",
        "전환 메모는 점수화하지 않고 다음 세션의 지원 조정 참고로만 사용합니다.",
    ]
    event_types = {event.event_type for event in capture.detected_events}
    if "gaze_diverted" in event_types or "posture_shifted" in event_types:
        observations.append("전환 전 first-then 예고와 짧은 휴식 신호를 더 일찍 제공합니다.")
    if "face_missing" in event_types:
        observations.append("이탈이 길어질 때는 과제를 즉시 축소하고 재진입 단서를 먼저 제시합니다.")
    return unique_preserving_order(observations)


def resolve_output_dir(args: argparse.Namespace, dossier: Dossier) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    run_id = args.run_id or f"{slugify(dossier.child_id)}-{time.strftime('%Y%m%d-%H%M%S')}"
    return (ONDAMM_LEARNING_EXPORTS / run_id).resolve()


def resolve_clips_dir(output_dir: Path) -> Path:
    return (output_dir / "event-clips").resolve()


def prepare_output_dirs(output_dir: Path, clips_dir: Path, *, record_events: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stale_event_metadata = output_dir / "event_recording.json"
    if stale_event_metadata.exists():
        stale_event_metadata.unlink()
    if clips_dir.exists():
        for pattern in ("*.mp4", "*.npz"):
            for path in clips_dir.glob(pattern):
                if path.is_file():
                    path.unlink()
    elif record_events:
        clips_dir.mkdir(parents=True, exist_ok=True)


def build_manifest(*, dossier: Dossier, capture: RunCapture, output_dir: Path, plan_paths: dict[str, Path], run_paths: dict[str, Path], event_metadata_path: Path | None) -> dict:
    return {
        "child_id": dossier.child_id,
        "child_name": dossier.display_name,
        "mode": capture.mode,
        "output_dir": str(output_dir),
        "record_events": event_metadata_path is not None,
        "dossier_auto_updated": False,
        "detected_event_count": len(capture.detected_events),
        "recorded_event_count": len(capture.recorded_events),
        "support_boundary_notice": "학습 프로그램/실행 요약은 지원용 기록이며 진단 또는 자동 canonical 기록이 아닙니다.",
        "raw_media_notice": RAW_MEDIA_NOTICE,
        "plan_outputs": {name: str(path) for name, path in plan_paths.items()},
        "run_summary_outputs": {name: str(path) for name, path in run_paths.items()},
        "event_metadata_output": str(event_metadata_path) if event_metadata_path else None,
        "event_clip_directory": capture.clip_directory,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child-id", required=True)
    parser.add_argument("--goal", default="시각 단서와 짧은 강화로 matching 활동 1회 참여 지원")
    parser.add_argument("--caregiver-note")
    parser.add_argument("--duration-seconds", type=float, default=12.0)
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--clip-fps", type=float, default=10.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--record-events", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--run-id")
    args = parser.parse_args()

    if args.clip_fps <= 0:
        raise ValueError("--clip-fps must be positive")
    if args.duration_seconds <= 0:
        raise ValueError("--duration-seconds must be positive")

    dossier = load_dossier(args.child_id)
    plan = build_learning_program_plan(
        dossier,
        goal=args.goal,
        caregiver_input=args.caregiver_note,
        created_at=deterministic_demo_started_at() if args.demo else None,
    )
    output_dir = resolve_output_dir(args, dossier)
    clips_dir = resolve_clips_dir(output_dir)
    prepare_output_dirs(output_dir, clips_dir, record_events=args.record_events)
    policy = EventRecordingPolicy()
    recorder = LocalEventClipRecorder(policy=policy, output_dir=clips_dir, recording_enabled=args.record_events)
    detector = SustainedEventDetector(policy=policy)
    capture = run_demo_mode(args, recorder=recorder, detector=detector) if args.demo else run_camera_mode(args, recorder=recorder, detector=detector)

    run_summary = build_learning_run_summary(
        plan,
        started_at=capture.started_at,
        finished_at=capture.finished_at,
        completed_step_titles=[],
        educator_notes=build_educator_notes(capture),
        reinforcement_observations=build_reinforcement_observations(plan, capture),
        transition_observations=build_transition_observations(plan, capture),
        caregiver_note=args.caregiver_note,
    )

    plan_paths = {
        "json": output_dir / "learning_plan.json",
        "markdown": output_dir / "learning_plan.md",
    }
    run_paths = {
        "json": output_dir / "learning_run_summary.json",
        "markdown": output_dir / "learning_run_summary.md",
    }
    event_metadata_path = output_dir / "event_recording.json" if args.record_events else None
    manifest_path = output_dir / "manifest.json"

    ONDAMM_LEARNING_EXPORTS.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(plan_paths["json"], serialize_learning_plan(plan))
    save_text(plan_paths["markdown"], render_learning_program_markdown(plan))
    save_json(run_paths["json"], serialize_run_summary(run_summary, capture, output_dir))
    save_text(run_paths["markdown"], render_learning_run_summary_markdown(run_summary))
    if event_metadata_path is not None:
        save_json(
            event_metadata_path,
            {
                "child_id": dossier.child_id,
                "mode": capture.mode,
                "recording_enabled": True,
                "policy": asdict(policy),
                "event_clip_directory": capture.clip_directory,
                "detected_event_count": len(capture.detected_events),
                "recorded_event_count": len(capture.recorded_events),
                "non_authoritative_notice": "지속 이벤트 메타데이터와 로컬 클립은 검토용 보조 기록이며 dossier에 자동 반영되지 않습니다.",
                "events": [event.to_dict() for event in capture.recorded_events],
            },
        )
    save_json(
        manifest_path,
        build_manifest(
            dossier=dossier,
            capture=capture,
            output_dir=output_dir,
            plan_paths=plan_paths,
            run_paths=run_paths,
            event_metadata_path=event_metadata_path,
        ),
    )

    print(f"saved_plan_json: {plan_paths['json']}")
    print(f"saved_plan_markdown: {plan_paths['markdown']}")
    print(f"saved_run_json: {run_paths['json']}")
    print(f"saved_run_markdown: {run_paths['markdown']}")
    if event_metadata_path is not None:
        print(f"saved_event_metadata: {event_metadata_path}")
        print(f"saved_event_clips: {capture.clip_directory}")
    print(f"saved_manifest: {manifest_path}")


if __name__ == "__main__":
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(__file__).resolve().parents[1] / "outputs" / ".matplotlib"),
    )
    main()
