from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from ondamm_event_recording import EventMetadata, EventObservation, EventRecordingPolicy, LocalEventClipRecorder, SustainedEventDetector
from ondamm_learning import build_learning_program_plan, build_learning_run_summary, render_learning_program_markdown, render_learning_run_summary_markdown
from ondamm_models import Dossier, unique_preserving_order, utc_now
from ondamm_paths import ONDAMM_EXPORTS, ONDAMM_LEARNING_EXPORTS
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
    temporal_enabled: bool = False
    temporal_checkpoint: str | None = None


def slugify(value: str) -> str:
    lowered = value.strip().lower()
    return re.sub(r"[^a-z0-9_-]+", "-", lowered).strip("-") or "run"


def dominant_key(counts: Counter[str], fallback: str) -> str:
    if not counts:
        return fallback
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def append_unique_events(target: list[EventMetadata], events: list[EventMetadata] | tuple[EventMetadata, ...]) -> None:
    known = {event.event_id for event in target}
    for event in events:
        if event.event_id not in known:
            target.append(event)
            known.add(event.event_id)


def resolve_temporal_checkpoint(value: str | None) -> Path | None:
    if value:
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing --temporal-checkpoint: {path}")
        return path
    root = Path(__file__).resolve().parents[1] / "outputs" / "micro_expression" / "v4_tcn"
    candidates = sorted(
        root.glob("encoder_*.pt"),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    return candidates[0].resolve() if candidates else None


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
    payload["temporal_enabled"] = capture.temporal_enabled
    payload["temporal_checkpoint"] = capture.temporal_checkpoint
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


def run_camera_mode(
    args: argparse.Namespace,
    *,
    recorder: LocalEventClipRecorder,
    detector: SustainedEventDetector,
    temporal_demo=None,
) -> RunCapture:
    import cv2

    from holistic_camera import classify_gaze_direction
    from micro_expression_signals import MicroExpressionSignalExtractor
    from ondamm_demo_overlay import render_demo_overlay
    from ondamm_facial_movement import analyze_facial_movements, rules_from_approved_profiles

    started_at = utc_now()
    wall_started = time.time()
    observations = ObservationAccumulator()
    detected_events: list[EventMetadata] = []
    recorded_events: list[EventMetadata] = []
    dossier = load_dossier(args.child_id)
    movement_rules = rules_from_approved_profiles(dossier.approved_facial_movement_profiles)
    extractor = MicroExpressionSignalExtractor(
        dino_every=args.dino_every,
        enable_dino=args.demo_dino,
    )
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        extractor.close()
        raise RuntimeError(f"Could not open camera index {args.camera}")

    frame_index = 0
    previous_loop = time.perf_counter()
    fps_ema = 0.0
    timestamp_ms = -1
    elapsed = 0.0
    print("demo_controls: ESC/Q stop" + (", B capture DINO baseline" if args.demo_dino else ""))
    try:
        while time.time() - wall_started < args.duration_seconds:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            elapsed = round(time.time() - wall_started, 3)
            now_loop = time.perf_counter()
            instant_fps = 1.0 / max(now_loop - previous_loop, 1e-6)
            previous_loop = now_loop
            fps_ema = instant_fps if frame_index == 0 else 0.9 * fps_ema + 0.1 * instant_fps
            timestamp_ms = max(timestamp_ms + 1, int(elapsed * 1000))
            signal = extractor.extract(frame, frame_index, timestamp_ms)
            face_present = bool(signal.get("face_detected"))
            gaze_zone = "unknown"
            gaze = signal.get("gaze")
            if isinstance(gaze, dict):
                gaze_zone = classify_gaze_direction(gaze["horizontal"], gaze["vertical"])
            posture_proxy = "unavailable"
            movement_labels: tuple[str, ...] = ()
            if face_present and signal.get("blendshapes"):
                movement_analysis = analyze_facial_movements(signal["blendshapes"], rules=movement_rules)
                movement_labels = tuple(movement_analysis.active_labels)
            observations.add(face_present=face_present, gaze_zone=gaze_zone, posture_proxy=posture_proxy)

            status_before = (
                temporal_demo.overlay_status(timestamp=elapsed)
                if temporal_demo is not None
                else {
                    "temporal_enabled": False,
                    "occurrence_threshold": args.temporal_min_occurrences,
                }
            )
            status_before["fps"] = fps_ema
            recorded_frame = (
                render_demo_overlay(frame, signal, status_before)
                if args.debug_overlay
                else frame
            )
            recorder.add_frame(frame=recorded_frame, timestamp=elapsed)
            detector_events = detector.add_observation(
                EventObservation(
                    timestamp=elapsed,
                    face_present=face_present,
                    gaze_zone=gaze_zone,
                    posture_proxy=posture_proxy,
                    facial_movement_labels=movement_labels,
                )
            )
            detected_events.extend(detector_events)
            recorded_events.extend(record_emitted_events(detector_events=detector_events, recorder=recorder, clip_fps=args.clip_fps))

            if temporal_demo is not None:
                temporal_result = temporal_demo.process(
                    timestamp=elapsed,
                    signal=signal,
                    frame_for_record=recorded_frame,
                )
                append_unique_events(detected_events, temporal_result.requested_events)
                append_unique_events(recorded_events, temporal_result.finalized_events)

            if not args.headless:
                if args.debug_overlay:
                    status_after = (
                        temporal_demo.overlay_status(timestamp=elapsed)
                        if temporal_demo is not None
                        else status_before
                    )
                    status_after["fps"] = fps_ema
                    preview = render_demo_overlay(frame, signal, status_after)
                else:
                    preview = frame.copy()
                    cv2.putText(preview, f"face={face_present}", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
                    cv2.putText(preview, f"gaze={gaze_zone}", (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
                    cv2.putText(preview, f"movement={','.join(movement_labels) or 'none'}", (16, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)
                cv2.imshow("ON DAMM live micro-motion demo", preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                if key in (ord("b"), ord("B")) and args.demo_dino:
                    extractor.capture_baseline()
            frame_index += 1
    finally:
        if temporal_demo is not None:
            temporal_result = temporal_demo.close(timestamp=elapsed)
            append_unique_events(detected_events, temporal_result.requested_events)
            append_unique_events(recorded_events, temporal_result.finalized_events)
        extractor.close()
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
    return RunCapture(
        mode="camera-temporal-demo" if temporal_demo is not None else "camera",
        started_at=started_at,
        finished_at=utc_now(),
        duration_seconds=round(time.time() - wall_started, 3),
        observations=observations,
        detected_events=detected_events,
        recorded_events=recorded_events,
        clip_directory=str(recorder.output_dir) if args.record_events and recorder.output_dir else None,
        temporal_enabled=temporal_demo is not None,
        temporal_checkpoint=str(temporal_demo.checkpoint_path) if temporal_demo is not None else None,
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
        "temporal_enabled": capture.temporal_enabled,
        "temporal_checkpoint": capture.temporal_checkpoint,
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
    parser.add_argument("--clip-fps", type=float, default=30.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--debug-overlay", action="store_true", help="Render and record the ON DAMM landmark/demo overlay")
    parser.add_argument("--demo-dino", action="store_true", help="Enable optional local DINO heatmap inference")
    parser.add_argument("--dino-every", type=int, default=3)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--record-events", action="store_true")
    parser.add_argument("--temporal-checkpoint", help="Frozen causal TCN encoder checkpoint; newest local encoder_*.pt is auto-selected")
    parser.add_argument("--no-temporal", action="store_true", help="Disable temporal pattern discovery even if a checkpoint exists")
    parser.add_argument("--require-temporal", action="store_true", help="Fail instead of falling back when no temporal checkpoint exists")
    parser.add_argument("--pattern-memory-root", default=str(ONDAMM_EXPORTS / "pattern-memory"))
    parser.add_argument("--temporal-calibration-seconds", type=float, default=3.0)
    parser.add_argument("--temporal-onset-z", type=float, default=4.0)
    parser.add_argument("--temporal-offset-z", type=float, default=2.0)
    parser.add_argument("--temporal-min-episode-seconds", type=float, default=0.2)
    parser.add_argument("--temporal-refractory-seconds", type=float, default=0.5)
    parser.add_argument("--temporal-min-occurrences", type=int, default=3)
    parser.add_argument("--temporal-strong-occurrences", type=int, default=5)
    parser.add_argument("--temporal-pre-seconds", type=float, default=1.5)
    parser.add_argument("--temporal-post-seconds", type=float, default=1.0)
    parser.add_argument(
        "--movement-label",
        action="append",
        default=[],
        help="Approved facial movement label to auto-record; repeat for multiple labels",
    )
    parser.add_argument("--movement-min-seconds", type=float, default=0.4)
    parser.add_argument("--output-dir")
    parser.add_argument("--run-id")
    args = parser.parse_args()

    if args.clip_fps <= 0:
        raise ValueError("--clip-fps must be positive")
    if args.duration_seconds <= 0:
        raise ValueError("--duration-seconds must be positive")
    if args.movement_min_seconds <= 0:
        raise ValueError("--movement-min-seconds must be positive")
    if args.dino_every <= 0:
        raise ValueError("--dino-every must be positive")
    if args.temporal_calibration_seconds < 0:
        raise ValueError("--temporal-calibration-seconds must be non-negative")
    if args.temporal_onset_z <= 0 or not 0 <= args.temporal_offset_z < args.temporal_onset_z:
        raise ValueError("temporal z thresholds require 0 <= offset < onset")
    if args.temporal_min_episode_seconds <= 0 or args.temporal_refractory_seconds < 0:
        raise ValueError("temporal episode durations are invalid")
    if args.temporal_min_occurrences < 2:
        raise ValueError("--temporal-min-occurrences must be at least two")
    if args.temporal_strong_occurrences < args.temporal_min_occurrences:
        raise ValueError("--temporal-strong-occurrences must be >= --temporal-min-occurrences")
    if args.temporal_pre_seconds < 0 or args.temporal_post_seconds < 0:
        raise ValueError("temporal clip pre/post seconds must be non-negative")

    dossier = load_dossier(args.child_id)
    if args.movement_label:
        from ondamm_facial_movement import rules_from_approved_profiles

        available_labels = {
            rule.label for rule in rules_from_approved_profiles(dossier.approved_facial_movement_profiles)
        }
        unknown_labels = sorted(set(args.movement_label) - available_labels)
        if unknown_labels:
            raise ValueError(f"Unknown --movement-label values: {', '.join(unknown_labels)}")
    plan = build_learning_program_plan(
        dossier,
        goal=args.goal,
        caregiver_input=args.caregiver_note,
        created_at=deterministic_demo_started_at() if args.demo else None,
    )
    output_dir = resolve_output_dir(args, dossier)
    clips_dir = resolve_clips_dir(output_dir)
    prepare_output_dirs(output_dir, clips_dir, record_events=args.record_events)
    event_metadata_path = output_dir / "event_recording.json" if args.record_events else None
    policy = EventRecordingPolicy(
        facial_movement_min_seconds=args.movement_min_seconds,
        target_facial_movement_labels=tuple(unique_preserving_order(args.movement_label)),
    )
    recorder = LocalEventClipRecorder(policy=policy, output_dir=clips_dir, recording_enabled=args.record_events)
    detector = SustainedEventDetector(policy=policy)
    temporal_demo = None
    if not args.demo and not args.no_temporal:
        checkpoint = resolve_temporal_checkpoint(args.temporal_checkpoint)
        if checkpoint is None:
            message = (
                "No temporal encoder checkpoint found. Skeleton/rule preview remains available, "
                "but UNKNOWN repeat counting is disabled. Run scripts/train_v4_tcn.py or pass "
                "--temporal-checkpoint."
            )
            if args.require_temporal:
                raise FileNotFoundError(message)
            print(f"temporal_warning: {message}")
        else:
            from ondamm_live_temporal_demo import LiveTemporalDemo

            temporal_demo = LiveTemporalDemo(
                child_id=dossier.child_id,
                checkpoint_path=checkpoint,
                pattern_memory_root=Path(args.pattern_memory_root),
                clips_dir=clips_dir,
                event_metadata_path=output_dir / "event_recording.json",
                record_events=args.record_events,
                clip_fps=args.clip_fps,
                calibration_seconds=args.temporal_calibration_seconds,
                onset_z=args.temporal_onset_z,
                offset_z=args.temporal_offset_z,
                min_episode_seconds=args.temporal_min_episode_seconds,
                refractory_seconds=args.temporal_refractory_seconds,
                min_occurrences_for_clip=args.temporal_min_occurrences,
                strong_candidate_occurrences=args.temporal_strong_occurrences,
                pre_seconds=args.temporal_pre_seconds,
                post_seconds=args.temporal_post_seconds,
            )
            print(f"temporal_checkpoint: {checkpoint}")
            print(f"pattern_memory: {Path(args.pattern_memory_root).expanduser().resolve() / dossier.child_id}")
    capture = (
        run_demo_mode(args, recorder=recorder, detector=detector)
        if args.demo
        else run_camera_mode(
            args,
            recorder=recorder,
            detector=detector,
            temporal_demo=temporal_demo,
        )
    )

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
                "temporal_policy": {
                    "enabled": capture.temporal_enabled,
                    "checkpoint": capture.temporal_checkpoint,
                    "calibration_seconds": args.temporal_calibration_seconds,
                    "episode_onset_z": args.temporal_onset_z,
                    "episode_offset_z": args.temporal_offset_z,
                    "min_episode_seconds": args.temporal_min_episode_seconds,
                    "refractory_seconds": args.temporal_refractory_seconds,
                    "min_occurrences_for_clip": args.temporal_min_occurrences,
                    "clip_pre_seconds": args.temporal_pre_seconds,
                    "clip_post_seconds": args.temporal_post_seconds,
                    "saved_frame_style": "debug-overlay" if args.debug_overlay else "raw-camera",
                },
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
