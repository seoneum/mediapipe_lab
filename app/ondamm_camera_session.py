"""Live camera session owned by the local ON DAMM web process.

No full-session video is persisted.

Pipeline:
    camera
      -> MediaPipe
      -> temporal encoder
      -> optional child metric head
      -> episode detector
      -> PatternMemory

Frames are kept only in the temporal runtime's bounded RAM ring buffer.
Event MP4 persistence is independently toggled by the web UI.
"""

from __future__ import annotations

import inspect
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any


class CameraSessionManager:
    def __init__(
        self,
        *,
        project_root: Path,
        pattern_memory_root: Path,
    ) -> None:
        self.project_root = (
            project_root.expanduser().resolve()
        )

        self.pattern_memory_root = (
            pattern_memory_root.expanduser().resolve()
        )

        self._condition = threading.Condition(
            threading.RLock()
        )

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._active_child_id: str | None = None
        self._session_id: str | None = None

        self._desired_event_recording = False
        self._abort_without_saving = False

        self._latest_jpeg: bytes | None = None
        self._frame_sequence = 0

        self._status: dict[str, Any] = {
            "running": False,
            "state": "stopped",
            "event_recording": False,
            "error": None,
        }

    # ---------------------------------------------------------
    # Public control API
    # ---------------------------------------------------------

    def start(
        self,
        *,
        child_id: str,
        camera: int = 0,
        width: int = 1280,
        height: int = 720,
    ) -> dict[str, Any]:
        from ondamm_rights import require_camera_session
        from ondamm_store import load_dossier

        child_id = str(child_id).strip()

        if not child_id:
            raise ValueError(
                "child_id is required"
            )

        if width <= 0 or height <= 0:
            raise ValueError(
                "camera width/height must be positive"
            )

        # Fail before opening any camera device.
        require_camera_session(
            load_dossier(child_id)
        )

        with self._condition:
            if (
                self._thread is not None
                and self._thread.is_alive()
            ):
                if (
                    self._active_child_id
                    == child_id
                ):
                    raise RuntimeError(
                        "camera is already running "
                        "for this child"
                    )

                raise RuntimeError(
                    "another child camera session "
                    "is already running"
                )

            self._stop_event = (
                threading.Event()
            )

            self._active_child_id = child_id

            self._session_id = (
                "camera-"
                + datetime.now().strftime(
                    "%Y%m%d-%H%M%S-%f"
                )[:-3]
            )

            self._desired_event_recording = (
                False
            )

            self._abort_without_saving = False

            self._latest_jpeg = None
            self._frame_sequence = 0

            self._status = {
                "running": True,
                "state": "starting",
                "child_id": child_id,
                "session_id":
                    self._session_id,
                "camera": int(camera),
                "width": int(width),
                "height": int(height),
                "event_recording": False,
                "error": None,
                "started_at":
                    datetime.now().isoformat(),
            }

            thread = threading.Thread(
                target=self._run,
                kwargs={
                    "child_id": child_id,
                    "session_id":
                        self._session_id,
                    "camera": int(camera),
                    "width": int(width),
                    "height": int(height),
                },
                name=(
                    "ondamm-camera-"
                    + child_id
                ),
                daemon=True,
            )

            self._thread = thread
            thread.start()

            self._condition.notify_all()

            return self._status_for_locked(
                child_id
            )

    def set_event_recording(
        self,
        *,
        child_id: str,
        enabled: bool,
    ) -> dict[str, Any]:
        enabled = bool(enabled)

        with self._condition:
            self._require_active_child_locked(
                child_id
            )

            # The actual LocalEventClipRecorder mutation
            # is performed by the camera worker thread.
            self._desired_event_recording = (
                enabled
            )

            self._status[
                "event_recording"
            ] = enabled

            self._condition.notify_all()

            return self._status_for_locked(
                child_id
            )

    def stop(
        self,
        *,
        child_id: str,
        discard: bool = False,
    ) -> dict[str, Any]:
        thread: threading.Thread | None

        with self._condition:
            if (
                self._thread is None
                or not self._thread.is_alive()
            ):
                return self._status_for_locked(
                    child_id
                )

            self._require_active_child_locked(
                child_id
            )

            if discard:
                self._abort_without_saving = True

            self._desired_event_recording = (
                False
            )

            self._stop_event.set()

            thread = self._thread

            self._status["state"] = (
                "stopping"
            )

            self._condition.notify_all()

        if (
            thread is not None
            and thread
            is not threading.current_thread()
        ):
            thread.join(timeout=5.0)

        return self.status(child_id)

    def stop_any(
        self,
        *,
        discard: bool = True,
    ) -> None:
        with self._condition:
            child_id = self._active_child_id

        if not child_id:
            return

        try:
            self.stop(
                child_id=child_id,
                discard=discard,
            )
        except RuntimeError:
            pass

    def status(
        self,
        child_id: str,
    ) -> dict[str, Any]:
        with self._condition:
            return self._status_for_locked(
                child_id
            )

    def wait_for_jpeg(
        self,
        *,
        child_id: str,
        after_sequence: int,
        timeout: float = 1.0,
    ) -> tuple[int, bytes] | None:
        deadline = (
            time.monotonic()
            + float(timeout)
        )

        with self._condition:
            while True:
                same_child = (
                    self._active_child_id
                    == child_id
                )

                if (
                    same_child
                    and self._latest_jpeg
                    is not None
                    and self._frame_sequence
                    > after_sequence
                ):
                    return (
                        self._frame_sequence,
                        self._latest_jpeg,
                    )

                thread_alive = (
                    self._thread is not None
                    and self._thread.is_alive()
                )

                if (
                    not thread_alive
                    or not same_child
                ):
                    return None

                remaining = (
                    deadline
                    - time.monotonic()
                )

                if remaining <= 0:
                    return None

                self._condition.wait(
                    timeout=remaining
                )

    # ---------------------------------------------------------
    # Worker
    # ---------------------------------------------------------

    def _run(
        self,
        *,
        child_id: str,
        session_id: str,
        camera: int,
        width: int,
        height: int,
    ) -> None:
        import cv2

        from micro_expression_signals import (
            MicroExpressionSignalExtractor,
        )
        from ondamm_demo_overlay import (
            render_demo_overlay,
        )
        from ondamm_live_temporal_demo import (
            LiveTemporalDemo,
        )
        from ondamm_rights import (
            RightsBlockedError,
            require_camera_session,
        )
        from ondamm_store import load_dossier

        base_checkpoint = (
            self.project_root
            / "outputs"
            / "micro_expression"
            / "v4_tcn"
            / "encoder_product.pt"
        )

        metric_checkpoint = (
            self.project_root
            / "outputs"
            / "micro_expression"
            / "children"
            / child_id
            / "metric_head"
            / "metric_head.pt"
        )

        if not base_checkpoint.is_file():
            self._fail(
                child_id,
                (
                    "missing temporal checkpoint: "
                    f"{base_checkpoint}"
                ),
            )
            return

        use_metric = (
            metric_checkpoint.is_file()
        )

        output_dir = (
            self.project_root
            / "outputs"
            / "ondamm"
            / "live-camera"
            / child_id
            / session_id
        )

        clips_dir = (
            output_dir
            / "event-clips"
        )

        event_metadata_path = (
            output_dir
            / "event_recording.json"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        extractor = None
        cap = None
        temporal_demo = None

        elapsed = 0.0

        try:
            # ----------------------------
            # Temporal runtime
            # ----------------------------

            kwargs: dict[str, Any] = {
                "child_id":
                    child_id,
                "session_id":
                    session_id,
                "checkpoint_path":
                    base_checkpoint,
                "pattern_memory_root":
                    self.pattern_memory_root,
                "clips_dir":
                    clips_dir,
                "event_metadata_path":
                    event_metadata_path,
                # Start in stream-only mode.
                "record_events":
                    False,
                "clip_fps":
                    30.0,
                "calibration_seconds":
                    3.0,
                "calibration_min_valid_samples":
                    60,
                "calibration_min_face_coverage":
                    0.80,
                "calibration_min_effective_seconds":
                    2.5,
                "face_loss_reset_seconds":
                    0.5,
                "onset_z":
                    4.0,
                "offset_z":
                    2.0,
                "min_episode_seconds":
                    0.2,
                "refractory_seconds":
                    0.5,
                "min_occurrences_for_clip":
                    3,
                "strong_candidate_occurrences":
                    5,
                "pre_seconds":
                    3.0,
                "post_seconds":
                    3.0,
                "review_frame_size":
                    (960, 540),
                "review_buffer_fps":
                    12.0,
            }

            signature = inspect.signature(
                LiveTemporalDemo.__init__
            )

            if use_metric:
                if (
                    "metric_checkpoint_path"
                    not in signature.parameters
                ):
                    raise RuntimeError(
                        "LiveTemporalDemo is missing "
                        "metric_checkpoint_path support"
                    )

                kwargs[
                    "metric_checkpoint_path"
                ] = metric_checkpoint

                # Child metric checkpoint decides
                # its own R3-selected threshold.
                kwargs[
                    "candidate_distance_threshold"
                ] = None
            else:
                kwargs[
                    "candidate_distance_threshold"
                ] = 0.05

            temporal_demo = (
                LiveTemporalDemo(**kwargs)
            )

            if not hasattr(
                temporal_demo,
                "set_event_recording",
            ):
                raise RuntimeError(
                    "LiveTemporalDemo is missing "
                    "set_event_recording()"
                )

            # ----------------------------
            # MediaPipe + camera
            # ----------------------------

            extractor = (
                MicroExpressionSignalExtractor(
                    dino_every=3,
                    enable_dino=False,
                )
            )

            backend = (
                cv2.CAP_AVFOUNDATION
                if __import__("sys").platform
                == "darwin"
                else cv2.CAP_ANY
            )

            cap = cv2.VideoCapture(
                camera,
                backend,
            )

            cap.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                width,
            )

            cap.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                height,
            )

            if not cap.isOpened():
                raise RuntimeError(
                    "could not open camera "
                    f"index {camera}"
                )

            with self._condition:
                self._status.update({
                    "running": True,
                    "state": "streaming",
                    "child_id": child_id,
                    "session_id":
                        session_id,
                    "event_recording":
                        False,
                    "metric_enabled":
                        use_metric,
                    "metric_checkpoint":
                        (
                            metric_checkpoint.name
                            if use_metric
                            else None
                        ),
                    "embedding_dimension":
                        temporal_demo
                        .encoder
                        .spec
                        .embedding_dim,
                    "output_dir":
                        str(output_dir),
                })

                self._condition.notify_all()

            wall_started = time.time()
            frame_index = 0
            last_applied_recording = False

            while not self._stop_event.is_set():
                # Apply persistence toggle only
                # inside the camera worker thread.
                with self._condition:
                    desired_recording = (
                        self._desired_event_recording
                    )

                if (
                    desired_recording
                    != last_applied_recording
                ):
                    temporal_demo.set_event_recording(
                        desired_recording
                    )

                    last_applied_recording = (
                        desired_recording
                    )

                ok, frame = cap.read()

                if not ok or frame is None:
                    continue

                # Rights remain live during capture.
                if frame_index % 15 == 0:
                    try:
                        require_camera_session(
                            load_dossier(
                                child_id
                            )
                        )
                    except RightsBlockedError as exc:
                        with self._condition:
                            self._abort_without_saving = (
                                True
                            )

                            self._status[
                                "rights_blocked"
                            ] = True

                            self._status[
                                "error"
                            ] = str(exc)

                        break

                elapsed = (
                    time.time()
                    - wall_started
                )

                timestamp_ms = int(
                    round(
                        elapsed
                        * 1000.0
                    )
                )

                signal = extractor.extract(
                    frame,
                    frame_index,
                    timestamp_ms,
                )

                # The bounded event buffer receives
                # the annotated live representation.
                status_before = (
                    temporal_demo.overlay_status(
                        timestamp=elapsed
                    )
                )

                frame_for_record = (
                    render_demo_overlay(
                        frame,
                        signal,
                        status_before,
                    )
                )

                temporal_demo.process(
                    timestamp=elapsed,
                    signal=signal,
                    frame_for_record=
                        frame_for_record,
                )

                status_after = (
                    temporal_demo.overlay_status(
                        timestamp=elapsed
                    )
                )

                status_after[
                    "event_recording"
                ] = (
                    last_applied_recording
                )

                preview = (
                    render_demo_overlay(
                        frame,
                        signal,
                        status_after,
                    )
                )

                ok_jpeg, encoded = (
                    cv2.imencode(
                        ".jpg",
                        preview,
                        [
                            int(
                                cv2.IMWRITE_JPEG_QUALITY
                            ),
                            84,
                        ],
                    )
                )

                if ok_jpeg:
                    self._publish_jpeg(
                        child_id=child_id,
                        jpeg=encoded.tobytes(),
                    )

                with self._condition:
                    self._status.update({
                        "running": True,
                        "state":
                            "streaming",
                        "event_recording":
                            last_applied_recording,
                        "calibration_ready":
                            status_after.get(
                                "calibration_ready"
                            ),
                        "warming_up":
                            status_after.get(
                                "warming_up"
                            ),
                        "warmup_frames":
                            status_after.get(
                                "warmup_frames"
                            ),
                        "warmup_required_frames":
                            status_after.get(
                                "warmup_required_frames"
                            ),
                        "motion_score":
                            status_after.get(
                                "motion_score"
                            ),
                        "motion_active":
                            status_after.get(
                                "motion_active"
                            ),
                        "lifecycle":
                            status_after.get(
                                "lifecycle"
                            ),
                        "candidate_id":
                            status_after.get(
                                "candidate_id"
                            ),
                        "pattern_id":
                            status_after.get(
                                "pattern_id"
                            ),
                        "occurrence_count":
                            status_after.get(
                                "occurrence_count",
                                0,
                            ),
                        "occurrence_threshold":
                            status_after.get(
                                "occurrence_threshold",
                                3,
                            ),
                        "event_saved":
                            status_after.get(
                                "event_saved",
                                False,
                            ),
                        "frame_index":
                            frame_index,
                        "elapsed_seconds":
                            round(
                                elapsed,
                                3,
                            ),
                    })

                    self._condition.notify_all()

                frame_index += 1

        except Exception as exc:
            with self._condition:
                self._abort_without_saving = (
                    True
                )

            self._fail(
                child_id,
                str(exc),
            )

        finally:
            try:
                if temporal_demo is not None:
                    if (
                        self._abort_without_saving
                    ):
                        temporal_demo.abort_without_saving()
                    else:
                        # Only already-complete event
                        # tails may be persisted.
                        temporal_demo.close(
                            timestamp=elapsed
                        )
            except Exception as exc:
                with self._condition:
                    if not self._status.get(
                        "error"
                    ):
                        self._status[
                            "error"
                        ] = str(exc)

            if extractor is not None:
                try:
                    extractor.close()
                except Exception:
                    pass

            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

            with self._condition:
                self._desired_event_recording = (
                    False
                )

                self._status[
                    "event_recording"
                ] = False

                self._status[
                    "running"
                ] = False

                if (
                    self._status.get("state")
                    != "error"
                ):
                    self._status[
                        "state"
                    ] = "stopped"

                self._status[
                    "stopped_at"
                ] = (
                    datetime.now().isoformat()
                )

                self._condition.notify_all()

    # ---------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------

    def _publish_jpeg(
        self,
        *,
        child_id: str,
        jpeg: bytes,
    ) -> None:
        with self._condition:
            if (
                self._active_child_id
                != child_id
            ):
                return

            self._latest_jpeg = jpeg
            self._frame_sequence += 1

            self._condition.notify_all()

    def _fail(
        self,
        child_id: str,
        message: str,
    ) -> None:
        with self._condition:
            if (
                self._active_child_id
                == child_id
            ):
                self._status[
                    "running"
                ] = False

                self._status[
                    "state"
                ] = "error"

                self._status[
                    "error"
                ] = str(message)

                self._condition.notify_all()

    def _require_active_child_locked(
        self,
        child_id: str,
    ) -> None:
        if (
            self._thread is None
            or not self._thread.is_alive()
            or self._active_child_id
            != child_id
        ):
            raise RuntimeError(
                "no active camera session "
                "for this child"
            )

    def _status_for_locked(
        self,
        child_id: str,
    ) -> dict[str, Any]:
        thread_alive = (
            self._thread is not None
            and self._thread.is_alive()
        )

        if (
            thread_alive
            and self._active_child_id
            != child_id
        ):
            return {
                "running": False,
                "state": "busy",
                "busy_by_other_child": True,
                "active_child_id":
                    self._active_child_id,
                "event_recording": False,
                "error": None,
            }

        result = dict(
            self._status
        )

        result[
            "running"
        ] = bool(
            thread_alive
            and self._active_child_id
            == child_id
        )

        result[
            "busy_by_other_child"
        ] = False

        return result
