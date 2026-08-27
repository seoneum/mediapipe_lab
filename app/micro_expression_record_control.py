from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parent.parent

OUTPUT_ROOT = (
    ROOT
    / "data"
    / "micro_expression"
    / "recordings"
)


# ============================================================
# Protocol definition
# ============================================================

@dataclass
class Phase:
    label: str
    duration: float
    instruction: str

    target_x: float = 0.5
    target_y: float = 0.5

    show_target: bool = True

    category: str = "control"


def build_protocol():
    """
    CONTROL / GAZE protocol

    target_x / target_y:
        normalized display coordinate

        (0, 0)       top-left
        (0.5, 0.5)   center
        (1, 1)       bottom-right
    """

    return [
        # ------------------------------------------------
        # Initial baseline
        # ------------------------------------------------
        Phase(
            label="CENTER_BASELINE",
            duration=10.0,
            instruction="Relax your face and look at the center dot.",
            target_x=0.5,
            target_y=0.5,
        ),

        # ------------------------------------------------
        # Blink
        # ------------------------------------------------
        Phase(
            label="CENTER_NEUTRAL_1",
            duration=3.0,
            instruction="Relax and keep looking at the center dot.",
        ),

        Phase(
            label="BLINK",
            duration=4.0,
            instruction="Blink naturally three times.",
        ),

        Phase(
            label="CENTER_NEUTRAL_2",
            duration=3.0,
            instruction="Relax and keep looking at the center dot.",
        ),

        # ------------------------------------------------
        # Gaze horizontal
        # ------------------------------------------------
        Phase(
            label="GAZE_LEFT",
            duration=2.5,
            instruction="Move only your eyes to the left dot.",
            target_x=0.20,
            target_y=0.50,
        ),

        Phase(
            label="CENTER_AFTER_GAZE_LEFT",
            duration=2.5,
            instruction="Return only your eyes to the center dot.",
            target_x=0.50,
            target_y=0.50,
        ),

        Phase(
            label="GAZE_RIGHT",
            duration=2.5,
            instruction="Move only your eyes to the right dot.",
            target_x=0.80,
            target_y=0.50,
        ),

        Phase(
            label="CENTER_AFTER_GAZE_RIGHT",
            duration=2.5,
            instruction="Return only your eyes to the center dot.",
            target_x=0.50,
            target_y=0.50,
        ),

        # ------------------------------------------------
        # Gaze vertical
        # ------------------------------------------------
        Phase(
            label="GAZE_UP",
            duration=2.5,
            instruction="Move only your eyes to the upper dot.",
            target_x=0.50,
            target_y=0.20,
        ),

        Phase(
            label="CENTER_AFTER_GAZE_UP",
            duration=2.5,
            instruction="Return only your eyes to the center dot.",
            target_x=0.50,
            target_y=0.50,
        ),

        Phase(
            label="GAZE_DOWN",
            duration=2.5,
            instruction="Move only your eyes to the lower dot.",
            target_x=0.50,
            target_y=0.80,
        ),

        Phase(
            label="CENTER_AFTER_GAZE_DOWN",
            duration=3.0,
            instruction="Return only your eyes to the center dot.",
            target_x=0.50,
            target_y=0.50,
        ),

        # ------------------------------------------------
        # Head yaw
        # ------------------------------------------------
        Phase(
            label="HEAD_LEFT",
            duration=2.5,
            instruction="Turn your head slightly left. Keep a neutral face.",
            show_target=False,
        ),

        Phase(
            label="CENTER_AFTER_HEAD_LEFT",
            duration=2.5,
            instruction="Return your head to center.",
            target_x=0.50,
            target_y=0.50,
        ),

        Phase(
            label="HEAD_RIGHT",
            duration=2.5,
            instruction="Turn your head slightly right. Keep a neutral face.",
            show_target=False,
        ),

        Phase(
            label="CENTER_AFTER_HEAD_RIGHT",
            duration=2.5,
            instruction="Return your head to center.",
            target_x=0.50,
            target_y=0.50,
        ),

        # ------------------------------------------------
        # Head pitch
        # ------------------------------------------------
        Phase(
            label="HEAD_UP",
            duration=2.5,
            instruction="Tilt your head slightly upward. Keep a neutral face.",
            show_target=False,
        ),

        Phase(
            label="CENTER_AFTER_HEAD_UP",
            duration=2.5,
            instruction="Return your head to center.",
            target_x=0.50,
            target_y=0.50,
        ),

        Phase(
            label="HEAD_DOWN",
            duration=2.5,
            instruction="Tilt your head slightly downward. Keep a neutral face.",
            show_target=False,
        ),

        Phase(
            label="CENTER_AFTER_HEAD_DOWN",
            duration=3.0,
            instruction="Return your head to center.",
            target_x=0.50,
            target_y=0.50,
        ),

        # ------------------------------------------------
        # Final baseline
        # ------------------------------------------------
        Phase(
            label="FINAL_CENTER",
            duration=5.0,
            instruction="Look at the center dot and relax.",
            target_x=0.50,
            target_y=0.50,
        ),
    ]


# ============================================================
# Drawing helpers
# ============================================================

def draw_translucent_rectangle(
    frame,
    x1,
    y1,
    x2,
    y2,
    *,
    alpha=0.42,
):
    """
    Draw translucent black box.
    """

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        (0, 0, 0),
        -1,
    )

    cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1.0 - alpha,
        0,
        frame,
    )


def draw_centered_text(
    frame,
    text,
    center_x,
    baseline_y,
    *,
    scale=0.8,
    thickness=2,
    color=(255, 255, 255),
):
    """
    Draw one line centered horizontally.
    """

    (text_w, text_h), _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        thickness,
    )

    x = int(
        center_x
        - text_w / 2
    )

    y = int(
        baseline_y
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_target(
    frame,
    x_norm,
    y_norm,
):
    """
    Fixation target.

    white outer ring
    red center
    """

    h, w = frame.shape[:2]

    x = int(
        x_norm * w
    )

    y = int(
        y_norm * h
    )

    # outer ring
    cv2.circle(
        frame,
        (x, y),
        22,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )

    # center
    cv2.circle(
        frame,
        (x, y),
        7,
        (0, 0, 255),
        -1,
        cv2.LINE_AA,
    )


def draw_progress_bar(
    frame,
    progress,
):
    h, w = frame.shape[:2]

    progress = max(
        0.0,
        min(
            1.0,
            progress,
        ),
    )

    x1 = 40
    x2 = w - 40

    y1 = h - 45
    y2 = h - 25

    # background
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (180, 180, 180),
        2,
    )

    filled = int(
        x1
        + (
            x2 - x1
        ) * progress
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (filled, y2),
        (255, 255, 255),
        -1,
    )


def draw_center_instruction(
    frame,
    *,
    title,
    instruction,
    remaining_text=None,
):
    """
    중앙에 계속 떠 있는 반투명 안내 UI.

    중요한 점:
        target dot은 이 함수 호출 후에 그려서
        중앙 fixation point가 항상 UI 위에 보이도록 한다.
    """

    h, w = frame.shape[:2]

    center_x = w // 2
    center_y = h // 2

    box_w = int(
        w * 0.74
    )

    box_h = int(
        h * 0.25
    )

    x1 = int(
        center_x
        - box_w / 2
    )

    y1 = int(
        center_y
        - box_h / 2
    )

    x2 = int(
        center_x
        + box_w / 2
    )

    y2 = int(
        center_y
        + box_h / 2
    )

    draw_translucent_rectangle(
        frame,
        x1,
        y1,
        x2,
        y2,
        alpha=0.40,
    )

    # ----------------------------------------------------
    # title
    # ----------------------------------------------------

    draw_centered_text(
        frame,
        title,
        center_x,
        center_y - 55,
        scale=0.95,
        thickness=2,
    )

    # ----------------------------------------------------
    # 중앙 fixation dot을 가리지 않도록
    # instruction은 점 아래쪽에 배치
    # ----------------------------------------------------

    draw_centered_text(
        frame,
        instruction,
        center_x,
        center_y + 55,
        scale=0.62,
        thickness=1,
    )

    # ----------------------------------------------------
    # countdown
    # ----------------------------------------------------

    if remaining_text is not None:
        draw_centered_text(
            frame,
            remaining_text,
            center_x,
            center_y + 92,
            scale=0.72,
            thickness=2,
        )


def draw_protocol_overlay(
    frame,
    phase,
    phase_idx,
    total_phases,
    elapsed,
):
    h, w = frame.shape[:2]

    remaining = max(
        0.0,
        phase.duration - elapsed,
    )

    progress = (
        elapsed
        / phase.duration
    )

    phase_title = (
        phase.label
        .replace("_", " ")
    )

    # ----------------------------------------------------
    # 1. 중앙 반투명 안내
    # ----------------------------------------------------

    draw_center_instruction(
        frame,
        title=phase_title,
        instruction=phase.instruction,
        remaining_text=f"{remaining:.1f} s",
    )

    # ----------------------------------------------------
    # 2. phase counter
    # ----------------------------------------------------

    counter_text = (
        f"{phase_idx + 1} / {total_phases}"
    )

    draw_centered_text(
        frame,
        counter_text,
        w // 2,
        36,
        scale=0.55,
        thickness=1,
    )

    # ----------------------------------------------------
    # 3. fixation target
    #
    # 반드시 instruction box보다 나중에 그린다.
    # 그래야 CENTER dot이 UI 위에 보인다.
    # ----------------------------------------------------

    if phase.show_target:
        draw_target(
            frame,
            phase.target_x,
            phase.target_y,
        )

    # ----------------------------------------------------
    # 4. progress bar
    # ----------------------------------------------------

    draw_progress_bar(
        frame,
        progress,
    )


# ============================================================
# Output
# ============================================================

def prepare_output(
    participant,
    session,
):
    session_dir = (
        OUTPUT_ROOT
        / participant
        / session
    )

    session_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    video_path = (
        session_dir
        / "control_gaze.mp4"
    )

    csv_path = (
        session_dir
        / "control_gaze_labels.csv"
    )

    metadata_path = (
        session_dir
        / "metadata.json"
    )

    return (
        session_dir,
        video_path,
        csv_path,
        metadata_path,
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--participant",
        required=True,
        help="e.g. p1",
    )

    parser.add_argument(
        "--session",
        default="s01",
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1920,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=1080,
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=60.0,
    )

    args = parser.parse_args()


    protocol = (
        build_protocol()
    )


    (
        session_dir,
        video_path,
        csv_path,
        metadata_path,
    ) = prepare_output(
        args.participant,
        args.session,
    )


    # ========================================================
    # Camera
    # ========================================================

    cap = cv2.VideoCapture(
        args.camera,
        cv2.CAP_AVFOUNDATION,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        args.width,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        args.height,
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        args.fps,
    )


    if not cap.isOpened():
        raise RuntimeError(
            "camera open failed"
        )


    actual_width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    actual_height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    camera_reported_fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )


    if (
        camera_reported_fps <= 1.0
        or camera_reported_fps > 240.0
    ):
        writer_fps = float(
            args.fps
        )

    else:
        writer_fps = (
            camera_reported_fps
        )


    print()
    print(
        "Camera:",
        actual_width,
        "x",
        actual_height,
    )

    print(
        "Camera reported FPS:",
        camera_reported_fps,
    )

    print(
        "Writer FPS:",
        writer_fps,
    )

    print(
        "Output:",
        session_dir,
    )


    # ========================================================
    # Video writer
    # ========================================================

    fourcc = (
        cv2.VideoWriter_fourcc(
            *"mp4v"
        )
    )

    writer = cv2.VideoWriter(
        str(video_path),
        fourcc,
        writer_fps,
        (
            actual_width,
            actual_height,
        ),
    )


    if not writer.isOpened():
        cap.release()

        raise RuntimeError(
            "VideoWriter open failed"
        )


    # ========================================================
    # Metadata
    # ========================================================

    metadata = {
        "participant": (
            args.participant
        ),

        "session": (
            args.session
        ),

        "camera_index": (
            args.camera
        ),

        "width": (
            actual_width
        ),

        "height": (
            actual_height
        ),

        "requested_fps": (
            args.fps
        ),

        "camera_reported_fps": (
            camera_reported_fps
        ),

        "writer_fps": (
            writer_fps
        ),

        "protocol": [
            {
                "index": i,
                "label": phase.label,
                "duration_s": (
                    phase.duration
                ),
                "instruction": (
                    phase.instruction
                ),
                "target_x": (
                    phase.target_x
                ),
                "target_y": (
                    phase.target_y
                ),
                "show_target": (
                    phase.show_target
                ),
                "category": (
                    phase.category
                ),
            }

            for i, phase
            in enumerate(protocol)
        ],
    }


    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )


    # ========================================================
    # CSV
    # ========================================================

    csv_file = open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    )


    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "frame_idx",

            "capture_timestamp_ms",

            "video_timestamp_ms",

            "phase_idx",

            "label",

            "phase_elapsed_ms",

            "phase_remaining_ms",

            "target_x_norm",

            "target_y_norm",

            "show_target",

            "instruction",
        ],
    )


    csv_writer.writeheader()


    # ========================================================
    # READY screen
    # ========================================================

    print()
    print("-------------------------------")
    print("SPACE : start protocol")
    print("Q     : quit")
    print("-------------------------------")
    print()


    cancelled_before_start = False


    while True:
        ok, frame = cap.read()

        if (
            not ok
            or frame is None
        ):
            writer.release()
            cap.release()
            csv_file.close()

            raise RuntimeError(
                "camera read failed"
            )


        preview = (
            frame.copy()
        )


        # 중앙 UI
        draw_center_instruction(
            preview,
            title="READY",
            instruction=(
                "Relax your face and look at the center dot."
            ),
            remaining_text="Press SPACE to start",
        )


        # fixation dot을 가장 마지막에 그려서
        # UI보다 위에 표시
        draw_target(
            preview,
            0.5,
            0.5,
        )


        cv2.imshow(
            "Control / Gaze Recording",
            preview,
        )


        key = (
            cv2.waitKey(1)
            & 0xFF
        )


        if key == 32:
            # SPACE
            break


        if key in (
            ord("q"),
            27,
        ):
            cancelled_before_start = True
            break


    if cancelled_before_start:
        writer.release()
        cap.release()
        csv_file.close()
        cv2.destroyAllWindows()

        print(
            "Cancelled."
        )

        return


    # ========================================================
    # Protocol recording
    # ========================================================

    recording_start = (
        time.monotonic()
    )

    frame_idx = 0

    aborted = False


    try:
        for phase_idx, phase in enumerate(
            protocol
        ):
            phase_start = (
                time.monotonic()
            )


            print(
                f"[{phase_idx + 1:02d}/"
                f"{len(protocol):02d}] "
                f"{phase.label}"
            )


            while True:
                # ------------------------------------------------
                # Capture first
                # ------------------------------------------------

                ok, frame = (
                    cap.read()
                )


                if (
                    not ok
                    or frame is None
                ):
                    aborted = True
                    break


                now = (
                    time.monotonic()
                )


                elapsed = (
                    now
                    - phase_start
                )


                if elapsed >= phase.duration:
                    break


                remaining = max(
                    0.0,
                    phase.duration
                    - elapsed,
                )


                capture_ms = int(
                    (
                        now
                        - recording_start
                    )
                    * 1000.0
                )


                video_ms = int(
                    (
                        frame_idx
                        / writer_fps
                    )
                    * 1000.0
                )


                # ================================================
                # RAW VIDEO
                #
                # IMPORTANT:
                # overlay를 그리기 전에 raw frame을 저장한다.
                #
                # 분석 영상에는:
                #     text 없음
                #     dot 없음
                #     progress bar 없음
                # ================================================

                writer.write(
                    frame
                )


                # ================================================
                # Ground-truth CSV
                # ================================================

                csv_writer.writerow(
                    {
                        "frame_idx": (
                            frame_idx
                        ),

                        "capture_timestamp_ms": (
                            capture_ms
                        ),

                        "video_timestamp_ms": (
                            video_ms
                        ),

                        "phase_idx": (
                            phase_idx
                        ),

                        "label": (
                            phase.label
                        ),

                        "phase_elapsed_ms": int(
                            elapsed
                            * 1000.0
                        ),

                        "phase_remaining_ms": int(
                            remaining
                            * 1000.0
                        ),

                        "target_x_norm": (
                            phase.target_x
                        ),

                        "target_y_norm": (
                            phase.target_y
                        ),

                        "show_target": int(
                            phase.show_target
                        ),

                        "instruction": (
                            phase.instruction
                        ),
                    }
                )


                # ================================================
                # DISPLAY
                #
                # raw frame 복사본에만 UI를 그린다.
                # ================================================

                display = (
                    frame.copy()
                )


                draw_protocol_overlay(
                    display,
                    phase,
                    phase_idx,
                    len(protocol),
                    elapsed,
                )


                cv2.imshow(
                    "Control / Gaze Recording",
                    display,
                )


                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )


                if key in (
                    ord("q"),
                    27,
                ):
                    aborted = True
                    break


                frame_idx += 1


            if aborted:
                break


    finally:
        writer.release()

        cap.release()

        csv_file.close()

        cv2.destroyAllWindows()


    # ========================================================
    # Result
    # ========================================================

    print()

    if aborted:
        print(
            "Recording aborted."
        )
    else:
        print(
            "Protocol finished."
        )


    print(
        "Frames:",
        frame_idx,
    )

    print(
        "Video:",
        video_path,
    )

    print(
        "Labels:",
        csv_path,
    )

    print(
        "Metadata:",
        metadata_path,
    )


if __name__ == "__main__":
    main()