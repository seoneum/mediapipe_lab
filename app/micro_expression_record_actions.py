from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
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
# Data structures
# ============================================================

@dataclass
class ActionSpec:
    name: str
    instruction: str


@dataclass
class Phase:
    label: str
    duration: float

    instruction: str

    action: str
    movement_phase: str

    repeat_idx: int

    is_neutral: bool


# ============================================================
# Natural facial actions
# ============================================================

UPPER_ACTIONS = [
    ActionSpec(
        name="brows_raise",
        instruction=(
            "Raise both eyebrows naturally."
        ),
    ),

    ActionSpec(
        name="brows_frown",
        instruction=(
            "Frown naturally by bringing "
            "your eyebrows slightly downward "
            "and toward the center."
        ),
    ),

    ActionSpec(
        name="eyes_squint",
        instruction=(
            "Squint both eyes naturally "
            "without fully closing them."
        ),
    ),

    ActionSpec(
        name="eyes_wide",
        instruction=(
            "Open both eyes naturally wider."
        ),
    ),
]


LOWER_ACTIONS = [
    ActionSpec(
        name="smile",
        instruction=(
            "Make a natural small smile."
        ),
    ),

    ActionSpec(
        name="mouth_frown",
        instruction=(
            "Lower the corners of your mouth "
            "naturally."
        ),
    ),

    ActionSpec(
        name="lip_press",
        instruction=(
            "Press your lips together gently."
        ),
    ),

    ActionSpec(
        name="lip_pucker",
        instruction=(
            "Pucker your lips naturally."
        ),
    ),

    ActionSpec(
        name="jaw_open",
        instruction=(
            "Open your mouth naturally "
            "by lowering the jaw."
        ),
    ),

    ActionSpec(
        name="nose_wrinkle",
        instruction=(
            "Wrinkle your nose naturally."
        ),
    ),
]


def get_actions(protocol: str):
    if protocol == "upper":
        return UPPER_ACTIONS

    if protocol == "lower":
        return LOWER_ACTIONS

    raise ValueError(
        f"Unknown protocol: {protocol}"
    )


# ============================================================
# Protocol generator
# ============================================================

def build_protocol(
    protocol: str,
    *,
    baseline_duration: float,
    neutral_duration: float,
    transition_duration: float,
    hold_duration: float,
    repeats: int,
):
    """
    Natural gradual facial-action protocol.

    For every action:

        neutral
        -> onset / gradual movement
        -> hold
        -> release / gradual return
        -> neutral

    No L1/L2/L3.

    The subtle movements are naturally contained
    inside the onset/release transition.
    """

    actions = get_actions(
        protocol
    )

    phases: list[Phase] = []


    # ========================================================
    # Initial neutral baseline
    # ========================================================

    phases.append(
        Phase(
            label="INITIAL_NEUTRAL",

            duration=baseline_duration,

            instruction=(
                "Relax your entire face. "
                "Keep your head still and "
                "look at the center dot."
            ),

            action="neutral",

            movement_phase="neutral",

            repeat_idx=0,

            is_neutral=True,
        )
    )


    # ========================================================
    # Action trials
    # ========================================================

    for action in actions:

        for repeat_idx in range(
            1,
            repeats + 1,
        ):

            # ------------------------------------------------
            # PRE neutral
            # ------------------------------------------------

            phases.append(
                Phase(
                    label=(
                        f"{action.name.upper()}"
                        f"_PRE_R{repeat_idx}"
                    ),

                    duration=neutral_duration,

                    instruction=(
                        "Relax your face completely. "
                        "Keep looking at the center dot."
                    ),

                    action=action.name,

                    movement_phase="pre_neutral",

                    repeat_idx=repeat_idx,

                    is_neutral=True,
                )
            )


            # ------------------------------------------------
            # ONSET
            # ------------------------------------------------

            phases.append(
                Phase(
                    label=(
                        f"{action.name.upper()}"
                        f"_ONSET_R{repeat_idx}"
                    ),

                    duration=transition_duration,

                    instruction=(
                        f"{action.instruction} "
                        "Move gradually and slowly "
                        "from neutral to a clear "
                        "but comfortable expression."
                    ),

                    action=action.name,

                    movement_phase="onset",

                    repeat_idx=repeat_idx,

                    is_neutral=False,
                )
            )


            # ------------------------------------------------
            # HOLD
            # ------------------------------------------------

            phases.append(
                Phase(
                    label=(
                        f"{action.name.upper()}"
                        f"_HOLD_R{repeat_idx}"
                    ),

                    duration=hold_duration,

                    instruction=(
                        "Hold the expression comfortably. "
                        "Do not exaggerate it."
                    ),

                    action=action.name,

                    movement_phase="hold",

                    repeat_idx=repeat_idx,

                    is_neutral=False,
                )
            )


            # ------------------------------------------------
            # RELEASE
            # ------------------------------------------------

            phases.append(
                Phase(
                    label=(
                        f"{action.name.upper()}"
                        f"_RELEASE_R{repeat_idx}"
                    ),

                    duration=transition_duration,

                    instruction=(
                        "Slowly return the face "
                        "back to neutral."
                    ),

                    action=action.name,

                    movement_phase="release",

                    repeat_idx=repeat_idx,

                    is_neutral=False,
                )
            )


            # ------------------------------------------------
            # POST neutral
            # ------------------------------------------------

            phases.append(
                Phase(
                    label=(
                        f"{action.name.upper()}"
                        f"_POST_R{repeat_idx}"
                    ),

                    duration=neutral_duration,

                    instruction=(
                        "Relax your face completely "
                        "and return to neutral."
                    ),

                    action=action.name,

                    movement_phase="post_neutral",

                    repeat_idx=repeat_idx,

                    is_neutral=True,
                )
            )


    # ========================================================
    # Final neutral
    # ========================================================

    phases.append(
        Phase(
            label="FINAL_NEUTRAL",

            duration=5.0,

            instruction=(
                "Relax your face and "
                "look at the center dot."
            ),

            action="neutral",

            movement_phase="neutral",

            repeat_idx=0,

            is_neutral=True,
        )
    )


    return phases


# ============================================================
# Intended protocol progress
# ============================================================

def get_intended_progress(
    phase: Phase,
    elapsed: float,
):
    """
    This is NOT measured facial intensity.

    It only describes where we are
    in the instructed movement.

    neutral:
        0

    onset:
        0 -> 1

    hold:
        1

    release:
        1 -> 0
    """

    if phase.duration <= 0:
        return 0.0

    progress = max(
        0.0,
        min(
            1.0,
            elapsed / phase.duration,
        ),
    )


    if phase.movement_phase == "onset":
        return progress


    if phase.movement_phase == "hold":
        return 1.0


    if phase.movement_phase == "release":
        return 1.0 - progress


    return 0.0


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
    alpha=0.40,
):
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (
            int(x1),
            int(y1),
        ),
        (
            int(x2),
            int(y2),
        ),
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
    (
        text_size,
        _,
    ) = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        thickness,
    )

    text_width = (
        text_size[0]
    )

    x = int(
        center_x
        - text_width / 2
    )

    cv2.putText(
        frame,
        text,
        (
            x,
            int(baseline_y),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_wrapped_centered_text(
    frame,
    text,
    center_x,
    start_y,
    *,
    max_width_px,
    scale=0.58,
    thickness=1,
    line_gap=27,
):
    words = text.split()

    lines = []

    current = ""


    for word in words:

        candidate = (
            word
            if not current
            else f"{current} {word}"
        )

        (
            text_size,
            _,
        ) = cv2.getTextSize(
            candidate,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            thickness,
        )


        if (
            text_size[0]
            <= max_width_px
        ):
            current = candidate

        else:

            if current:
                lines.append(
                    current
                )

            current = word


    if current:
        lines.append(
            current
        )


    y = start_y


    for line in lines:

        draw_centered_text(
            frame,
            line,
            center_x,
            y,
            scale=scale,
            thickness=thickness,
        )

        y += line_gap


    return y


def draw_target(
    frame,
    x_norm=0.5,
    y_norm=0.5,
):
    h, w = frame.shape[:2]

    x = int(
        x_norm * w
    )

    y = int(
        y_norm * h
    )


    cv2.circle(
        frame,
        (x, y),
        22,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )


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
        )
        * progress
    )


    cv2.rectangle(
        frame,
        (x1, y1),
        (
            filled,
            y2,
        ),
        (255, 255, 255),
        -1,
    )


def draw_intensity_bar(
    frame,
    intended_progress,
):
    """
    UI용 cue.

    실제 얼굴 intensity가 아니다.
    현재 프로토콜에서 의도한 movement progress.
    """

    h, w = frame.shape[:2]

    bar_width = int(
        w * 0.35
    )

    x1 = int(
        w / 2
        - bar_width / 2
    )

    x2 = int(
        w / 2
        + bar_width / 2
    )

    y1 = int(
        h * 0.72
    )

    y2 = y1 + 18


    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (200, 200, 200),
        2,
    )


    filled = int(
        x1
        + (
            x2 - x1
        )
        * intended_progress
    )


    cv2.rectangle(
        frame,
        (x1, y1),
        (
            filled,
            y2,
        ),
        (255, 255, 255),
        -1,
    )


def draw_phase_overlay(
    frame,
    phase,
    phase_idx,
    total_phases,
    elapsed,
):
    h, w = frame.shape[:2]


    remaining = max(
        0.0,
        phase.duration
        - elapsed,
    )


    phase_progress = max(
        0.0,
        min(
            1.0,
            elapsed
            / max(
                phase.duration,
                1e-6,
            ),
        ),
    )


    intended_progress = (
        get_intended_progress(
            phase,
            elapsed,
        )
    )


    center_x = (
        w // 2
    )

    center_y = (
        h // 2
    )


    # --------------------------------------------------------
    # Translucent instruction box
    # --------------------------------------------------------

    box_width = int(
        w * 0.82
    )

    box_height = int(
        h * 0.36
    )


    x1 = (
        center_x
        - box_width // 2
    )

    x2 = (
        center_x
        + box_width // 2
    )

    y1 = (
        center_y
        - box_height // 2
    )

    y2 = (
        center_y
        + box_height // 2
    )


    draw_translucent_rectangle(
        frame,
        x1,
        y1,
        x2,
        y2,
        alpha=0.40,
    )


    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    if phase.is_neutral:

        title = "NEUTRAL"

    else:

        title = (
            f"{phase.action.upper()} "
            f"| "
            f"{phase.movement_phase.upper()} "
            f"| R{phase.repeat_idx}"
        )


    draw_centered_text(
        frame,
        title,
        center_x,
        center_y - 80,
        scale=0.86,
        thickness=2,
    )


    # --------------------------------------------------------
    # Instruction
    # --------------------------------------------------------

    draw_wrapped_centered_text(
        frame,
        phase.instruction,
        center_x,
        center_y + 45,
        max_width_px=int(
            box_width * 0.88
        ),
        scale=0.58,
        thickness=1,
        line_gap=27,
    )


    # --------------------------------------------------------
    # Remaining time
    # --------------------------------------------------------

    draw_centered_text(
        frame,
        f"{remaining:.1f} s",
        center_x,
        center_y + 125,
        scale=0.70,
        thickness=2,
    )


    # --------------------------------------------------------
    # Intended gradual movement bar
    # --------------------------------------------------------

    if (
        phase.movement_phase
        in (
            "onset",
            "hold",
            "release",
        )
    ):
        draw_intensity_bar(
            frame,
            intended_progress,
        )


    # --------------------------------------------------------
    # Phase count
    # --------------------------------------------------------

    draw_centered_text(
        frame,
        (
            f"{phase_idx + 1} "
            f"/ {total_phases}"
        ),
        center_x,
        35,
        scale=0.50,
        thickness=1,
    )


    # --------------------------------------------------------
    # Center fixation dot
    # --------------------------------------------------------

    draw_target(
        frame,
        0.5,
        0.5,
    )


    # --------------------------------------------------------
    # Current phase progress
    # --------------------------------------------------------

    draw_progress_bar(
        frame,
        phase_progress,
    )


# ============================================================
# Output paths
# ============================================================

def prepare_output(
    participant,
    session,
    protocol,
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


    stem = (
        f"{protocol}_face"
    )


    video_path = (
        session_dir
        / f"{stem}.mp4"
    )


    csv_path = (
        session_dir
        / f"{stem}_labels.csv"
    )


    metadata_path = (
        session_dir
        / f"{stem}_metadata.json"
    )


    return (
        session_dir,
        video_path,
        csv_path,
        metadata_path,
    )


def check_overwrite(
    paths,
    overwrite,
):
    existing = [
        path
        for path in paths
        if path.exists()
    ]


    if (
        existing
        and not overwrite
    ):

        existing_text = "\n".join(
            str(path)
            for path in existing
        )


        raise FileExistsError(
            "Output files already exist.\n"
            "Use --overwrite to replace them:\n"
            f"{existing_text}"
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
        "--protocol",
        choices=(
            "upper",
            "lower",
        ),
        required=True,
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


    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help=(
            "Number of repetitions "
            "per natural facial action."
        ),
    )


    parser.add_argument(
        "--baseline-duration",
        type=float,
        default=10.0,
    )


    parser.add_argument(
        "--neutral-duration",
        type=float,
        default=2.0,
    )


    parser.add_argument(
        "--transition-duration",
        type=float,
        default=2.0,
    )


    parser.add_argument(
        "--hold-duration",
        type=float,
        default=1.0,
    )


    parser.add_argument(
        "--overwrite",
        action="store_true",
    )


    args = parser.parse_args()


    if args.repeats < 1:
        raise ValueError(
            "--repeats must be >= 1"
        )


    protocol = build_protocol(
        args.protocol,

        baseline_duration=(
            args.baseline_duration
        ),

        neutral_duration=(
            args.neutral_duration
        ),

        transition_duration=(
            args.transition_duration
        ),

        hold_duration=(
            args.hold_duration
        ),

        repeats=(
            args.repeats
        ),
    )


    (
        session_dir,
        video_path,
        csv_path,
        metadata_path,
    ) = prepare_output(
        args.participant,
        args.session,
        args.protocol,
    )


    check_overwrite(
        (
            video_path,
            csv_path,
            metadata_path,
        ),
        args.overwrite,
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
        f"Participant : {args.participant}"
    )

    print(
        f"Protocol    : {args.protocol}"
    )

    print(
        f"Resolution  : "
        f"{actual_width} x {actual_height}"
    )

    print(
        f"Writer FPS  : {writer_fps}"
    )

    print(
        f"Repeats     : {args.repeats}"
    )

    print(
        f"Output      : {session_dir}"
    )

    print()


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

        "protocol": (
            args.protocol
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

        "repeats": (
            args.repeats
        ),

        "baseline_duration_s": (
            args.baseline_duration
        ),

        "neutral_duration_s": (
            args.neutral_duration
        ),

        "transition_duration_s": (
            args.transition_duration
        ),

        "hold_duration_s": (
            args.hold_duration
        ),

        "actions": [
            asdict(action)
            for action
            in get_actions(
                args.protocol
            )
        ],

        "phases": [
            asdict(phase)
            for phase
            in protocol
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

            "protocol",

            "action",

            "movement_phase",

            "repeat_idx",

            "is_neutral",

            "phase_elapsed_ms",

            "phase_remaining_ms",

            "phase_progress",

            "intended_progress",

            "instruction",
        ],
    )


    csv_writer.writeheader()


    # ========================================================
    # READY screen
    # ========================================================

    print(
        "SPACE : start"
    )

    print(
        "Q     : quit"
    )

    print()


    cancelled = False


    while True:

        ok, frame = (
            cap.read()
        )


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


        preview = frame.copy()

        h, w = preview.shape[:2]

        center_x = w // 2
        center_y = h // 2


        box_width = int(
            w * 0.78
        )

        box_height = int(
            h * 0.28
        )


        draw_translucent_rectangle(
            preview,

            center_x
            - box_width // 2,

            center_y
            - box_height // 2,

            center_x
            + box_width // 2,

            center_y
            + box_height // 2,

            alpha=0.40,
        )


        draw_centered_text(
            preview,

            (
                f"{args.protocol.upper()} "
                f"FACE READY"
            ),

            center_x,

            center_y - 55,

            scale=0.95,

            thickness=2,
        )


        draw_centered_text(
            preview,

            (
                "Perform each movement "
                "slowly and naturally."
            ),

            center_x,

            center_y + 48,

            scale=0.62,

            thickness=1,
        )


        draw_centered_text(
            preview,

            "Press SPACE to start",

            center_x,

            center_y + 88,

            scale=0.74,

            thickness=2,
        )


        draw_target(
            preview,
            0.5,
            0.5,
        )


        cv2.imshow(
            "Micro Expression Action Recording",
            preview,
        )


        key = (
            cv2.waitKey(1)
            & 0xFF
        )


        if key == 32:
            break


        if key in (
            ord("q"),
            27,
        ):

            cancelled = True

            break


    if cancelled:

        writer.release()

        cap.release()

        csv_file.close()

        cv2.destroyAllWindows()

        print(
            "Cancelled."
        )

        return


    # ========================================================
    # Recording
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
                f"[{phase_idx + 1:03d}/"
                f"{len(protocol):03d}] "
                f"{phase.label}"
            )


            while True:

                ok, frame = cap.read()


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


                if (
                    elapsed
                    >= phase.duration
                ):
                    break


                remaining = max(
                    0.0,
                    phase.duration
                    - elapsed,
                )


                phase_progress = max(
                    0.0,
                    min(
                        1.0,
                        elapsed
                        / max(
                            phase.duration,
                            1e-6,
                        ),
                    ),
                )


                intended_progress = (
                    get_intended_progress(
                        phase,
                        elapsed,
                    )
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


                # ============================================
                # RAW VIDEO
                #
                # overlay 전 원본 frame 저장
                # ============================================

                writer.write(
                    frame
                )


                # ============================================
                # Ground-truth / protocol CSV
                # ============================================

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

                        "protocol": (
                            args.protocol
                        ),

                        "action": (
                            phase.action
                        ),

                        "movement_phase": (
                            phase.movement_phase
                        ),

                        "repeat_idx": (
                            phase.repeat_idx
                        ),

                        "is_neutral": int(
                            phase.is_neutral
                        ),

                        "phase_elapsed_ms": int(
                            elapsed * 1000.0
                        ),

                        "phase_remaining_ms": int(
                            remaining * 1000.0
                        ),

                        "phase_progress": (
                            phase_progress
                        ),

                        "intended_progress": (
                            intended_progress
                        ),

                        "instruction": (
                            phase.instruction
                        ),
                    }
                )


                # ============================================
                # DISPLAY ONLY
                # ============================================

                display = (
                    frame.copy()
                )


                draw_phase_overlay(
                    display,
                    phase,
                    phase_idx,
                    len(protocol),
                    elapsed,
                )


                cv2.imshow(
                    "Micro Expression Action Recording",
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
        f"Frames   : {frame_idx}"
    )

    print(
        f"Video    : {video_path}"
    )

    print(
        f"Labels   : {csv_path}"
    )

    print(
        f"Metadata : {metadata_path}"
    )


if __name__ == "__main__":
    main()