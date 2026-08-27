from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent

if str(
    ROOT / "app"
) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT / "app"),
    )


from micro_expression_signals import (
    LEFT_BROW,
    LEFT_EYE,
    MOUTH,
    RIGHT_BROW,
    RIGHT_EYE,
    MicroExpressionSignalExtractor,
)


# ============================================================
# Drawing helpers
# ============================================================

def points_to_pixels(
    points,
    width,
    height,
):
    pixels = (
        points[:, :2].copy()
    )

    pixels[:, 0] *= width
    pixels[:, 1] *= height

    return np.rint(
        pixels
    ).astype(
        np.int32
    )


def draw_landmarks(
    frame,
    points,
):
    h, w = frame.shape[:2]

    pixels = points_to_pixels(
        points,
        w,
        h,
    )

    for x, y in pixels:
        if (
            0 <= x < w
            and 0 <= y < h
        ):
            cv2.circle(
                frame,
                (
                    int(x),
                    int(y),
                ),
                1,
                (0, 255, 0),
                -1,
                cv2.LINE_AA,
            )


def draw_region(
    frame,
    points,
    indices,
    color,
    alpha=0.10,
):
    h, w = frame.shape[:2]

    pixels = points_to_pixels(
        points,
        w,
        h,
    )

    valid = [
        i
        for i in indices
        if i < len(pixels)
    ]

    if len(valid) < 3:
        return

    pts = pixels[
        valid
    ].astype(
        np.int32
    )

    hull = cv2.convexHull(
        pts
    )

    overlay = frame.copy()

    cv2.fillConvexPoly(
        overlay,
        hull,
        color,
    )

    cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1.0 - alpha,
        0,
        frame,
    )

    cv2.polylines(
        frame,
        [hull],
        True,
        color,
        1,
        cv2.LINE_AA,
    )


def normalize_heatmap(
    diff,
):
    if diff is None:
        return None

    low = float(
        np.percentile(
            diff,
            10,
        )
    )

    high = float(
        np.percentile(
            diff,
            95,
        )
    )

    if high <= low + 1e-8:
        return np.zeros_like(
            diff,
            dtype=np.float32,
        )

    normalized = (
        diff - low
    ) / (
        high - low
    )

    return np.clip(
        normalized,
        0.0,
        1.0,
    ).astype(
        np.float32
    )


def draw_dino_heatmap(
    frame,
    diff,
    bbox,
):
    if (
        diff is None
        or bbox is None
    ):
        return

    (
        x1,
        y1,
        x2,
        y2,
    ) = bbox

    width = (
        x2 - x1
    )

    height = (
        y2 - y1
    )

    if (
        width <= 0
        or height <= 0
    ):
        return

    heat = normalize_heatmap(
        diff
    )

    heat = cv2.resize(
        heat,
        (
            width,
            height,
        ),
        interpolation=(
            cv2.INTER_CUBIC
        ),
    )

    heat_u8 = np.uint8(
        np.clip(
            heat * 255.0,
            0,
            255,
        )
    )

    colored = cv2.applyColorMap(
        heat_u8,
        cv2.COLORMAP_TURBO,
    )

    # 변화량이 큰 부분일수록
    # overlay를 강하게 적용
    alpha = (
        heat[..., None]
        * 0.50
    ).astype(
        np.float32
    )

    roi = frame[
        y1:y2,
        x1:x2,
    ].astype(
        np.float32
    )

    colored = colored.astype(
        np.float32
    )

    blended = (
        roi
        * (
            1.0 - alpha
        )
        + colored
        * alpha
    )

    frame[
        y1:y2,
        x1:x2,
    ] = np.clip(
        blended,
        0,
        255,
    ).astype(
        np.uint8
    )


def put_line(
    frame,
    text,
    y,
    *,
    color=(255, 255, 255),
    scale=0.50,
    thickness=1,
):
    cv2.putText(
        frame,
        text,
        (20, int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

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
        "--dino-every",
        type=int,
        default=1,
        help=(
            "DINO inference interval. "
            "1 = every frame"
        ),
    )

    args = parser.parse_args()


    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

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

    if not cap.isOpened():
        raise RuntimeError(
            f"camera {args.camera} open failed"
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

    print(
        f"camera: "
        f"{actual_width} x "
        f"{actual_height}"
    )


    # --------------------------------------------------------
    # Signal extractor
    # --------------------------------------------------------

    extractor = (
        MicroExpressionSignalExtractor(
            dino_every=(
                args.dino_every
            )
        )
    )


    print()
    print("Controls")
    print("---------------------------")
    print("B : capture neutral baseline")
    print("R : reset DINO baseline")
    print("Q : quit")
    print()


    start = time.monotonic()

    frame_idx = 0

    prev_loop_time = (
        time.perf_counter()
    )

    fps_ema = 0.0


    try:
        while True:
            ok, frame = cap.read()

            if (
                not ok
                or frame is None
            ):
                print(
                    "camera read failed"
                )
                break


            # ------------------------------------------------
            # FPS
            # ------------------------------------------------

            now_loop = (
                time.perf_counter()
            )

            dt = max(
                now_loop
                - prev_loop_time,
                1e-6,
            )

            prev_loop_time = (
                now_loop
            )

            instant_fps = (
                1.0 / dt
            )

            if frame_idx == 0:
                fps_ema = instant_fps
            else:
                fps_ema = (
                    0.90 * fps_ema
                    + 0.10 * instant_fps
                )


            timestamp_ms = int(
                (
                    time.monotonic()
                    - start
                )
                * 1000
            )


            # ------------------------------------------------
            # Extract
            # ------------------------------------------------

            signal = (
                extractor.extract(
                    frame,
                    frame_idx,
                    timestamp_ms,
                )
            )


            # ------------------------------------------------
            # Header
            # ------------------------------------------------

            baseline_text = (
                "BASELINE: SET"
                if extractor.has_baseline
                else "BASELINE: PRESS B"
            )

            baseline_color = (
                (0, 255, 0)
                if extractor.has_baseline
                else (0, 255, 255)
            )

            put_line(
                frame,
                baseline_text,
                28,
                color=baseline_color,
                scale=0.65,
                thickness=2,
            )

            put_line(
                frame,
                f"FPS={fps_ema:.1f}",
                50,
            )


            # ------------------------------------------------
            # Face detected
            # ------------------------------------------------

            if signal[
                "face_detected"
            ]:
                points = signal[
                    "landmarks"
                ]


                # ============================================
                # DINO heatmap first
                # ============================================

                draw_dino_heatmap(
                    frame,
                    signal[
                        "dino_change_map"
                    ],
                    signal[
                        "bbox"
                    ],
                )


                # ============================================
                # MediaPipe landmarks / ROI
                # ============================================

                draw_landmarks(
                    frame,
                    points,
                )

                draw_region(
                    frame,
                    points,
                    LEFT_EYE,
                    (255, 100, 0),
                )

                draw_region(
                    frame,
                    points,
                    RIGHT_EYE,
                    (255, 100, 0),
                )

                draw_region(
                    frame,
                    points,
                    LEFT_BROW,
                    (0, 200, 255),
                )

                draw_region(
                    frame,
                    points,
                    RIGHT_BROW,
                    (0, 200, 255),
                )

                draw_region(
                    frame,
                    points,
                    MOUTH,
                    (180, 0, 255),
                )


                # ============================================
                # Face bbox
                # ============================================

                bbox = signal[
                    "bbox"
                ]

                if bbox is not None:
                    (
                        x1,
                        y1,
                        x2,
                        y2,
                    ) = bbox

                    cv2.rectangle(
                        frame,
                        (
                            x1,
                            y1,
                        ),
                        (
                            x2,
                            y2,
                        ),
                        (255, 255, 255),
                        1,
                    )


                # ============================================
                # General facial information
                # ============================================

                face_pct = (
                    signal[
                        "face_ratio"
                    ]
                    * 100.0
                )

                put_line(
                    frame,
                    (
                        f"face={face_pct:.1f}% "
                        f"blink={signal['blink']:.3f}"
                    ),
                    75,
                )

                put_line(
                    frame,
                    (
                        f"head "
                        f"yaw={signal['yaw_deg']:+.1f} "
                        f"pitch={signal['pitch_deg']:+.1f} "
                        f"roll={signal['roll_deg']:+.1f}"
                    ),
                    96,
                )


                gaze = signal[
                    "gaze"
                ]

                if gaze is not None:
                    put_line(
                        frame,
                        (
                            f"gaze "
                            f"h={gaze['horizontal']:.3f} "
                            f"v={gaze['vertical']:.3f}"
                        ),
                        117,
                    )
                else:
                    put_line(
                        frame,
                        "gaze unavailable",
                        117,
                    )


                # ============================================
                # Region motion
                # ============================================

                put_line(
                    frame,
                    (
                        "motion "
                        f"mouth={signal['motion_mouth']:.5f} "
                        f"Leye={signal['motion_left_eye']:.5f} "
                        f"Reye={signal['motion_right_eye']:.5f}"
                    ),
                    138,
                )


                put_line(
                    frame,
                    (
                        "brow MAG "
                        f"L={signal['motion_left_brow']:.5f} "
                        f"R={signal['motion_right_brow']:.5f}"
                    ),
                    159,
                )


                # ============================================
                # Eyebrow direction
                # ============================================

                put_line(
                    frame,
                    (
                        "brow UP   "
                        f"L={signal['brow_up_left']:.5f} "
                        f"R={signal['brow_up_right']:.5f}"
                    ),
                    180,
                    color=(0, 255, 255),
                )

                put_line(
                    frame,
                    (
                        "brow DOWN "
                        f"L={signal['brow_down_left']:.5f} "
                        f"R={signal['brow_down_right']:.5f}"
                    ),
                    201,
                    color=(0, 200, 255),
                )


                # ============================================
                # DINO
                # ============================================

                put_line(
                    frame,
                    (
                        "DINO "
                        f"mean={signal['dino_change_mean']:.4f} "
                        f"max={signal['dino_change_max']:.4f} "
                        f"{signal['dino_inference_ms']:.1f}ms"
                    ),
                    222,
                )


                # ============================================
                # MediaPipe blendshapes
                # ============================================

                bs = signal[
                    "blendshapes"
                ]

                shown = [
                    # mouth
                    "mouthSmileLeft",
                    "mouthSmileRight",

                    "mouthPressLeft",
                    "mouthPressRight",

                    "mouthFrownLeft",
                    "mouthFrownRight",

                    # eyes
                    "eyeSquintLeft",
                    "eyeSquintRight",

                    "eyeWideLeft",
                    "eyeWideRight",

                    # eyebrow raise
                    "browInnerUp",
                    "browOuterUpLeft",
                    "browOuterUpRight",

                    # eyebrow lower
                    "browDownLeft",
                    "browDownRight",

                    # nose
                    "noseSneerLeft",
                    "noseSneerRight",

                    # jaw
                    "jawOpen",
                ]


                y = 255

                for name in shown:
                    value = bs.get(
                        name,
                        0.0,
                    )

                    put_line(
                        frame,
                        (
                            f"{name}: "
                            f"{value:.3f}"
                        ),
                        y,
                    )

                    y += 19


            else:
                put_line(
                    frame,
                    "FACE NOT DETECTED",
                    80,
                    color=(0, 0, 255),
                    scale=0.7,
                    thickness=2,
                )


            # ------------------------------------------------
            # Controls
            # ------------------------------------------------

            put_line(
                frame,
                "B=baseline   R=reset   Q=quit",
                frame.shape[0] - 25,
                scale=0.60,
            )


            cv2.imshow(
                "MediaPipe + DINOv3 Micro Expression",
                frame,
            )


            key = (
                cv2.waitKey(1)
                & 0xFF
            )


            if key == ord("b"):
                ok_baseline = (
                    extractor
                    .capture_baseline()
                )

                if ok_baseline:
                    print(
                        "DINO baseline: SET"
                    )
                else:
                    print(
                        "DINO baseline: "
                        "NO FEATURE YET"
                    )


            elif key == ord("r"):
                extractor.reset_baseline()

                print(
                    "DINO baseline: RESET"
                )


            elif key in (
                ord("q"),
                27,
            ):
                break


            frame_idx += 1


    finally:
        extractor.close()

        cap.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()