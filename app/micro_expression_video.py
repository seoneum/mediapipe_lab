from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = REPO_ROOT / "models" / "face_landmarker.task"


def landmark_to_pixel(lm, width: int, height: int) -> tuple[int, int]:
    x = int(round(lm.x * width))
    y = int(round(lm.y * height))

    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))

    return x, y


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    csv_path = Path(args.csv)
    model_path = Path(args.model)

    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {input_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))

    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"input      : {input_path}")
    print(f"resolution : {width}x{height}")
    print(f"fps        : {fps:.3f}")
    print(f"frames     : {n_frames}")

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Cannot create: {output_path}")

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(model_path)
        ),

        # 녹화 영상이므로 VIDEO
        running_mode=RunningMode.VIDEO,

        # 한 영상에서 분석 대상 얼굴 1명
        num_faces=1,

        # 52개 blendshape 출력
        output_face_blendshapes=True,

        # head motion 제거에 나중에 사용
        output_facial_transformation_matrixes=True,

        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    rows: list[dict] = []

    with FaceLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0

        while True:
            ok, frame = cap.read()

            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb),
            )

            # 실제 영상 시간 기준 timestamp
            timestamp_ms = int(round(frame_idx * 1000.0 / fps))

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms,
            )

            vis = frame.copy()

            row = {
                "frame_idx": frame_idx,
                "timestamp_ms": timestamp_ms,
                "face_detected": 0,
            }

            if result.face_landmarks:
                row["face_detected"] = 1

                landmarks = result.face_landmarks[0]

                # -----------------------------
                # 478 landmark 전부 CSV 저장
                # -----------------------------
                for i, lm in enumerate(landmarks):
                    row[f"lm_{i}_x"] = float(lm.x)
                    row[f"lm_{i}_y"] = float(lm.y)
                    row[f"lm_{i}_z"] = float(lm.z)

                # -----------------------------
                # 우선 모든 landmark를 점으로 표시
                # -----------------------------
                for lm in landmarks:
                    x, y = landmark_to_pixel(
                        lm,
                        width,
                        height,
                    )

                    cv2.circle(
                        vis,
                        (x, y),
                        1,
                        (0, 255, 0),
                        -1,
                        cv2.LINE_AA,
                    )

            # ---------------------------------
            # 52 blendshape 저장
            # ---------------------------------
            if result.face_blendshapes:
                for category in result.face_blendshapes[0]:
                    name = category.category_name
                    row[f"bs_{name}"] = float(category.score)

            # ---------------------------------
            # 4x4 facial transform 저장
            # ---------------------------------
            matrices = getattr(
                result,
                "facial_transformation_matrixes",
                None,
            )

            if matrices:
                matrix = np.asarray(
                    matrices[0],
                    dtype=np.float64,
                )

                for r in range(matrix.shape[0]):
                    for c in range(matrix.shape[1]):
                        row[f"T_{r}{c}"] = float(matrix[r, c])

            rows.append(row)

            cv2.putText(
                vis,
                f"frame {frame_idx}  t={timestamp_ms / 1000:.3f}s",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(vis)

            frame_idx += 1

            if frame_idx % 100 == 0:
                print(f"{frame_idx}/{n_frames}")

    cap.release()
    writer.release()

    # frame마다 key가 조금 다를 수 있어서 전체 header union 생성
    fieldnames: list[str] = []

    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as fh:
        writer_csv = csv.DictWriter(
            fh,
            fieldnames=fieldnames,
        )

        writer_csv.writeheader()
        writer_csv.writerows(rows)

    print()
    print("DONE")
    print(f"video : {output_path}")
    print(f"csv   : {csv_path}")
    print(f"rows  : {len(rows)}")
    print(f"cols  : {len(fieldnames)}")


if __name__ == "__main__":
    main()
