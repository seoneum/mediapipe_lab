from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


ROOT = Path(__file__).resolve().parent.parent

MP_MODEL = ROOT / "models" / "face_landmarker.task"
DINO_MODEL = ROOT / "models" / "dinov3" / "vits16"


# ============================================================
# Face regions
# ============================================================

LEFT_EYE = [
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246,
]

RIGHT_EYE = [
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398,
]

LEFT_BROW = [
    70, 63, 105, 66, 107,
    55, 65, 52, 53, 46,
]

RIGHT_BROW = [
    336, 296, 334, 293, 300,
    285, 295, 282, 283, 276,
]

MOUTH = [
    61, 146, 91, 181, 84, 17,
    314, 405, 321, 375, 291,
    308, 324, 318, 402, 317,
    14, 87, 178, 88, 95, 78,
]


# 얼굴 정규화를 위한 비교적 안정적인 눈 landmark
LEFT_EYE_CENTER = [33, 133, 159, 145]
RIGHT_EYE_CENTER = [362, 263, 386, 374]


# iris
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

LEFT_EYE_CORNERS = (33, 133)
RIGHT_EYE_CORNERS = (362, 263)


# ============================================================
# Geometry
# ============================================================

def rotation_matrix_to_euler_degrees(matrix):
    """
    MediaPipe facial transformation matrix에서
    yaw / pitch / roll을 degree 단위로 추출한다.
    """

    arr = np.asarray(
        matrix,
        dtype=np.float64,
    )

    if arr.shape not in (
        (3, 3),
        (4, 4),
    ):
        raise ValueError(
            f"unexpected transform shape: {arr.shape}"
        )

    r00 = arr[0, 0]
    r10 = arr[1, 0]

    r20 = arr[2, 0]
    r21 = arr[2, 1]
    r22 = arr[2, 2]

    yaw = math.degrees(
        math.atan2(
            -r20,
            math.sqrt(
                r00 * r00
                + r10 * r10
            ),
        )
    )

    pitch = math.degrees(
        math.atan2(
            r21,
            r22,
        )
    )

    roll = math.degrees(
        math.atan2(
            r10,
            r00,
        )
    )

    return (
        float(yaw),
        float(pitch),
        float(roll),
    )


def landmarks_to_array(landmarks):
    """
    MediaPipe landmark list
        ->
    [N, 3] float32 ndarray
    """

    return np.asarray(
        [
            [
                lm.x,
                lm.y,
                lm.z,
            ]
            for lm in landmarks
        ],
        dtype=np.float32,
    )


def average_xy(points, indices):
    valid = [
        i
        for i in indices
        if i < len(points)
    ]

    if not valid:
        return None

    return points[
        valid,
        :2,
    ].mean(axis=0)


def canonicalize_landmarks(points):
    """
    두 눈 중심을 기준으로 landmark 좌표를 정규화한다.

    제거:
        - translation
        - global scale
        - in-plane roll

    완전한 3D head pose 제거는 아니지만,
    raw image coordinate 차분보다 훨씬 안정적이다.
    """

    xy = points[:, :2].copy()

    left_eye = average_xy(
        points,
        LEFT_EYE_CENTER,
    )

    right_eye = average_xy(
        points,
        RIGHT_EYE_CENTER,
    )

    if (
        left_eye is None
        or right_eye is None
    ):
        return xy.astype(
            np.float32
        )

    center = (
        left_eye + right_eye
    ) / 2.0

    eye_vector = (
        right_eye - left_eye
    )

    scale = float(
        np.linalg.norm(
            eye_vector
        )
    )

    scale = max(
        scale,
        1e-6,
    )

    angle = math.atan2(
        float(eye_vector[1]),
        float(eye_vector[0]),
    )

    # 얼굴 roll을 제거
    c = math.cos(-angle)
    s = math.sin(-angle)

    rotation = np.asarray(
        [
            [c, -s],
            [s, c],
        ],
        dtype=np.float32,
    )

    normalized = (
        (xy - center)
        @ rotation.T
    ) / scale

    return normalized.astype(
        np.float32
    )


def region_motion(
    magnitudes,
    indices,
):
    """
    특정 얼굴 영역 landmark들의
    평균 이동량 magnitude.
    """

    valid = [
        i
        for i in indices
        if i < len(magnitudes)
    ]

    if not valid:
        return 0.0

    return float(
        np.mean(
            magnitudes[valid]
        )
    )


def region_vertical_motion(
    displacement,
    indices,
):
    """
    특정 영역의 수직 방향 motion.

    canonical coordinate에서도
    image coordinate의 y 방향은 유지된다.

        dy < 0 : 위
        dy > 0 : 아래

    반환:
        signed : signed mean dy
        up     : 위쪽 이동량
        down   : 아래쪽 이동량
    """

    valid = [
        i
        for i in indices
        if i < len(displacement)
    ]

    if not valid:
        return {
            "signed": 0.0,
            "up": 0.0,
            "down": 0.0,
        }

    mean_dy = float(
        np.mean(
            displacement[
                valid,
                1,
            ]
        )
    )

    return {
        "signed": mean_dy,

        # image y가 작아지면 위
        "up": max(
            0.0,
            -mean_dy,
        ),

        # image y가 커지면 아래
        "down": max(
            0.0,
            mean_dy,
        ),
    }


# ============================================================
# Gaze
# ============================================================

def eye_gaze_ratio(
    points,
    iris_indices,
    corner_indices,
):
    required = max(
        max(iris_indices),
        *corner_indices,
    )

    if len(points) <= required:
        return None

    iris = average_xy(
        points,
        iris_indices,
    )

    if iris is None:
        return None

    corner_a = points[
        corner_indices[0],
        :2,
    ]

    corner_b = points[
        corner_indices[1],
        :2,
    ]

    left_x = min(
        float(corner_a[0]),
        float(corner_b[0]),
    )

    right_x = max(
        float(corner_a[0]),
        float(corner_b[0]),
    )

    center_y = (
        float(corner_a[1])
        + float(corner_b[1])
    ) / 2.0

    eye_width = max(
        right_x - left_x,
        1e-5,
    )

    horizontal = (
        float(iris[0])
        - left_x
    ) / eye_width

    vertical = (
        float(iris[1])
        - center_y
    ) / eye_width

    return (
        horizontal,
        vertical,
    )


def estimate_gaze(points):
    left = eye_gaze_ratio(
        points,
        LEFT_IRIS,
        LEFT_EYE_CORNERS,
    )

    right = eye_gaze_ratio(
        points,
        RIGHT_IRIS,
        RIGHT_EYE_CORNERS,
    )

    if (
        left is None
        or right is None
    ):
        return None

    return {
        "horizontal": float(
            (
                left[0]
                + right[0]
            ) / 2.0
        ),

        "vertical": float(
            (
                left[1]
                + right[1]
            ) / 2.0
        ),
    }


# ============================================================
# Face crop
# ============================================================

def get_face_crop(
    frame,
    points,
    margin=0.18,
):
    """
    landmark bbox를 기준으로
    square face crop을 만든다.
    """

    h, w = frame.shape[:2]

    xs = points[:, 0] * w
    ys = points[:, 1] * h

    raw_x1 = float(
        xs.min()
    )

    raw_y1 = float(
        ys.min()
    )

    raw_x2 = float(
        xs.max()
    )

    raw_y2 = float(
        ys.max()
    )

    face_ratio = (
        raw_y2 - raw_y1
    ) / float(h)

    cx = (
        raw_x1 + raw_x2
    ) / 2.0

    cy = (
        raw_y1 + raw_y2
    ) / 2.0

    side = max(
        raw_x2 - raw_x1,
        raw_y2 - raw_y1,
    )

    side *= (
        1.0
        + 2.0 * margin
    )

    x1 = int(
        round(
            cx - side / 2
        )
    )

    y1 = int(
        round(
            cy - side / 2
        )
    )

    x2 = int(
        round(
            cx + side / 2
        )
    )

    y2 = int(
        round(
            cy + side / 2
        )
    )

    x1 = max(
        0,
        x1,
    )

    y1 = max(
        0,
        y1,
    )

    x2 = min(
        w,
        x2,
    )

    y2 = min(
        h,
        y2,
    )

    if (
        x2 <= x1
        or y2 <= y1
    ):
        return (
            None,
            None,
            float(face_ratio),
        )

    crop = frame[
        y1:y2,
        x1:x2,
    ].copy()

    return (
        crop,
        (
            x1,
            y1,
            x2,
            y2,
        ),
        float(face_ratio),
    )


# ============================================================
# DINOv3
# ============================================================

class DinoSignals:
    def __init__(
        self,
        model_path=DINO_MODEL,
    ):
        self.device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        print(
            f"DINO device: {self.device}"
        )

        self.processor = (
            AutoImageProcessor
            .from_pretrained(
                str(model_path),
                local_files_only=True,
            )
        )

        self.model = (
            AutoModel
            .from_pretrained(
                str(model_path),
                local_files_only=True,
            )
            .to(self.device)
        )

        self.model.eval()

        self.register_tokens = int(
            getattr(
                self.model.config,
                "num_register_tokens",
                4,
            )
        )

        self.latest_features = None
        self.baseline = None

        self.last_inference_ms = 0.0


    @torch.inference_mode()
    def extract(
        self,
        bgr_crop,
    ):
        t0 = time.perf_counter()

        rgb = cv2.cvtColor(
            bgr_crop,
            cv2.COLOR_BGR2RGB,
        )

        image = Image.fromarray(
            rgb
        )

        inputs = self.processor(
            images=image,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(
                self.device
            )
            for key, value
            in inputs.items()
        }

        output = self.model(
            **inputs
        )

        tokens = (
            output.last_hidden_state
        )

        # CLS token + register tokens 제거
        patches = tokens[
            :,
            1 + self.register_tokens:,
            :
        ]

        # cosine similarity 계산을 위해 normalize
        patches = F.normalize(
            patches.float(),
            dim=-1,
        )

        n = int(
            patches.shape[1]
        )

        grid = int(
            round(
                math.sqrt(n)
            )
        )

        if grid * grid != n:
            raise RuntimeError(
                f"DINO patch count {n} "
                f"is not a square grid"
            )

        patches = patches.reshape(
            grid,
            grid,
            -1,
        )

        self.latest_features = (
            patches.detach().cpu()
        )

        self.last_inference_ms = (
            time.perf_counter()
            - t0
        ) * 1000.0

        return self.latest_features


    def capture_baseline(self):
        if self.latest_features is None:
            return False

        self.baseline = (
            self.latest_features.clone()
        )

        return True


    def reset_baseline(self):
        self.baseline = None


    @property
    def has_baseline(self):
        return (
            self.baseline
            is not None
        )


    def change_map(self):
        if (
            self.latest_features is None
            or self.baseline is None
        ):
            return None

        similarity = (
            self.latest_features
            * self.baseline
        ).sum(dim=-1)

        diff = (
            1.0
            - similarity
        )

        # floating-point 오차로 음수가 될 수 있으므로 clamp
        diff = torch.clamp(
            diff,
            min=0.0,
        )

        return (
            diff.numpy()
            .astype(np.float32)
        )


class DisabledDinoSignals:
    """No-op DINO adapter for MediaPipe-only live/demo operation."""

    baseline = None
    latest_features = None
    last_inference_ms = 0.0

    def extract(self, bgr_crop):
        return None

    def capture_baseline(self):
        return False

    def reset_baseline(self):
        return None

    @property
    def has_baseline(self):
        return False

    def change_map(self):
        return None


# ============================================================
# Unified extractor
# ============================================================

class MicroExpressionSignalExtractor:
    def __init__(
        self,
        *,
        dino_every=3,
        model_path=MP_MODEL,
        enable_dino=True,
    ):
        self.dino_every = max(
            1,
            int(dino_every),
        )

        self.dino = (
            DinoSignals()
            if enable_dino
            else DisabledDinoSignals()
        )
        self.dino_enabled = bool(enable_dino)

        self.prev_canonical = None

        if not Path(
            model_path
        ).is_file():
            raise FileNotFoundError(
                model_path
            )

        options = (
            mp.tasks.vision
            .FaceLandmarkerOptions(
                base_options=(
                    mp.tasks.BaseOptions(
                        model_asset_path=str(
                            model_path
                        ),
                        delegate=(
                            mp.tasks
                            .BaseOptions
                            .Delegate
                            .CPU
                        ),
                    )
                ),

                running_mode=(
                    mp.tasks.vision
                    .RunningMode.VIDEO
                ),

                num_faces=1,

                output_face_blendshapes=True,

                output_facial_transformation_matrixes=True,

                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )

        self.landmarker = (
            mp.tasks.vision
            .FaceLandmarker
            .create_from_options(
                options
            )
        )


    def extract(
        self,
        frame,
        frame_idx,
        timestamp_ms,
    ):
        # ----------------------------------------------------
        # MediaPipe inference
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        image = mp.Image(
            image_format=(
                mp.ImageFormat.SRGB
            ),
            data=np.ascontiguousarray(
                rgb
            ),
        )

        result = (
            self.landmarker
            .detect_for_video(
                image,
                int(timestamp_ms),
            )
        )


        # ----------------------------------------------------
        # No face
        # ----------------------------------------------------

        if not result.face_landmarks:
            self.prev_canonical = None

            return {
                "frame_idx": int(
                    frame_idx
                ),

                "timestamp_ms": int(
                    timestamp_ms
                ),

                "face_detected": False,
            }


        # ----------------------------------------------------
        # Landmarks
        # ----------------------------------------------------

        landmarks = (
            result.face_landmarks[0]
        )

        points = landmarks_to_array(
            landmarks
        )

        canonical = canonicalize_landmarks(
            points
        )


        # ----------------------------------------------------
        # Temporal landmark displacement
        # ----------------------------------------------------

        if self.prev_canonical is None:
            displacement = np.zeros_like(
                canonical
            )
        else:
            displacement = (
                canonical
                - self.prev_canonical
            )

        self.prev_canonical = (
            canonical.copy()
        )


        motion_mag = np.linalg.norm(
            displacement,
            axis=1,
        )


        # ----------------------------------------------------
        # Eyebrow directional motion
        # ----------------------------------------------------

        left_brow_vertical = (
            region_vertical_motion(
                displacement,
                LEFT_BROW,
            )
        )

        right_brow_vertical = (
            region_vertical_motion(
                displacement,
                RIGHT_BROW,
            )
        )


        # ----------------------------------------------------
        # MediaPipe blendshapes
        # ----------------------------------------------------

        blendshapes = {}

        if result.face_blendshapes:
            blendshapes = {
                item.category_name:
                float(item.score)

                for item
                in result.face_blendshapes[0]
            }


        blink = max(
            blendshapes.get(
                "eyeBlinkLeft",
                0.0,
            ),

            blendshapes.get(
                "eyeBlinkRight",
                0.0,
            ),
        )


        # ----------------------------------------------------
        # Head pose
        # ----------------------------------------------------

        yaw = 0.0
        pitch = 0.0
        roll = 0.0

        matrices = getattr(
            result,
            "facial_transformation_matrixes",
            None,
        )

        if matrices:
            (
                yaw,
                pitch,
                roll,
            ) = (
                rotation_matrix_to_euler_degrees(
                    matrices[0]
                )
            )


        # ----------------------------------------------------
        # Gaze
        # ----------------------------------------------------

        gaze = estimate_gaze(
            points
        )


        # ----------------------------------------------------
        # Face crop + DINO
        # ----------------------------------------------------

        (
            crop,
            bbox,
            face_ratio,
        ) = get_face_crop(
            frame,
            points,
        )


        if (
            self.dino_enabled
            and
            crop is not None
            and frame_idx
            % self.dino_every
            == 0
        ):
            self.dino.extract(
                crop
            )


        dino_change = (
            self.dino.change_map()
        )


        dino_mean = 0.0
        dino_max = 0.0

        if dino_change is not None:
            dino_mean = float(
                dino_change.mean()
            )

            dino_max = float(
                dino_change.max()
            )


        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        return {
            "frame_idx": int(
                frame_idx
            ),

            "timestamp_ms": int(
                timestamp_ms
            ),

            "face_detected": True,

            # raw geometry
            "landmarks": points,

            # head-normalized 2D geometry
            "canonical_landmarks": canonical,

            "bbox": bbox,

            "face_ratio": float(
                face_ratio
            ),

            # MediaPipe semantic facial actions
            "blendshapes": blendshapes,

            # head pose
            "yaw_deg": float(yaw),
            "pitch_deg": float(pitch),
            "roll_deg": float(roll),

            # eyes
            "blink": float(
                blink
            ),

            "gaze": gaze,

            # global motion
            "motion_mean": float(
                motion_mag.mean()
            ),

            "motion_max": float(
                motion_mag.max()
            ),

            # region motion magnitude
            "motion_mouth": region_motion(
                motion_mag,
                MOUTH,
            ),

            "motion_left_eye": region_motion(
                motion_mag,
                LEFT_EYE,
            ),

            "motion_right_eye": region_motion(
                motion_mag,
                RIGHT_EYE,
            ),

            "motion_left_brow": region_motion(
                motion_mag,
                LEFT_BROW,
            ),

            "motion_right_brow": region_motion(
                motion_mag,
                RIGHT_BROW,
            ),

            # eyebrow directional motion
            "brow_up_left": (
                left_brow_vertical[
                    "up"
                ]
            ),

            "brow_down_left": (
                left_brow_vertical[
                    "down"
                ]
            ),

            "brow_up_right": (
                right_brow_vertical[
                    "up"
                ]
            ),

            "brow_down_right": (
                right_brow_vertical[
                    "down"
                ]
            ),

            # signed 값:
            # negative = up
            # positive = down
            "brow_vertical_left": (
                left_brow_vertical[
                    "signed"
                ]
            ),

            "brow_vertical_right": (
                right_brow_vertical[
                    "signed"
                ]
            ),

            # DINO
            "dino_change_map": (
                dino_change
            ),

            "dino_change_mean": float(
                dino_mean
            ),

            "dino_change_max": float(
                dino_max
            ),

            "dino_inference_ms": float(
                self.dino.last_inference_ms
            ),
        }


    def capture_baseline(self):
        return (
            self.dino
            .capture_baseline()
        )


    def reset_baseline(self):
        self.dino.reset_baseline()


    @property
    def has_baseline(self):
        return (
            self.dino.has_baseline
        )


    def close(self):
        self.landmarker.close()
