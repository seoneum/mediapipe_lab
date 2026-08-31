from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import FancyBboxPatch


# ============================================================
# CONFIG
# ============================================================

OUT = Path("outputs/video_visuals/dino_visual.mp4")
OUT.parent.mkdir(parents=True, exist_ok=True)

FPS = 30
SECONDS = 8
FRAMES = FPS * SECONDS

REGIONS = [
    "GLOBAL",
    "BROW",
    "EYES",
    "NOSE",
    "MOUTH",
]

rng = np.random.default_rng(42)

# 소개 영상용 feature visualization.
# 실제 DINO vector를 표현하는 "개념 시각화"이며
# 측정 결과 자체를 의미하지 않는다.
base_vectors = {
    region: rng.normal(
        0,
        1,
        size=32,
    )
    for region in REGIONS
}


# ============================================================
# FIGURE
# ============================================================

fig = plt.figure(
    figsize=(16, 9),
    facecolor="#080b10",
)

ax = fig.add_axes([0, 0, 1, 1])

ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")

fig.text(
    0.05,
    0.92,
    "DINOv3",
    fontsize=32,
    color="white",
    weight="bold",
)

fig.text(
    0.05,
    0.875,
    "Region-level visual feature extraction",
    fontsize=15,
    color="#9aa4b2",
)


# ============================================================
# FACE
# ============================================================

face_box = FancyBboxPatch(
    (0.8, 1.3),
    5.0,
    6.6,
    boxstyle="round,pad=0.04,rounding_size=0.25",
    linewidth=1.5,
    edgecolor="#525c6b",
    facecolor="#111722",
)

ax.add_patch(face_box)

ax.text(
    3.3,
    7.45,
    "FACE FRAME",
    ha="center",
    color="#a8b1bf",
    fontsize=11,
)


# 얼굴 윤곽
theta = np.linspace(0, 2 * np.pi, 200)

face_x = 3.3 + 1.45 * np.cos(theta)
face_y = 4.6 + 2.25 * np.sin(theta)

ax.plot(
    face_x,
    face_y,
    linewidth=2,
    color="#d4d9e1",
)

# 눈
ax.plot(
    [2.35, 2.75],
    [5.15, 5.18],
    linewidth=4,
    solid_capstyle="round",
    color="#d4d9e1",
)

ax.plot(
    [3.85, 4.25],
    [5.18, 5.15],
    linewidth=4,
    solid_capstyle="round",
    color="#d4d9e1",
)

# 눈썹
ax.plot(
    [2.25, 2.8],
    [5.75, 5.85],
    linewidth=2,
    color="#d4d9e1",
)

ax.plot(
    [3.8, 4.35],
    [5.85, 5.75],
    linewidth=2,
    color="#d4d9e1",
)

# 코
ax.plot(
    [3.3, 3.15, 3.42],
    [5.0, 4.25, 4.2],
    linewidth=1.5,
    color="#d4d9e1",
)

# 입
mouth_x = np.linspace(2.65, 3.95, 100)

mouth_y = (
    3.55
    - 0.16
    * np.cos(
        np.linspace(0, np.pi, 100)
    )
)

ax.plot(
    mouth_x,
    mouth_y,
    linewidth=2.5,
    color="#d4d9e1",
)


# ============================================================
# REGION POINTS
# ============================================================

region_points = {
    "GLOBAL": (4.8, 6.9),
    "BROW": (4.4, 5.85),
    "EYES": (4.45, 5.2),
    "NOSE": (4.3, 4.35),
    "MOUTH": (4.15, 3.55),
}

region_lines = {}

for i, region in enumerate(REGIONS):

    y = 7.0 - i * 1.18

    box = FancyBboxPatch(
        (7.0, y - 0.38),
        7.7,
        0.82,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        linewidth=1,
        edgecolor="#394253",
        facecolor="#101621",
    )

    ax.add_patch(box)

    ax.text(
        7.3,
        y,
        region,
        va="center",
        color="#e2e6ec",
        fontsize=11,
        weight="bold",
    )

    x0, y0 = region_points[region]

    line, = ax.plot(
        [x0, 7.0],
        [y0, y],
        linewidth=1.5,
        alpha=0.35,
        color="#7f8fa6",
    )

    region_lines[region] = line


# ============================================================
# FEATURE BARS
# ============================================================

feature_lines = {}

for i, region in enumerate(REGIONS):

    y = 7.0 - i * 1.18

    xs = np.linspace(
        8.8,
        14.2,
        32,
    )

    line, = ax.plot(
        xs,
        np.full_like(xs, y),
        linewidth=5,
        solid_capstyle="butt",
    )

    feature_lines[region] = (
        xs,
        line,
    )


status_text = ax.text(
    7.0,
    0.65,
    "",
    color="#9aa4b2",
    fontsize=13,
)

embedding_text = ax.text(
    14.7,
    4.55,
    "",
    rotation=90,
    ha="center",
    va="center",
    color="#c7ced8",
    fontsize=12,
)


# ============================================================
# UPDATE
# ============================================================

def update(frame_idx):

    t = frame_idx / FPS

    # feature들이 0~1.5초 사이에 등장
    appear = np.clip(
        (t - 0.8) / 1.5,
        0,
        1,
    )

    for i, region in enumerate(REGIONS):

        xs, line = feature_lines[region]

        vec = base_vectors[region].copy()

        # 살아 움직이는 느낌
        vec += (
            0.35
            * np.sin(
                t * 2.0
                + np.arange(32) * 0.34
                + i
            )
        )

        vec = np.tanh(vec)

        y_base = 7.0 - i * 1.18

        ys = (
            y_base
            + vec
            * 0.28
            * appear
        )

        line.set_data(
            xs,
            ys,
        )

        line.set_alpha(
            0.15
            + 0.85 * appear
        )

        region_lines[
            region
        ].set_alpha(
            0.15
            + 0.45 * appear
        )

    if t < 0.8:

        status = "Detecting facial regions..."

    elif t < 2.5:

        status = "Extracting region-level visual representations"

    elif t < 5.5:

        status = "DINOv3  →  high-dimensional visual features"

    else:

        status = "Visual change represented as embeddings"

    status_text.set_text(status)

    embedding_text.set_text(
        "VISUAL\nEMBEDDING"
        if t > 4
        else ""
    )

    return []


animation = FuncAnimation(
    fig,
    update,
    frames=FRAMES,
    interval=1000 / FPS,
)

writer = FFMpegWriter(
    fps=FPS,
    bitrate=8000,
)

animation.save(
    OUT,
    writer=writer,
    dpi=140,
)

plt.close(fig)

print("saved:", OUT)
