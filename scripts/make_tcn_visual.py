from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import FancyBboxPatch, Circle


# ============================================================
# CONFIG
# ============================================================

OUT = Path(
    "outputs/video_visuals/tcn_visual.mp4"
)

OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

FPS = 30
SECONDS = 10
N_FRAMES = FPS * SECONDS

rng = np.random.default_rng(7)


# ============================================================
# FIGURE
# ============================================================

fig = plt.figure(
    figsize=(16, 9),
    facecolor="#080b10",
)

ax = fig.add_axes(
    [0, 0, 1, 1]
)

ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")


fig.text(
    0.05,
    0.925,
    "CAUSAL TCN",
    color="white",
    fontsize=30,
    weight="bold",
)

fig.text(
    0.05,
    0.88,
    "Temporal facial-pattern representation",
    color="#9aa4b2",
    fontsize=15,
)


# ============================================================
# TIMELINE
# ============================================================

timeline_y = 7.1

ax.plot(
    [0.9, 6.3],
    [timeline_y, timeline_y],
    linewidth=2,
    color="#3b4655",
)


FRAME_COUNT = 12

frame_xs = np.linspace(
    1.0,
    6.2,
    FRAME_COUNT,
)

timeline_circles = []

for i, x in enumerate(frame_xs):

    circle = Circle(
        (x, timeline_y),
        0.10,
        facecolor="#171f2c",
        edgecolor="#657087",
        linewidth=1,
    )

    ax.add_patch(circle)

    timeline_circles.append(
        circle
    )


ax.text(
    0.9,
    7.65,
    "60 frames  ≈  2 seconds",
    color="#c7ced8",
    fontsize=13,
)

ax.text(
    0.9,
    6.55,
    "past",
    color="#667286",
    fontsize=10,
)

ax.text(
    5.85,
    6.55,
    "now",
    color="#667286",
    fontsize=10,
)


# ============================================================
# CAUSAL ARROW
# ============================================================

ax.annotate(
    "",
    xy=(6.55, timeline_y),
    xytext=(6.0, timeline_y),
    arrowprops=dict(
        arrowstyle="->",
        linewidth=2,
        color="#9aa4b2",
    ),
)


# ============================================================
# TCN BLOCKS
# ============================================================

blocks = []

block_specs = [
    (
        7.0,
        6.55,
        1.25,
        1.15,
        "Dilated\nConv 1",
    ),
    (
        8.7,
        5.95,
        1.25,
        1.15,
        "Dilated\nConv 2",
    ),
    (
        10.4,
        5.35,
        1.25,
        1.15,
        "Temporal\nFeatures",
    ),
]

for x, y, w, h, label in block_specs:

    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=(
            "round,pad=0.04,"
            "rounding_size=0.15"
        ),
        linewidth=1.3,
        edgecolor="#526071",
        facecolor="#111923",
    )

    ax.add_patch(box)

    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        color="#e0e5eb",
        fontsize=11,
    )

    blocks.append(box)


# 연결선
ax.annotate(
    "",
    xy=(8.7, 6.5),
    xytext=(8.25, 6.9),
    arrowprops=dict(
        arrowstyle="->",
        color="#606d80",
        linewidth=1.5,
    ),
)

ax.annotate(
    "",
    xy=(10.4, 5.9),
    xytext=(9.95, 6.3),
    arrowprops=dict(
        arrowstyle="->",
        color="#606d80",
        linewidth=1.5,
    ),
)


# ============================================================
# 64D
# ============================================================

box_64 = FancyBboxPatch(
    (12.0, 5.1),
    1.25,
    1.25,
    boxstyle="round,pad=0.04,rounding_size=0.15",
    linewidth=1.4,
    edgecolor="#647388",
    facecolor="#151d29",
)

ax.add_patch(box_64)

ax.text(
    12.625,
    5.72,
    "64D",
    ha="center",
    va="center",
    color="white",
    fontsize=18,
    weight="bold",
)

ax.text(
    12.625,
    5.28,
    "TCN embedding",
    ha="center",
    color="#8995a7",
    fontsize=8,
)


# ============================================================
# PERSONALIZED METRIC
# ============================================================

metric_box = FancyBboxPatch(
    (13.7, 4.85),
    1.45,
    1.75,
    boxstyle="round,pad=0.04,rounding_size=0.16",
    linewidth=1.3,
    edgecolor="#7b8495",
    facecolor="#181e28",
)

ax.add_patch(metric_box)

ax.text(
    14.425,
    5.95,
    "PERSONAL",
    ha="center",
    color="#cbd2dc",
    fontsize=10,
    weight="bold",
)

ax.text(
    14.425,
    5.60,
    "METRIC",
    ha="center",
    color="#cbd2dc",
    fontsize=10,
    weight="bold",
)

ax.text(
    14.425,
    5.12,
    "64D → 32D",
    ha="center",
    color="#8d99aa",
    fontsize=9,
)


ax.annotate(
    "",
    xy=(13.7, 5.7),
    xytext=(13.25, 5.7),
    arrowprops=dict(
        arrowstyle="->",
        color="#657286",
        linewidth=1.5,
    ),
)


# ============================================================
# CLUSTER SPACE
# ============================================================

cluster_center = np.array(
    [12.7, 2.35]
)

cluster_points = np.array(
    [
        [12.25, 2.20],
        [12.55, 2.55],
        [12.85, 2.15],
        [13.05, 2.45],
        [12.65, 1.95],
    ]
)

other_cluster = np.array(
    [
        [14.25, 2.1],
        [14.55, 2.35],
        [14.35, 2.65],
        [14.75, 2.05],
    ]
)

point_artists = []

for p in cluster_points:

    artist = ax.scatter(
        p[0],
        p[1],
        s=65,
        alpha=0.0,
    )

    point_artists.append(
        artist
    )

other_artists = []

for p in other_cluster:

    artist = ax.scatter(
        p[0],
        p[1],
        s=55,
        alpha=0.0,
        marker="s",
    )

    other_artists.append(
        artist
    )


new_point = ax.scatter(
    [10.8],
    [1.4],
    s=180,
    marker="x",
    linewidth=3,
    alpha=0.0,
)


ax.text(
    11.0,
    3.25,
    "Personalized pattern space",
    color="#c6ced8",
    fontsize=12,
)

cluster_label = ax.text(
    12.65,
    1.4,
    "",
    ha="center",
    color="#adb7c5",
    fontsize=10,
)

unknown_label = ax.text(
    10.25,
    1.0,
    "",
    color="#adb7c5",
    fontsize=10,
)


status = ax.text(
    0.9,
    0.55,
    "",
    color="#9aa4b2",
    fontsize=13,
)


# ============================================================
# UPDATE
# ============================================================

def smoothstep(x):
    x = np.clip(x, 0, 1)

    return (
        x
        * x
        * (3 - 2 * x)
    )


def update(frame_idx):

    t = frame_idx / FPS

    # --------------------------------------------------------
    # Timeline
    # --------------------------------------------------------

    for i, circle in enumerate(
        timeline_circles
    ):

        trigger = (
            0.6
            + i * 0.08
        )

        alpha = smoothstep(
            (t - trigger) / 0.35
        )

        # 움직임 구간 강조
        motion_strength = (
            np.exp(
                -0.5
                * (
                    (i - 7)
                    / 2.0
                ) ** 2
            )
        )

        base = (
            0.12
            + 0.58
            * motion_strength
            * alpha
        )

        circle.set_facecolor(
            (
                base,
                base + 0.08,
                base + 0.14,
            )
        )

    # --------------------------------------------------------
    # TCN blocks
    # --------------------------------------------------------

    for i, block in enumerate(
        blocks
    ):

        start = (
            2.2
            + i * 0.55
        )

        strength = smoothstep(
            (t - start) / 0.4
        )

        block.set_alpha(
            0.35
            + 0.65 * strength
        )

    # --------------------------------------------------------
    # clusters
    # --------------------------------------------------------

    cluster_alpha = smoothstep(
        (t - 5.0) / 1.0
    )

    for artist in point_artists:
        artist.set_alpha(
            0.75
            * cluster_alpha
        )

    for artist in other_artists:
        artist.set_alpha(
            0.55
            * cluster_alpha
        )

    # --------------------------------------------------------
    # New embedding moves into cluster
    # --------------------------------------------------------

    move = smoothstep(
        (t - 6.3) / 2.0
    )

    start = np.array(
        [10.8, 1.4]
    )

    end = np.array(
        [12.55, 2.28]
    )

    pos = (
        start * (1 - move)
        + end * move
    )

    new_point.set_offsets(
        pos.reshape(1, 2)
    )

    new_point.set_alpha(
        smoothstep(
            (t - 5.8) / 0.5
        )
    )

    # --------------------------------------------------------
    # labels
    # --------------------------------------------------------

    if t < 2.0:

        text = (
            "Collecting recent facial movement"
        )

    elif t < 4.2:

        text = (
            "Causal convolutions model "
            "temporal change"
        )

    elif t < 5.8:

        text = (
            "Temporal sequence → "
            "64D representation"
        )

    elif t < 7.8:

        text = (
            "Child-specific metric projection "
            "→ 32D"
        )

    else:

        text = (
            "Repeated movement patterns "
            "form nearby clusters"
        )

    status.set_text(text)

    if t > 7.5:

        cluster_label.set_text(
            "known / repeating pattern"
        )

        unknown_label.set_text(
            "new episode"
        )

    else:

        cluster_label.set_text("")
        unknown_label.set_text("")

    return []


animation = FuncAnimation(
    fig,
    update,
    frames=N_FRAMES,
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
