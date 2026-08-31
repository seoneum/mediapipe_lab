from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import (
    FancyBboxPatch,
    Rectangle,
    Circle,
)


OUT = Path(
    "outputs/video_visuals/dino_theory_30s.mp4"
)
OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

FPS = 30
SECONDS = 30
TOTAL_FRAMES = FPS * SECONDS


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


# ============================================================
# Helpers
# ============================================================

artists = []


def add_artist(obj):
    artists.append(obj)
    return obj


def clear_scene():
    for obj in artists:
        try:
            obj.remove()
        except Exception:
            pass

    artists.clear()


def text(
    x,
    y,
    value,
    *,
    size=20,
    weight="normal",
    alpha=1.0,
    ha="left",
):
    obj = ax.text(
        x,
        y,
        value,
        fontsize=size,
        color="white",
        weight=weight,
        alpha=alpha,
        ha=ha,
        va="center",
    )

    return add_artist(obj)


def box(
    x,
    y,
    w,
    h,
    label=None,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=(
            "round,pad=0.04,"
            "rounding_size=0.15"
        ),
        edgecolor="#5d6879",
        facecolor="#121923",
        linewidth=1.5,
    )

    ax.add_patch(patch)
    add_artist(patch)

    if label:
        text(
            x + w / 2,
            y + h / 2,
            label,
            size=14,
            ha="center",
        )

    return patch


def arrow(
    x1,
    y1,
    x2,
    y2,
):
    obj = ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="->",
            linewidth=2,
            color="#8a96a8",
        ),
    )

    return add_artist(obj)


def draw_face(
    cx=4,
    cy=4.4,
    scale=1.0,
):
    theta = np.linspace(
        0,
        2 * np.pi,
        200,
    )

    line, = ax.plot(
        cx + 1.35 * scale * np.cos(theta),
        cy + 2.0 * scale * np.sin(theta),
        linewidth=2,
        color="#d0d5dd",
    )

    add_artist(line)

    # eyes
    for dx in (-0.55, 0.55):
        line, = ax.plot(
            [
                cx + dx - 0.2,
                cx + dx + 0.2,
            ],
            [
                cy + 0.45,
                cy + 0.45,
            ],
            linewidth=4,
            color="#d0d5dd",
        )
        add_artist(line)

    # mouth
    xs = np.linspace(
        cx - 0.6,
        cx + 0.6,
        80,
    )

    ys = (
        cy - 0.75
        - 0.12
        * np.cos(
            np.linspace(
                0,
                np.pi,
                80,
            )
        )
    )

    line, = ax.plot(
        xs,
        ys,
        linewidth=2.5,
        color="#d0d5dd",
    )

    add_artist(line)


def scene_title(
    title,
    subtitle,
):
    text(
        0.8,
        8.25,
        title,
        size=29,
        weight="bold",
    )

    text(
        0.8,
        7.78,
        subtitle,
        size=13,
        alpha=0.60,
    )


# ============================================================
# Scenes
# ============================================================

current_scene = None


def draw_scene(scene):
    global current_scene

    if scene == current_scene:
        return

    clear_scene()

    current_scene = scene

    # --------------------------------------------------------
    # Scene 0 : problem
    # --------------------------------------------------------

    if scene == 0:

        scene_title(
            "DINOv3",
            "How can a computer represent visual change?",
        )

        draw_face(
            cx=4.2,
            cy=4.2,
            scale=1.25,
        )

        text(
            8.0,
            5.6,
            "Human",
            size=17,
            weight="bold",
        )

        text(
            8.0,
            5.1,
            '"The eyebrow moved slightly."',
            size=21,
        )

        text(
            8.0,
            3.8,
            "Computer",
            size=17,
            weight="bold",
        )

        text(
            8.0,
            3.3,
            "Hundreds of thousands of RGB values",
            size=20,
        )

        text(
            8.0,
            2.4,
            "How do pixels become meaningful features?",
            size=19,
            weight="bold",
        )

    # --------------------------------------------------------
    # Scene 1 : patches
    # --------------------------------------------------------

    elif scene == 1:

        scene_title(
            "STEP 1 — Patch",
            "Vision Transformer divides an image into small patches.",
        )

        start_x = 2.0
        start_y = 1.9

        rows = 5
        cols = 5

        size = 0.85

        for r in range(rows):
            for c in range(cols):

                rect = Rectangle(
                    (
                        start_x + c * size,
                        start_y + r * size,
                    ),
                    size,
                    size,
                    edgecolor="#657286",
                    facecolor="#151d28",
                    linewidth=1,
                )

                ax.add_patch(rect)
                add_artist(rect)

        text(
            8.3,
            5.7,
            "Image",
            size=18,
            weight="bold",
        )

        arrow(
            9.0,
            5.1,
            9.0,
            4.2,
        )

        text(
            8.25,
            3.8,
            "Patch tokens",
            size=18,
            weight="bold",
        )

        text(
            8.25,
            3.1,
            "x₁  x₂  x₃  ...  xₙ",
            size=25,
        )

        text(
            8.25,
            2.2,
            "Each patch becomes one token.",
            size=16,
        )

    # --------------------------------------------------------
    # Scene 2 : embedding
    # --------------------------------------------------------

    elif scene == 2:

        scene_title(
            "STEP 2 — Embedding",
            "Pixels are transformed into a numerical representation.",
        )

        box(
            1.3,
            3.4,
            2.7,
            2.1,
            "IMAGE\nPATCH",
        )

        arrow(
            4.2,
            4.45,
            5.4,
            4.45,
        )

        box(
            5.5,
            3.4,
            2.5,
            2.1,
            "DINOv3\nENCODER",
        )

        arrow(
            8.2,
            4.45,
            9.4,
            4.45,
        )

        box(
            9.5,
            2.7,
            3.8,
            3.5,
        )

        values = [
            "+0.23",
            "-0.81",
            "+0.15",
            "+0.44",
            " ... ",
            "-0.18",
        ]

        for i, value in enumerate(values):

            text(
                10.7,
                5.55 - i * 0.52,
                value,
                size=15,
                ha="center",
            )

        text(
            11.4,
            1.85,
            "Visual embedding  z ∈ ℝᴰ",
            size=21,
            weight="bold",
            ha="center",
        )

    # --------------------------------------------------------
    # Scene 3 : self supervised
    # --------------------------------------------------------

    elif scene == 3:

        scene_title(
            "STEP 3 — Self-Supervised Learning",
            "Different views of the same image should have similar representations.",
        )

        box(
            1.1,
            5.0,
            2.3,
            1.4,
            "Original\nFace",
        )

        arrow(
            3.4,
            5.7,
            5.0,
            6.3,
        )

        arrow(
            3.4,
            5.7,
            5.0,
            4.3,
        )

        box(
            5.1,
            5.6,
            2.1,
            1.3,
            "Crop",
        )

        box(
            5.1,
            3.5,
            2.1,
            1.3,
            "Color / View",
        )

        arrow(
            7.3,
            6.2,
            8.5,
            6.2,
        )

        arrow(
            7.3,
            4.1,
            8.5,
            4.1,
        )

        box(
            8.6,
            5.55,
            2.1,
            1.3,
            "Student",
        )

        box(
            8.6,
            3.45,
            2.1,
            1.3,
            "Teacher",
        )

        arrow(
            10.8,
            6.2,
            12.1,
            5.3,
        )

        arrow(
            10.8,
            4.1,
            12.1,
            5.0,
        )

        text(
            12.25,
            5.15,
            "z₁ ≈ z₂",
            size=28,
            weight="bold",
        )

        text(
            8.0,
            1.9,
            "The model learns visual structure without manual labels.",
            size=18,
            weight="bold",
            ha="center",
        )

    # --------------------------------------------------------
    # Scene 4 : face use
    # --------------------------------------------------------

    elif scene == 4:

        scene_title(
            "DINOv3 in facial analysis",
            "Represent visual changes without assigning an emotion label.",
        )

        labels = [
            "BROW",
            "EYES",
            "NOSE",
            "MOUTH",
        ]

        for i, label in enumerate(labels):

            box(
                1.0,
                6.1 - i * 1.25,
                2.3,
                0.8,
                label,
            )

            arrow(
                3.4,
                6.5 - i * 1.25,
                4.4,
                6.5 - i * 1.25,
            )

            box(
                4.5,
                6.05 - i * 1.25,
                2.7,
                0.9,
                "DINO FEATURE",
            )

        box(
            9.0,
            5.1,
            2.1,
            1.2,
            "Neutral\nz₁",
        )

        box(
            9.0,
            2.9,
            2.1,
            1.2,
            "Movement\nz₂",
        )

        arrow(
            11.2,
            5.7,
            12.4,
            4.7,
        )

        arrow(
            11.2,
            3.5,
            12.4,
            4.4,
        )

        text(
            12.5,
            4.55,
            "distance(z₁, z₂)",
            size=21,
            weight="bold",
        )

        text(
            8.2,
            1.35,
            "Not: “What emotion is this?”",
            size=17,
        )

        text(
            8.2,
            0.85,
            "But: “What visually changed?”",
            size=22,
            weight="bold",
        )


# ============================================================
# Animation
# ============================================================

def update(frame):

    t = frame / FPS

    if t < 5:
        scene = 0

    elif t < 10:
        scene = 1

    elif t < 16:
        scene = 2

    elif t < 23:
        scene = 3

    else:
        scene = 4

    draw_scene(scene)

    return artists


animation = FuncAnimation(
    fig,
    update,
    frames=TOTAL_FRAMES,
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
