from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import FancyBboxPatch, Circle


OUT = Path(
    "outputs/video_visuals/tcn_theory_30s.mp4"
)

OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

FPS = 30
SECONDS = 30
TOTAL = FPS * SECONDS


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


artists = []
current_scene = None


def add(obj):
    artists.append(obj)
    return obj


def clear():
    for obj in artists:
        try:
            obj.remove()
        except Exception:
            pass

    artists.clear()


def txt(
    x,
    y,
    s,
    size=18,
    bold=False,
    alpha=1,
    ha="left",
):
    obj = ax.text(
        x,
        y,
        s,
        color="white",
        fontsize=size,
        weight=(
            "bold"
            if bold
            else "normal"
        ),
        alpha=alpha,
        ha=ha,
        va="center",
    )

    add(obj)

    return obj


def box(
    x,
    y,
    w,
    h,
    label,
):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=(
            "round,pad=0.04,"
            "rounding_size=0.14"
        ),
        facecolor="#121923",
        edgecolor="#647184",
        linewidth=1.4,
    )

    ax.add_patch(p)
    add(p)

    txt(
        x + w / 2,
        y + h / 2,
        label,
        size=13,
        bold=True,
        ha="center",
    )

    return p


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
            color="#8d99aa",
            linewidth=2,
        ),
    )

    add(obj)


def title(
    a,
    b,
):
    txt(
        0.8,
        8.25,
        a,
        size=29,
        bold=True,
    )

    txt(
        0.8,
        7.75,
        b,
        size=13,
        alpha=0.58,
    )


def node(
    x,
    y,
    r=0.11,
):
    c = Circle(
        (x, y),
        r,
        facecolor="#d0d6df",
        edgecolor="none",
    )

    ax.add_patch(c)
    add(c)

    return c


def line(
    x1,
    y1,
    x2,
    y2,
    alpha=0.5,
):
    obj, = ax.plot(
        [x1, x2],
        [y1, y2],
        linewidth=1.3,
        alpha=alpha,
        color="#718096",
    )

    add(obj)


def draw_scene(scene):

    global current_scene

    if scene == current_scene:
        return

    clear()
    current_scene = scene

    # ========================================================
    # Scene 0
    # ========================================================

    if scene == 0:

        title(
            "Why Temporal Models?",
            "A movement cannot be understood from one frame.",
        )

        box(
            2.0,
            3.0,
            2.5,
            2.5,
            "FRAME t\n\n😐",
        )

        txt(
            6.1,
            5.4,
            "One frame",
            size=19,
            bold=True,
        )

        txt(
            6.1,
            4.7,
            "Position can be observed.",
            size=18,
        )

        txt(
            6.1,
            3.9,
            "But movement requires",
            size=18,
        )

        txt(
            6.1,
            3.25,
            "CHANGE OVER TIME",
            size=25,
            bold=True,
        )

    # ========================================================
    # Scene 1 temporal convolution
    # ========================================================

    elif scene == 1:

        title(
            "Temporal Convolution",
            "Convolution is applied along the time axis.",
        )

        xs = np.linspace(
            2,
            8,
            7,
        )

        for i, x in enumerate(xs):

            node(
                x,
                5.5,
            )

            txt(
                x,
                4.95,
                f"xₜ₋{6-i}",
                size=12,
                ha="center",
            )

        line(
            xs[-3],
            5.3,
            6.8,
            3.9,
        )

        line(
            xs[-2],
            5.3,
            6.8,
            3.9,
        )

        line(
            xs[-1],
            5.3,
            6.8,
            3.9,
        )

        box(
            5.8,
            3.0,
            2.0,
            1.0,
            "Conv",
        )

        arrow(
            6.8,
            3.0,
            6.8,
            2.1,
        )

        txt(
            6.8,
            1.7,
            "hₜ",
            size=26,
            bold=True,
            ha="center",
        )

        txt(
            10.1,
            4.8,
            "hₜ = Σ wₖ xₜ₋ₖ",
            size=25,
            bold=True,
        )

        txt(
            10.1,
            3.8,
            "Nearby frames jointly",
            size=17,
        )

        txt(
            10.1,
            3.25,
            "describe temporal change.",
            size=17,
        )

    # ========================================================
    # Scene 2 causal
    # ========================================================

    elif scene == 2:

        title(
            "Causal TCN",
            "Real-time inference must not use future frames.",
        )

        xs = np.linspace(
            1.2,
            14.6,
            14,
        )

        current = 8

        for i, x in enumerate(xs):

            c = node(
                x,
                4.7,
            )

            if i > current:
                c.set_alpha(
                    0.18
                )

            if i == current:
                c.set_radius(
                    0.17
                )

        txt(
            xs[1],
            5.45,
            "PAST",
            size=14,
            alpha=0.65,
        )

        txt(
            xs[current],
            5.45,
            "NOW",
            size=14,
            bold=True,
            ha="center",
        )

        txt(
            xs[-3],
            5.45,
            "FUTURE",
            size=14,
            alpha=0.35,
        )

        line(
            xs[0],
            3.8,
            xs[current],
            3.8,
            alpha=0.85,
        )

        txt(
            5.3,
            3.2,
            "USED",
            size=18,
            bold=True,
        )

        line(
            xs[current + 1],
            3.8,
            xs[-1],
            3.8,
            alpha=0.2,
        )

        txt(
            12.1,
            3.2,
            "NOT USED",
            size=18,
            alpha=0.35,
        )

        txt(
            8.0,
            1.8,
            "Output at time t depends only on x≤t",
            size=23,
            bold=True,
            ha="center",
        )

    # ========================================================
    # Scene 3 dilation
    # ========================================================

    elif scene == 3:

        title(
            "Dilated Convolution",
            "A larger temporal receptive field with fewer layers.",
        )

        layers = [
            (
                6.1,
                1,
            ),
            (
                4.6,
                2,
            ),
            (
                3.1,
                4,
            ),
        ]

        xs = np.linspace(
            2,
            13.5,
            13,
        )

        for y, dilation in layers:

            txt(
                0.8,
                y,
                f"d={dilation}",
                size=13,
                alpha=0.7,
            )

            for i, x in enumerate(xs):

                if i % dilation == 0:

                    node(
                        x,
                        y,
                        r=0.08,
                    )

        txt(
            8.0,
            1.4,
            "small network  →  long temporal context",
            size=22,
            bold=True,
            ha="center",
        )

    # ========================================================
    # Scene 4 ON DAMM architecture
    # ========================================================

    elif scene == 4:

        title(
            "ON DAMM Temporal Encoder",
            "Approximately two seconds of facial motion are represented as one vector.",
        )

        box(
            0.8,
            4.0,
            2.8,
            1.8,
            "79D FEATURES\n× 60 FRAMES",
        )

        arrow(
            3.7,
            4.9,
            5.0,
            4.9,
        )

        box(
            5.1,
            4.0,
            2.5,
            1.8,
            "CAUSAL\nTCN",
        )

        arrow(
            7.7,
            4.9,
            8.8,
            4.9,
        )

        box(
            8.9,
            4.0,
            2.0,
            1.8,
            "64D",
        )

        arrow(
            11.0,
            4.9,
            12.0,
            4.9,
        )

        box(
            12.1,
            3.65,
            2.7,
            2.5,
            "PERSONAL\nMETRIC\n64D → 32D",
        )

        txt(
            8.0,
            2.35,
            "A temporal episode becomes a point",
            size=19,
            ha="center",
        )

        txt(
            8.0,
            1.75,
            "in a personalized embedding space.",
            size=22,
            bold=True,
            ha="center",
        )

    # ========================================================
    # Scene 5 pattern space
    # ========================================================

    elif scene == 5:

        title(
            "Pattern Memory",
            "Repeated episodes form nearby clusters.",
        )

        rng = np.random.default_rng(
            42
        )

        center_a = np.array(
            [5.0, 4.5]
        )

        center_b = np.array(
            [11.0, 4.2]
        )

        A = (
            center_a
            + rng.normal(
                0,
                0.35,
                size=(7, 2),
            )
        )

        B = (
            center_b
            + rng.normal(
                0,
                0.4,
                size=(6, 2),
            )
        )

        for p in A:

            node(
                p[0],
                p[1],
                r=0.10,
            )

        for p in B:

            c = node(
                p[0],
                p[1],
                r=0.10,
            )

            c.set_alpha(
                0.45
            )

        txt(
            5.0,
            3.2,
            "Pattern A",
            size=15,
            ha="center",
        )

        txt(
            11.0,
            3.0,
            "Pattern B",
            size=15,
            alpha=0.6,
            ha="center",
        )

        # new occurrences

        for i, p in enumerate(
            [
                (
                    2.3,
                    6.7,
                ),
                (
                    2.8,
                    6.35,
                ),
                (
                    3.25,
                    6.7,
                ),
            ]
        ):

            txt(
                p[0],
                p[1],
                "×",
                size=26,
                bold=True,
                ha="center",
            )

            txt(
                p[0],
                p[1] - 0.45,
                str(
                    i + 1
                ),
                size=11,
                ha="center",
            )

        arrow(
            3.6,
            6.2,
            4.5,
            5.2,
        )

        txt(
            8.0,
            1.5,
            "Same temporal behavior → small cosine distance",
            size=22,
            bold=True,
            ha="center",
        )

        txt(
            8.0,
            0.85,
            "Repeated candidate → human review",
            size=18,
            ha="center",
        )


def update(frame):

    t = (
        frame
        / FPS
    )

    if t < 5:
        scene = 0

    elif t < 10:
        scene = 1

    elif t < 15:
        scene = 2

    elif t < 20:
        scene = 3

    elif t < 25:
        scene = 4

    else:
        scene = 5

    draw_scene(
        scene
    )

    return artists


animation = FuncAnimation(
    fig,
    update,
    frames=TOTAL,
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

print(
    "saved:",
    OUT,
)
