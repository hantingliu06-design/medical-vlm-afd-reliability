from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
from matplotlib.transforms import Bbox


OUTPUT_DIR = Path("latex_template") / "Images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)

INK = "#17365D"
MUTED = "#667085"
ARROW = "#52647A"
BLUE = "#315B9A"
BLUE_TINT = "#F2F6FC"
ORANGE = "#B36B18"
ORANGE_TINT = "#FFF7E8"
GREEN = "#3D7E2B"
GREEN_TINT = "#F1F8EE"
RED = "#B44335"
RED_TINT = "#FDF2F0"
PURPLE = "#7042A0"
PURPLE_TINT = "#F7F2FB"

FIT_CHECKS = []


def rounded(ax, x, y, width, height, face, edge, line=1.0, radius=0.012):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        linewidth=line,
        edgecolor=edge,
        facecolor=face,
        zorder=2,
    )
    ax.add_patch(patch)
    return patch


def text_in_patch(ax, patch, x, y, text, size, bold=False, linespacing=1.08):
    artist = ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=size,
        fontweight="bold" if bold else "normal",
        color=INK,
        linespacing=linespacing,
        zorder=4,
    )
    FIT_CHECKS.append((patch, artist))
    return artist


def badge(ax, x, y, label, colour):
    pill = rounded(ax, x, y, 0.025, 0.034, colour, colour, line=0.8, radius=0.009)
    artist = ax.text(
        x + 0.0125,
        y + 0.017,
        label,
        ha="center",
        va="center",
        fontsize=5.8,
        fontweight="bold",
        color="white",
        zorder=5,
    )
    FIT_CHECKS.append((pill, artist))


def section(ax, x, y, width, height, title, stage, face, edge, title_size=9.0):
    patch = rounded(ax, x, y, width, height, face, edge, line=1.1, radius=0.014)
    title_artist = ax.text(
        x + width / 2,
        y + height - 0.033,
        title,
        ha="center",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color=INK,
        linespacing=1.0,
        zorder=4,
    )
    FIT_CHECKS.append((patch, title_artist))
    return patch


def chip(ax, x, y, width, height, text, edge=BLUE, face="white", size=7.4, bold=False):
    patch = rounded(ax, x, y, width, height, face, edge, line=0.72, radius=0.008)
    text_in_patch(ax, patch, x + width / 2, y + height / 2, text, size, bold=bold)
    return patch


def arrow(ax, start, end, colour=ARROW, line=1.1):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=line,
            color=colour,
            shrinkA=0,
            shrinkB=0,
            zorder=3,
        )
    )


def orthogonal_arrow(ax, points, colour=ARROW, line=1.1):
    # Every start/end coordinate is deliberately offset from its nearby frame.
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    ax.plot(xs[:-1], ys[:-1], color=colour, linewidth=line, zorder=1.5)
    arrow(ax, points[-2], points[-1], colour=colour, line=line)


def validate_text_fit(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    failures = []
    for patch, artist in FIT_CHECKS:
        patch_box = patch.get_window_extent(renderer)
        safe_box = Bbox.from_extents(
            patch_box.x0 + 3,
            patch_box.y0 + 3,
            patch_box.x1 - 3,
            patch_box.y1 - 3,
        )
        text_box = artist.get_window_extent(renderer)
        if not (
            safe_box.contains(text_box.x0, text_box.y0)
            and safe_box.contains(text_box.x1, text_box.y1)
        ):
            failures.append(artist.get_text())
    if failures:
        raise RuntimeError("Text exceeded its card boundary: " + "; ".join(failures))


def draw_framework():
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Main pipeline. Gaps are part of the layout and are never occupied by cards.
    section(ax, 0.020, 0.350, 0.110, 0.320, "Medical VQA", "01", BLUE_TINT, BLUE, 8.8)
    chip(ax, 0.038, 0.505, 0.074, 0.058, "Image", size=7.8)
    chip(ax, 0.038, 0.425, 0.074, 0.058, "Question", size=7.8)
    ax.text(0.075, 0.382, "optional context", ha="center", va="center", fontsize=5.8, color=MUTED)

    prompt = rounded(ax, 0.170, 0.425, 0.085, 0.160, ORANGE_TINT, ORANGE, line=1.1, radius=0.014)
    text_in_patch(ax, prompt, 0.2125, 0.505, "Prompt", 8.5, bold=True)

    section(ax, 0.295, 0.285, 0.140, 0.440, "Zero-Shot VLMs", "03", BLUE_TINT, BLUE, 8.5)
    chip(ax, 0.315, 0.545, 0.100, 0.060, "Qwen2.5-VL-3B", size=6.9)
    chip(ax, 0.315, 0.455, 0.100, 0.060, "LLaVA-1.5-7B", size=6.9)
    chip(ax, 0.315, 0.365, 0.100, 0.060, "MedGemma-4B", size=6.9)

    section(ax, 0.475, 0.325, 0.130, 0.370, "Answer\nGeneration", "04", BLUE_TINT, BLUE, 7.5)
    chip(ax, 0.495, 0.515, 0.090, 0.075, "1 Greedy\nAnswer", size=6.8, bold=True)
    chip(ax, 0.495, 0.405, 0.090, 0.075, "3 Sampled\nAnswers", size=6.8, bold=True)

    # Parallel evaluation outcomes.
    section(ax, 0.660, 0.700, 0.180, 0.235, "Answer Quality", "A", GREEN_TINT, GREEN, 8.8)
    chip(ax, 0.678, 0.800, 0.067, 0.050, "BLEU-1", edge=GREEN, size=7.0)
    chip(ax, 0.757, 0.800, 0.067, 0.050, "BLEU-2", edge=GREEN, size=7.0)
    chip(ax, 0.678, 0.730, 0.067, 0.050, "ROUGE-L", edge=GREEN, size=7.0)
    chip(ax, 0.757, 0.730, 0.067, 0.050, "METEOR", edge=GREEN, size=7.0)

    section(ax, 0.640, 0.085, 0.200, 0.500, "Reliability Analysis", "B", RED_TINT, RED, 8.5)
    chip(
        ax,
        0.660,
        0.395,
        0.160,
        0.075,
        "Failure label\nROUGE-L < 0.2 AND\nMETEOR < 0.1",
        edge=PURPLE,
        face=PURPLE_TINT,
        size=5.7,
        bold=True,
    )
    chip(
        ax,
        0.660,
        0.255,
        0.160,
        0.105,
        "Detector scores\nAFD frequency\nSemantic AFD\nQuestion-aligned uncertainty",
        edge=RED,
        size=5.5,
        bold=True,
    )
    chip(
        ax,
        0.680,
        0.130,
        0.120,
        0.075,
        "AUROC  ·  AUPRC\nSelective coverage",
        edge=PURPLE,
        face=PURPLE_TINT,
        size=6.5,
        bold=True,
    )

    # Threshold decision. It is visually separated from offline evaluation.
    diamond = Polygon(
        [[0.915, 0.440], [0.960, 0.350], [0.915, 0.260], [0.870, 0.350]],
        closed=True,
        facecolor=BLUE_TINT,
        edgecolor=BLUE,
        linewidth=1.05,
        zorder=2,
    )
    ax.add_patch(diamond)
    decision_text = ax.text(0.915, 0.350, "$s_i>t$?", ha="center", va="center", fontsize=7.8, fontweight="bold", color=INK, zorder=4)
    FIT_CHECKS.append((diamond, decision_text))
    chip(ax, 0.875, 0.610, 0.080, 0.065, "Accept", edge=GREEN, face=GREEN_TINT, size=7.2, bold=True)
    chip(ax, 0.860, 0.065, 0.110, 0.085, "Refer for\nHuman Review", edge=ORANGE, face=ORANGE_TINT, size=6.6, bold=True)

    # Floating arrows: every endpoint stays visibly clear of a frame.
    arrow(ax, (0.138, 0.505), (0.162, 0.505))
    arrow(ax, (0.263, 0.505), (0.287, 0.505))
    arrow(ax, (0.443, 0.505), (0.467, 0.505))

    orthogonal_arrow(ax, [(0.613, 0.552), (0.630, 0.552), (0.630, 0.825), (0.652, 0.825)])
    orthogonal_arrow(ax, [(0.613, 0.552), (0.618, 0.552), (0.618, 0.432), (0.632, 0.432)])
    orthogonal_arrow(ax, [(0.613, 0.442), (0.608, 0.442), (0.608, 0.312), (0.632, 0.312)])

    arrow(ax, (0.848, 0.312), (0.862, 0.344))
    arrow(ax, (0.915, 0.448), (0.915, 0.602))
    arrow(ax, (0.915, 0.252), (0.915, 0.158))
    ax.text(0.932, 0.525, "No", ha="left", va="center", fontsize=6.2, color=INK)
    ax.text(0.932, 0.205, "Yes", ha="left", va="center", fontsize=6.2, color=INK)

    validate_text_fit(fig)
    fig.savefig(OUTPUT_DIR / "reliability_centred_vlm_framework_polished_draft.pdf", bbox_inches="tight", pad_inches=0.06)
    fig.savefig(OUTPUT_DIR / "reliability_centred_vlm_framework_polished_draft.png", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


if __name__ == "__main__":
    draw_framework()
    print("Saved polished methodology framework to", OUTPUT_DIR)
