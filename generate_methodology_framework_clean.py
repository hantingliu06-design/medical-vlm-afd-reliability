from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


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
BLUE = "#315B9A"
BLUE_TINT = "#EFF5FC"
ORANGE = "#B36B18"
ORANGE_TINT = "#FFF7E8"
GREEN = "#3D7E2B"
GREEN_TINT = "#EFF8EC"
RED = "#B44335"
RED_TINT = "#FDF1EF"
PURPLE = "#7042A0"
PURPLE_TINT = "#F6F0FB"
GREY = "#667085"
GREY_TINT = "#F5F6F8"


def rounded(ax, x, y, width, height, face, edge, line=1.05, radius=0.012):
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


def section(ax, x, y, width, height, title, face=BLUE_TINT, edge=BLUE, title_size=10):
    rounded(ax, x, y, width, height, face, edge, line=1.15, radius=0.014)
    ax.text(
        x + width / 2,
        y + height - 0.038,
        title,
        ha="center",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color=INK,
        zorder=4,
    )


def chip(ax, x, y, width, height, text, edge=BLUE, face="white", size=8, bold=False):
    rounded(ax, x, y, width, height, face, edge, line=0.75, radius=0.008)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        fontweight="bold" if bold else "normal",
        color=INK,
        linespacing=1.08,
        zorder=4,
    )


def arrow(ax, start, end, colour=INK, line=1.15, connection="arc3"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=line,
            color=colour,
            connectionstyle=connection,
            shrinkA=0,
            shrinkB=0,
            zorder=3,
        )
    )


def orthogonal_arrow(ax, points, colour=INK, line=1.15):
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    ax.plot(x_values[:-1], y_values[:-1], color=colour, linewidth=line, zorder=1.5)
    arrow(ax, points[-2], points[-1], colour=colour, line=line)


def draw_framework():
    fig, ax = plt.subplots(figsize=(11.5, 5.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Main left-to-right path.
    section(ax, 0.020, 0.345, 0.120, 0.330, "Medical VQA", title_size=9.0)
    chip(ax, 0.038, 0.510, 0.084, 0.060, "Image", size=8.3)
    chip(ax, 0.038, 0.430, 0.084, 0.060, "Question", size=8.3)
    ax.text(0.080, 0.380, "optional context", ha="center", va="center", fontsize=6.0, color=GREY)

    rounded(ax, 0.175, 0.425, 0.095, 0.160, ORANGE_TINT, ORANGE, line=1.1, radius=0.014)
    ax.text(0.2225, 0.505, "Prompt", ha="center", va="center", fontsize=8.5, fontweight="bold", color=INK)

    section(ax, 0.310, 0.275, 0.150, 0.460, "Zero-Shot VLMs", title_size=8.8)
    chip(ax, 0.330, 0.550, 0.110, 0.064, "Qwen2.5-VL-3B", size=7.2)
    chip(ax, 0.330, 0.455, 0.110, 0.064, "LLaVA-1.5-7B", size=7.2)
    chip(ax, 0.330, 0.360, 0.110, 0.064, "MedGemma-4B", size=7.2)

    section(ax, 0.500, 0.315, 0.140, 0.380, "Answer Generation", title_size=8.3)
    chip(ax, 0.520, 0.520, 0.100, 0.078, "1 Greedy\nAnswer", size=7.0, bold=True)
    chip(ax, 0.520, 0.405, 0.100, 0.078, "3 Sampled\nAnswers", size=7.0, bold=True)

    # Answer-quality branch.
    section(ax, 0.680, 0.685, 0.185, 0.245, "Answer Quality", face=GREEN_TINT, edge=GREEN, title_size=9.0)
    chip(ax, 0.697, 0.795, 0.070, 0.052, "BLEU-1", edge=GREEN, size=7.3)
    chip(ax, 0.778, 0.795, 0.070, 0.052, "BLEU-2", edge=GREEN, size=7.3)
    chip(ax, 0.697, 0.720, 0.070, 0.052, "ROUGE-L", edge=GREEN, size=7.3)
    chip(ax, 0.778, 0.720, 0.070, 0.052, "METEOR", edge=GREEN, size=7.3)

    # Reliability branch. Failure label and detector scores remain separate;
    # both are used in offline evaluation, while only the score is thresholded.
    section(ax, 0.660, 0.075, 0.215, 0.500, "Reliability Analysis", face=RED_TINT, edge=RED, title_size=8.8)
    chip(
        ax,
        0.675,
        0.385,
        0.185,
        0.080,
        "Failure label\nROUGE-L < 0.2 AND METEOR < 0.1",
        edge=PURPLE,
        face=PURPLE_TINT,
        size=6.2,
        bold=True,
    )
    chip(
        ax,
        0.675,
        0.255,
        0.185,
        0.100,
        "Detector scores\nAFD frequency  ·  Semantic AFD\nQuestion-aligned uncertainty",
        edge=RED,
        size=6.2,
        bold=True,
    )
    chip(
        ax,
        0.695,
        0.130,
        0.145,
        0.085,
        "AUROC  ·  AUPRC\nSelective coverage",
        edge=PURPLE,
        face=PURPLE_TINT,
        size=7.1,
        bold=True,
    )

    # Selective-use decision inspired by the second reference image.
    diamond = Polygon(
        [[0.935, 0.445], [0.982, 0.355], [0.935, 0.265], [0.888, 0.355]],
        closed=True,
        facecolor=BLUE_TINT,
        edgecolor=BLUE,
        linewidth=1.05,
        zorder=2,
    )
    ax.add_patch(diamond)
    ax.text(0.935, 0.355, "$s_i>t$?", ha="center", va="center", fontsize=8.0, fontweight="bold", color=INK, zorder=4)
    chip(ax, 0.895, 0.590, 0.080, 0.070, "Accept", edge=GREEN, face=GREEN_TINT, size=7.5, bold=True)
    chip(ax, 0.880, 0.070, 0.110, 0.090, "Refer for\nHuman Review", edge=ORANGE, face=ORANGE_TINT, size=6.8, bold=True)

    # Main flow arrows occupy only the gaps between cards.
    arrow(ax, (0.140, 0.505), (0.175, 0.505))
    arrow(ax, (0.270, 0.505), (0.310, 0.505))
    arrow(ax, (0.460, 0.505), (0.500, 0.505))

    # Greedy answer supports answer quality and defines the failure endpoint.
    orthogonal_arrow(ax, [(0.620, 0.559), (0.650, 0.559), (0.650, 0.807), (0.680, 0.807)])
    orthogonal_arrow(ax, [(0.620, 0.559), (0.646, 0.559), (0.646, 0.425), (0.675, 0.425)])

    # Sampled answers create the detector scores; the line uses only whitespace.
    orthogonal_arrow(ax, [(0.620, 0.444), (0.640, 0.444), (0.640, 0.305), (0.675, 0.305)])

    # The detector score, not AUROC or AUPRC, is compared with the threshold.
    arrow(ax, (0.860, 0.305), (0.888, 0.355))
    arrow(ax, (0.935, 0.445), (0.935, 0.590))
    arrow(ax, (0.935, 0.265), (0.935, 0.160))
    ax.text(0.948, 0.515, "No", ha="left", va="center", fontsize=6.4, color=INK)
    ax.text(0.948, 0.213, "Yes", ha="left", va="center", fontsize=6.4, color=INK)

    fig.savefig(OUTPUT_DIR / "reliability_centred_vlm_framework_clean_draft.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUTPUT_DIR / "reliability_centred_vlm_framework_clean_draft.png", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    draw_framework()
    print("Saved clean methodology framework to", OUTPUT_DIR)
