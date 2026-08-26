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

NAVY = "#17365D"
BLUE_EDGE = "#2F5597"
BLUE_FILL = "#EEF4FB"
GREEN_EDGE = "#3B7D23"
GREEN_FILL = "#EEF7EA"
RED_EDGE = "#B34033"
RED_FILL = "#FCEFED"
PURPLE_EDGE = "#6A3D9A"
PURPLE_FILL = "#F5EFFA"
AMBER_EDGE = "#A66A00"
AMBER_FILL = "#FFF4D6"


def rounded_box(ax, x, y, w, h, face, edge, linewidth=1.0, radius=0.012):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        linewidth=linewidth,
        edgecolor=edge,
        facecolor=face,
        zorder=2,
    )
    ax.add_patch(patch)


def panel(ax, x, y, w, h, title, face=BLUE_FILL, edge=BLUE_EDGE, fontsize=9.2):
    rounded_box(ax, x, y, w, h, face, edge, linewidth=1.1, radius=0.014)
    ax.text(
        x + w / 2,
        y + h - 0.035,
        title,
        ha="center",
        va="top",
        fontsize=fontsize,
        fontweight="bold",
        color=NAVY,
        linespacing=1.0,
        zorder=4,
    )


def inner_box(ax, x, y, w, h, text, face="white", edge=BLUE_EDGE, fontsize=7.6, bold=False):
    rounded_box(ax, x, y, w, h, face, edge, linewidth=0.75, radius=0.008)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold" if bold else "normal",
        color=NAVY,
        linespacing=1.05,
        zorder=4,
    )


def arrow(ax, start, end, colour=NAVY, linewidth=1.15, style="arc3"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=linewidth,
            color=colour,
            connectionstyle=style,
            shrinkA=0,
            shrinkB=0,
            zorder=3,
        )
    )


def poly_arrow(ax, points, colour=NAVY, linewidth=1.15):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs[:-1], ys[:-1], color=colour, linewidth=linewidth, zorder=1.5)
    arrow(ax, points[-2], points[-1], colour=colour, linewidth=linewidth)


def make_framework():
    # Simple first-version-style overview. The PDF is vector based and the PNG
    # is retained as a 300 dpi workbench preview.
    fig, ax = plt.subplots(figsize=(10.0, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Medical VQA and Prompt are separate, parallel stages in the main flow.
    panel(ax, 0.015, 0.355, 0.115, 0.300, "Medical VQA", fontsize=9.0)
    inner_box(ax, 0.030, 0.505, 0.085, 0.060, "Image")
    inner_box(ax, 0.030, 0.425, 0.085, 0.060, "Question")
    ax.text(0.0725, 0.380, "optional context", ha="center", va="center", fontsize=5.8, color=NAVY)

    rounded_box(ax, 0.155, 0.435, 0.105, 0.140, BLUE_FILL, BLUE_EDGE, linewidth=1.1, radius=0.014)
    ax.text(
        0.2075,
        0.505,
        "Prompt\nFixed within\neach dataset",
        ha="center",
        va="center",
        fontsize=6.4,
        fontweight="bold",
        color=NAVY,
        linespacing=1.15,
        zorder=4,
    )

    panel(ax, 0.295, 0.270, 0.145, 0.460, "Zero-Shot VLMs", fontsize=8.8)
    inner_box(ax, 0.315, 0.550, 0.105, 0.065, "Qwen2.5-VL-3B", fontsize=6.8)
    inner_box(ax, 0.315, 0.455, 0.105, 0.065, "LLaVA-1.5-7B", fontsize=6.8)
    inner_box(ax, 0.315, 0.360, 0.105, 0.065, "MedGemma-4B", fontsize=6.8)

    panel(ax, 0.480, 0.315, 0.145, 0.370, "Answer Generation", fontsize=8.5)
    inner_box(ax, 0.500, 0.515, 0.105, 0.075, "1 Greedy\nAnswer", face="#F7FBFF", fontsize=7.0, bold=True)
    inner_box(ax, 0.500, 0.405, 0.105, 0.075, "3 Sampled\nAnswers", face="#F7FBFF", fontsize=7.0, bold=True)

    panel(ax, 0.665, 0.655, 0.205, 0.270, "Answer Quality", face=GREEN_FILL, edge=GREEN_EDGE, fontsize=9.0)
    inner_box(ax, 0.685, 0.770, 0.075, 0.055, "BLEU-1", face="white", edge=GREEN_EDGE, fontsize=7.0)
    inner_box(ax, 0.775, 0.770, 0.075, 0.055, "BLEU-2", face="white", edge=GREEN_EDGE, fontsize=7.0)
    inner_box(ax, 0.685, 0.695, 0.075, 0.055, "ROUGE-L", face="white", edge=GREEN_EDGE, fontsize=7.0)
    inner_box(ax, 0.775, 0.695, 0.075, 0.055, "METEOR", face="white", edge=GREEN_EDGE, fontsize=7.0)

    panel(
        ax,
        0.650,
        0.075,
        0.235,
        0.500,
        "Reliability and Failure Detection",
        face=RED_FILL,
        edge=RED_EDGE,
        fontsize=8.2,
    )
    inner_box(
        ax,
        0.670,
        0.395,
        0.195,
        0.080,
        "Operational failure label\nROUGE-L < 0.2 AND METEOR < 0.1",
        face="white",
        edge=PURPLE_EDGE,
        fontsize=6.2,
    )
    inner_box(
        ax,
        0.670,
        0.235,
        0.195,
        0.135,
        "AFD frequency   |   Semantic AFD\nQuestion-aligned uncertainty\nRandom baseline (reference)",
        face="white",
        edge=RED_EDGE,
        fontsize=6.4,
    )
    inner_box(
        ax,
        0.690,
        0.115,
        0.155,
        0.080,
        "AUROC   |   AUPRC\nSelective coverage",
        face=PURPLE_FILL,
        edge=PURPLE_EDGE,
        fontsize=6.5,
        bold=True,
    )

    # The surgical example contributes only this generic threshold decision.
    diamond = Polygon(
        [[0.940, 0.455], [0.985, 0.365], [0.940, 0.275], [0.895, 0.365]],
        closed=True,
        facecolor="#F7FBFF",
        edgecolor=BLUE_EDGE,
        linewidth=1.0,
        zorder=2,
    )
    ax.add_patch(diamond)
    ax.text(0.940, 0.365, "$s_i>t$?", ha="center", va="center", fontsize=7.8, fontweight="bold", color=NAVY, zorder=4)
    inner_box(ax, 0.900, 0.575, 0.085, 0.070, "Accept", face=GREEN_FILL, edge=GREEN_EDGE, fontsize=7.0, bold=True)
    inner_box(ax, 0.890, 0.080, 0.100, 0.090, "Refer for\nHuman Review", face=AMBER_FILL, edge=AMBER_EDGE, fontsize=6.4, bold=True)

    arrow(ax, (0.130, 0.505), (0.155, 0.505))
    arrow(ax, (0.260, 0.505), (0.295, 0.505))
    arrow(ax, (0.440, 0.505), (0.480, 0.505))

    # Answer generation splits into the two outcomes of the study.
    poly_arrow(ax, [(0.625, 0.550), (0.640, 0.550), (0.640, 0.790), (0.665, 0.790)])
    poly_arrow(ax, [(0.605, 0.552), (0.635, 0.552), (0.635, 0.435), (0.650, 0.435)])
    poly_arrow(ax, [(0.605, 0.442), (0.638, 0.442), (0.638, 0.305), (0.650, 0.305)])

    # The detector score, not AUROC or AUPRC, is compared with a threshold.
    arrow(ax, (0.885, 0.325), (0.895, 0.365))
    arrow(ax, (0.940, 0.455), (0.942, 0.575))
    arrow(ax, (0.940, 0.275), (0.940, 0.170))
    ax.text(0.951, 0.515, "No", ha="left", va="center", fontsize=6.2, color=NAVY)
    ax.text(0.951, 0.225, "Yes", ha="left", va="center", fontsize=6.2, color=NAVY)

    fig.savefig(OUTPUT_DIR / "reliability_centred_vlm_framework_simple_draft.pdf", bbox_inches="tight", pad_inches=0.035)
    fig.savefig(OUTPUT_DIR / "reliability_centred_vlm_framework_simple_draft.png", bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


if __name__ == "__main__":
    make_framework()
    print("Saved simplified methodology framework to", OUTPUT_DIR)
