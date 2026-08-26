from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


OUTPUT_DIR = Path("latex_template") / "Images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7.5,
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
PURPLE_EDGE = "#6A3D9A"
PURPLE_FILL = "#F5EFFA"
RED_EDGE = "#B34033"
RED_FILL = "#FCEFED"
GREY_EDGE = "#6B7280"
GREY_FILL = "#F3F4F6"
AMBER_EDGE = "#A66A00"
AMBER_FILL = "#FFF4D6"


def rounded_box(ax, x, y, w, h, face, edge, lw=0.9, radius=0.012):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        zorder=2,
    )
    ax.add_patch(patch)
    return patch


def labelled_panel(ax, x, y, w, h, title, face=BLUE_FILL, edge=BLUE_EDGE):
    rounded_box(ax, x, y, w, h, face, edge, lw=1.05, radius=0.014)
    ax.text(
        x + w / 2,
        y + h - 0.035,
        title,
        ha="center",
        va="top",
        fontsize=8.6,
        fontweight="bold",
        color=NAVY,
        zorder=4,
    )


def inner_box(ax, x, y, w, h, text, face="white", edge=BLUE_EDGE, fontsize=7.1, bold=False):
    rounded_box(ax, x, y, w, h, face, edge, lw=0.75, radius=0.009)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold" if bold else "normal",
        color=NAVY,
        linespacing=1.15,
        zorder=4,
    )


def arrow(ax, start, end, colour=NAVY, lw=1.05, style="arc3"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=lw,
            color=colour,
            connectionstyle=style,
            shrinkA=0,
            shrinkB=0,
            zorder=3,
        )
    )


def poly_arrow(ax, points, colour=NAVY, lw=1.05):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs[:-1], ys[:-1], color=colour, linewidth=lw, zorder=1.5)
    arrow(ax, points[-2], points[-1], colour=colour, lw=lw)


def make_framework():
    # The two-tier arrangement keeps the text readable when the PDF is placed
    # at dissertation text width.  It is one figure, not two disconnected
    # diagrams: the arrows link answer generation to both evaluation tiers.
    fig, ax = plt.subplots(figsize=(7.15, 6.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Top row: inputs, model inference, and generated outputs.
    labelled_panel(ax, 0.020, 0.650, 0.235, 0.325, "Medical VQA Inputs")
    inner_box(ax, 0.038, 0.825, 0.199, 0.055, "PathVQA (primary)")
    inner_box(ax, 0.038, 0.755, 0.199, 0.055, "VQA-RAD (external)")
    inner_box(
        ax,
        0.038,
        0.670,
        0.199,
        0.070,
        "ProstateMM-CHIMERA\n(external; + clinical context)",
        fontsize=5.6,
    )

    labelled_panel(ax, 0.310, 0.650, 0.310, 0.325, "Zero-Shot VLM Inference")
    inner_box(ax, 0.332, 0.835, 0.266, 0.050, "Fixed prompt within each dataset", fontsize=6.0)
    inner_box(ax, 0.332, 0.770, 0.266, 0.050, "Qwen2.5-VL-3B")
    inner_box(ax, 0.332, 0.705, 0.266, 0.050, "LLaVA-1.5-7B")
    inner_box(ax, 0.332, 0.655, 0.266, 0.038, "MedGemma-4B")

    labelled_panel(ax, 0.680, 0.650, 0.300, 0.325, "Answer Generation")
    inner_box(ax, 0.710, 0.790, 0.240, 0.075, "1 Greedy Answer", face="#F7FBFF", bold=True)
    inner_box(ax, 0.710, 0.680, 0.240, 0.075, "3 Sampled Answers", face="#F7FBFF", bold=True)

    labelled_panel(ax, 0.020, 0.385, 0.270, 0.200, "Answer Quality", face=GREEN_FILL, edge=GREEN_EDGE)
    inner_box(ax, 0.040, 0.475, 0.105, 0.050, "BLEU-1", face="white", edge=GREEN_EDGE)
    inner_box(ax, 0.165, 0.475, 0.105, 0.050, "BLEU-2", face="white", edge=GREEN_EDGE)
    inner_box(ax, 0.040, 0.410, 0.105, 0.050, "ROUGE-L", face="white", edge=GREEN_EDGE)
    inner_box(ax, 0.165, 0.410, 0.105, 0.050, "METEOR", face="white", edge=GREEN_EDGE)

    arrow(ax, (0.255, 0.805), (0.310, 0.805))
    arrow(ax, (0.620, 0.805), (0.680, 0.805))

    # Bottom row: failure endpoint, detector scores, offline evaluation, and
    # an explicitly separate illustrative threshold decision.
    labelled_panel(ax, 0.020, 0.105, 0.270, 0.220, "Operational Failure", face=PURPLE_FILL, edge=PURPLE_EDGE)
    inner_box(
        ax,
        0.045,
        0.145,
        0.220,
        0.080,
        "$F_i=1$ if\nROUGE-L $<0.2$ AND\nMETEOR $<0.1$",
        face="white",
        edge=PURPLE_EDGE,
        fontsize=6.0,
    )

    labelled_panel(ax, 0.350, 0.105, 0.310, 0.480, "Failure-Detection Scores", face=RED_FILL, edge=RED_EDGE)
    inner_box(ax, 0.375, 0.440, 0.260, 0.060, "AFD Frequency", face="white", edge=RED_EDGE)
    inner_box(ax, 0.375, 0.360, 0.260, 0.060, "Semantic AFD", face="white", edge=RED_EDGE)
    inner_box(ax, 0.375, 0.280, 0.260, 0.060, "Question-Aligned Uncertainty", face="white", edge=RED_EDGE, fontsize=6.5)
    inner_box(ax, 0.405, 0.175, 0.200, 0.065, "Random Baseline\n(reference only)", face=GREY_FILL, edge=GREY_EDGE, fontsize=6.2)

    labelled_panel(ax, 0.705, 0.260, 0.145, 0.300, "Reliability\nEvaluation", face=PURPLE_FILL, edge=PURPLE_EDGE)
    inner_box(ax, 0.722, 0.405, 0.111, 0.050, "AUROC", face="white", edge=PURPLE_EDGE, bold=True)
    inner_box(ax, 0.722, 0.340, 0.111, 0.050, "AUPRC", face="white", edge=PURPLE_EDGE, bold=True)
    inner_box(ax, 0.717, 0.270, 0.121, 0.060, "Selective\nCoverage", face="white", edge=PURPLE_EDGE, fontsize=5.6)

    diamond = Polygon(
        [[0.920, 0.370], [0.975, 0.300], [0.920, 0.230], [0.865, 0.300]],
        closed=True,
        facecolor="#F7FBFF",
        edgecolor=BLUE_EDGE,
        linewidth=1.0,
        zorder=2,
    )
    ax.add_patch(diamond)
    ax.text(0.920, 0.300, "$s_i>t$?", ha="center", va="center", fontsize=7.5, fontweight="bold", color=NAVY, zorder=4)
    inner_box(ax, 0.870, 0.455, 0.120, 0.065, "Accept", face=GREEN_FILL, edge=GREEN_EDGE, bold=True)
    inner_box(ax, 0.850, 0.090, 0.145, 0.085, "Refer for\nHuman Review", face=AMBER_FILL, edge=AMBER_EDGE, fontsize=6.5, bold=True)

    # Main-answer branch defines quality and the operational endpoint.
    poly_arrow(ax, [(0.830, 0.790), (0.830, 0.615), (0.155, 0.615), (0.155, 0.585)])
    # Sampled-answer branch creates detector scores.
    poly_arrow(ax, [(0.830, 0.680), (0.830, 0.605), (0.505, 0.605), (0.505, 0.585)])
    # Offline detector evaluation compares scores with the pre-defined label.
    poly_arrow(ax, [(0.830, 0.790), (0.830, 0.625), (0.305, 0.625), (0.305, 0.215), (0.290, 0.215)])
    poly_arrow(ax, [(0.155, 0.105), (0.155, 0.065), (0.690, 0.065), (0.690, 0.365), (0.705, 0.365)])
    arrow(ax, (0.660, 0.390), (0.705, 0.390))
    # A threshold acts on the detector score, not on AUROC or AUPRC.
    arrow(ax, (0.660, 0.205), (0.875, 0.250), style="arc3,rad=-0.24")
    arrow(ax, (0.920, 0.370), (0.930, 0.455))
    arrow(ax, (0.920, 0.230), (0.922, 0.175))
    ax.text(0.940, 0.410, "No", ha="left", va="center", fontsize=6.2, color=NAVY)
    ax.text(0.940, 0.205, "Yes", ha="left", va="center", fontsize=6.2, color=NAVY)

    fig.savefig(OUTPUT_DIR / "reliability_centred_vlm_framework_detailed_draft.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUTPUT_DIR / "reliability_centred_vlm_framework_detailed_draft.png", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    make_framework()
    print("Saved methodology framework to", OUTPUT_DIR)
