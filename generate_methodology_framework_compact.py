from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
from matplotlib.transforms import Bbox


OUTPUT_DIR = Path("latex_template") / "Images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
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


def fitted_text(ax, patch, x, y, text, size, bold=False, linespacing=1.05):
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


def section(ax, x, y, width, height, title, face, edge, title_size=8.2):
    patch = rounded(ax, x, y, width, height, face, edge, line=1.1, radius=0.014)
    title_artist = ax.text(
        x + width / 2,
        y + height - 0.030,
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


def chip(ax, x, y, width, height, text, edge=BLUE, face="white", size=6.8, bold=False):
    patch = rounded(ax, x, y, width, height, face, edge, line=0.72, radius=0.008)
    fitted_text(ax, patch, x + width / 2, y + height / 2, text, size, bold=bold)
    return patch


def arrow(ax, start, end, colour=ARROW, line=1.05):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8.5,
            linewidth=line,
            color=colour,
            shrinkA=0,
            shrinkB=0,
            zorder=3,
        )
    )


def orthogonal_arrow(ax, points, colour=ARROW, line=1.05):
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

    # Compact main path with deliberately enlarged gaps between cards.
    section(ax, 0.030, 0.380, 0.105, 0.280, "Medical VQA", BLUE_TINT, BLUE, 8.2)
    chip(ax, 0.048, 0.500, 0.069, 0.055, "Image +\nQuestion", size=6.0)
    chip(ax, 0.048, 0.420, 0.069, 0.055, "Optional\ncontext", edge="#7E8DA5", face="white", size=5.6)

    prompt = rounded(ax, 0.185, 0.435, 0.075, 0.170, ORANGE_TINT, ORANGE, line=1.1, radius=0.014)
    fitted_text(ax, prompt, 0.2225, 0.520, "Prompt", 8.1, bold=True)

    section(ax, 0.315, 0.320, 0.125, 0.390, "Zero-Shot VLMs", BLUE_TINT, BLUE, 8.0)
    chip(ax, 0.334, 0.535, 0.087, 0.064, "Qwen2.5-VL-\n3B-Instruct", size=5.8)
    chip(ax, 0.334, 0.445, 0.087, 0.064, "LLaVA-\n1.5-7B", size=5.9)
    chip(ax, 0.334, 0.355, 0.087, 0.064, "MedGemma-\n4B-it", size=5.9)

    section(ax, 0.495, 0.350, 0.115, 0.340, "Answer\nGeneration", BLUE_TINT, BLUE, 7.2)
    chip(ax, 0.514, 0.510, 0.077, 0.068, "1 Greedy\nAnswer", size=6.3, bold=True)
    chip(ax, 0.514, 0.410, 0.077, 0.068, "3 Sampled\nAnswers", size=6.3, bold=True)

    # Compact, visually balanced evaluation branches.
    section(ax, 0.690, 0.700, 0.160, 0.215, "Answer Quality", GREEN_TINT, GREEN, 8.2)
    chip(ax, 0.707, 0.790, 0.058, 0.045, "BLEU-1", edge=GREEN, size=6.5)
    chip(ax, 0.777, 0.790, 0.058, 0.045, "BLEU-2", edge=GREEN, size=6.5)
    chip(ax, 0.707, 0.725, 0.058, 0.045, "ROUGE-L", edge=GREEN, size=6.5)
    chip(ax, 0.777, 0.725, 0.058, 0.045, "METEOR", edge=GREEN, size=6.5)

    section(ax, 0.675, 0.205, 0.175, 0.245, "Reliability Analysis", RED_TINT, RED, 7.9)
    chip(ax, 0.692, 0.315, 0.066, 0.050, "Failure label", edge=PURPLE, face=PURPLE_TINT, size=6.0, bold=True)
    chip(ax, 0.770, 0.315, 0.063, 0.050, "AFD scores", edge=RED, size=6.0, bold=True)
    chip(
        ax,
        0.707,
        0.235,
        0.110,
        0.050,
        "AUROC · AUPRC\nSelective coverage",
        edge=PURPLE,
        face=PURPLE_TINT,
        size=5.7,
        bold=True,
    )

    # Small, separate threshold decision.
    diamond = Polygon(
        [[0.930, 0.425], [0.970, 0.345], [0.930, 0.265], [0.890, 0.345]],
        closed=True,
        facecolor=BLUE_TINT,
        edgecolor=BLUE,
        linewidth=1.0,
        zorder=2,
    )
    ax.add_patch(diamond)
    decision_text = ax.text(0.930, 0.345, "$s_i>t$?", ha="center", va="center", fontsize=7.2, fontweight="bold", color=INK, zorder=4)
    FIT_CHECKS.append((diamond, decision_text))
    chip(ax, 0.895, 0.575, 0.070, 0.060, "Accept", edge=GREEN, face=GREEN_TINT, size=6.8, bold=True)
    chip(ax, 0.880, 0.075, 0.100, 0.075, "Refer for\nHuman Review", edge=ORANGE, face=ORANGE_TINT, size=6.2, bold=True)

    # Floating arrows use the whitespace between cards; no endpoint touches a frame.
    arrow(ax, (0.145, 0.520), (0.175, 0.520))
    arrow(ax, (0.270, 0.520), (0.305, 0.520))
    arrow(ax, (0.450, 0.520), (0.485, 0.520))

    orthogonal_arrow(ax, [(0.620, 0.544), (0.645, 0.544), (0.645, 0.807), (0.680, 0.807)])
    orthogonal_arrow(ax, [(0.620, 0.444), (0.640, 0.444), (0.640, 0.310), (0.665, 0.310)])

    arrow(ax, (0.860, 0.315), (0.880, 0.340))
    arrow(ax, (0.930, 0.435), (0.930, 0.565))
    arrow(ax, (0.930, 0.255), (0.930, 0.160))
    ax.text(0.945, 0.500, "No", ha="left", va="center", fontsize=5.9, color=INK)
    ax.text(0.945, 0.205, "Yes", ha="left", va="center", fontsize=5.9, color=INK)

    validate_text_fit(fig)
    fig.savefig(OUTPUT_DIR / "reliability_centred_vlm_framework.pdf", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUTPUT_DIR / "reliability_centred_vlm_framework.png", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


if __name__ == "__main__":
    draw_framework()
    print("Saved compact methodology framework to", OUTPUT_DIR)
