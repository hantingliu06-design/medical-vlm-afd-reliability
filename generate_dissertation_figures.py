from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Rectangle
import pandas as pd


ROOT = Path.cwd()
CSV_PATH = Path("tmp") / "afd_summary_all_datasets_coverage_10_30_50_70_90.csv"
IMAGE_DIR = Path("latex_template") / "Images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        # DejaVu Sans is embedded in the vector PDF and remains legible after scaling.
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": None,
    }
)


def make_performance_rejection_curve():
    data = pd.read_csv(str(CSV_PATH), encoding="utf-8")
    data["Model label"] = data["Model"].replace(
        {
            "Qwen/Qwen2.5-VL-3B-Instruct": "Qwen2.5-VL-3B-Instruct",
            "qwen2_5_vl_3b": "Qwen2.5-VL-3B-Instruct",
            "llava_1_5_7b": "LLaVA-1.5-7B",
            "medgemma_4b_it": "MedGemma-4B-it",
        }
    )
    method_labels = {
        "Question-aligned entropy": "Question-aligned uncertainty",
        "AFD frequency": "AFD frequency",
        "Semantic AFD": "Semantic AFD",
        "Random baseline": "Random baseline",
    }
    methods = list(method_labels)
    colours = {
        "Question-aligned entropy": "#0072B2",
        "AFD frequency": "#D55E00",
        "Semantic AFD": "#009E73",
        "Random baseline": "#4D4D4D",
    }
    markers = {
        "Question-aligned entropy": "o",
        "AFD frequency": "s",
        "Semantic AFD": "^",
        "Random baseline": "D",
    }
    line_styles = {
        "Question-aligned entropy": "-",
        "AFD frequency": "--",
        "Semantic AFD": "-.",
        "Random baseline": ":",
    }

    # The complete summary reports accepted quality at 10%, 30%, 50%,
    # 70%, and 90% coverage. Exclusion is 100% minus coverage.
    coverages = [90, 70, 50, 30, 10]
    exclusion = [10, 30, 50, 70, 90]
    datasets = ["PathVQA", "ProstateMM-CHIMERA"]
    models = ["Qwen2.5-VL-3B-Instruct", "LLaVA-1.5-7B", "MedGemma-4B-it"]
    file_stems = {
        "Qwen2.5-VL-3B-Instruct": "performance_rejection_qwen",
        "LLaVA-1.5-7B": "performance_rejection_llava",
        "MedGemma-4B-it": "performance_rejection_medgemma",
    }

    for model in models:
        fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.7), sharey=True)
        for panel_index, (ax, dataset) in enumerate(zip(axes, datasets)):
            model_data = data[
                (data["Model label"] == model)
                & (data["Dataset"] == dataset)
                & (
                    data["Split"].eq("test_full")
                    if dataset == "PathVQA"
                    else data["Split"].eq("test")
                )
            ]
            for method in methods:
                row = model_data[model_data["Method"] == method]
                if row.empty:
                    continue
                row = row.iloc[0]
                values = [
                    100.0 * float(row[f"Accepted ROUGE-L @{coverage}%"])
                    for coverage in coverages
                ]
                ax.plot(
                    exclusion,
                    values,
                    color=colours[method],
                    marker=markers[method],
                    linestyle=line_styles[method],
                    linewidth=1.8,
                    markersize=5.2,
                    markeredgewidth=0.8,
                    label=method_labels[method],
                )
            ax.set_title(f"({chr(97 + panel_index)}) {dataset}")
            ax.set_xlabel("Exclusion ratio (%)")
            ax.set_xticks(exclusion)
            ax.grid(True, axis="y", linewidth=0.5, alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_ylim(0, 70)
            ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70])

        axes[0].set_ylabel("Accepted answer quality (ROUGE-L, %)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=4,
            frameon=False,
        )
        fig.subplots_adjust(top=0.79, wspace=0.17, bottom=0.20, left=0.09, right=0.985)
        stem = file_stems[model]
        fig.savefig(IMAGE_DIR / f"{stem}.png", bbox_inches=None)
        fig.savefig(IMAGE_DIR / f"{stem}.pdf", bbox_inches=None)
        plt.close(fig)


def rounded_box(ax, xy, width, height, text, face, edge="#4b5563", fontsize=9):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.1,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        wrap=True,
    )


def arrow(ax, start, end, colour="#374151", connectionstyle="arc3"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.0,
            color=colour,
            connectionstyle=connectionstyle,
        )
    )


def make_framework_diagram():
    fig, ax = plt.subplots(figsize=(12.0, 4.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.01,
        0.97,
        "Zero-shot VLM evaluation framework for answer quality and failure detection",
        fontsize=12,
        fontweight="bold",
        va="top",
    )

    # Input panel with a simple tissue-like thumbnail.
    rounded_box(ax, (0.02, 0.54), 0.14, 0.27, "Image\n+\nQuestion", "#f7f3e8")
    ax.add_patch(Rectangle((0.045, 0.62), 0.09, 0.115, facecolor="#f5c2b8", edgecolor="#8c4b42", linewidth=0.8))
    ax.add_patch(Ellipse((0.09, 0.68), 0.07, 0.035, facecolor="#c86f65", edgecolor="#8c4b42", linewidth=0.6))
    ax.add_patch(Ellipse((0.07, 0.66), 0.02, 0.014, facecolor="#7c3f50", edgecolor="none"))
    ax.add_patch(Ellipse((0.11, 0.70), 0.018, 0.012, facecolor="#7c3f50", edgecolor="none"))
    ax.text(0.09, 0.585, "medical image + question", ha="center", va="center", fontsize=6.8)

    rounded_box(ax, (0.21, 0.54), 0.15, 0.27, "Three VLMs\nQwen2.5-VL\nLLaVA\nMedGemma", "#e9f2fb")
    rounded_box(ax, (0.42, 0.63), 0.16, 0.18, "One greedy answer\n$\\hat{y}_i^{(0)}$", "#eaf6ea")
    rounded_box(ax, (0.42, 0.38), 0.16, 0.18, "Three sampled answers\n$\\hat{y}_i^{(1)},\\ldots,\\hat{y}_i^{(3)}$", "#fff4dc")
    rounded_box(ax, (0.66, 0.67), 0.15, 0.14, "Answer quality\nBLEU-1/2\nROUGE-L, METEOR", "#f0ecfb", fontsize=8)
    rounded_box(ax, (0.66, 0.41), 0.15, 0.14, "Operational failure\nF_i = 1 if ROUGE-L < .2\nand METEOR < .1", "#fbe9e9", fontsize=7.2)
    rounded_box(ax, (0.66, 0.16), 0.15, 0.14, "AFD signals\nfrequency\nsemantic\nquestion-aligned", "#e8f4f4", fontsize=8)
    rounded_box(ax, (0.87, 0.39), 0.11, 0.22, "AUROC\nAUPRC\ncoverage\ncurves", "#f4f4f4", fontsize=8.5)

    arrow(ax, (0.16, 0.675), (0.21, 0.675))
    arrow(ax, (0.36, 0.70), (0.42, 0.72))
    arrow(ax, (0.36, 0.61), (0.42, 0.47))
    arrow(ax, (0.58, 0.72), (0.66, 0.74))
    arrow(ax, (0.58, 0.47), (0.66, 0.48))
    arrow(ax, (0.58, 0.47), (0.66, 0.23))
    arrow(ax, (0.81, 0.74), (0.87, 0.54), connectionstyle="arc3,rad=-0.12")
    arrow(ax, (0.81, 0.48), (0.87, 0.49))
    arrow(ax, (0.81, 0.23), (0.87, 0.46), connectionstyle="arc3,rad=0.12")
    ax.text(0.375, 0.77, "zero-shot inference", ha="center", fontsize=7.5, color="#374151")
    ax.text(0.605, 0.60, "main answer", ha="center", fontsize=7.2, color="#374151")
    ax.text(0.605, 0.34, "sampled-output variability", ha="center", fontsize=7.2, color="#374151")

    fig.savefig(IMAGE_DIR / "afd_evaluation_framework.png")
    fig.savefig(IMAGE_DIR / "afd_evaluation_framework.pdf")
    plt.close(fig)


if __name__ == "__main__":
    make_performance_rejection_curve()
    print("Saved figures to", IMAGE_DIR)
