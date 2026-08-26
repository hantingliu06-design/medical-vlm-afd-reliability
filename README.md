# AFD Reliability Evaluation for Medical Vision-Language Models

This repository contains reproducible analysis code and selected aggregate results for a zero-shot medical visual question answering study. It compares Qwen2.5-VL, LLaVA-1.5 and MedGemma on PathVQA, VQA-RAD and ProstateMM/CHIMERA, and evaluates AFD-based uncertainty signals against answer-quality failures.

The project reports BLEU, ROUGE-L and METEOR answer quality, operational failure labels, AUROC/AUPRC, and selective prediction results at 10%, 30%, 50%, 70% and 90% coverage. AFD post-processing reads saved model outputs; it does not run inference by itself.

## Layout

```text
.
├── AFD_Coverage_10_30_50_70_90_All_Datasets.py
├── PathVQA_Qwen_MedGemma_LLaVA_ZeroShot_AFD.py
├── ProstateMM_CHIMERA_Qwen_MedGemma_LLaVA_ZeroShot_AFD.py
├── *.ipynb                         # Colab/Jupyter versions
├── generate_dissertation_figures.py
├── tmp/                             # compact aggregate tables
├── latex_template/                  # dissertation source and figures
├── data/README.md
├── results/README.md
└── docs/reproducibility.md
```

## Quick start

1. Create a Python 3.10/3.11 environment and install `requirements.txt`.
2. Obtain datasets and model access separately. Do not commit medical images, patient records, tokens, model caches or Drive exports.
3. Run the inference scripts in Google Colab or on a CUDA machine after setting dataset/output paths in their configuration cells.
4. Run `AFD_Coverage_10_30_50_70_90_All_Datasets.py` on saved JSON outputs.
5. Run `python generate_dissertation_figures.py` to regenerate figures from the compact summary in `tmp/`.

The notebooks are convenient for Colab; the `.py` files are the version-controlled source of truth. Set `INSTALL_DEPENDENCIES = False` after first setup and restart the runtime when requested.

## Reproducibility and scope

All experiments use seed 42 where supported. Exact scores depend on model revisions, quantisation, GPU/CUDA versions, dataset versions and the saved output files selected by post-processing. See [`docs/reproducibility.md`](docs/reproducibility.md) and [`results/README.md`](results/README.md).

## Data, ethics and licensing

Datasets and pretrained models remain subject to their original licences, terms of use and ethics requirements. This repository does not redistribute them. See [`data/README.md`](data/README.md). Add a project licence only after confirming compatibility with included dependencies and result sources.

## Citation

If you use this code or dissertation, cite the project using [`CITATION.cff`](CITATION.cff), as well as the original dataset and model papers.
