# AFD Reliability Evaluation for Medical Vision-Language Models

This repository contains the code and selected results for a medical visual question answering (VQA) reliability study. It compares three vision-language models (VLMs):

- Qwen2.5-VL-3B-Instruct
- MedGemma-4B-it
- LLaVA-1.5-7B

The evaluated datasets are PathVQA, VQA-RAD, and ProstateMM/CHIMERA. The models first answer questions in a zero-shot setting. Multiple uncertainty signals are then used to identify potentially incorrect answers. The project reports BLEU, ROUGE-L, METEOR, failure AUROC, failure AUPRC, and selective prediction metrics at different coverage levels.

## Research Pipeline

The project can be understood as the following pipeline:

```text
Medical image + question
          |
          v
Three VLMs generate answers
          |
          +-- Main answer: answer-quality evaluation
          +-- K sampled answers: uncertainty estimation
          |
          v
AFD / entropy / disagreement scoring
          |
          v
Failure label based on answer quality
          |
          v
AUROC, AUPRC, and coverage-rejection curves
```

The AFD post-processing script reads previously saved JSON inference outputs. It does not download the datasets or run VLM inference by itself.

## How to Read This Repository

### Fastest reading path

If you only want to understand the project quickly, read the files in this order:

1. This README for the research goal, workflow, and file roles.
2. [Final PathVQA results](tmp/clean_final_pathvqa_afd_table.csv).
3. [Final ProstateMM/CHIMERA results](tmp/clean_final_prostatemm_afd_table.csv).
4. [AFD post-processing script](AFD_Coverage_10_30_50_70_90_All_Datasets.py).
5. [Dissertation figure-generation script](generate_dissertation_figures.py).

### To understand model inference

Read the [PathVQA inference script](PathVQA_Qwen_MedGemma_LLaVA_ZeroShot_AFD.py), the [ProstateMM/CHIMERA inference script](ProstateMM_CHIMERA_Qwen_MedGemma_LLaVA_ZeroShot_AFD.py), and their corresponding notebooks. Each inference script contains environment setup, seed and device configuration, data loading, model loading, prompt construction, batched generation, checkpointing, answer-quality evaluation, and result export.

### To understand the dissertation

Read [latex_template/](latex_template/), [references.bib](references.bib), and [OVERLEAF_IMPORT_GUIDE_CN.md](OVERLEAF_IMPORT_GUIDE_CN.md).

## File Guide

### Core experiment and analysis scripts

| File | Purpose |
| --- | --- |
| `PathVQA_Qwen_MedGemma_LLaVA_ZeroShot_AFD.py` | Runs zero-shot inference, sampling, checkpointing, and answer evaluation for the three VLMs on the PathVQA test split. |
| `ProstateMM_CHIMERA_Qwen_MedGemma_LLaVA_ZeroShot_AFD.py` | Loads ProstateMM/CHIMERA data and runs inference and reliability evaluation for the three VLMs. The default split is test. |
| `AFD_Coverage_10_30_50_70_90_All_Datasets.py` | Reads JSON outputs and computes AFD frequency, Semantic AFD, answer disagreement, question-aligned entropy, random baseline, failure AUROC/AUPRC, and coverage metrics. |
| `generate_dissertation_figures.py` | Reads the main aggregate CSV and generates dissertation performance-rejection curves and the evaluation framework diagram. |

### Notebook files

| File type | Purpose |
| --- | --- |
| `*_AFD.ipynb` | Interactive Google Colab/Jupyter versions of the corresponding Python scripts. |
| `AFD_Coverage_10_30_50_70_90_All_Datasets.ipynb` | Notebook version of the AFD post-processing pipeline. |

The `.py` files are the recommended source of truth for long-term maintenance. The notebooks are useful for interactive execution, debugging, and Colab experiment records.

### Result files

| File | Purpose |
| --- | --- |
| `tmp/afd_summary_all_datasets_coverage_10_30_50_70_90.csv` | Main aggregate table across datasets, models, uncertainty methods, and coverage levels. |
| `tmp/clean_final_pathvqa_afd_table.csv` | Compact presentation table for PathVQA. |
| `tmp/clean_final_prostatemm_afd_table.csv` | Compact presentation table for ProstateMM/CHIMERA. |
| `results/README.md` | Result schema, provenance, and reproduction notes. |

Record-level JSON files, full scored CSV exports, and raw medical images are intentionally excluded from the public repository.

### Dissertation and figure files

| File or directory | Purpose |
| --- | --- |
| `latex_template/` | Dissertation chapters, references, and figures. |
| `generate_methodology_framework*.py` | Historical versions of the methodology-framework figure scripts. Keep only the final version when preparing a clean release. |
| `references.bib` | Project-level BibTeX references. |
| `CITATION.cff` | GitHub citation metadata template; replace the author and repository placeholders. |

### Project-management files

| File | Purpose |
| --- | --- |
| `requirements.txt` | Python dependencies. PyTorch and CUDA still need to be matched to the target machine. |
| `.gitignore` | Prevents caches, model weights, raw data, credentials, and large files from entering Git. |
| `data/README.md` | Dataset access, model access, privacy, and licensing notes. |
| `docs/reproducibility.md` | Experiment reproduction checklist. |
| `docs/upload_checklist.md` | Pre-publication checklist for GitHub. |

## Environment Requirements

Recommended environment:

- Python 3.10 or 3.11
- Linux, Google Colab, or an NVIDIA GPU machine
- Compatible CUDA, PyTorch, and bitsandbytes versions
- Sufficient GPU memory for 4-bit quantised models

Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell does not allow activation, install directly into the environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Install `bitsandbytes` separately if required, using a build compatible with your CUDA environment. Do not force GPU-only dependencies into an incompatible CPU environment.

## Reproducing PathVQA or ProstateMM/CHIMERA

### 1. Prepare datasets and model access

1. Obtain each dataset from its official source.
2. Request access to the required models through Hugging Face.
3. Store the Hugging Face token in Colab Secrets or an environment variable.
4. Prepare Google Drive or another writable location for model caches and JSON outputs.

Never commit tokens, dataset archives, patient-level records, or model caches.

### 2. Run inference

Run a small smoke test before the complete test split. In the selected script or notebook, review:

- `INSTALL_DEPENDENCIES`
- `RUN_ID`
- `DATASET_SPLIT` or `SPLIT_SELECTION`
- `MODEL_BATCH_SIZES`
- `K` or the number of sampled answers
- Google Drive output directories
- Model IDs

After the first dependency installation and runtime restart, set `INSTALL_DEPENDENCIES = False` and execute from the beginning. The inference scripts save resumable JSON outputs. Keep these files because the later AFD analysis depends on them.

PathVQA entry point:

```text
PathVQA_Qwen_MedGemma_LLaVA_ZeroShot_AFD.py
```

ProstateMM/CHIMERA entry point:

```text
ProstateMM_CHIMERA_Qwen_MedGemma_LLaVA_ZeroShot_AFD.py
```

### 3. Run AFD post-processing

Open:

```text
AFD_Coverage_10_30_50_70_90_All_Datasets.py
```

Review `MY_DRIVE_DIR`, `OUTPUT_DIRS`, and `AFD_OUTPUT_DIR`. By default, the script selects the largest complete run for each dataset/model pair, preventing smoke tests and final runs from being mixed.

The script computes:

- AFD frequency
- Semantic AFD
- Answer disagreement
- Question-aligned entropy
- Random baseline
- Failure AUROC
- Failure AUPRC
- Selective metrics at 10%, 30%, 50%, 70%, and 90% coverage

Copy only audited aggregate CSV files into `tmp/` or `results/`. Do not copy patient-level records.

### 4. Regenerate dissertation figures

From the repository root:

```powershell
python generate_dissertation_figures.py
```

The script reads:

```text
tmp/afd_summary_all_datasets_coverage_10_30_50_70_90.csv
```

and writes generated figures to:

```text
latex_template/Images/
```

## Current Reproduction Scope

The public repository currently supports inspection and reproduction of:

- The committed aggregate result tables
- PathVQA inference
- ProstateMM/CHIMERA inference
- AFD post-processing
- Dissertation figure generation

There is currently no standalone VQA-RAD inference script in this repository. VQA-RAD aggregate results can be inspected, but a complete rerun from raw data requires adding the corresponding data-loading, model-inference, and output-generation code.

Even with identical code, results may change because of model revisions, dataset versions, GPU/CUDA/PyTorch versions, batch size, quantisation, random seeds, sampling parameters, and the exact JSON output files selected for post-processing. For each formal experiment, record the git commit, model revision, dataset version, hardware, dependency versions, seed, batch size, quantisation method, and `K`.

## Data, Ethics, and Licensing

This project does not redistribute raw medical images, patient-level records, model weights, or dataset archives. All datasets and pretrained models remain subject to their original licences, data-use agreements, and ethical requirements.

Before publishing a release:

1. Remove tokens, private paths, and debug output from notebooks.
2. Check that result tables contain no patient identifiers or traceable information.
3. Replace the author, institution, and repository placeholders in [CITATION.cff](CITATION.cff).
4. Choose a project licence compatible with the code, datasets, and models.
5. Create a Git tag and document the experiment version in the release notes.

## Citation

If you use this code or its results, cite [CITATION.cff](CITATION.cff), together with the original PathVQA, VQA-RAD, ProstateMM/CHIMERA, and model papers.

