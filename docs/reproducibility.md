# Reproducibility checklist

1. Use Python 3.10 or 3.11 with `requirements.txt` and a CUDA-compatible PyTorch installation for GPU inference.
2. Download datasets from official sources and obtain model access through Hugging Face. Store credentials in runtime secrets.
3. Configure `MY_DRIVE_DIR`, dataset locations, output directories, model IDs and split selection in the inference scripts.
4. Run each model with documented seed, batch size, quantisation and sampled-answer count (`K`). Keep raw JSON outputs privately.
5. Run `AFD_Coverage_10_30_50_70_90_All_Datasets.py` to compute uncertainty scores, failure labels and selective metrics.
6. Copy only audited aggregate CSVs into `tmp/` or `results/`, then run `generate_dissertation_figures.py`.

For each release, archive the git commit, dependency versions, dataset/model revisions, hardware, command/configuration and checksums of private inputs. Model and dataset updates can change results without code changes.
