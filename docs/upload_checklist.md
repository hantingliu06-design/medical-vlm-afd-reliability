# GitHub upload checklist

## Include after review

- The three `.py` experiment/evaluation scripts and their matching notebooks.
- `generate_dissertation_figures.py` and the final methodology/figure source that is actually used.
- Compact aggregate CSVs in `tmp/` or `results/`.
- `latex_template/` only if the dissertation source is intended to be public.
- `README.md`, `requirements.txt`, `.gitignore`, `CITATION.cff` and reproducibility notes.

## Keep local or archive privately

- `Untitled2.ipynb` and files named as copies, drafts or smoke tests until their purpose is documented.
- Record-level scored CSVs and raw JSON outputs, even when they are below GitHub's 100 MB limit.
- Raw medical images, dataset archives, patient/task records and Hugging Face/Google Drive caches.
- `MSc_Dissertation_Overleaf_Ready_*.zip` and duplicate `release/overleaf_ready_*` snapshots; keep one source tree and create release archives outside git.

## Before publishing

- Remove tokens, absolute local paths and personal identifiers from notebooks and outputs.
- Confirm dataset, model and image licences permit redistribution of every committed artifact.
- Replace the author/repository placeholders in `CITATION.cff` and choose an explicit project licence.
- Add a release tag and a short changelog describing dataset/model revisions and known limitations.
