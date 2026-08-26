# Data and model access

This repository intentionally excludes raw images, patient-level records, downloaded archives and model weights. They may contain restricted clinical data or be too large for GitHub.

Before running, record the exact dataset release, split, preprocessing steps and access URL in an experiment log. Keep local paths in a private configuration file or environment variables; never commit tokens or Google Drive mount paths.

Evaluated sources:

- PathVQA (test split; released summary contains 6,719 evaluated samples).
- VQA-RAD (test split; released summary contains 451 evaluated samples).
- ProstateMM/CHIMERA (scripts support test and all-records modes).

Check each dataset's official licence and data-use agreement. Do not publish record-level predictions, images or identifiers unless redistribution is explicitly permitted and data are de-identified.
