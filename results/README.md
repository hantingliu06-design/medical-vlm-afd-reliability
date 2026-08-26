# Results

Checked-in tables are compact aggregate exports for review and plotting. They include model/method identifiers, sample counts, answer-quality metrics, failure AUROC/AUPRC, mean uncertainty and selective metrics at several coverage levels.

`tmp/afd_summary_all_datasets_coverage_10_30_50_70_90.csv` is the main cross-dataset summary used by `generate_dissertation_figures.py`; its current rows cover PathVQA and ProstateMM/CHIMERA. The `clean_final_*_afd_table.csv` files are presentation-ready subsets for those two datasets.

The VQA-RAD model-specific summaries are:

- `tmp/vqa_rad_qwen2_5_vl_3b_summary.csv`
- `tmp/vqa_rad_medgemma_4b_it_summary.csv`
- `tmp/vqa_rad_llava_1_5_7b_summary.csv`

The corresponding inference workflow is provided in `VQA_RAD_Qwen_MedGemma_LLaVA_ZeroShot_AFD.ipynb`.

Record-level scored outputs and raw JSON inference files are deliberately not tracked. To reproduce a table, place corresponding private JSON outputs in configured directories and run the AFD post-processing script. Preserve dataset version, model revision, split, seed, `K`, batch size, quantisation and evaluation date.
