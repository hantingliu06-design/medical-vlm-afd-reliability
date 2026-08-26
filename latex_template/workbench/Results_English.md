# Chapter 4: Results

This chapter reports answer quality, operational failure detection, and
their relationship across the three datasets. All bounded values are
displayed as percentages and differences as percentage points;
calculations remain on standard normalised scales. Results are
descriptive because one inference run is available and no confidence
intervals or significance tests were calculated.

Overall Experimental Results
============================

Table \[tab:overall-result-pattern\] summarises the central result.
PathVQA gives a metric-dependent answer ranking: Qwen2.5-VL-3B-Instruct
leads BLEU-1 and BLEU-2, while MedGemma-4B-it leads ROUGE-L and METEOR;
Qwen2.5-VL-3B-Instruct has the highest detector AUROC. VQA-RAD shows
clearer divergence because MedGemma-4B-it leads three answer metrics but
Qwen2.5-VL-3B-Instruct leads failure detection. ProstateMM-CHIMERA
instead shows alignment, with MedGemma-4B-it leading three answer
metrics and AUROC. The relationship is therefore not constant across
metrics or datasets, and the small patient-linked prostate result
requires particular caution.

[lXXrl]{} Dataset & Answer-quality leader & Best detector (model;
method) & AUROC & Relationship\
PathVQA & Qwen2.5-VL-3B-Instruct on BLEU; MedGemma-4B-it on
ROUGE-L/METEOR & Qwen2.5-VL-3B-Instruct; question-aligned & 81.20% &
Metric-dependent\
VQA-RAD & MedGemma-4B-it on three metrics & Qwen2.5-VL-3B-Instruct;
Semantic AFD & 69.43% & **Divergent**\
ProstateMM-CHIMERA & MedGemma-4B-it on three metrics & MedGemma-4B-it;
question-aligned & 93.88% & **Aligned**\

Answer Quality on PathVQA
=========================

Table \[tab:answer-quality-all\] shows all answer results. On PathVQA,
Qwen2.5-VL-3B-Instruct leads BLEU-1 at 25.29% and BLEU-2 at 1.38%,
whereas MedGemma-4B-it leads ROUGE-L at 34.77% and METEOR at 17.73%;
LLaVA-1.5-7B is lowest on all four metrics. Relative to
Qwen2.5-VL-3B-Instruct, MedGemma-4B-it is 3.03 percentage points higher
on ROUGE-L and 1.63 points higher on METEOR but 5.78 points lower on
BLEU-1 and 0.95 points lower on BLEU-2. Answer strength on this dataset
therefore depends on the metric, so all four measures are retained
rather than combined into one accuracy value.

[lXrrrr]{} Dataset & Model & BLEU-1 & BLEU-2 & ROUGE-L & METEOR\
& Qwen2.5-VL-3B-Instruct & **25.29%** & **1.38%** & &\
& LLaVA-1.5-7B & & & 26.75% & 13.53%\
& MedGemma-4B-it & 19.51% & 0.43% & **34.77%** & **17.73%**\
& Qwen2.5-VL-3B-Instruct & 39.37% & **8.49%** & 48.16% & 25.94%\
& LLaVA-1.5-7B & 38.41% & 4.88% & 40.25% & 20.42%\
& MedGemma-4B-it & **40.18%** & & **58.66%** & **31.05%**\
& Qwen2.5-VL-3B-Instruct & & & & **22.48%**\
& LLaVA-1.5-7B & 19.00% & 2.18% & 15.27% & 13.90%\
& MedGemma-4B-it & **62.24%** & **28.57%** & **21.86%** &\

Answer Quality on VQA-RAD
=========================

On VQA-RAD, MedGemma-4B-it leads BLEU-1, ROUGE-L, and METEOR, while
Qwen2.5-VL-3B-Instruct leads BLEU-2 and LLaVA-1.5-7B is lowest
throughout. MedGemma-4B-it’s ROUGE-L of 58.66% is 10.50 percentage
points above Qwen2.5-VL-3B-Instruct and 18.41 points above LLaVA-1.5-7B.
Its METEOR of 31.05% is also highest. These within-dataset rankings
support MedGemma-4B-it as the strongest answer generator on three
measures, but higher absolute values than PathVQA do not prove that
radiology is easier because answer style and reference distributions
differ.

Operational Failure Distribution
================================

The final per-record exports allow operational failure prevalence to be
reported for the complete test splits. These values are the proportions of
records whose greedy answer falls below both fixed reference-matching
thresholds, not clinical error rates. MedGemma-4B-it has the lowest value
on all three datasets. Failure burden and failure ranking are different
properties, and AUPRC is interpreted cautiously because it depends on
positive-class prevalence.

[lXrr]{} Dataset & Model & Test records & Failure prevalence\
PathVQA & Qwen2.5-VL-3B-Instruct & 6,719 & 66.88%\
PathVQA & LLaVA-1.5-7B & 6,719 & 71.96%\
PathVQA & MedGemma-4B-it & 6,719 & **62.94%**\
VQA-RAD & Qwen2.5-VL-3B-Instruct & 451 & 47.23%\
VQA-RAD & LLaVA-1.5-7B & 451 & 58.09%\
VQA-RAD & MedGemma-4B-it & 451 & **36.81%**\
ProstateMM-CHIMERA & Qwen2.5-VL-3B-Instruct & 42 & 54.76%\
ProstateMM-CHIMERA & LLaVA-1.5-7B & 42 & 42.86%\
ProstateMM-CHIMERA & MedGemma-4B-it & 42 & **33.33%**\

Failure Detection Results
=========================

On PathVQA, Qwen2.5-VL-3B-Instruct has the highest AUROC for every
informed detector and reaches 81.20% with question-aligned uncertainty
(Table \[tab:pathvqa-failure-detection\]). AFD frequency is best by
AUROC for LLaVA-1.5-7B at 69.55% and MedGemma-4B-it at 68.63%, whereas
question-aligned uncertainty gives their highest AUPRC. Random AUROCs
remain close to 50%. At 50% coverage, all selected best-AUROC conditions
improve accepted ROUGE-L and reduce failure rate relative to random
ranking (Table \[tab:selective-gain-50\]); the largest PathVQA change is
Qwen2.5-VL-3B-Instruct, with ROUGE-L rising by 21.65 points and failure
rate falling by 21.52 points.

[XXrr]{} Model & Detector & AUROC & AUPRC\
Qwen2.5-VL-3B-Instruct & Question-aligned uncertainty & **81.20%** &
**90.71%**\
Qwen2.5-VL-3B-Instruct & AFD frequency & &\
Qwen2.5-VL-3B-Instruct & Semantic AFD & 78.59% & 85.61%\
Qwen2.5-VL-3B-Instruct & Random baseline & 50.91% & 66.78%\
LLaVA-1.5-7B & Question-aligned uncertainty & 64.76% & **83.39%**\
LLaVA-1.5-7B & AFD frequency & **69.55%** &\
LLaVA-1.5-7B & Semantic AFD & & 80.97%\
LLaVA-1.5-7B & Random baseline & 50.14% & 71.78%\
MedGemma-4B-it & Question-aligned uncertainty & 62.96% & **77.92%**\
MedGemma-4B-it & AFD frequency & **68.63%** &\
MedGemma-4B-it & Semantic AFD & & 72.73%\
MedGemma-4B-it & Random baseline & 50.42% & 62.87%\

[lXXrr]{} Dataset & Model & Best detector & $\Delta$ROUGE-L &
Failure-rate reduction\
PathVQA & Qwen2.5-VL-3B-Instruct & Question-aligned & **21.65%** &
**21.52%**\
PathVQA & LLaVA-1.5-7B & AFD frequency & 10.89% & 10.32%\
PathVQA & MedGemma-4B-it & AFD frequency & 12.79% & 11.88%\
VQA-RAD & Qwen2.5-VL-3B-Instruct & Semantic AFD & **12.48%** &
**11.51%**\
VQA-RAD & LLaVA-1.5-7B & Semantic AFD & 8.09% & 7.07%\
VQA-RAD & MedGemma-4B-it & Question-aligned & 3.11% & 4.87%\
ProstateMM-CHIMERA & Qwen2.5-VL-3B-Instruct & Question-aligned & 6.13% &
14.28%\
ProstateMM-CHIMERA & LLaVA-1.5-7B & Question-aligned & 7.26% &
**28.57%**\
ProstateMM-CHIMERA & MedGemma-4B-it & Question-aligned & **11.68%** &
**28.57%**\

Figures \[fig:performance-rejection-qwen\]–\[fig:performance-rejection-medgemma\]
show that informed detectors usually improve PathVQA accepted ROUGE-L as
exclusion increases. On ProstateMM-CHIMERA, question-aligned uncertainty
improves all three models, whereas AFD frequency can perform below
random for Qwen2.5-VL-3B-Instruct and LLaVA-1.5-7B, supporting
dataset-specific detector selection.

![Accepted ROUGE-L by exclusion for Qwen2.5-VL-3B-Instruct on PathVQA
and
ProstateMM-CHIMERA.[]{data-label="fig:performance-rejection-qwen"}](../Images/performance_rejection_qwen.pdf){width="92.00000%"}

![Accepted ROUGE-L by exclusion for LLaVA-1.5-7B on PathVQA and
ProstateMM-CHIMERA.[]{data-label="fig:performance-rejection-llava"}](../Images/performance_rejection_llava.pdf){width="92.00000%"}

![Accepted ROUGE-L by exclusion for MedGemma-4B-it on PathVQA and
ProstateMM-CHIMERA.[]{data-label="fig:performance-rejection-medgemma"}](../Images/performance_rejection_medgemma.pdf){width="92.00000%"}

Model Ranking Across Dimensions
===============================

Table \[tab:best-detector-all\] gives the best detector for every model
and dataset. PathVQA ranks Qwen2.5-VL-3B-Instruct, LLaVA-1.5-7B, then
MedGemma-4B-it by AUROC, agreeing with BLEU but conflicting with ROUGE-L
and METEOR. The clearest contrast is between Qwen2.5-VL-3B-Instruct and
MedGemma-4B-it: MedGemma-4B-it leads the latter answer metrics, yet
Qwen2.5-VL-3B-Instruct’s best AUROC is 12.57 percentage points higher.
LLaVA-1.5-7B also has the lowest answer scores but a slightly higher
PathVQA AUROC than MedGemma-4B-it. Average answer quality therefore does
not determine failure ranking.

[lXXrr]{} Dataset & Model & Best detector & AUROC & AUPRC\
PathVQA & Qwen2.5-VL-3B-Instruct & Question-aligned & **81.20%** &
**90.71%**\
PathVQA & LLaVA-1.5-7B & AFD frequency & &\
PathVQA & MedGemma-4B-it & AFD frequency & 68.63% & 75.61%\
VQA-RAD & Qwen2.5-VL-3B-Instruct & Semantic AFD & **69.43%** &\
VQA-RAD & LLaVA-1.5-7B & Semantic AFD & 63.67% & **73.53%**\
VQA-RAD & MedGemma-4B-it & Question-aligned & & 61.47%\
ProstateMM-CHIMERA & Qwen2.5-VL-3B-Instruct & Question-aligned & 81.69%
&\
ProstateMM-CHIMERA & LLaVA-1.5-7B & Question-aligned & & 86.12%\
ProstateMM-CHIMERA & MedGemma-4B-it & Question-aligned & **93.88%** &
**89.59%**\

Overall Validation
==================

VQA-RAD confirms divergence: MedGemma-4B-it leads three answer metrics,
but Qwen2.5-VL-3B-Instruct has the highest AUROC at 69.43%. At 50%
coverage, its Semantic AFD raises accepted ROUGE-L from 48.07% to 60.55%
and lowers failure rate from 47.35% to 35.84%. ProstateMM-CHIMERA
reverses the pattern: MedGemma-4B-it leads three answer metrics and
failure detection at 93.88% AUROC; its question-aligned detector raises
accepted ROUGE-L from 27.71% to 39.39% and reduces the 50%-coverage
failure rate from 28.57% to 0.00%. This aligned result is based on only
21 accepted task records. Detector choice also changes across settings,
so neither a model nor a method is uniformly dominant.

Representative Quantitative Cases
=================================

Table \[tab:representative-quantitative-cases\] provides one verified
50%-coverage case per dataset using aggregate outputs. Each informed
detector improves accepted ROUGE-L and lowers failure rate relative to
an equally sized random subset. The PathVQA case gives gains of 21.65
and 21.52 percentage points, VQA-RAD gives 12.48 and 11.51 points, and
ProstateMM-CHIMERA gives an 11.68-point ROUGE-L gain and a 28.57-point
failure-rate reduction. These are quantitative operating-point cases,
not individual clinical examples; item-level images and answers will be
reported only if complete scored records can be verified.

[lXrrr]{} Dataset & Model and detector & Records & Accepted ROUGE-L &
Failure rate\
PathVQA & Qwen2.5-VL-3B-Instruct; question-aligned & 3,360 & 32.20%
$\rightarrow$ 53.85% & 66.40% $\rightarrow$ 44.88%\
VQA-RAD & Qwen2.5-VL-3B-Instruct; Semantic AFD & 226 & 48.07%
$\rightarrow$ 60.55% & 47.35% $\rightarrow$ 35.84%\
ProstateMM-CHIMERA & MedGemma-4B-it; question-aligned & 21 & 27.71%
$\rightarrow$ 39.39% & 28.57% $\rightarrow$ 0.00%\

Hypothesis and Objective Support
=================================

| Hypothesis or objective | Evidence | Main result | Interpretation |
|---|---|---|---|
| H1: answer quality and failure awareness can diverge | Cross-dataset answer and detector rankings | VQA-RAD and PathVQA show different leaders; ProstateMM-CHIMERA is aligned | Supported conditionally |
| H2: detector performance depends on model and dataset | AUROC/AUPRC for three models and four retained methods | The best detector changes across model--dataset pairs | Supported descriptively |
| H3: informative detection improves selective operation | Accepted ROUGE-L and failure rate at matched coverage | Selected detectors improve the 50% operating point over random ranking | Descriptively supported |
| Objective: compare utility and reliability separately | Separate greedy-answer and sampled-answer paths | No combined score is used | Achieved |

The support is descriptive because the study uses one inference run, one seed,
and a metric-defined failure endpoint. It is not statistical confirmation or
clinical validation.

Summary of Findings
===================

Three findings answer the research question. First, answer quality is
metric-dependent, especially on PathVQA. Second, Qwen2.5-VL-3B-Instruct
has the strongest failure-detection AUROC on PathVQA and VQA-RAD even
when MedGemma-4B-it leads broader answer-alignment metrics. Third, this
divergence is not universal because MedGemma-4B-it leads both dimensions
on the small ProstateMM-CHIMERA test. Question-aligned uncertainty is
also useful but conditional: it is best for Qwen2.5-VL-3B-Instruct on
PathVQA and all three prostate conditions, but not for every PathVQA or
VQA-RAD model. Better answer generation therefore does not necessarily
imply better failure awareness, and both dimensions must be reported
separately.


