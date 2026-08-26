# Chapter 3: Methods

This chapter implements the comparison defined in Chapter 2. Answer
quality and operational failure detection are evaluated separately, and
the failure endpoint is fixed before detector performance is measured
[@carlini2026qasnne].

Study Design
============

The study is quantitative, retrospective, and preclinical.
Qwen2.5-VL-3B-Instruct, LLaVA-1.5-7B, and MedGemma-4B-it are evaluated
zero-shot on each dataset using one inference run. One greedy answer per
item is compared with the reference, while three sampled answers are
used to construct black-box failure scores. No combined
performance-and-reliability score is created because the research
question concerns disagreement between the two dimensions. The models
differ in size, training, architecture, and medical specialisation, so
the experiment describes the behaviour of the released checkpoints under
a controlled interface rather than the causal effect of one model
component.

Models and Configuration
========================

The selected checkpoints represent three instruction-following VLM
families. Each retains its supported processor and chat template, but
all receives the same content within a dataset. Four-bit loading is used
to make inference feasible and is not varied as an experimental
condition. No model is fine-tuned on PathVQA, VQA-RAD, or
ProstateMM-CHIMERA.

[&gt;p[3.1cm]{}cX]{} Model & Parameters & Model family and structure\
Qwen2.5-VL-3B-Instruct & approximately 3B & Qwen2.5-VL multimodal
language model with a visual encoder, multimodal projection, and
instruction-tuned language backbone [@bai2023qwenvl].\
LLaVA-1.5-7B & approximately 7B & CLIP-based visual encoder connected to
a 7B language model through a learned projection layer
[@liu2023visualinstruction].\
MedGemma-4B-it & approximately 4B & Instruction-tuned medical adaptation
of the Gemma multimodal family for biomedical image and text tasks
[@sellergren2025medgemma].\

Evaluation Framework
====================

Each item contains image $x_i$, question $q_i$, and reference $y_i$;
ProstateMM-CHIMERA also includes clinical context $z_i$. The prompt,
test items, answer rules, metric implementation, failure thresholds, and
detector settings are fixed across models within a dataset. The greedy
answer defines answer quality and the operational label, whereas the
sampled answers define the detector scores.
Figure \[fig:reliability-centred-framework\] shows this separation and
the final comparison of answer-quality and failure-detection rankings.

![Reliability-centred evaluation framework. One greedy answer is used
for answer quality and the operational failure label; three sampled
answers produce black-box detector scores evaluated using AUROC, AUPRC,
and selective analysis. The accept-or-review branch is illustrative, not
a validated clinical
policy.[]{data-label="fig:reliability-centred-framework"}](../Images/reliability_centred_vlm_framework.pdf){width="\textwidth"}

Inference Protocol
==================

No training examples are placed in the prompt and no model weights are
updated. Prompts request only yes or no for binary questions and a short
medical phrase for open questions. The greedy answer and three
stochastic answers are generated as
$$\hat{y}_i^{(0)}=\operatorname{GreedyDecode}[p_\theta(y\mid x_i,q_i,z_i)], \qquad
\hat{y}_i^{(k)}\sim p_\theta(y\mid x_i,q_i,z_i;\tau,p),\; k=1,2,3,$$
with temperature $\tau=0.7$, nucleus probability $p=90\%$, and seed 42.
Maximum new tokens are 24 for PathVQA, 32 for VQA-RAD, and 48 for
ProstateMM-CHIMERA. Cleaning removes assistant labels, answer prefixes,
extra whitespace, and text after the first non-empty line without
manually correcting content. One run is used, so generation-seed
variability is not estimated.

Answer Quality
==============

After lower-casing, punctuation removal, and whitespace normalisation,
the greedy answer is scored with BLEU-1, BLEU-2, ROUGE-L, and METEOR
[@papineni2002bleu; @lin2004rouge; @banerjee2005meteor]. BLEU uses
clipped n-gram precision $p_n$ and brevity penalty $\mathrm{BP}$,
$$\mathrm{BLEU}_N=\mathrm{BP}\exp\!\left(\sum_{n=1}^{N}w_n\log p_n\right),$$
while item-level ROUGE-L is the F-measure of precision and recall
derived from the longest common subsequence $L_i$,
$$P_{L,i}=\frac{L_i}{|\hat y_i^{(0)}|},\quad R_{L,i}=\frac{L_i}{|y_i|},\quad
\mathrm{ROUGE\mbox{-}L}_i=\frac{(1+\beta^2)P_{L,i}R_{L,i}}{R_{L,i}+\beta^2P_{L,i}}.$$
The four metrics are reported separately because they reward different
properties. Standard normalised values are displayed as percentages.

Operational Failure Definition
==============================

The binary endpoint is defined from the greedy answer:
$$F_i=\mathbb{I}[\mathrm{ROUGE\mbox{-}L}_i<20\%\;\land\;\mathrm{METEOR}_i<10\%].$$
The conjunction and thresholds are fixed across all models and datasets.
This is a reproducible label for low reference agreement, not an
expert-adjudicated hallucination or clinical error. It may penalise
valid paraphrases or miss medically important errors that retain
reference words. Calculating the label from the greedy answer and the
predictor from sampled answers also prevents the same samples from
defining both target and score.

Failure Detection Methods
=========================

Three non-random scores and one random baseline are retained. Larger
values always indicate greater uncertainty; answer disagreement
calculated during development is omitted because it overlaps with the
retained semantic condition.

AFD Frequency
-------------

After normalising case, punctuation, and yes/no variants, let $c_i(a)$
be the count of answer $a$ among $K=3$ samples. Exact-agreement
uncertainty is $$u_i^{\mathrm{freq}}=1-\frac{\max_a c_i(a)}{K}.$$ It is
transparent but treats paraphrases as different answers.

Semantic AFD
------------

Answers are embedded using BAAI/bge-small-en-v1.5 and L2 normalised.
Samples with cosine similarity of at least 80% are connected, and
connected components form semantic groups $\mathcal C_i$. The score is
$$u_i^{\mathrm{sem}}=1-\frac{\max_{C\in\mathcal C_i}|C|}{K}.$$ This
permits paraphrases to agree but depends on the embedding and threshold.

Question Aligned Uncertainty
----------------------------

This score combines mean question-answer alignment with mean consistency
between sampled answers. For normalised embedding $e(\cdot)$,
$$s_i^{\mathrm{qa}}=\frac{1}{K}\sum_{k=1}^{K}\frac{1+\cos(e(q_i),e(\hat y_i^{(k)}))}{2},\qquad
s_i^{\mathrm{cons}}=\frac{1}{\binom K2}\sum_{j<k}\frac{1+\cos(e(\hat y_i^{(j)}),e(\hat y_i^{(k)}))}{2},$$
and $u_i^{\mathrm{qa}}=1-s_i^{\mathrm{qa}}s_i^{\mathrm{cons}}$, clipped
to $[0\%,100\%]$. Low alignment or consistency raises uncertainty. The
score follows the question-validity motivation of Carlini et al. but is
a simpler product of mean similarities, not their QA-SNNE algorithm
[@carlini2026qasnne].
The original implementation labels this quantity ``Question-aligned entropy''.
Because it is not a probability entropy, this dissertation reports it as
Question-aligned uncertainty for clarity.

Random Baseline
---------------

A deterministic pseudo-random score in $[0\%,100\%]$ is generated from
seed 42 and the item index. It represents a ranking with no answer
information; one realisation need not give exactly 50% AUROC.

Reliability Analysis
====================

AUROC is the primary detector metric and equals
$\Pr(u^+>u^-)+\tfrac12\Pr(u^+=u^-)$; AUPRC is secondary because it
changes with failure prevalence. Detectors are first compared within
each model and dataset, after which the best-AUROC model ranking is
compared with answer-quality rankings across datasets. Selective
analysis sorts items from low to high uncertainty. At coverage $c$, the
first $\lceil cN\rceil$ items form accepted set $A_c$, with
$$Q_R(c)=\frac{1}{|A_c|}\sum_{i\in A_c}\mathrm{ROUGE\mbox{-}L}_i,\qquad
\phi(c)=\frac{1}{|A_c|}\sum_{i\in A_c}F_i.$$ Accepted ROUGE-L $Q_R(c)$
and failure rate $\phi(c)$ are compared with random ranking at fixed
coverages. AUROC measures ordering, not calibrated probability, and no
clinical operating threshold is selected.

Reproducibility and Ethics
==========================

The implementation fixes model identifiers, seed, splits, prompts,
decoding values, normalisation, metric libraries, failure thresholds,
embedding model, semantic threshold, and coverage levels; outputs are
stored by dataset and model. No new patient data are collected, and
ProstateMM-CHIMERA identifiers are retained only to preserve grouping.
The evaluation is not a clinical trial. Its main ethical risk is
overstatement: a high AUROC against a lexical proxy does not prove that
a model recognises clinically harmful answers, and any future use would
require expert labels, patient-level statistics, subgroup assessment,
calibrated thresholds, and prospective workflow testing.


