# Chapter 5: Discussion

This chapter interprets the findings rather than repeating every value.
It explains why answer quality and failure detection can produce
different rankings, how the pattern changes across datasets, and what
the evidence can and cannot support.

Main Findings
=============

The main finding is that stronger answer generation does not guarantee
stronger operational failure detection. VQA-RAD shows the clearest
divergence: MedGemma-4B-it leads three answer metrics, while
Qwen2.5-VL-3B-Instruct has the highest detector AUROC. PathVQA shows the
same contrast under ROUGE-L and METEOR, although Qwen2.5-VL-3B-Instruct
leads BLEU and AUROC, demonstrating that the conclusion also depends on
how answer quality is represented. ProstateMM-CHIMERA produces alignment
instead, with MedGemma-4B-it leading three answer measures and AUROC.
Detector rankings also change: frequency is strongest for two PathVQA
models, Semantic AFD for two VQA-RAD models, and question-aligned
uncertainty for all three prostate conditions. The selective results
give these rankings practical meaning because the selected detectors
improve retained ROUGE-L and reduce the metric-defined failure rate
relative to random selection.

Divergence Between Answer Performance and Reliability
=====================================================

The divergence follows from the different questions asked by the
metrics. Answer measures estimate average similarity between one greedy
output and a reference; AUROC tests whether sampled-output behaviour
separates lower-quality greedy answers from the rest. A model can
therefore generate stronger answers on average while assigning similar
detector scores to its good and bad cases, whereas a weaker generator
can expose failures through greater surface variation, semantic
variation, or loss of question alignment. A lower failure burden is also
not the same as better detection: MedGemma-4B-it has the lowest
displayed random-subset failure rates, yet Qwen2.5-VL-3B-Instruct
separates failures more clearly on PathVQA and VQA-RAD. The term
reliability here means only this ranking ability. It does not mean that
Qwen2.5-VL-3B-Instruct is clinically safer, because clinical safety also
includes grounding, harmful omissions, calibration, robustness, subgroup
behaviour, and consequences for patients.

Interpretation of Model Differences
===================================

The three checkpoints combine different parameter scales, visual
encoders, training mixtures, and degrees of medical specialisation.
MedGemma-4B-it’s stronger ROUGE-L and METEOR may be compatible with
medical instruction tuning that produces terminology closer to the
references, but stable domain language could also make some wrong
answers appear consistent. Qwen2.5-VL-3B-Instruct’s sampled outputs may
change more clearly when its greedy answer is weak, making black-box
failure ranking easier even when average answer overlap is lower.
LLaVA-1.5-7B reinforces the distinction because it has the weakest
answer quality but does not always have the weakest AUROC. These
explanations are plausible rather than causal. Aggregate results cannot
identify which training or architectural feature created the pattern,
and a useful detector cannot compensate for poor underlying answers.
Model selection for selective use would require strength on both
dimensions.

Question Aligned Uncertainty
============================

Question-aligned uncertainty was included because internally consistent
answers can still fail to address the requested information, a
distinction developed in surgical VQA work [@carlini2026qasnne]. It
performs best for Qwen2.5-VL-3B-Instruct on PathVQA and for all three
models on ProstateMM-CHIMERA, suggesting that question relevance can add
information when tasks contain longer, context-dependent targets. It is
not universally strongest: frequency leads two PathVQA models and
Semantic AFD leads two VQA-RAD models. Cosine alignment can be
influenced by question length, binary answers, medical vocabulary, and
the embedding space, so it is not a direct test of clinical validity.
The implemented score is also a simple product of mean question-answer
alignment and answer consistency, not the QA-SNNE algorithm. Its
contribution is as one comparative detector condition, not as a new
method.

Generalisation Across Datasets
==============================

No pair of datasets gives the same complete model ranking by best AUROC.
Qwen2.5-VL-3B-Instruct leads PathVQA and VQA-RAD, while MedGemma-4B-it
leads ProstateMM-CHIMERA; the best detector also changes with the
setting. This supports the view that failure awareness is behaviour
produced by a model-task combination rather than a fixed attribute of a
checkpoint. Pathology, radiology, and contextual prostate assessment
differ in modality, answer form, and available context, all of which may
change sampling behaviour. The prostate result is especially uncertain
because 42 task records come from 14 patients and three records are
linked to each patient. Its high AUROC demonstrates separation within
that small package, not patient-level generalisation. The changed
ranking is therefore a finding rather than a failed validation and is
consistent with evidence that uncertainty behaviour changes under
dataset shift [@ovadia2019uncertainty].

Relationship to Previous Literature
===================================

The findings connect medical VQA, uncertainty, and selective prediction.
PathVQA and VQA-RAD established reference-based evaluation of free-form
medical answers, but such scores cannot show whether low-quality outputs
are recognisable [@he2020pathvqa; @lau2018vqarad]. Semantic entropy and
pairwise semantic approaches show that meaning-level variation can
reveal failures beyond token disagreement
[@farquhar2024semanticentropy; @nguyen2025pairwisesemantic]; Semantic
AFD is useful here, particularly on VQA-RAD, but its changing rank
agrees with reviews that no uncertainty method is uniformly best
[@shorinwa2025uqsurvey]. The conditional benefit of question alignment
also supports the distinction between consistency and question validity
in surgical VQA [@carlini2026qasnne]. Finally, improved retained subsets
follow the purpose of selective prediction [@geifman2017selective],
although this study evaluates ranking and rejection rather than
probability calibration [@guo2017calibration].

Implications for Healthcare VLM Evaluation
==========================================

Healthcare VLMs should be evaluated on at least two explicit axes:
answer utility and failure awareness. Answer metrics should be reported
transparently and supplemented by expert review when possible, while
detector performance should be tested against a defined endpoint and at
relevant coverage levels. The two axes also create four useful
evaluation profiles. A model may answer well and expose its failures
clearly, answer well but fail silently, answer poorly while still
identifying many weak cases, or perform poorly on both dimensions. Only
the first profile provides a reasonable starting point for selective
review, although it still requires clinical validation. This view
prevents a high average answer score from hiding a weak referral signal
and prevents a high AUROC from disguising poor underlying utility. AUROC
alone is insufficient for workflow design because a referral policy must
also consider retained answer quality, residual failures, reviewer
workload, and the harm of missed cases. Thresholds cannot be transferred
directly between tasks because the best model and detector change across
the three datasets and failure prevalence affects AUPRC and referral
value. Most importantly, the present results do not justify calling any
model clinically safe. They show only that some black-box scores rank a
lexical subset of poor answers better than chance under the tested
protocol.

Limitations
===========

The largest limitation is the operational label: ROUGE-L below 20% and
METEOR below 10% is reproducible but can penalise valid paraphrases,
miss meaning-changing errors with shared words, and cannot confirm
visual grounding or clinical harm. Only three sampled answers, one seed,
and one inference run are available, so confidence intervals,
repeated-run variability, and significance tests are absent. Complete
scored exports support the reported prevalence and selective summaries,
but they do not provide expert adjudication, clinical error categories,
or image-linked qualitative review.
Dataset scale also varies sharply, and the prostate records are small
and patient-linked without grouped resampling. Public benchmarks may
overlap with model pre-training, and zero-shot evaluation does not
isolate architecture, parameter scale, medical training, quantisation,
or chat-template effects. The results therefore describe three named
checkpoints under one protocol rather than population-level or causal
effects.

Future Research
===============

Future work should link the scored records to source images and expert
review so that correct retention, detected failure, false alarm, and
confident failure can be assigned clinically meaningful categories.
Stronger uncertainty estimates should use more sampled answers,
repeated seeds, confidence intervals, and patient-grouped resampling;
detector thresholds should be chosen on validation data and tested
separately. Analysis by yes/no status, answer length, modality, and
clinical topic could test why frequency, semantic grouping, and question
alignment change rank. White-box probabilities, visual grounding, and
perturbation-based signals should also be compared with the current
black-box scores. Finally, larger multi-cohort and prospective studies
must measure subgroup performance, clinician workload, missed harmful
errors, and user reliance before selective referral can support a real
clinical workflow.

