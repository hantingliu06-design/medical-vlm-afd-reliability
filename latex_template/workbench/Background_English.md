# Chapter 1: Literature Review

Literature Review
=================

Healthcare Visual Question Answering
------------------------------------

VQA requires a model to answer a natural-language question about an
image, so the question changes which visual evidence is relevant
[@agrawal2016vqa]. Medical datasets adapted this interface to clinical
images: VQA-RAD introduced clinician-written radiology questions,
PathVQA linked pathology images with questions about organs and
findings, and SLAKE added semantic labels and external medical knowledge
[@lau2018vqarad; @he2020pathvqa; @liu2021slake]. Surgical-VQA later
examined instruments, anatomy, actions, and locations in operative
scenes [@seenivasan2022surgicalvqa]. These benchmarks made focused
medical image questioning possible, but they usually reduce a clinical
problem to one image, one short question, and a limited set of
references. A plausible answer may therefore score well without
demonstrating complete clinical reasoning or safe case-level behaviour.

General Vision Language Models
------------------------------

Earlier medical VQA systems were commonly trained for one dataset and
answer space, whereas modern VLMs connect large visual encoders and
language models. The Transformer enabled general attention-based
sequence modelling, CLIP aligned image and text representations, and
BLIP-2 connected frozen visual and language components through a learned
interface [@vaswani2017attention; @radford2021clip; @li2023blip2].
Visual instruction tuning then allowed LLaVA and Qwen-VL to answer new
image-based instructions without a task-specific classification head
[@liu2023visualinstruction; @bai2023qwenvl]. Medical adaptations such as
LLaVA-Med and MedGemma add biomedical data and instructions
[@li2023llavamed; @sellergren2025medgemma]. Their broader capability is
useful, but opaque training mixtures and flexible generation make visual
grounding and domain validation more important, not less
[@moor2023foundation; @wiens2019donoharm].

Zero Shot Evaluation
--------------------

Zero-shot evaluation tests a model without labelled training examples
from the target task. Instruction-tuned VLMs can follow a new
image-and-text prompt at inference time, which supports direct
comparison across medical datasets
[@liu2023visualinstruction; @bai2023qwenvl]. However, zero-shot does not
mean that the model has never encountered related benchmark images,
captions, or questions during pre-training. Results also depend on
prompt wording, chat templates, decoding settings, and output length; a
model may return a word, sentence, or explanation for the same content.
Consequently, zero-shot results describe the behaviour of a named
checkpoint under a stated protocol. They do not isolate architecture,
parameter scale, or training data as a causal explanation, and they do
not establish transfer to clinical practice
[@li2023llavamed; @sellergren2025medgemma].

Reference Based Answer Evaluation
---------------------------------

Free-form VQA answers are commonly compared with references using BLEU,
ROUGE-L, and METEOR. BLEU measures clipped n-gram precision, ROUGE-L
uses the longest common subsequence, and METEOR aligns words with a
fragmentation penalty
[@papineni2002bleu; @lin2004rouge; @banerjee2005meteor]. The metrics
reward different properties, so models can change rank across them. This
is especially important for short medical answers, where a correct
synonym may have low lexical overlap and an incorrect answer may retain
much of the reference wording [@lau2018vqarad; @he2020pathvqa].
Aggregate scores also hide systematic case-level failures and do not
show whether the image was used or whether the model recognises
uncertainty [@ribeiro2020checklist]. Reference metrics are therefore
useful measures of answer quality, but not complete measures of
reliability.

Hallucination and Failure Detection
-----------------------------------

VLM hallucination includes generated objects, attributes, or relations
that are unsupported by the image [@liu2024hallucinationsurvey].
Reference-free and perturbation-based approaches attempt to detect such
failures without using the correct answer at inference time
[@li2024referencefree; @zhang2024vluncertainty]. Uncertainty
quantification instead measures how uncertain the model or output
appears, while automatic failure detection asks whether a score ranks a
defined set of errors above non-errors. This distinction requires an
explicit endpoint because a score can vary widely without separating
correct and incorrect answers. Confidence is also often poorly
calibrated, and higher accuracy does not guarantee that confidence has
more reliable meaning [@guo2017calibration; @jiang2021whenknow].
Fluency, confidence, answer quality, and failure detection are therefore
related but different properties.

Sampling and Semantic Uncertainty
---------------------------------

Black-box uncertainty can be estimated by sampling several answers to
the same input. Exact frequency is transparent but treats paraphrases as
disagreement and may assign low uncertainty to a repeated error.
Semantic entropy and pairwise semantic methods compare meanings rather
than surface forms, making them better suited to free-form answers
[@farquhar2024semanticentropy; @nguyen2025pairwisesemantic]. Their
behaviour still depends on the semantic representation, grouping rule,
sample count, and decoding protocol. More importantly, agreement among
answers does not guarantee that they address the question. Surgical VQA
research therefore introduced question alignment into the uncertainty
calculation and showed that answer consistency and question validity
should be separated [@carlini2026qasnne]. No sampling method can be
treated as universally reliable without evaluation against a stated
failure label.

Selective Prediction and Dataset Shift
--------------------------------------

Selective prediction gives failure scores an operational use by
accepting lower-risk cases and referring or rejecting the rest, creating
a trade-off between coverage and error [@geifman2017selective]. A useful
detector should improve the quality of retained answers as high-risk
cases are removed, but this behaviour may weaken when the test
distribution changes. Selective question answering and broader
uncertainty studies report that the relationship between confidence and
correctness can shift across domains
[@kamath2020selectiveqa; @ovadia2019uncertainty]. External-validation
studies in radiology similarly find performance reductions outside
development data [@yu2022externalvalidation]. Image modality, question
type, answer length, clinical context, and failure prevalence can all
change detector behaviour, so ranking and selective performance must be
re-evaluated in each target setting.

Unresolved Research Gap
-----------------------

The literature shows progress in both medical answer generation and
reliability evaluation, but it usually asks either which model produces
the best answers or which uncertainty method detects failures within one
setting. Reference metrics do not establish grounding or case-level
failure awareness, while sampling and semantic scores can be stable yet
wrong and can change under dataset shift
[@farquhar2024semanticentropy; @carlini2026qasnne; @ovadia2019uncertainty].
There is less evidence on whether different healthcare VLMs retain the
same model ranking when answer quality and failure detection are
evaluated separately across medical datasets. This unresolved
relationship motivates a dual evaluation: answer utility is measured
from the greedy response, failure awareness is measured from detector
ranking and selective behaviour, and neither is inferred from the other.

