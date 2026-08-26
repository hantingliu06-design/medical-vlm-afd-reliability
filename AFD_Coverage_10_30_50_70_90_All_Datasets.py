# %%
# ============================================================
# AFD post-processing for three existing datasets
#
# This script does NOT run VLM inference.
# It reads already saved JSON output files and recomputes:
#   - AFD frequency
#   - Semantic AFD
#   - Answer disagreement
#   - Question-aligned entropy
#   - Random baseline
#   - Failure AUROC / AUPRC
#   - Selective metrics at 10%, 30%, 50%, 70%, and 90% coverage
#
# The largest complete run is selected for each dataset/model pair
# by default. This avoids mixing small smoke-test files with final runs.
# ============================================================


# %%
# ============================================================
# CELL 1 - Install only post-processing dependencies if needed
# ============================================================

import os
import sys
import subprocess


INSTALL_DEPENDENCIES = False


def run_pip(*packages):
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--upgrade",
            *packages,
        ]
    )


REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "tqdm": "tqdm",
    "sklearn": "scikit-learn",
    "transformers": "transformers",
    "rouge_score": "rouge-score",
    "nltk": "nltk",
}


def install_missing_dependencies():
    import importlib.util

    missing_packages = [
        package_name
        for import_name, package_name in REQUIRED_PACKAGES.items()
        if importlib.util.find_spec(import_name) is None
    ]

    if missing_packages:
        print(
            "Installing missing packages:",
            ", ".join(missing_packages),
        )
        run_pip(*missing_packages)


if INSTALL_DEPENDENCIES:
    run_pip(*REQUIRED_PACKAGES.values())
else:
    install_missing_dependencies()


# %%
# ============================================================
# CELL 2 - Imports, Drive, and configuration
# ============================================================

import gc
import json
import math
import random
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from tqdm.auto import tqdm
from rouge_score import rouge_scorer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore")


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    COMPUTE_DTYPE = (
        torch.bfloat16
        if hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
        else torch.float16
    )
else:
    COMPUTE_DTYPE = torch.float32

print("Device:", DEVICE)
if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


DRIVE_MOUNT_DIR = "/content/gdrive"
MY_DRIVE_DIR = os.path.join(
    DRIVE_MOUNT_DIR,
    "MyDrive",
)

try:
    from google.colab import drive

    if not os.path.isdir(MY_DRIVE_DIR):
        drive.mount(
            DRIVE_MOUNT_DIR,
            force_remount=False,
        )
except ImportError:
    print("Google Colab drive module is unavailable.")


if not os.path.isdir(MY_DRIVE_DIR):
    raise RuntimeError(
        "Google Drive is not mounted. "
        "Mount Drive before running this cell."
    )


HF_HOME = os.path.join(
    MY_DRIVE_DIR,
    "hf_cache",
)
HF_HUB_CACHE = os.path.join(
    HF_HOME,
    "hub",
)

OUTPUT_DIRS = {
    # Change only these three paths if your folders have different names.
    "PathVQA": os.path.join(
        MY_DRIVE_DIR,
        "pathvqa_afd_full_comparison",
    ),
    "VQA-RAD": os.path.join(
        MY_DRIVE_DIR,
        "vqa_rad_afd_comparison",
    ),
    "ProstateMM-CHIMERA": os.path.join(
        MY_DRIVE_DIR,
        "prostatemm_chimera_afd_comparison",
    ),
}

AFD_OUTPUT_DIR = os.path.join(
    MY_DRIVE_DIR,
    "afd_coverage_10_30_50_70_90_all_datasets",
)

for directory in [
    HF_HOME,
    HF_HUB_CACHE,
    AFD_OUTPUT_DIR,
]:
    os.makedirs(
        directory,
        exist_ok=True,
    )

os.environ["HF_HOME"] = HF_HOME
os.environ["HF_HUB_CACHE"] = HF_HUB_CACHE


# Selective coverage:
# 10%, 30%, 50%, 70%, 90%
SELECTIVE_COVERAGES = (
    0.10,
    0.30,
    0.50,
    0.70,
    0.90,
)

FAILURE_ROUGE_L_THRESHOLD = 0.2
FAILURE_METEOR_THRESHOLD = 0.1

EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"
EMBED_BATCH_SIZE = 64
EMBED_MAX_LENGTH = 128
SEMANTIC_SIMILARITY_THRESHOLD = 0.80

# "largest" keeps the largest complete run per dataset/model pair.
# Change to "all" if you want to evaluate every complete JSON file.
RUN_SELECTION = "largest"


# %%
# ============================================================
# CELL 3 - NLTK resources
# ============================================================

import nltk


def ensure_nltk_resources():
    nltk_data_dir = os.path.join(
        MY_DRIVE_DIR,
        "nltk_data",
    )
    os.makedirs(
        nltk_data_dir,
        exist_ok=True,
    )

    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.insert(
            0,
            nltk_data_dir,
        )

    os.environ["NLTK_ALLOW_PROXIED_URLOPEN"] = "1"

    try:
        from nltk import pathsec

        pathsec.ALLOW_PROXIED_FETCH = True
    except ImportError:
        pass

    resources = {
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }

    for resource_path, download_name in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            downloaded = nltk.download(
                download_name,
                download_dir=nltk_data_dir,
                quiet=False,
            )
            if not downloaded:
                raise RuntimeError(
                    f"Could not download NLTK resource: "
                    f"{download_name}"
                )


ensure_nltk_resources()

from nltk.translate.meteor_score import meteor_score


# %%
# ============================================================
# CELL 4 - Text metrics and normalisation
# ============================================================


def normalize_text(text):
    if text is None:
        return ""

    text = str(text).lower().strip()
    text = re.sub(
        r"[^a-z0-9+\-\s]",
        " ",
        text,
    )
    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()
    return text


def normalize_for_frequency(text):
    text = normalize_text(text)

    yes_set = {
        "yes",
        "yeah",
        "yep",
        "true",
        "present",
    }
    no_set = {
        "no",
        "nope",
        "false",
        "absent",
    }

    if text in yes_set:
        return "yes"
    if text in no_set:
        return "no"

    text = re.sub(
        r"\b(?:the|a|an)\b",
        " ",
        text,
    )
    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


sample_rouge_scorer = rouge_scorer.RougeScorer(
    ["rougeL"],
    use_stemmer=True,
)


def compute_single_metrics(
    prediction,
    reference,
):
    prediction = normalize_text(
        prediction
    ) or "empty"
    reference = normalize_text(
        reference
    ) or "empty"

    prediction_tokens = prediction.split()
    reference_tokens = reference.split()

    rouge_l = sample_rouge_scorer.score(
        reference,
        prediction,
    )["rougeL"].fmeasure

    meteor_value = meteor_score(
        [reference_tokens],
        prediction_tokens,
    )

    failure = int(
        rouge_l < FAILURE_ROUGE_L_THRESHOLD
        and meteor_value < FAILURE_METEOR_THRESHOLD
    )

    return {
        "rougeL": float(rouge_l),
        "meteor": float(meteor_value),
        "failure": failure,
    }


def safe_auroc(
    labels,
    scores,
):
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    if (
        len(labels) == 0
        or np.unique(labels).size < 2
    ):
        return np.nan

    return float(
        roc_auc_score(
            labels,
            scores,
        )
    )


def safe_auprc(
    labels,
    scores,
):
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    if (
        len(labels) == 0
        or np.unique(labels).size < 2
    ):
        return np.nan

    return float(
        average_precision_score(
            labels,
            scores,
        )
    )


# %%
# ============================================================
# CELL 5 - Load semantic embedding model
# ============================================================

print("Loading embedding model:", EMBED_MODEL_ID)

embed_tokenizer = AutoTokenizer.from_pretrained(
    EMBED_MODEL_ID,
    cache_dir=HF_HUB_CACHE,
)

embed_model = AutoModel.from_pretrained(
    EMBED_MODEL_ID,
    cache_dir=HF_HUB_CACHE,
    torch_dtype=(
        torch.float16
        if DEVICE == "cuda"
        else torch.float32
    ),
).to(DEVICE).eval()

print("Embedding model loaded.")


@torch.inference_mode()
def encode_texts(
    texts,
    batch_size=EMBED_BATCH_SIZE,
):
    if not texts:
        return np.empty(
            (
                0,
                embed_model.config.hidden_size,
            ),
            dtype=np.float32,
        )

    all_embeddings = []

    for start in range(
        0,
        len(texts),
        batch_size,
    ):
        batch_texts = [
            normalize_text(text)
            or "empty"
            for text in texts[
                start:start + batch_size
            ]
        ]

        encoded = embed_tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=EMBED_MAX_LENGTH,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(DEVICE)
            for key, value in encoded.items()
        }

        outputs = embed_model(
            **encoded
        )
        token_embeddings = outputs.last_hidden_state
        attention_mask = encoded[
            "attention_mask"
        ]
        mask = attention_mask.unsqueeze(
            -1
        ).expand(
            token_embeddings.size()
        ).float()

        pooled = torch.sum(
            token_embeddings * mask,
            dim=1,
        ) / torch.clamp(
            mask.sum(dim=1),
            min=1e-9,
        )

        pooled = F.normalize(
            pooled,
            p=2,
            dim=1,
        )

        all_embeddings.append(
            pooled.detach()
            .cpu()
            .float()
            .numpy()
        )

    return np.concatenate(
        all_embeddings,
        axis=0,
    )


def mean_pairwise_similarity(
    embeddings,
):
    if len(embeddings) <= 1:
        return 1.0

    similarity_matrix = (
        embeddings @ embeddings.T
    )
    upper_indices = np.triu_indices(
        len(embeddings),
        k=1,
    )

    return float(
        np.mean(
            similarity_matrix[
                upper_indices
            ]
        )
    )


# %%
# ============================================================
# CELL 6 - AFD scoring methods
# ============================================================


def score_random(
    record,
    features,
):
    record_rng = random.Random(
        SEED + int(record.get("index", 0))
    )
    return float(record_rng.random())


def score_afd_frequency(
    record,
    features,
):
    sampled = record.get(
        "sampled_answers",
        [],
    )

    if not sampled:
        return 1.0

    normalised = [
        normalize_for_frequency(answer)
        for answer in sampled
    ]

    counts = {}
    for position, answer in enumerate(
        normalised
    ):
        key = (
            answer
            or f"__empty_{position}"
        )
        counts[key] = (
            counts.get(key, 0)
            + 1
        )

    most_common_count = max(
        counts.values()
    )
    reliability = (
        most_common_count
        / len(normalised)
    )
    return float(
        np.clip(
            1.0 - reliability,
            0.0,
            1.0,
        )
    )


def score_semantic_afd(
    record,
    features,
):
    answer_embeddings = features[
        "answer_embeddings"
    ]
    number_of_answers = len(
        answer_embeddings
    )

    if number_of_answers <= 1:
        return 0.0

    similarity_matrix = (
        answer_embeddings
        @ answer_embeddings.T
    )
    visited = np.zeros(
        number_of_answers,
        dtype=bool,
    )
    cluster_sizes = []

    for start_index in range(
        number_of_answers
    ):
        if visited[start_index]:
            continue

        stack = [start_index]
        visited[start_index] = True
        cluster_size = 0

        while stack:
            current_index = stack.pop()
            cluster_size += 1

            neighbours = np.where(
                similarity_matrix[
                    current_index
                ]
                >= SEMANTIC_SIMILARITY_THRESHOLD
            )[0]

            for neighbour_index in neighbours:
                if not visited[
                    neighbour_index
                ]:
                    visited[
                        neighbour_index
                    ] = True
                    stack.append(
                        int(neighbour_index)
                    )

        cluster_sizes.append(
            cluster_size
        )

    largest_cluster = max(
        cluster_sizes
    )
    return float(
        np.clip(
            1.0
            - largest_cluster
            / number_of_answers,
            0.0,
            1.0,
        )
    )


def score_answer_disagreement(
    record,
    features,
):
    answer_embeddings = features[
        "answer_embeddings"
    ]

    if len(answer_embeddings) == 0:
        return 1.0

    consistency = (
        mean_pairwise_similarity(
            answer_embeddings
        )
        + 1.0
    ) / 2.0

    return float(
        np.clip(
            1.0 - consistency,
            0.0,
            1.0,
        )
    )


def score_question_aligned_entropy(
    record,
    features,
):
    question_embedding = features[
        "question_embedding"
    ]
    answer_embeddings = features[
        "answer_embeddings"
    ]

    if len(answer_embeddings) == 0:
        return 1.0

    qa_similarities = (
        answer_embeddings
        @ question_embedding
    )
    qa_alignment = float(
        np.mean(
            (qa_similarities + 1.0)
            / 2.0
        )
    )

    answer_consistency = (
        mean_pairwise_similarity(
            answer_embeddings
        )
        + 1.0
    ) / 2.0

    reliability = (
        qa_alignment
        * answer_consistency
    )
    return float(
        np.clip(
            1.0 - reliability,
            0.0,
            1.0,
        )
    )


METHODS = {
    "Random baseline": score_random,
    "AFD frequency": score_afd_frequency,
    "Semantic AFD": score_semantic_afd,
    "Answer disagreement": score_answer_disagreement,
    "Question-aligned entropy": (
        score_question_aligned_entropy
    ),
}


# %%
# ============================================================
# CELL 7 - JSON compatibility and file selection
# ============================================================


def load_output_payload(
    json_path,
):
    with open(
        json_path,
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    if (
        isinstance(payload, dict)
        and "records" in payload
    ):
        return (
            payload.get(
                "metadata",
                {},
            ),
            list(payload["records"]),
        )

    return {}, list(payload)


def get_first_value(
    record,
    keys,
    default="",
):
    for key in keys:
        value = record.get(key)
        if value is not None:
            return value
    return default


def standardise_record(
    record,
    index,
):
    sampled_answers = get_first_value(
        record,
        [
            "sampled_answers",
            "sampled_predictions",
            "samples",
        ],
        default=[],
    )

    if isinstance(sampled_answers, str):
        try:
            sampled_answers = json.loads(
                sampled_answers
            )
        except json.JSONDecodeError:
            sampled_answers = [
                sampled_answers
            ]

    if not isinstance(
        sampled_answers,
        list,
    ):
        sampled_answers = list(
            sampled_answers
        )

    return {
        **record,
        "index": int(
            record.get(
                "index",
                index,
            )
        ),
        "question": str(
            get_first_value(
                record,
                [
                    "question",
                    "Question",
                    "prompt",
                ],
                default="",
            )
        ),
        "ground_truth": str(
            get_first_value(
                record,
                [
                    "ground_truth",
                    "primary_reference",
                    "reference",
                    "answer",
                    "gt_answer",
                ],
                default="",
            )
        ),
        "greedy_prediction": str(
            get_first_value(
                record,
                [
                    "greedy_prediction",
                    "prediction",
                    "generated_answer",
                    "model_answer",
                ],
                default="",
            )
        ),
        "sampled_answers": [
            str(answer)
            for answer in sampled_answers
        ],
    }


def is_complete(
    metadata,
    records,
):
    expected_size = metadata.get(
        "eval_size"
    )

    if expected_size is None:
        return len(records) > 0

    return len(records) == int(
        expected_size
    )


def infer_model_name(
    metadata,
    records,
    json_path,
):
    value = get_first_value(
        metadata,
        [
            "model_key",
            "model",
            "model_id",
        ],
        default=None,
    )

    if value:
        return str(value)

    if records:
        value = get_first_value(
            records[0],
            [
                "model_key",
                "model",
                "model_id",
            ],
            default=None,
        )
        if value:
            return str(value)

    return Path(
        json_path
    ).stem


def discover_json_files(
    dataset_dir,
):
    if not os.path.isdir(dataset_dir):
        print(
            "Directory not found:",
            dataset_dir,
        )
        return []

    return sorted(
        [
            os.path.join(
                dataset_dir,
                file_name,
            )
            for file_name in os.listdir(
                dataset_dir
            )
            if file_name.startswith(
                "outputs_"
            )
            and file_name.endswith(
                ".json"
            )
        ]
    )


def select_json_files(
    dataset_name,
    dataset_dir,
):
    candidates = []

    for json_path in discover_json_files(
        dataset_dir
    ):
        try:
            metadata, records = (
                load_output_payload(
                    json_path
                )
            )
        except Exception as error:
            print(
                "Could not read:",
                json_path,
                repr(error),
            )
            continue

        if not is_complete(
            metadata,
            records,
        ):
            print(
                "Skipping incomplete:",
                os.path.basename(
                    json_path
                ),
                f"({len(records)}/"
                f"{metadata.get('eval_size')})",
            )
            continue

        model_name = infer_model_name(
            metadata,
            records,
            json_path,
        )

        candidates.append(
            {
                "dataset": dataset_name,
                "json_path": json_path,
                "metadata": metadata,
                "records": records,
                "model": model_name,
                "sample_count": len(
                    records
                ),
            }
        )

    if RUN_SELECTION == "all":
        return candidates

    selected = {}
    for candidate in candidates:
        key = (
            candidate["dataset"],
            candidate["model"],
        )
        previous = selected.get(key)

        if (
            previous is None
            or candidate["sample_count"]
            > previous["sample_count"]
        ):
            selected[key] = candidate

    return list(
        selected.values()
    )


selected_runs = []

for dataset_name, dataset_dir in (
    OUTPUT_DIRS.items()
):
    print("=" * 80)
    print(
        f"Searching {dataset_name}:",
        dataset_dir,
    )

    selected_dataset_runs = (
        select_json_files(
            dataset_name,
            dataset_dir,
        )
    )
    selected_runs.extend(
        selected_dataset_runs
    )

    for run in selected_dataset_runs:
        print(
            "Selected:",
            os.path.basename(
                run["json_path"]
            ),
            "| samples:",
            run["sample_count"],
            "| model:",
            run["model"],
        )

if not selected_runs:
    raise RuntimeError(
        "No complete output JSON files were found. "
        "Check OUTPUT_DIRS."
    )


# %%
# ============================================================
# CELL 8 - Selective metrics: 10%, 30%, 50%, 70%, 90%
# ============================================================


def selective_metrics(
    records,
    uncertainty_scores,
):
    evaluation_df = pd.DataFrame(
        records
    ).copy()
    evaluation_df[
        "uncertainty"
    ] = uncertainty_scores

    evaluation_df = (
        evaluation_df
        .sort_values(
            "uncertainty",
            ascending=True,
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    output = {}

    for coverage in (
        SELECTIVE_COVERAGES
    ):
        accepted_count = max(
            1,
            int(
                math.ceil(
                    len(evaluation_df)
                    * coverage
                )
            ),
        )
        accepted_df = (
            evaluation_df.iloc[
                :accepted_count
            ]
        )
        percentage = int(
            round(
                coverage
                * 100
            )
        )

        output[
            f"Accepted samples @{percentage}%"
        ] = int(
            len(accepted_df)
        )
        output[
            f"Accepted ROUGE-L @{percentage}%"
        ] = float(
            accepted_df[
                "rougeL"
            ].mean()
        )
        output[
            f"Accepted METEOR @{percentage}%"
        ] = float(
            accepted_df[
                "meteor"
            ].mean()
        )
        output[
            f"Accepted failure rate @{percentage}%"
        ] = float(
            accepted_df[
                "failure"
            ].mean()
        )

    return output


# %%
# ============================================================
# CELL 9 - Evaluate selected JSON outputs
# ============================================================


def evaluate_one_run(
    run,
):
    dataset_name = run[
        "dataset"
    ]
    json_path = run[
        "json_path"
    ]
    metadata = run[
        "metadata"
    ]
    raw_records = run[
        "records"
    ]

    print("=" * 80)
    print(
        "Evaluating:",
        dataset_name,
        "|",
        os.path.basename(
            json_path
        ),
    )
    print("=" * 80)

    records = [
        standardise_record(
            record,
            index,
        )
        for index, record in enumerate(
            raw_records
        )
    ]
    records = sorted(
        records,
        key=lambda item: int(
            item.get(
                "index",
                0,
            )
        ),
    )

    scored_records = []

    for record in tqdm(
        records,
        desc=f"{dataset_name} "
        "per-sample metrics",
    ):
        item = dict(
            record
        )
        item.update(
            compute_single_metrics(
                item[
                    "greedy_prediction"
                ],
                item[
                    "ground_truth"
                ],
            )
        )
        scored_records.append(
            item
        )

    y_true = [
        item[
            "failure"
        ]
        for item in scored_records
    ]

    semantic_texts = []
    semantic_ranges = []

    for record in scored_records:
        start_index = len(
            semantic_texts
        )
        sampled_answers = record[
            "sampled_answers"
        ]

        semantic_texts.append(
            record[
                "question"
            ]
        )
        semantic_texts.extend(
            sampled_answers
        )
        semantic_ranges.append(
            (
                start_index,
                len(
                    sampled_answers
                ),
            )
        )

    all_embeddings = encode_texts(
        semantic_texts
    )

    semantic_features = []
    for start_index, answer_count in (
        semantic_ranges
    ):
        semantic_features.append(
            {
                "question_embedding": (
                    all_embeddings[
                        start_index
                    ]
                ),
                "answer_embeddings": (
                    all_embeddings[
                        start_index + 1:
                        start_index + 1
                        + answer_count
                    ]
                ),
            }
        )

    summary_rows = []
    scored_rows = []

    for method_name, method_function in (
        METHODS.items()
    ):
        print(
            "Running:",
            dataset_name,
            "|",
            run["model"],
            "|",
            method_name,
        )

        scores = np.asarray(
            [
                method_function(
                    record,
                    features,
                )
                for record, features in zip(
                    scored_records,
                    semantic_features,
                )
            ],
            dtype=np.float64,
        )

        summary_row = {
            "Dataset": dataset_name,
            "Model": run["model"],
            "Model ID": metadata.get(
                "model_id",
                "",
            ),
            "Split": metadata.get(
                "split",
                "",
            ),
            "Evaluation samples": len(
                scored_records
            ),
            "Method": method_name,
            "Failure AUROC": safe_auroc(
                y_true,
                scores,
            ),
            "Failure AUPRC": safe_auprc(
                y_true,
                scores,
            ),
            "Mean uncertainty": float(
                np.mean(scores)
            ),
            "Source JSON": os.path.basename(
                json_path
            ),
        }

        summary_row.update(
            selective_metrics(
                scored_records,
                scores,
            )
        )
        summary_rows.append(
            summary_row
        )

        for record, score in zip(
            scored_records,
            scores,
        ):
            scored_rows.append(
                {
                    "Dataset": dataset_name,
                    "Model": run["model"],
                    "Method": method_name,
                    "index": record.get(
                        "index"
                    ),
                    "question": record.get(
                        "question"
                    ),
                    "ground_truth": record.get(
                        "ground_truth"
                    ),
                    "greedy_prediction": record.get(
                        "greedy_prediction"
                    ),
                    "sampled_answers": json.dumps(
                        record.get(
                            "sampled_answers",
                            [],
                        ),
                        ensure_ascii=False,
                    ),
                    "rougeL": record.get(
                        "rougeL"
                    ),
                    "meteor": record.get(
                        "meteor"
                    ),
                    "failure": record.get(
                        "failure"
                    ),
                    "uncertainty": float(
                        score
                    ),
                }
            )

    return (
        pd.DataFrame(
            summary_rows
        ),
        pd.DataFrame(
            scored_rows
        ),
    )


all_summary = []
all_scored = []

for run in selected_runs:
    summary_df, scored_df = (
        evaluate_one_run(
            run
        )
    )
    all_summary.append(
        summary_df
    )
    all_scored.append(
        scored_df
    )


final_summary_df = pd.concat(
    all_summary,
    ignore_index=True,
)
final_scored_df = pd.concat(
    all_scored,
    ignore_index=True,
)


# %%
# ============================================================
# CELL 10 - Save clean tables
# ============================================================


numeric_columns = (
    final_summary_df
    .select_dtypes(
        include=[np.number]
    )
    .columns
)
final_summary_df.loc[
    :,
    numeric_columns,
] = final_summary_df.loc[
    :,
    numeric_columns,
].round(4)

final_summary_df = (
    final_summary_df
    .sort_values(
        [
            "Dataset",
            "Model",
            "Failure AUPRC",
        ],
        ascending=[
            True,
            True,
            False,
        ],
        na_position="last",
    )
    .reset_index(
        drop=True
    )
)

summary_path = os.path.join(
    AFD_OUTPUT_DIR,
    "afd_summary_all_datasets_coverage_10_30_50_70_90.csv",
)
scored_path = os.path.join(
    AFD_OUTPUT_DIR,
    "afd_scored_records_all_datasets_coverage_10_30_50_70_90.csv",
)

final_summary_df.to_csv(
    summary_path,
    index=False,
)
final_scored_df.to_csv(
    scored_path,
    index=False,
)

print("=" * 80)
print("AFD evaluation finished.")
print("Summary saved to:", summary_path)
print("Scored records saved to:", scored_path)
print("=" * 80)

display(final_summary_df)
