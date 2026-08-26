# %%
# ============================================================
# ProstateMM-VQA / CHIMERA zero-shot reliability evaluation
# Qwen2.5-VL + MedGemma + LLaVA + AFD
#
# This notebook-style Python file:
#   1. Installs the required packages when requested.
#   2. Mounts Google Drive and extracts ProstateMM-VQA.
#   3. Reads train/validation/test JSONL records.
#   4. Uses the dataset-provided system prompt and clinical context.
#   5. Downloads all three VLMs before inference starts.
#   6. Runs 4-bit, batched, zero-shot inference.
#   7. Saves after every batch and supports resume.
#   8. Computes BLEU, ROUGE-L, METEOR and AFD metrics.
#   9. Produces overall and task-specific CSV summaries.
#
# The default evaluation split is "test" (42 records, 14 patients).
# Set SPLIT_SELECTION = "all" to evaluate all 285 task records.
#
# Important:
#   ProstateMM-VQA contains 95 patients and 285 task instances.
#   Each patient contributes three task records. These are not 285
#   independent patients.
# ============================================================


# %%
# ============================================================
# CELL 1 - Safe environment setup
# ============================================================

import os
import sys
import subprocess


# Set True only when packages need to be installed.
# After installation, restart the Colab runtime and set it back to False.
INSTALL_DEPENDENCIES = False


def run_pip(*packages, extra_args=()):
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--no-cache-dir",
        *extra_args,
        *packages,
    ]
    print("Running:", " ".join(command))
    subprocess.check_call(command)


if INSTALL_DEPENDENCIES:
    # Do not uninstall PyTorch here. Colab already provides the CUDA build.
    run_pip(
        "numpy<2.0",
        "pandas==2.2.2",
        "scipy==1.13.1",
        "scikit-learn==1.5.2",
        "matplotlib",
        "pillow",
        "tqdm",
        "datasets==2.21.0",
        "transformers==4.51.3",
        "tokenizers==0.21.1",
        "huggingface-hub==0.30.2",
        "accelerate==1.6.0",
        "evaluate==0.4.3",
        "qwen-vl-utils",
        "nltk==3.9.1",
        "rouge-score==0.1.2",
        "sentencepiece",
        "protobuf",
        "einops",
        "safetensors",
    )

    run_pip(
        "bitsandbytes>=0.46.1",
        extra_args=("--no-deps",),
    )

    print("=" * 80)
    print("Dependencies installed.")
    print("Restart the Colab runtime before continuing.")
    print("=" * 80)


# %%
# ============================================================
# CELL 2 - Imports, Drive, dataset extraction, and configuration
# ============================================================

import gc
import json
import math
import random
import re
import shutil
import tarfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from IPython.display import display

warnings.filterwarnings("ignore")


# -----------------------------
# Reproducibility
# -----------------------------

SEED = 42
RUN_ID = "prostamm_chimera_zero_shot_v1"
FORCE_RERUN = False
CONTINUE_AFTER_MODEL_ERROR = True

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# -----------------------------
# Device and dtype
# -----------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)

if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU memory:",
        round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3,
            1,
        ),
        "GB",
    )

if DEVICE == "cuda":
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        COMPUTE_DTYPE = torch.bfloat16
    else:
        COMPUTE_DTYPE = torch.float16
else:
    COMPUTE_DTYPE = torch.float32


# -----------------------------
# Google Drive
# -----------------------------

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
        "Mount Drive before running this notebook."
    )


# -----------------------------
# Drive directories
# -----------------------------

HF_HOME = os.path.join(
    MY_DRIVE_DIR,
    "hf_cache",
)
HF_HUB_CACHE = os.path.join(
    HF_HOME,
    "hub",
)
TRANSFORMERS_CACHE = os.path.join(
    HF_HOME,
    "transformers",
)

OUTPUT_DIR = os.path.join(
    MY_DRIVE_DIR,
    "prostatemm_chimera_afd_comparison",
)

MODEL_STORAGE_DIR = os.path.join(
    MY_DRIVE_DIR,
    "pathvqa_models",
)

DATA_STORAGE_DIR = os.path.join(
    MY_DRIVE_DIR,
    "prostatemm_vqa_data",
)

for directory in [
    HF_HOME,
    HF_HUB_CACHE,
    TRANSFORMERS_CACHE,
    OUTPUT_DIR,
    MODEL_STORAGE_DIR,
    DATA_STORAGE_DIR,
]:
    os.makedirs(
        directory,
        exist_ok=True,
    )

os.environ["HF_HOME"] = HF_HOME
os.environ["HF_HUB_CACHE"] = HF_HUB_CACHE
os.environ["TRANSFORMERS_CACHE"] = TRANSFORMERS_CACHE

print("HF cache:", HF_HOME)
print("Output directory:", OUTPUT_DIR)
print("Data directory:", DATA_STORAGE_DIR)


# -----------------------------
# Hugging Face authentication
# -----------------------------

HF_TOKEN = os.environ.get(
    "HF_TOKEN",
    "",
)

if HF_TOKEN:
    from huggingface_hub import login

    login(
        token=HF_TOKEN,
        add_to_git_credential=False,
    )
    print("HF_TOKEN loaded.")
else:
    print(
        "HF_TOKEN was not found. "
        "Public models may work without login. "
        "MedGemma requires approved access."
    )


# -----------------------------
# Dataset archive configuration
# -----------------------------

# Upload ProstateMM_VQA_share.tar.gz to Google Drive first.
# This path assumes it is placed directly in My Drive.
DATA_ARCHIVE_PATH = os.path.join(
    MY_DRIVE_DIR,
    "ProstateMM_VQA_share.tar.gz",
)

# Alternative locations are checked automatically.
ALTERNATIVE_ARCHIVE_PATHS = [
    "/content/ProstateMM_VQA_share.tar.gz",
    os.path.join(
        MY_DRIVE_DIR,
        "prostatemm",
        "ProstateMM_VQA_share.tar.gz",
    ),
]


def find_dataset_archive():
    candidates = [
        DATA_ARCHIVE_PATH,
        *ALTERNATIVE_ARCHIVE_PATHS,
    ]

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    # Also search subdirectories in My Drive. This is useful when the
    # archive was uploaded into a folder rather than the Drive root.
    for search_root in [
        MY_DRIVE_DIR,
        "/content",
    ]:
        if not os.path.isdir(search_root):
            continue

        for root, directories, files in os.walk(search_root):
            for file_name in files:
                if file_name == "ProstateMM_VQA_share.tar.gz":
                    return os.path.join(root, file_name)

    return None


def upload_dataset_archive_if_needed():
    """
    Let the user upload the archive directly from the local computer
    when it is not available in Google Drive.
    """

    archive_path = find_dataset_archive()

    if archive_path is not None:
        return archive_path

    try:
        from google.colab import files
    except ImportError as error:
        raise FileNotFoundError(
            "The dataset archive was not found. "
            "In Colab, upload ProstateMM_VQA_share.tar.gz "
            "to Google Drive or set DATA_ARCHIVE_PATH to its "
            "exact /content/gdrive/MyDrive path."
        ) from error

    print("=" * 80)
    print(
        "ProstateMM_VQA_share.tar.gz was not found in Drive."
    )
    print(
        "Please select the archive from your local computer "
        "in the upload window."
    )
    print("=" * 80)

    uploaded_files = files.upload()

    if not uploaded_files:
        raise FileNotFoundError(
            "No file was uploaded. "
            "Please upload ProstateMM_VQA_share.tar.gz."
        )

    archive_name = None

    for uploaded_name in uploaded_files:
        if uploaded_name.endswith(
            "ProstateMM_VQA_share.tar.gz"
        ):
            archive_name = uploaded_name
            break

    if archive_name is None:
        archive_name = next(iter(uploaded_files))

    uploaded_path = os.path.join(
        "/content",
        os.path.basename(archive_name),
    )

    # files.upload() normally writes the file itself, but this explicit
    # write also works in notebook environments that only return bytes.
    if not os.path.isfile(uploaded_path):
        with open(
            uploaded_path,
            "wb",
        ) as file:
            file.write(
                uploaded_files[archive_name]
            )

    if not os.path.isfile(uploaded_path):
        raise FileNotFoundError(
            f"Uploaded archive could not be found: "
            f"{uploaded_path}"
        )

    print("Uploaded dataset archive:", uploaded_path)
    return uploaded_path


def safe_extract_tar_gz(archive_path, destination_dir):
    """
    Extract the archive while preventing path traversal.
    """

    destination_dir = os.path.abspath(destination_dir)
    os.makedirs(
        destination_dir,
        exist_ok=True,
    )

    with tarfile.open(
        archive_path,
        mode="r:gz",
    ) as archive:
        for member in archive.getmembers():
            member_target = os.path.abspath(
                os.path.join(
                    destination_dir,
                    member.name,
                )
            )

            if os.path.commonpath(
                [destination_dir, member_target]
            ) != destination_dir:
                raise RuntimeError(
                    f"Unsafe archive member: {member.name}"
                )

        archive.extractall(destination_dir)


def locate_package_root(extraction_dir):
    expected_directories = [
        os.path.join(
            extraction_dir,
            "ProstateMM_VQA_share",
        ),
        extraction_dir,
    ]

    for candidate in expected_directories:
        if (
            os.path.isdir(candidate)
            and os.path.isfile(
                os.path.join(
                    candidate,
                    "test.jsonl",
                )
            )
            and os.path.isdir(
                os.path.join(
                    candidate,
                    "images_256",
                )
            )
        ):
            return candidate

    for root, directories, files in os.walk(extraction_dir):
        if (
            "test.jsonl" in files
            and "images_256" in directories
        ):
            return root

    return None


archive_path = upload_dataset_archive_if_needed()

print("Dataset archive:", archive_path)

package_root = locate_package_root(
    DATA_STORAGE_DIR
)

if package_root is None:
    print("Extracting ProstateMM-VQA archive...")
    safe_extract_tar_gz(
        archive_path,
        DATA_STORAGE_DIR,
    )
    package_root = locate_package_root(
        DATA_STORAGE_DIR
    )

if package_root is None:
    raise RuntimeError(
        "The archive was extracted, but the expected "
        "ProstateMM-VQA package structure was not found."
    )

print("Dataset package root:", package_root)


# -----------------------------
# Dataset split configuration
# -----------------------------

# Recommended for the final independent comparison:
#   "test"       -> 42 records, 14 patients
#   "validation" -> 42 records, 14 patients
#   "train"      -> 201 records, 67 patients
#   "all"        -> 285 task records, 95 patients
SPLIT_SELECTION = "test"

# None means all records in the selected split.
MAX_EVAL_SAMPLES = None

K_SAMPLED_ANSWERS = 3
TEMPERATURE = 0.7
TOP_P = 0.9
USE_4BIT = True

FAILURE_ROUGE_L_THRESHOLD = 0.2
FAILURE_METEOR_THRESHOLD = 0.1

SELECTIVE_COVERAGES = (
    0.50,
    0.70,
    0.90,
)


# -----------------------------
# Model configuration
# -----------------------------

MODEL_BATCH_SIZES = {
    # ProstateMM includes longer clinical-context prompts.
    # Use a conservative batch size for stable multimodal generation.
    "qwen2_5_vl_3b": 2,
    "medgemma_4b_it": 4,
    "llava_1_5_7b": 4,
}

MODEL_MAX_NEW_TOKENS = {
    "qwen2_5_vl_3b": 48,
    "medgemma_4b_it": 48,
    "llava_1_5_7b": 48,
}

MODEL_REGISTRY = {
    "qwen2_5_vl_3b": {
        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "local_path": os.path.join(
            MODEL_STORAGE_DIR,
            "Qwen_Qwen2_5_VL_3B_Instruct",
        ),
        "family": "qwen2_5_vl",
    },
    "medgemma_4b_it": {
        "model_id": "google/medgemma-4b-it",
        "local_path": os.path.join(
            MODEL_STORAGE_DIR,
            "google_medgemma_4b_it",
        ),
        "family": "medgemma",
    },
    "llava_1_5_7b": {
        "model_id": "llava-hf/llava-1.5-7b-hf",
        "local_path": os.path.join(
            MODEL_STORAGE_DIR,
            "llava_hf_llava_1_5_7b_hf",
        ),
        "family": "llava",
    },
}


# %%
# ============================================================
# CELL 3 - Read ProstateMM JSONL and build the DataLoader
# ============================================================


def read_jsonl(path):
    records = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()
            if not line:
                continue

            try:
                records.append(
                    json.loads(line)
                )
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Invalid JSONL at {path}:{line_number}"
                ) from error

    return records


def load_selected_records(package_dir, split_selection):
    if split_selection == "all":
        split_names = [
            "train",
            "validation",
            "test",
        ]
    elif split_selection in {
        "train",
        "validation",
        "test",
    }:
        split_names = [split_selection]
    else:
        raise ValueError(
            "SPLIT_SELECTION must be train, validation, "
            "test, or all."
        )

    selected_records = []

    for split_name in split_names:
        split_path = os.path.join(
            package_dir,
            f"{split_name}.jsonl",
        )

        if not os.path.isfile(split_path):
            raise FileNotFoundError(
                f"Missing split file: {split_path}"
            )

        split_records = read_jsonl(
            split_path
        )
        selected_records.extend(
            split_records
        )

        print(
            f"Loaded {split_name}: "
            f"{len(split_records)} records"
        )

    if MAX_EVAL_SAMPLES is not None:
        selected_records = selected_records[
            :int(MAX_EVAL_SAMPLES)
        ]

    return selected_records


raw_records = load_selected_records(
    package_root,
    SPLIT_SELECTION,
)

if not raw_records:
    raise RuntimeError(
        "No records were loaded."
    )

required_record_fields = {
    "sample_id",
    "patient_id",
    "split",
    "task",
    "image_path",
    "question",
    "system_prompt",
    "clinical_context",
    "primary_reference",
}

missing_fields = (
    required_record_fields
    - set(raw_records[0].keys())
)

if missing_fields:
    raise RuntimeError(
        f"Required fields are missing: "
        f"{sorted(missing_fields)}"
    )


def resolve_image_path(package_dir, relative_path):
    relative_path = str(relative_path)
    image_path = os.path.abspath(
        os.path.join(
            package_dir,
            relative_path,
        )
    )
    package_dir_abs = os.path.abspath(
        package_dir
    )

    if os.path.commonpath(
        [package_dir_abs, image_path]
    ) != package_dir_abs:
        raise RuntimeError(
            f"Unsafe image path: {relative_path}"
        )

    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    return image_path


for record in raw_records:
    record["_absolute_image_path"] = resolve_image_path(
        package_root,
        record["image_path"],
    )

EVAL_SIZE = len(raw_records)
SPLIT_NAME = SPLIT_SELECTION

unique_patients = sorted(
    {
        str(record["patient_id"])
        for record in raw_records
    }
)

task_counts = (
    pd.Series(
        [
            record["task"]
            for record in raw_records
        ]
    )
    .value_counts()
    .to_dict()
)

print("=" * 80)
print("ProstateMM-VQA configuration")
print("=" * 80)
print("Dataset package:", package_root)
print("Selected split:", SPLIT_SELECTION)
print("Evaluation records:", EVAL_SIZE)
print("Unique patients:", len(unique_patients))
print("Task counts:", task_counts)
print("K sampled answers:", K_SAMPLED_ANSWERS)
print("Use 4-bit:", USE_4BIT)
print("=" * 80)


class ProstateMMVQADataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        image = Image.open(
            record["_absolute_image_path"]
        ).convert("RGB")

        return {
            "index": int(index),
            "sample_id": str(record["sample_id"]),
            "patient_id": str(record["patient_id"]),
            "split": str(record["split"]),
            "task": str(record["task"]),
            "image": image,
            "image_path": str(record["image_path"]),
            "system_prompt": str(record["system_prompt"]),
            "question": str(record["question"]),
            "clinical_context": record.get(
                "clinical_context",
                {},
            ),
            "target_facts": record.get(
                "target_facts",
                {},
            ),
            "primary_reference": str(
                record["primary_reference"]
            ),
            "alternative_references": record.get(
                "alternative_references",
                [],
            ),
        }


def collate_prostatemm_batch(batch):
    return {
        "index": [
            item["index"]
            for item in batch
        ],
        "sample_id": [
            item["sample_id"]
            for item in batch
        ],
        "patient_id": [
            item["patient_id"]
            for item in batch
        ],
        "split": [
            item["split"]
            for item in batch
        ],
        "task": [
            item["task"]
            for item in batch
        ],
        "image": [
            item["image"]
            for item in batch
        ],
        "image_path": [
            item["image_path"]
            for item in batch
        ],
        "system_prompt": [
            item["system_prompt"]
            for item in batch
        ],
        "question": [
            item["question"]
            for item in batch
        ],
        "clinical_context": [
            item["clinical_context"]
            for item in batch
        ],
        "target_facts": [
            item["target_facts"]
            for item in batch
        ],
        "primary_reference": [
            item["primary_reference"]
            for item in batch
        ],
        "alternative_references": [
            item["alternative_references"]
            for item in batch
        ],
    }


eval_dataset = ProstateMMVQADataset(
    raw_records
)


def serialise_clinical_context(context):
    if not isinstance(context, dict):
        return str(context)

    return json.dumps(
        context,
        ensure_ascii=False,
        sort_keys=True,
    )


def build_user_text(
    question,
    clinical_context,
):
    return (
        f"Question: {str(question).strip()}\n\n"
        "Clinical context:\n"
        f"{serialise_clinical_context(clinical_context)}\n\n"
        "Answer:"
    )


def clean_answer(text, task=None):
    if text is None:
        return ""

    text = str(text).strip()

    answer_match = re.search(
        r"<answer>\s*(.*?)\s*</answer>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if answer_match:
        text = answer_match.group(1)

    text = re.sub(
        r"^\s*(assistant|answer|final answer)\s*[:：]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]
    text = lines[0] if lines else ""
    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    for prefix in [
        "the answer is ",
        "it is ",
        "this is ",
        "the image shows ",
        "this image shows ",
    ]:
        if text.lower().startswith(prefix):
            text = text[len(prefix):].strip()

    # The bcr task is explicitly binary. Only normalise this task to yes/no.
    if task == "bcr_prediction":
        match = re.search(
            r"\b(yes|no)\b",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            text = match.group(1)
        elif re.search(
            r"\bnot predicted\b|\bno biochemical recurrence\b",
            text,
            flags=re.IGNORECASE,
        ):
            text = "no"
        elif re.search(
            r"\bpredicted\b|\bbiochemical recurrence is present\b",
            text,
            flags=re.IGNORECASE,
        ):
            text = "yes"

    text = text.strip(
        " \t\n\r.,;:!?()[]{}<>。；：！？"
    )

    return text.lower()


def safe_model_name(model_id):
    return (
        str(model_id)
        .replace("/", "_")
        .replace(".", "_")
        .replace("-", "_")
    )


def get_records_path(model_id, model_key):
    file_name = (
        f"outputs_prostatemm_{model_key}_"
        f"{safe_model_name(model_id)}_"
        f"{SPLIT_NAME}_N{EVAL_SIZE}_"
        f"K{K_SAMPLED_ANSWERS}_"
        f"{RUN_ID}.json"
    )
    return os.path.join(
        OUTPUT_DIR,
        file_name,
    )


# %%
# ============================================================
# CELL 4 - Model loading and ProstateMM multimodal prompts
# ============================================================

import importlib.metadata


def ensure_supported_bitsandbytes():
    required_version = "0.46.1"

    try:
        installed_version = importlib.metadata.version(
            "bitsandbytes"
        )
    except importlib.metadata.PackageNotFoundError:
        installed_version = None

    needs_install = installed_version is None

    if installed_version is not None:
        from packaging.version import Version

        needs_install = (
            Version(installed_version)
            < Version(required_version)
        )

    if needs_install:
        print(
            "Installing bitsandbytes>="
            f"{required_version}. "
            f"Current version: {installed_version}"
        )
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--upgrade",
                "--no-cache-dir",
                "--no-deps",
                f"bitsandbytes>={required_version}",
            ]
        )

        print(
            "bitsandbytes was upgraded. "
            "If the old version is still visible, "
            "restart the Runtime once and rerun this cell."
        )
    else:
        print(
            "bitsandbytes version:",
            installed_version,
        )


ensure_supported_bitsandbytes()


from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    LlavaForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
)

try:
    from qwen_vl_utils import process_vision_info
except ModuleNotFoundError:
    print(
        "qwen-vl-utils is missing. Installing it now..."
    )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-cache-dir",
            "qwen-vl-utils",
        ]
    )
    from qwen_vl_utils import process_vision_info


def set_padding_side_left(processor):
    if (
        hasattr(processor, "tokenizer")
        and processor.tokenizer is not None
    ):
        processor.tokenizer.padding_side = "left"

        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = (
                processor.tokenizer.eos_token
            )


def move_inputs_to_device(inputs, device):
    moved = {}

    for key, value in inputs.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value

    return moved


def record_prompt_parts(
    system_prompt,
    question,
    clinical_context,
):
    system_prompt = str(
        system_prompt
    ).strip()

    if not system_prompt:
        system_prompt = (
            "You are a clinical pathology assistant "
            "specialising in prostate cancer assessment. "
            "Provide a concise answer using the image, "
            "question, and clinical context."
        )

    user_text = build_user_text(
        question,
        clinical_context,
    )

    return system_prompt, user_text


def build_qwen_messages(
    images,
    questions,
    system_prompts,
    clinical_contexts,
):
    messages_batch = []

    for image, question, system_prompt, context in zip(
        images,
        questions,
        system_prompts,
        clinical_contexts,
    ):
        system_text, user_text = record_prompt_parts(
            system_prompt,
            question,
            context,
        )

        messages_batch.append(
            [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system_text,
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {
                            "type": "text",
                            "text": user_text,
                        },
                    ],
                },
            ]
        )

    return messages_batch


def build_gemma_messages(
    questions,
    system_prompts,
    clinical_contexts,
):
    messages_batch = []

    for question, system_prompt, context in zip(
        questions,
        system_prompts,
        clinical_contexts,
    ):
        system_text, user_text = record_prompt_parts(
            system_prompt,
            question,
            context,
        )

        messages_batch.append(
            [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system_text,
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                        },
                        {
                            "type": "text",
                            "text": user_text,
                        },
                    ],
                },
            ]
        )

    return messages_batch


def build_llava_prompts(
    questions,
    system_prompts,
    clinical_contexts,
):
    prompts = []

    for question, system_prompt, context in zip(
        questions,
        system_prompts,
        clinical_contexts,
    ):
        system_text, user_text = record_prompt_parts(
            system_prompt,
            question,
            context,
        )

        prompts.append(
            "USER: <image>\n"
            f"{system_text}\n\n"
            f"{user_text}\n"
            "ASSISTANT:"
        )

    return prompts


def format_chat_messages(
    processor,
    messages_batch,
):
    return [
        processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        for messages in messages_batch
    ]


def get_generation_kwargs(
    processor,
    do_sample,
    max_new_tokens,
):
    generation_kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
    }

    pad_token_id = None

    if (
        hasattr(processor, "tokenizer")
        and processor.tokenizer is not None
    ):
        pad_token_id = (
            processor.tokenizer.pad_token_id
        )

        if pad_token_id is None:
            pad_token_id = (
                processor.tokenizer.eos_token_id
            )

    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = (
            pad_token_id
        )

    if do_sample:
        generation_kwargs.update(
            {
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
            }
        )

    return generation_kwargs


def decode_generated_ids(
    processor,
    inputs,
    output_ids,
    questions,
    tasks,
):
    generated_ids = [
        output_ids[index][
            len(inputs["input_ids"][index]):
        ]
        for index in range(len(output_ids))
    ]

    decoded = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    return [
        clean_answer(
            text,
            task,
        )
        for text, task in zip(
            decoded,
            tasks,
        )
    ]


@torch.inference_mode()
def generate_gemma_single(
    model,
    processor,
    image,
    question,
    system_prompt,
    clinical_context,
    task,
    do_sample,
    max_new_tokens,
):
    messages_batch = build_gemma_messages(
        [question],
        [system_prompt],
        [clinical_context],
    )
    texts = format_chat_messages(
        processor,
        messages_batch,
    )

    inputs = processor(
        text=texts,
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    inputs = move_inputs_to_device(
        inputs,
        DEVICE,
    )

    output_ids = model.generate(
        **inputs,
        **get_generation_kwargs(
            processor,
            do_sample,
            max_new_tokens,
        ),
    )

    return decode_generated_ids(
        processor,
        inputs,
        output_ids,
        [question],
        [task],
    )[0]


@torch.inference_mode()
def generate_batch(
    model,
    processor,
    family,
    images,
    questions,
    system_prompts,
    clinical_contexts,
    tasks,
    do_sample=False,
    max_new_tokens=48,
):
    if not (
        len(images)
        == len(questions)
        == len(system_prompts)
        == len(clinical_contexts)
        == len(tasks)
    ):
        raise ValueError(
            "Images, questions, system prompts, "
            "contexts, and tasks must have equal length."
        )

    if family == "qwen2_5_vl":
        messages_batch = build_qwen_messages(
            images,
            questions,
            system_prompts,
            clinical_contexts,
        )
        formatted_texts = format_chat_messages(
            processor,
            messages_batch,
        )

        image_inputs, video_inputs = (
            process_vision_info(messages_batch)
        )

        inputs = processor(
            text=formatted_texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

    elif family == "medgemma":
        messages_batch = build_gemma_messages(
            questions,
            system_prompts,
            clinical_contexts,
        )
        formatted_texts = format_chat_messages(
            processor,
            messages_batch,
        )

        # Gemma 3 requires one image group per text sample.
        nested_images = [
            [image]
            for image in images
        ]

        try:
            inputs = processor(
                text=formatted_texts,
                images=nested_images,
                padding=True,
                return_tensors="pt",
            )
        except ValueError as error:
            if (
                "inconsistently sized batches"
                not in str(error).lower()
            ):
                raise

            # Compatibility fallback for strict Gemma processors.
            return [
                generate_gemma_single(
                    model=model,
                    processor=processor,
                    image=image,
                    question=question,
                    system_prompt=system_prompt,
                    clinical_context=context,
                    task=task,
                    do_sample=do_sample,
                    max_new_tokens=max_new_tokens,
                )
                for image, question, system_prompt, context, task
                in zip(
                    images,
                    questions,
                    system_prompts,
                    clinical_contexts,
                    tasks,
                )
            ]

    elif family == "llava":
        prompts = build_llava_prompts(
            questions,
            system_prompts,
            clinical_contexts,
        )

        inputs = processor(
            text=prompts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

    else:
        raise ValueError(
            f"Unknown model family: {family}"
        )

    inputs = move_inputs_to_device(
        inputs,
        DEVICE,
    )

    output_ids = model.generate(
        **inputs,
        **get_generation_kwargs(
            processor,
            do_sample,
            max_new_tokens,
        ),
    )

    return decode_generated_ids(
        processor,
        inputs,
        output_ids,
        questions,
        tasks,
    )


def load_model_and_processor(
    model_config,
):
    model_id = model_config["model_id"]
    family = model_config["family"]
    local_path = model_config["local_path"]

    local_config_path = os.path.join(
        local_path,
        "config.json",
    )

    if os.path.isfile(local_config_path):
        model_source = local_path
        local_files_only = True
    else:
        model_source = model_id
        local_files_only = False

    print("=" * 80)
    print("Loading model:", model_id)
    print("Family:", family)
    print("Source:", model_source)
    print("Local files only:", local_files_only)
    print("4-bit:", USE_4BIT)
    print("=" * 80)

    processor = AutoProcessor.from_pretrained(
        model_source,
        cache_dir=HF_HUB_CACHE,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    set_padding_side_left(processor)

    quantization_config = None

    if USE_4BIT:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=COMPUTE_DTYPE,
            bnb_4bit_use_double_quant=True,
        )

    model_kwargs = {
        "low_cpu_mem_usage": True,
        "torch_dtype": COMPUTE_DTYPE,
        "quantization_config": quantization_config,
        "cache_dir": HF_HUB_CACHE,
        "local_files_only": local_files_only,
        "trust_remote_code": True,
    }

    # The quantised models fit comfortably on the available GPU.
    # Forcing a single-GPU map avoids slow CPU offloading caused by
    # device_map="auto" on some Colab runtimes.
    if DEVICE == "cuda":
        model_kwargs["device_map"] = {"": 0}

    if family == "qwen2_5_vl":
        model = (
            Qwen2_5_VLForConditionalGeneration
            .from_pretrained(
                model_source,
                **model_kwargs,
            )
        )
    elif family == "llava":
        model = (
            LlavaForConditionalGeneration
            .from_pretrained(
                model_source,
                **model_kwargs,
            )
        )
    elif family == "medgemma":
        model = (
            AutoModelForImageTextToText
            .from_pretrained(
                model_source,
                **model_kwargs,
            )
        )
    else:
        raise ValueError(
            f"Unknown model family: {family}"
        )

    model.eval()
    model.generation_config.use_cache = True

    if hasattr(model, "hf_device_map"):
        print("Model device map:", model.hf_device_map)

    print("Model loaded:", model_id)
    return model, processor


# %%
# ============================================================
# CELL 5 - Download all models before inference
# ============================================================

from huggingface_hub import snapshot_download


DOWNLOAD_MODELS_BEFORE_INFERENCE = True


def ensure_all_models_downloaded():
    if not DOWNLOAD_MODELS_BEFORE_INFERENCE:
        print("Model pre-download is disabled.")
        return

    token = (
        globals().get("HF_TOKEN")
        or os.environ.get("HF_TOKEN")
        or None
    )

    print("=" * 80)
    print("Preparing all models before inference")
    print("=" * 80)

    for model_key, model_config in MODEL_REGISTRY.items():
        model_id = model_config["model_id"]
        local_path = model_config["local_path"]
        config_path = os.path.join(
            local_path,
            "config.json",
        )

        if os.path.isfile(config_path):
            print(
                f"[Already available] {model_key}\n"
                f"Local path: {local_path}"
            )
            continue

        print("=" * 80)
        print(f"[Downloading] {model_key}")
        print("Model ID:", model_id)
        print("Destination:", local_path)
        print("=" * 80)

        os.makedirs(
            local_path,
            exist_ok=True,
        )

        try:
            snapshot_download(
                repo_id=model_id,
                local_dir=local_path,
                cache_dir=HF_HUB_CACHE,
                token=token,
                resume_download=True,
            )
        except Exception as error:
            print(
                f"Failed to download {model_id}:",
                repr(error),
            )

            if "medgemma" in model_id.lower():
                raise RuntimeError(
                    "MedGemma download failed. "
                    "Check Hugging Face approval and login."
                ) from error

            raise

        if not os.path.isfile(config_path):
            raise RuntimeError(
                f"config.json is missing after download: "
                f"{local_path}"
            )

        print(
            f"[Downloaded successfully] {model_key}\n"
            f"Local path: {local_path}"
        )

    print("=" * 80)
    print("All three models are available locally.")
    print("Inference will start now.")
    print("=" * 80)


ensure_all_models_downloaded()


# %%
# ============================================================
# CELL 6 - Three-model inference with checkpoint/resume
# ============================================================


def save_records(
    path,
    metadata,
    records,
):
    records = sorted(
        records,
        key=lambda item: int(
            item.get("index", -1)
        ),
    )

    payload = {
        "metadata": metadata,
        "records": records,
    }

    temporary_path = path + ".tmp"

    with open(
        temporary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )

    os.replace(
        temporary_path,
        path,
    )


def load_existing_records(
    path,
    expected_metadata,
):
    if not os.path.isfile(path):
        return []

    try:
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:
            payload = json.load(file)

        if (
            isinstance(payload, dict)
            and "records" in payload
        ):
            old_metadata = payload.get(
                "metadata",
                {},
            )
            records = payload.get(
                "records",
                [],
            )

            important_keys = [
                "dataset",
                "split",
                "eval_size",
                "model_key",
                "model_id",
                "k_sampled_answers",
                "max_new_tokens",
                "use_4bit",
                "run_id",
            ]

            metadata_matches = all(
                old_metadata.get(key)
                == expected_metadata.get(key)
                for key in important_keys
            )

            if not metadata_matches:
                print(
                    "Existing cache metadata differs. "
                    "This run file will be regenerated."
                )
                return []

            print(
                f"Loaded existing cache: "
                f"{len(records)} records"
            )
            return list(records)

        if isinstance(payload, list):
            print(
                f"Loaded old-format cache: "
                f"{len(payload)} records"
            )
            return list(payload)

    except Exception as error:
        print(
            "Failed to load cache:",
            repr(error),
        )

    return []


MODEL_CONFIGS = [
    {
        "model_key": model_key,
        **config,
        "batch_size": MODEL_BATCH_SIZES[model_key],
        "max_new_tokens": MODEL_MAX_NEW_TOKENS[model_key],
    }
    for model_key, config
    in MODEL_REGISTRY.items()
]

all_model_output_files = []
model_errors = []


for model_config in MODEL_CONFIGS:
    model_key = model_config["model_key"]
    model_id = model_config["model_id"]
    family = model_config["family"]
    batch_size = int(
        model_config["batch_size"]
    )
    max_new_tokens = int(
        model_config["max_new_tokens"]
    )

    metadata = {
        "dataset": "ProstateMM-VQA",
        "dataset_source": "CHIMERA",
        "dataset_package_root": package_root,
        "split": SPLIT_NAME,
        "eval_size": EVAL_SIZE,
        "unique_patients": len(unique_patients),
        "model_key": model_key,
        "model_id": model_id,
        "family": family,
        "batch_size": batch_size,
        "k_sampled_answers": K_SAMPLED_ANSWERS,
        "max_new_tokens": max_new_tokens,
        "use_4bit": USE_4BIT,
        "seed": SEED,
        "run_id": RUN_ID,
    }

    records_path = get_records_path(
        model_id,
        model_key,
    )

    if FORCE_RERUN:
        records = []
        print(
            "FORCE_RERUN=True: "
            "ignoring existing cache."
        )
    else:
        records = load_existing_records(
            records_path,
            metadata,
        )

    records_by_index = {
        int(record["index"]): record
        for record in records
        if "index" in record
    }
    done_indices = set(
        records_by_index.keys()
    )

    print("=" * 80)
    print("Current model:", model_id)
    print(
        f"Cached records: "
        f"{len(done_indices)}/{EVAL_SIZE}"
    )
    print("Save path:", records_path)
    print("=" * 80)

    if len(done_indices) >= EVAL_SIZE:
        print(
            "Complete cache found. "
            "Skipping inference for this model."
        )
        all_model_output_files.append(
            records_path
        )
        continue

    model = None
    processor = None

    try:
        model, processor = load_model_and_processor(
            model_config
        )

        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_prostatemm_batch,
        )

        total_batches = math.ceil(
            EVAL_SIZE / batch_size
        )

        for batch in tqdm(
            eval_loader,
            total=total_batches,
            desc=f"Inference: {model_key}",
        ):
            batch_indices = [
                int(index)
                for index in batch["index"]
            ]

            if all(
                index in done_indices
                for index in batch_indices
            ):
                continue

            active_positions = [
                position
                for position, index
                in enumerate(batch_indices)
                if index not in done_indices
            ]

            images = [
                batch["image"][position]
                for position in active_positions
            ]
            questions = [
                batch["question"][position]
                for position in active_positions
            ]
            system_prompts = [
                batch["system_prompt"][position]
                for position in active_positions
            ]
            clinical_contexts = [
                batch["clinical_context"][position]
                for position in active_positions
            ]
            tasks = [
                batch["task"][position]
                for position in active_positions
            ]
            indices = [
                batch_indices[position]
                for position in active_positions
            ]

            try:
                print(
                    f"\n[{model_key}] batch "
                    f"{indices[0]}-{indices[-1]}: "
                    "greedy generation",
                    flush=True,
                )

                greedy_predictions = generate_batch(
                    model=model,
                    processor=processor,
                    family=family,
                    images=images,
                    questions=questions,
                    system_prompts=system_prompts,
                    clinical_contexts=clinical_contexts,
                    tasks=tasks,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                )

                sampled_predictions_by_k = []

                for sample_number in range(
                    K_SAMPLED_ANSWERS
                ):
                    print(
                        f"[{model_key}] batch "
                        f"{indices[0]}-{indices[-1]}: "
                        f"sample {sample_number + 1}/"
                        f"{K_SAMPLED_ANSWERS}",
                        flush=True,
                    )

                    sampled_predictions = (
                        generate_batch(
                            model=model,
                            processor=processor,
                            family=family,
                            images=images,
                            questions=questions,
                            system_prompts=system_prompts,
                            clinical_contexts=clinical_contexts,
                            tasks=tasks,
                            do_sample=True,
                            max_new_tokens=max_new_tokens,
                        )
                    )
                    sampled_predictions_by_k.append(
                        sampled_predictions
                    )

                for position, index in enumerate(
                    indices
                ):
                    sampled_answers = [
                        sampled_predictions_by_k[k][
                            position
                        ]
                        for k in range(
                            K_SAMPLED_ANSWERS
                        )
                    ]

                    source_record = raw_records[
                        int(index)
                    ]

                    records_by_index[
                        int(index)
                    ] = {
                        "index": int(index),
                        "sample_id": source_record[
                            "sample_id"
                        ],
                        "patient_id": source_record[
                            "patient_id"
                        ],
                        "split": source_record[
                            "split"
                        ],
                        "task": source_record[
                            "task"
                        ],
                        "image_path": source_record[
                            "image_path"
                        ],
                        "question": questions[
                            position
                        ],
                        "clinical_context": (
                            clinical_contexts[
                                position
                            ]
                        ),
                        "target_facts": source_record.get(
                            "target_facts",
                            {},
                        ),
                        "ground_truth": source_record[
                            "primary_reference"
                        ],
                        "reference_answers": source_record.get(
                            "alternative_references",
                            [],
                        ),
                        "greedy_prediction": (
                            greedy_predictions[
                                position
                            ]
                        ),
                        "sampled_answers": sampled_answers,
                        "model_id": model_id,
                        "model_key": model_key,
                        "family": family,
                    }
                    done_indices.add(
                        int(index)
                    )

                save_records(
                    records_path,
                    metadata,
                    list(records_by_index.values()),
                )

            except RuntimeError as error:
                print(
                    "RuntimeError:",
                    repr(error),
                )

                if "out of memory" in str(error).lower():
                    print(
                        "CUDA OOM. Reduce this model "
                        "batch size in MODEL_BATCH_SIZES."
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                save_records(
                    records_path,
                    metadata,
                    list(records_by_index.values()),
                )
                raise

            except Exception as error:
                print(
                    "Batch error:",
                    repr(error),
                )
                save_records(
                    records_path,
                    metadata,
                    list(records_by_index.values()),
                )
                raise

        save_records(
            records_path,
            metadata,
            list(records_by_index.values()),
        )
        all_model_output_files.append(
            records_path
        )

        print("=" * 80)
        print("Finished model:", model_id)
        print(
            "Total saved records:",
            len(records_by_index),
        )
        print("Saved to:", records_path)
        print("=" * 80)

    except Exception as error:
        model_errors.append(
            {
                "model_key": model_key,
                "model_id": model_id,
                "error": repr(error),
            }
        )
        print(
            f"Model {model_id} failed. "
            f"Progress was saved to {records_path}."
        )

        if not CONTINUE_AFTER_MODEL_ERROR:
            raise

    finally:
        if model is not None:
            del model
        if processor is not None:
            del processor

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


print("=" * 80)
print("Inference stage finished.")
print("Complete output files:")

for output_file in all_model_output_files:
    print(output_file)

if model_errors:
    print("Model errors:")
    for model_error in model_errors:
        print(model_error)

print("=" * 80)


# %%
# ============================================================
# CELL 7 - AFD metrics and final summaries
# ============================================================

import evaluate
import torch.nn.functional as F

os.environ["NLTK_ALLOW_PROXIED_URLOPEN"] = "1"
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

    # Colab's trusted proxy can trigger NLTK's SSRF protection.
    try:
        from nltk import pathsec

        pathsec.ALLOW_PROXIED_FETCH = True
    except ImportError:
        pass

    required_resources = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }

    for resource_path, download_name in required_resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            downloaded = nltk.download(
                download_name,
                download_dir=nltk_data_dir,
                quiet=True,
            )

            if not downloaded:
                raise RuntimeError(
                    f"Could not download NLTK resource: "
                    f"{download_name}"
                )


ensure_nltk_resources()

from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)
from transformers import AutoTokenizer, AutoModel

import nltk

for resource_name in [
    "wordnet",
    "omw-1.4",
    "punkt",
]:
    nltk.download(
        resource_name,
        quiet=True,
    )


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


def get_nlp_mettics(
    references,
    hypotheses,
):
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")

    references_norm = [
        normalize_text(reference)
        or "empty"
        for reference in references
    ]
    hypotheses_norm = [
        normalize_text(hypothesis)
        or "empty"
        for hypothesis in hypotheses
    ]

    results_bleu = bleu.compute(
        predictions=hypotheses_norm,
        references=[
            [reference]
            for reference in references_norm
        ],
    )
    results_rouge = rouge.compute(
        predictions=hypotheses_norm,
        references=references_norm,
    )
    results_meteor = meteor.compute(
        predictions=hypotheses_norm,
        references=references_norm,
    )

    return {
        "BLEU-1": float(
            results_bleu["precisions"][0]
        ),
        "BLEU-2": float(
            results_bleu["precisions"][1]
        ),
        "ROUGE-L": float(
            results_rouge["rougeL"]
        ),
        "METEOR": float(
            results_meteor["meteor"]
        ),
    }


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


# -----------------------------
# BGE embedding model
# -----------------------------

EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"
EMBED_BATCH_SIZE = 64
EMBED_MAX_LENGTH = 128

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


@torch.inference_mode()
def encode_texts(
    texts,
    batch_size=EMBED_BATCH_SIZE,
):
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
        token_embeddings = (
            outputs.last_hidden_state
        )
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
            pooled.detach().cpu().float().numpy()
        )

    if not all_embeddings:
        return np.empty(
            (
                0,
                embed_model.config.hidden_size,
            ),
            dtype=np.float32,
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


SEMANTIC_SIMILARITY_THRESHOLD = 0.80


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


def selective_metrics(
    records,
    uncertainty_scores,
):
    evaluation_df = pd.DataFrame(
        records
    ).copy()
    evaluation_df["uncertainty"] = (
        uncertainty_scores
    )

    evaluation_df = (
        evaluation_df
        .sort_values(
            "uncertainty",
            ascending=True,
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    output = {}

    for coverage in SELECTIVE_COVERAGES:
        accepted_count = max(
            1,
            int(
                math.ceil(
                    len(evaluation_df)
                    * coverage
                )
            ),
        )
        accepted_df = evaluation_df.iloc[
            :accepted_count
        ]
        percentage = int(
            round(coverage * 100)
        )

        output[
            f"Accepted samples @{percentage}%"
        ] = int(len(accepted_df))
        output[
            f"Accepted ROUGE-L @{percentage}%"
        ] = float(
            accepted_df["rougeL"].mean()
        )
        output[
            f"Accepted METEOR @{percentage}%"
        ] = float(
            accepted_df["meteor"].mean()
        )
        output[
            f"Accepted failure rate @{percentage}%"
        ] = float(
            accepted_df["failure"].mean()
        )

    return output


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


def evaluate_output_json(
    json_path,
):
    print("=" * 80)
    print("Evaluating:", json_path)
    print("=" * 80)

    metadata, records = load_output_payload(
        json_path
    )

    expected_size = metadata.get(
        "eval_size"
    )
    if (
        expected_size is not None
        and len(records)
        != int(expected_size)
    ):
        print(
            "Skipping incomplete output:",
            f"{len(records)}/{expected_size}",
        )
        return None, None

    records = sorted(
        records,
        key=lambda item: int(
            item.get("index", 0)
        ),
    )

    model_id = metadata.get(
        "model_id",
        records[0].get(
            "model_id",
            "unknown",
        ),
    )
    model_key = metadata.get(
        "model_key",
        records[0].get(
            "model_key",
            "unknown",
        ),
    )

    references = [
        record.get(
            "ground_truth",
            "",
        )
        for record in records
    ]
    hypotheses = [
        record.get(
            "greedy_prediction",
            "",
        )
        for record in records
    ]

    generation_metrics = get_nlp_mettics(
        references,
        hypotheses,
    )
    print(
        "Generation metrics:",
        generation_metrics,
    )

    scored_records = []

    for record in tqdm(
        records,
        desc="Per-sample metrics",
    ):
        item = dict(record)
        item.update(
            compute_single_metrics(
                item.get(
                    "greedy_prediction",
                    "",
                ),
                item.get(
                    "ground_truth",
                    "",
                ),
            )
        )
        scored_records.append(item)

    y_true = [
        item["failure"]
        for item in scored_records
    ]

    semantic_texts = []
    semantic_ranges = []

    for record in scored_records:
        start_index = len(
            semantic_texts
        )
        sampled_answers = record.get(
            "sampled_answers",
            [],
        )

        semantic_texts.append(
            record.get(
                "question",
                "",
            )
        )
        semantic_texts.extend(
            sampled_answers
        )
        semantic_ranges.append(
            (
                start_index,
                len(sampled_answers),
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
    task_summary_rows = []
    scored_rows = []

    for method_name, method_function in (
        METHODS.items()
    ):
        print(
            "Running method:",
            method_name,
        )

        scores = np.asarray(
            [
                method_function(
                    record,
                    features,
                )
                for record, features
                in zip(
                    scored_records,
                    semantic_features,
                )
            ],
            dtype=np.float64,
        )

        summary_row = {
            "Model": model_key,
            "Model ID": model_id,
            "Dataset": "ProstateMM-VQA",
            "Source": "CHIMERA",
            "Split": metadata.get(
                "split",
                SPLIT_NAME,
            ),
            "Method": method_name,
            "Evaluation samples": len(
                scored_records
            ),
            "Unique patients": len(
                {
                    record.get(
                        "patient_id"
                    )
                    for record in scored_records
                }
            ),
            "BLEU-1": generation_metrics[
                "BLEU-1"
            ],
            "BLEU-2": generation_metrics[
                "BLEU-2"
            ],
            "ROUGE-L": generation_metrics[
                "ROUGE-L"
            ],
            "METEOR": generation_metrics[
                "METEOR"
            ],
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

        task_values = sorted(
            {
                record.get(
                    "task",
                    "unknown",
                )
                for record in scored_records
            }
        )

        for task_name in task_values:
            task_indices = [
                index
                for index, record in enumerate(
                    scored_records
                )
                if record.get(
                    "task",
                    "unknown",
                )
                == task_name
            ]
            task_records = [
                scored_records[index]
                for index in task_indices
            ]
            task_scores = scores[
                task_indices
            ]
            task_labels = [
                scored_records[index][
                    "failure"
                ]
                for index in task_indices
            ]
            task_generation = get_nlp_mettics(
                [
                    record.get(
                        "ground_truth",
                        "",
                    )
                    for record in task_records
                ],
                [
                    record.get(
                        "greedy_prediction",
                        "",
                    )
                    for record in task_records
                ],
            )

            task_row = {
                "Model": model_key,
                "Model ID": model_id,
                "Task": task_name,
                "Split": metadata.get(
                    "split",
                    SPLIT_NAME,
                ),
                "Method": method_name,
                "Evaluation samples": len(
                    task_records
                ),
                "Unique patients": len(
                    {
                        record.get(
                            "patient_id"
                        )
                        for record in task_records
                    }
                ),
                "BLEU-1": task_generation[
                    "BLEU-1"
                ],
                "BLEU-2": task_generation[
                    "BLEU-2"
                ],
                "ROUGE-L": task_generation[
                    "ROUGE-L"
                ],
                "METEOR": task_generation[
                    "METEOR"
                ],
                "Failure AUROC": safe_auroc(
                    task_labels,
                    task_scores,
                ),
                "Failure AUPRC": safe_auprc(
                    task_labels,
                    task_scores,
                ),
                "Mean uncertainty": float(
                    np.mean(task_scores)
                ),
            }
            task_row.update(
                selective_metrics(
                    task_records,
                    task_scores,
                )
            )
            task_summary_rows.append(
                task_row
            )

        for record, score in zip(
            scored_records,
            scores,
        ):
            scored_rows.append(
                {
                    "index": record.get(
                        "index"
                    ),
                    "sample_id": record.get(
                        "sample_id"
                    ),
                    "patient_id": record.get(
                        "patient_id"
                    ),
                    "task": record.get(
                        "task"
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
                    "Method": method_name,
                    "uncertainty": float(
                        score
                    ),
                }
            )

    summary_df = pd.DataFrame(
        summary_rows
    )
    task_summary_df = pd.DataFrame(
        task_summary_rows
    )
    scored_df = pd.DataFrame(
        scored_rows
    )

    summary_df = summary_df.sort_values(
        "Failure AUPRC",
        ascending=False,
        na_position="last",
    ).reset_index(
        drop=True
    )
    task_summary_df = task_summary_df.sort_values(
        [
            "Task",
            "Failure AUPRC",
        ],
        ascending=[
            True,
            False,
        ],
        na_position="last",
    ).reset_index(
        drop=True
    )

    base_name = os.path.splitext(
        os.path.basename(
            json_path
        )
    )[0]

    summary_path = os.path.join(
        OUTPUT_DIR,
        f"afd_summary_{base_name}.csv",
    )
    task_summary_path = os.path.join(
        OUTPUT_DIR,
        f"afd_task_summary_{base_name}.csv",
    )
    scored_path = os.path.join(
        OUTPUT_DIR,
        f"afd_scored_records_{base_name}.csv",
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )
    task_summary_df.to_csv(
        task_summary_path,
        index=False,
    )
    scored_df.to_csv(
        scored_path,
        index=False,
    )

    print("Saved summary:", summary_path)
    print(
        "Saved task summary:",
        task_summary_path,
    )
    print(
        "Saved scored records:",
        scored_path,
    )

    return (
        summary_df,
        task_summary_df,
    )


current_json_files = [
    os.path.join(
        OUTPUT_DIR,
        file_name,
    )
    for file_name in os.listdir(
        OUTPUT_DIR
    )
    if (
        file_name.startswith(
            "outputs_prostatemm_"
        )
        and file_name.endswith(".json")
        and RUN_ID in file_name
    )
]
current_json_files = sorted(
    current_json_files
)

if not current_json_files:
    raise RuntimeError(
        "No ProstateMM output JSON files found."
    )

all_summary = []
all_task_summary = []

for json_file in current_json_files:
    summary_result = evaluate_output_json(
        json_file
    )

    if summary_result[0] is not None:
        all_summary.append(
            summary_result[0]
        )
        all_task_summary.append(
            summary_result[1]
        )

if not all_summary:
    raise RuntimeError(
        "No complete ProstateMM model outputs "
        "were available for AFD evaluation."
    )

final_summary_df = pd.concat(
    all_summary,
    ignore_index=True,
)
final_task_summary_df = pd.concat(
    all_task_summary,
    ignore_index=True,
)

final_summary_df = final_summary_df.sort_values(
    [
        "Model",
        "Failure AUPRC",
    ],
    ascending=[
        True,
        False,
    ],
    na_position="last",
).reset_index(
    drop=True
)

final_task_summary_df = (
    final_task_summary_df
    .sort_values(
        [
            "Model",
            "Task",
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

for dataframe in [
    final_summary_df,
    final_task_summary_df,
]:
    numeric_columns = dataframe.select_dtypes(
        include=[np.number]
    ).columns
    dataframe.loc[
        :,
        numeric_columns,
    ] = dataframe.loc[
        :,
        numeric_columns,
    ].round(4)

final_summary_path = os.path.join(
    OUTPUT_DIR,
    "final_prostatemm_chimera_afd_summary.csv",
)
final_task_summary_path = os.path.join(
    OUTPUT_DIR,
    "final_prostatemm_chimera_task_afd_summary.csv",
)

final_summary_df.to_csv(
    final_summary_path,
    index=False,
)
final_task_summary_df.to_csv(
    final_task_summary_path,
    index=False,
)

print("=" * 80)
print(
    "Final overall summary:",
    final_summary_path,
)
print(
    "Final task summary:",
    final_task_summary_path,
)
print("=" * 80)

display(final_summary_df)
display(final_task_summary_df)


# %%
# ============================================================
# CELL 8 - Clean tables for presentation and optional archive
# ============================================================

pd.set_option(
    "display.max_columns",
    None,
)
pd.set_option(
    "display.width",
    2600,
)
pd.set_option(
    "display.max_colwidth",
    60,
)


clean_columns = [
    "Model",
    "Method",
    "BLEU-1",
    "BLEU-2",
    "ROUGE-L",
    "METEOR",
    "Failure AUROC",
    "Failure AUPRC",
    "Mean uncertainty",
    "Accepted samples @50%",
    "Accepted ROUGE-L @50%",
    "Accepted METEOR @50%",
    "Accepted failure rate @50%",
    "Accepted samples @70%",
    "Accepted ROUGE-L @70%",
    "Accepted METEOR @70%",
    "Accepted failure rate @70%",
    "Accepted samples @90%",
    "Accepted ROUGE-L @90%",
    "Accepted METEOR @90%",
    "Accepted failure rate @90%",
]

available_clean_columns = [
    column
    for column in clean_columns
    if column in final_summary_df.columns
]

clean_summary_df = final_summary_df[
    available_clean_columns
].copy()

clean_summary_path = os.path.join(
    OUTPUT_DIR,
    "clean_final_prostatemm_afd_table.csv",
)
clean_summary_df.to_csv(
    clean_summary_path,
    index=False,
)

clean_task_columns = [
    "Model",
    "Task",
    "Method",
    "BLEU-1",
    "BLEU-2",
    "ROUGE-L",
    "METEOR",
    "Failure AUROC",
    "Failure AUPRC",
    "Mean uncertainty",
]

available_clean_task_columns = [
    column
    for column in clean_task_columns
    if column in final_task_summary_df.columns
]

clean_task_summary_df = final_task_summary_df[
    available_clean_task_columns
].copy()

clean_task_summary_path = os.path.join(
    OUTPUT_DIR,
    "clean_final_prostatemm_task_afd_table.csv",
)
clean_task_summary_df.to_csv(
    clean_task_summary_path,
    index=False,
)

print("Clean overall table:", clean_summary_path)
print("Clean task table:", clean_task_summary_path)

display(clean_summary_df)
display(clean_task_summary_df)


# Optional archive. Set to "" to disable.
ARCHIVE_RUN_NUMBER = ""

if ARCHIVE_RUN_NUMBER.strip():
    if not ARCHIVE_RUN_NUMBER.strip().isdigit():
        raise ValueError(
            "ARCHIVE_RUN_NUMBER must be a positive integer."
        )

    archive_dir = os.path.join(
        OUTPUT_DIR,
        "experiment_archive",
        f"prostatemm_run_{int(ARCHIVE_RUN_NUMBER)}",
    )
    os.makedirs(
        archive_dir,
        exist_ok=True,
    )

    files_to_archive = [
        *all_model_output_files,
        final_summary_path,
        final_task_summary_path,
        clean_summary_path,
        clean_task_summary_path,
    ]

    archived_files = []

    for source_path in files_to_archive:
        if not os.path.isfile(source_path):
            continue

        destination_path = os.path.join(
            archive_dir,
            os.path.basename(source_path),
        )
        shutil.copy2(
            source_path,
            destination_path,
        )
        archived_files.append(
            destination_path
        )

    manifest = {
        "run_number": int(
            ARCHIVE_RUN_NUMBER
        ),
        "dataset": "ProstateMM-VQA",
        "source": "CHIMERA",
        "split": SPLIT_NAME,
        "evaluation_records": EVAL_SIZE,
        "unique_patients": len(
            unique_patients
        ),
        "models": [
            config["model_key"]
            for config in MODEL_CONFIGS
        ],
        "files": archived_files,
    }

    manifest_path = os.path.join(
        archive_dir,
        "manifest.json",
    )
    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("Archive saved:", archive_dir)
