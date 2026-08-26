# %%
# ============================================================
# PathVQA full-test zero-shot comparison
# Qwen2.5-VL + MedGemma + LLaVA + AFD reliability evaluation
#
# This file is organised as Jupyter/Colab cells.
# Run the cells from top to bottom.
#
# Default evaluation:
#   - PathVQA test split
#   - all available test samples
#   - three models
#   - 4-bit quantisation
#   - K sampled answers per question
#   - checkpoint/resume after every batch
# ============================================================


# %%
# ============================================================
# CELL 1 - Safe environment setup
# ============================================================

import os
import sys
import subprocess


INSTALL_DEPENDENCIES = True


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
        "bitsandbytes==0.45.4",
        extra_args=("--no-deps",),
    )

    print("=" * 80)
    print("Dependencies installed.")
    print("If this is the first run, restart the Colab runtime now.")
    print("Then set INSTALL_DEPENDENCIES = False and continue.")
    print("=" * 80)


# %%
# ============================================================
# CELL 2 - Imports, Drive, cache, authentication, and config
# ============================================================

import os
import gc
import json
import math
import random
import re
import warnings

import numpy as np
import pandas as pd
import torch

from PIL import Image
from tqdm.auto import tqdm
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from IPython.display import display

warnings.filterwarnings("ignore")


# -----------------------------
# Reproducibility
# -----------------------------

SEED = 42
RUN_ID = "pathvqa_full_zero_shot_v1"
FORCE_RERUN = False
CONTINUE_AFTER_MODEL_ERROR = True

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# -----------------------------
# Device
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
MY_DRIVE_DIR = os.path.join(DRIVE_MOUNT_DIR, "MyDrive")

try:
    from google.colab import drive

    if not os.path.isdir(MY_DRIVE_DIR):
        drive.mount(DRIVE_MOUNT_DIR, force_remount=False)
except ImportError:
    print("Google Colab drive module is unavailable.")
    print("The code will continue only if the output directory already exists.")


if not os.path.isdir(MY_DRIVE_DIR):
    print("Warning: Google Drive is not mounted.")


# -----------------------------
# Hugging Face cache and output
# -----------------------------

HF_HOME = os.path.join(MY_DRIVE_DIR, "hf_cache")
HF_HUB_CACHE = os.path.join(HF_HOME, "hub")
TRANSFORMERS_CACHE = os.path.join(HF_HOME, "transformers")

OUTPUT_DIR = os.path.join(
    MY_DRIVE_DIR,
    "pathvqa_afd_full_comparison",
)

MODEL_STORAGE_DIR = os.path.join(
    MY_DRIVE_DIR,
    "pathvqa_models",
)

for directory in [
    HF_HOME,
    HF_HUB_CACHE,
    TRANSFORMERS_CACHE,
    OUTPUT_DIR,
    MODEL_STORAGE_DIR,
]:
    os.makedirs(directory, exist_ok=True)

os.environ["HF_HOME"] = HF_HOME
os.environ["HF_HUB_CACHE"] = HF_HUB_CACHE
os.environ["TRANSFORMERS_CACHE"] = TRANSFORMERS_CACHE

print("HF cache:", HF_HOME)
print("Output directory:", OUTPUT_DIR)
print("Model storage directory:", MODEL_STORAGE_DIR)


# -----------------------------
# Optional Hugging Face login
# -----------------------------

HF_TOKEN = os.environ.get("HF_TOKEN", "")

if HF_TOKEN:
    from huggingface_hub import login

    login(
        token=HF_TOKEN,
        add_to_git_credential=False,
    )
    print("Hugging Face token loaded from HF_TOKEN.")
else:
    print(
        "HF_TOKEN was not found. Public models may work without login; "
        "MedGemma requires approved access and authentication."
    )
    print("If needed, run `from huggingface_hub import login; login()` manually.")


# -----------------------------
# Experiment configuration
# -----------------------------

DATASET_NAME = "flaviagiammarino/path-vqa"
DATASET_TAG = "pathvqa"
DATASET_SPLIT = "test"

# None means the complete PathVQA test split.
# For a quick trial, set this to 20, 200, or 600.
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


# Stable starting batch sizes.
# On A100, Qwen can usually be increased to 8 or 16.
# For MedGemma, use 2-4 if the processor causes high memory use.
# For LLaVA, use 2-8 depending on available memory.
MODEL_BATCH_SIZES = {
    "qwen2_5_vl_3b": 8,
    "medgemma_4b_it": 4,
    "llava_1_5_7b": 4,
}

MODEL_MAX_NEW_TOKENS = {
    "qwen2_5_vl_3b": 24,
    "medgemma_4b_it": 24,
    "llava_1_5_7b": 24,
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


# This is the complete test split used for the final comparison.
PATHVQA_TEST_FILES = (
    "hf://datasets/flaviagiammarino/"
    "path-vqa/data/test-*.parquet"
)

pathvqa_test = load_dataset(
    "parquet",
    data_files={
        "test": PATHVQA_TEST_FILES,
    },
    split="test",
)

required_columns = {"image", "question", "answer"}
missing_columns = required_columns - set(pathvqa_test.column_names)

if missing_columns:
    raise RuntimeError(
        f"PathVQA columns missing: {sorted(missing_columns)}"
    )

print("Original PathVQA test samples:", len(pathvqa_test))
print("Columns:", pathvqa_test.column_names)

if MAX_EVAL_SAMPLES is None:
    eval_hf_dataset = pathvqa_test
    SPLIT_NAME = "test_full"
else:
    selected_size = min(
        int(MAX_EVAL_SAMPLES),
        len(pathvqa_test),
    )
    eval_hf_dataset = pathvqa_test.select(
        range(selected_size)
    )
    SPLIT_NAME = f"test_first_{selected_size}"

EVAL_SIZE = len(eval_hf_dataset)


# %%
# ============================================================
# CELL 3 - PathVQA DataLoader and model-independent prompt
# ============================================================


class PathVQAEvalDataset(Dataset):
    """Keep PIL images and text fields for multimodal processors."""

    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]

        return {
            "index": int(index),
            "image": sample["image"].convert("RGB"),
            "question": str(sample["question"]),
            "answer": str(sample["answer"]),
        }


def collate_pathvqa_batch(batch):
    return {
        "index": [item["index"] for item in batch],
        "image": [item["image"] for item in batch],
        "question": [item["question"] for item in batch],
        "answer": [item["answer"] for item in batch],
    }


eval_dataset = PathVQAEvalDataset(eval_hf_dataset)


SYSTEM_MESSAGE = (
    "You are a professional pathologist specialised in histopathology. "
    "You are answering visual questions based on H&E-stained pathology images. "
    "Use the given image, the question, and appropriate pathology knowledge. "
    "For yes/no questions, answer only 'yes' or 'no'. "
    "For questions beginning with what, where, which, who, when, why, or how, "
    "do not answer yes or no. "
    "For open-ended questions, give the shortest accurate medical phrase or key term. "
    "Do not repeat the question. "
    "Do not explain your reasoning. "
    "Do not provide unrelated information. "
    "Only output the final answer."
)


def is_yes_no_question(question):
    question = str(question).strip().lower()

    if re.match(
        r"^(what|where|which|who|when|why|how)\b",
        question,
    ):
        return False

    return bool(
        re.match(
            r"^(is|are|am|was|were|do|does|did|can|could|"
            r"has|have|had|will|would|should|may|might)\b",
            question,
        )
    )


def clean_answer(text, question=None):
    """
    Clean model output without changing the semantic answer.
    Yes/no extraction is applied only to detected yes/no questions.
    """

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

    text = re.sub(r"\s+", " ", text).strip()

    removable_prefixes = [
        "the answer is ",
        "it is ",
        "this is ",
        "the image shows ",
        "this image shows ",
    ]

    changed = True
    while changed:
        changed = False
        lower_text = text.lower()

        for prefix in removable_prefixes:
            if lower_text.startswith(prefix):
                text = text[len(prefix):].strip()
                changed = True
                break

    if question is not None and is_yes_no_question(question):
        yes_no_match = re.search(
            r"\b(yes|no)\b",
            text,
            flags=re.IGNORECASE,
        )
        if yes_no_match:
            text = yes_no_match.group(1)

    text = text.strip(
        " \t\n\r.,;:!?()[]{}<>。；：！？"
    )

    return text.lower()


def build_question_text(question):
    return (
        f"Question: {str(question).strip()}\n"
        "Answer:"
    )


def safe_model_name(model_id):
    return (
        str(model_id)
        .replace("/", "_")
        .replace(".", "_")
        .replace("-", "_")
    )


def get_records_path(model_id, model_key):
    file_name = (
        f"outputs_{DATASET_TAG}_{model_key}_"
        f"{safe_model_name(model_id)}_"
        f"{SPLIT_NAME}_N{EVAL_SIZE}_"
        f"K{K_SAMPLED_ANSWERS}_"
        f"{RUN_ID}.json"
    )
    return os.path.join(OUTPUT_DIR, file_name)


print("=" * 80)
print("PathVQA experiment configuration")
print("=" * 80)
print("Dataset:", DATASET_NAME)
print("Split:", SPLIT_NAME)
print("Evaluation samples:", EVAL_SIZE)
print("K sampled answers:", K_SAMPLED_ANSWERS)
print("Use 4-bit:", USE_4BIT)
print("Output directory:", OUTPUT_DIR)
print("=" * 80)


# %%
# ============================================================
# CELL 4 - Model loading and batched generation
# ============================================================

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    LlavaForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
)

from qwen_vl_utils import process_vision_info


def set_padding_side_left(processor):
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"

        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token


def move_inputs_to_device(inputs, device):
    moved = {}

    for key, value in inputs.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value

    return moved


def build_qwen_messages(images, questions):
    messages_batch = []

    for image, question in zip(images, questions):
        messages_batch.append(
            [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": SYSTEM_MESSAGE,
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
                            "text": build_question_text(question),
                        },
                    ],
                },
            ]
        )

    return messages_batch


def build_gemma_messages(questions):
    messages_batch = []

    for question in questions:
        messages_batch.append(
            [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": SYSTEM_MESSAGE,
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
                            "text": build_question_text(question),
                        },
                    ],
                },
            ]
        )

    return messages_batch


def build_llava_prompts(questions):
    prompts = []

    for question in questions:
        prompts.append(
            "USER: <image>\n"
            f"{SYSTEM_MESSAGE}\n\n"
            f"{build_question_text(question)}\n"
            "ASSISTANT:"
        )

    return prompts


def format_chat_messages(processor, messages_batch):
    return [
        processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        for messages in messages_batch
    ]


def get_generation_kwargs(processor, do_sample, max_new_tokens):
    pad_token_id = None

    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        pad_token_id = processor.tokenizer.pad_token_id

        if pad_token_id is None:
            pad_token_id = processor.tokenizer.eos_token_id

    generation_kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
    }

    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = pad_token_id

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
):
    generated_ids = [
        output_ids[index][len(inputs["input_ids"][index]):]
        for index in range(len(output_ids))
    ]

    decoded = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    return [
        clean_answer(text, question)
        for text, question in zip(decoded, questions)
    ]


@torch.inference_mode()
def generate_gemma_single(
    model,
    processor,
    image,
    question,
    do_sample,
    max_new_tokens,
):
    messages_batch = build_gemma_messages([question])
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
    inputs = move_inputs_to_device(inputs, DEVICE)

    output_ids = model.generate(
        **inputs,
        **get_generation_kwargs(
            processor,
            do_sample,
            max_new_tokens,
        ),
    )

    decoded = decode_generated_ids(
        processor,
        inputs,
        output_ids,
        [question],
    )

    return decoded[0]


@torch.inference_mode()
def generate_batch(
    model,
    processor,
    family,
    images,
    questions,
    do_sample=False,
    max_new_tokens=24,
):
    if len(images) != len(questions):
        raise ValueError(
            f"Received {len(images)} images but "
            f"{len(questions)} questions."
        )

    if family == "qwen2_5_vl":
        messages_batch = build_qwen_messages(
            images,
            questions,
        )
        formatted_texts = format_chat_messages(
            processor,
            messages_batch,
        )

        image_inputs, video_inputs = process_vision_info(
            messages_batch
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
            questions
        )
        formatted_texts = format_chat_messages(
            processor,
            messages_batch,
        )

        # Gemma 3 expects one image list per text sample.
        # This avoids the common "images (1) and text (N)" error.
        nested_images = [[image] for image in images]

        try:
            inputs = processor(
                text=formatted_texts,
                images=nested_images,
                padding=True,
                return_tensors="pt",
            )
        except ValueError as error:
            if "inconsistently sized batches" not in str(error).lower():
                raise

            # Stable fallback for processor versions with strict batching.
            return [
                generate_gemma_single(
                    model=model,
                    processor=processor,
                    image=image,
                    question=question,
                    do_sample=do_sample,
                    max_new_tokens=max_new_tokens,
                )
                for image, question in zip(images, questions)
            ]

    elif family == "llava":
        prompts = build_llava_prompts(questions)

        inputs = processor(
            text=prompts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

    else:
        raise ValueError(f"Unknown model family: {family}")

    inputs = move_inputs_to_device(inputs, DEVICE)

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
    )


def load_model_and_processor(model_config):
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
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "torch_dtype": COMPUTE_DTYPE,
        "quantization_config": quantization_config,
        "cache_dir": HF_HUB_CACHE,
        "local_files_only": local_files_only,
        "trust_remote_code": True,
    }

    if family == "qwen2_5_vl":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_source,
            **model_kwargs,
        )
    elif family == "llava":
        model = LlavaForConditionalGeneration.from_pretrained(
            model_source,
            **model_kwargs,
        )
    elif family == "medgemma":
        model = AutoModelForImageTextToText.from_pretrained(
            model_source,
            **model_kwargs,
        )
    else:
        raise ValueError(f"Unknown model family: {family}")

    model.eval()

    print("Model loaded:", model_id)
    return model, processor


# %%
# ============================================================
# CELL 5 - Three-model inference with checkpoint/resume
# First download all three models to Google Drive
# ============================================================

from huggingface_hub import snapshot_download


DOWNLOAD_MODELS_BEFORE_INFERENCE = True


def ensure_all_models_downloaded():
    """
    Download Qwen, MedGemma, and LLaVA before inference starts.
    Existing local model directories are reused automatically.
    """

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
            print("=" * 80)
            print(f"Failed to download {model_id}")
            print(repr(error))
            print("=" * 80)

            if "medgemma" in model_id.lower():
                raise RuntimeError(
                    "MedGemma download failed. "
                    "Make sure you have accepted the MedGemma "
                    "license and logged in to Hugging Face."
                ) from error

            raise

        if not os.path.isfile(config_path):
            raise RuntimeError(
                f"Download finished but config.json is missing: "
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


def save_records(path, metadata, records):
    records = sorted(
        records,
        key=lambda item: int(item.get("index", -1)),
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


def load_existing_records(path, expected_metadata):
    if not os.path.isfile(path):
        return []

    try:
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:
            payload = json.load(file)

        if isinstance(payload, dict) and "records" in payload:
            old_metadata = payload.get(
                "metadata",
                {},
            )
            records = payload.get(
                "records",
                [],
            )

            important_metadata_keys = [
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
                for key in important_metadata_keys
            )

            if not metadata_matches:
                print(
                    "Existing cache metadata differs. "
                    "A new run will overwrite this run file."
                )
                return []

            print(
                f"Loaded existing cache: {len(records)} records"
            )
            return list(records)

        if isinstance(payload, list):
            print(
                f"Loaded old-format cache: {len(payload)} records"
            )
            return list(payload)

    except Exception as error:
        print("Failed to load cache:", repr(error))

    return []


MODEL_CONFIGS = [
    {
        "model_key": model_key,
        **config,
        "batch_size": MODEL_BATCH_SIZES[model_key],
        "max_new_tokens": MODEL_MAX_NEW_TOKENS[model_key],
    }
    for model_key, config in MODEL_REGISTRY.items()
]


all_model_output_files = []
model_errors = []


for model_config in MODEL_CONFIGS:
    model_key = model_config["model_key"]
    model_id = model_config["model_id"]
    family = model_config["family"]
    batch_size = int(model_config["batch_size"])
    max_new_tokens = int(model_config["max_new_tokens"])

    metadata = {
        "dataset": DATASET_NAME,
        "dataset_tag": DATASET_TAG,
        "split": SPLIT_NAME,
        "eval_size": EVAL_SIZE,
        "model_key": model_key,
        "model_id": model_id,
        "family": family,
        "batch_size": batch_size,
        "k_sampled_answers": K_SAMPLED_ANSWERS,
        "max_new_tokens": max_new_tokens,
        "system_message": SYSTEM_MESSAGE,
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
        print("FORCE_RERUN=True: ignoring old cache.")
    else:
        records = load_existing_records(
            records_path,
            metadata,
        )

    # De-duplicate records by sample index.
    records_by_index = {
        int(record["index"]): record
        for record in records
        if "index" in record
    }
    records = list(records_by_index.values())
    done_indices = set(records_by_index.keys())

    print("=" * 80)
    print("Current model:", model_id)
    print(
        f"Cached records: {len(done_indices)}/{EVAL_SIZE}"
    )
    print("Save path:", records_path)
    print("=" * 80)

    if len(done_indices) >= EVAL_SIZE:
        print(
            "Complete cache found. "
            "Skipping model loading for this model."
        )
        all_model_output_files.append(records_path)
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
            collate_fn=collate_pathvqa_batch,
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
            answers = [
                batch["answer"][position]
                for position in active_positions
            ]
            indices = [
                batch_indices[position]
                for position in active_positions
            ]

            try:
                greedy_predictions = generate_batch(
                    model=model,
                    processor=processor,
                    family=family,
                    images=images,
                    questions=questions,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                )

                sampled_predictions_by_k = []

                for _ in range(K_SAMPLED_ANSWERS):
                    sampled_predictions = generate_batch(
                        model=model,
                        processor=processor,
                        family=family,
                        images=images,
                        questions=questions,
                        do_sample=True,
                        max_new_tokens=max_new_tokens,
                    )
                    sampled_predictions_by_k.append(
                        sampled_predictions
                    )

                for position, index in enumerate(indices):
                    sampled_answers = [
                        sampled_predictions_by_k[k][position]
                        for k in range(K_SAMPLED_ANSWERS)
                    ]

                    records_by_index[int(index)] = {
                        "index": int(index),
                        "question": questions[position],
                        "ground_truth": answers[position],
                        "greedy_prediction": greedy_predictions[position],
                        "sampled_answers": sampled_answers,
                        "model_id": model_id,
                        "model_key": model_key,
                        "family": family,
                    }
                    done_indices.add(int(index))

                records = list(records_by_index.values())
                save_records(
                    records_path,
                    metadata,
                    records,
                )

            except RuntimeError as error:
                print("RuntimeError:", repr(error))

                if "out of memory" in str(error).lower():
                    print(
                        "CUDA OOM. Reduce this model batch size "
                        "in MODEL_BATCH_SIZES."
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                records = list(records_by_index.values())
                save_records(
                    records_path,
                    metadata,
                    records,
                )
                raise

            except Exception as error:
                print("Batch error:", repr(error))
                records = list(records_by_index.values())
                save_records(
                    records_path,
                    metadata,
                    records,
                )
                raise

        records = list(records_by_index.values())
        save_records(
            records_path,
            metadata,
            records,
        )
        all_model_output_files.append(records_path)

        print("=" * 80)
        print("Finished model:", model_id)
        print("Total saved records:", len(records))
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
# CELL 6 - AFD evaluation and final metrics
# ============================================================

import evaluate
import torch.nn.functional as F

from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sklearn.metrics import roc_auc_score, average_precision_score
from transformers import AutoTokenizer, AutoModel


for resource_name in [
    "wordnet",
    "omw-1.4",
    "punkt",
]:
    import nltk

    nltk.download(
        resource_name,
        quiet=True,
    )


print("AFD output directory:", OUTPUT_DIR)
print("AFD device:", DEVICE)


# -----------------------------
# Text normalisation
# -----------------------------

def normalize_text(text):
    if text is None:
        return ""

    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
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
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------------
# HF evaluate metrics
# Keep the lab notebook function name exactly.
# -----------------------------

def get_nlp_mettics(references, hypotheses):
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")

    references_norm = [
        normalize_text(reference) or "empty"
        for reference in references
    ]
    hypotheses_norm = [
        normalize_text(hypothesis) or "empty"
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
        "BLEU-1": float(results_bleu["precisions"][0]),
        "BLEU-2": float(results_bleu["precisions"][1]),
        "ROUGE-L": float(results_rouge["rougeL"]),
        "METEOR": float(results_meteor["meteor"]),
    }


sample_rouge_scorer = rouge_scorer.RougeScorer(
    ["rougeL"],
    use_stemmer=True,
)


def compute_single_metrics(prediction, reference):
    prediction = normalize_text(prediction) or "empty"
    reference = normalize_text(reference) or "empty"

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


def safe_auroc(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    if len(labels) == 0 or np.unique(labels).size < 2:
        return np.nan

    return float(roc_auc_score(labels, scores))


def safe_auprc(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    if len(labels) == 0 or np.unique(labels).size < 2:
        return np.nan

    return float(average_precision_score(labels, scores))


# -----------------------------
# BGE embeddings for semantic AFD
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

print("Embedding model loaded.")


@torch.inference_mode()
def encode_texts(texts, batch_size=EMBED_BATCH_SIZE):
    if not texts:
        return np.empty(
            (0, embed_model.config.hidden_size),
            dtype=np.float32,
        )

    all_embeddings = []

    for start in range(0, len(texts), batch_size):
        batch_texts = [
            normalize_text(text) or "empty"
            for text in texts[start:start + batch_size]
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

        outputs = embed_model(**encoded)
        token_embeddings = outputs.last_hidden_state
        attention_mask = encoded["attention_mask"]
        mask = attention_mask.unsqueeze(-1).expand(
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

    return np.concatenate(
        all_embeddings,
        axis=0,
    )


def mean_pairwise_similarity(embeddings):
    if len(embeddings) <= 1:
        return 1.0

    similarity_matrix = embeddings @ embeddings.T
    upper_indices = np.triu_indices(
        len(embeddings),
        k=1,
    )

    return float(
        np.mean(similarity_matrix[upper_indices])
    )


print("Preparing semantic embeddings...")

semantic_texts = []
semantic_ranges = []


def load_output_payload(json_path):
    with open(
        json_path,
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    if isinstance(payload, dict) and "records" in payload:
        return payload.get("metadata", {}), list(payload["records"])

    return {}, list(payload)


def output_is_complete(metadata, records):
    expected_size = metadata.get("eval_size")

    if expected_size is None:
        return len(records) > 0

    return len(records) == int(expected_size)


current_json_files = [
    os.path.join(OUTPUT_DIR, file_name)
    for file_name in os.listdir(OUTPUT_DIR)
    if file_name.startswith(f"outputs_{DATASET_TAG}_")
    and file_name.endswith(".json")
    and RUN_ID in file_name
]

current_json_files = sorted(current_json_files)

print("Candidate output files:")
for json_file in current_json_files:
    print("-", json_file)


if not current_json_files:
    raise RuntimeError(
        "No current PathVQA output JSON files were found. "
        "Run the inference cell first."
    )


def score_random(record, features):
    record_rng = random.Random(
        SEED + int(record.get("index", 0))
    )
    return float(record_rng.random())


def score_afd_frequency(record, features):
    sampled = record.get("sampled_answers", [])

    if not sampled:
        return 1.0

    normalized = [
        normalize_for_frequency(answer)
        for answer in sampled
    ]

    counts = {}
    for position, answer in enumerate(normalized):
        key = answer or f"__empty_{position}"
        counts[key] = counts.get(key, 0) + 1

    most_common_count = max(counts.values())
    reliability = most_common_count / len(normalized)
    return float(np.clip(1.0 - reliability, 0.0, 1.0))


SEMANTIC_SIMILARITY_THRESHOLD = 0.80


def score_semantic_afd(record, features):
    answer_embeddings = features["answer_embeddings"]
    number_of_answers = len(answer_embeddings)

    if number_of_answers <= 1:
        return 0.0

    similarity_matrix = answer_embeddings @ answer_embeddings.T
    visited = np.zeros(
        number_of_answers,
        dtype=bool,
    )
    cluster_sizes = []

    for start_index in range(number_of_answers):
        if visited[start_index]:
            continue

        stack = [start_index]
        visited[start_index] = True
        cluster_size = 0

        while stack:
            current_index = stack.pop()
            cluster_size += 1

            neighbours = np.where(
                similarity_matrix[current_index]
                >= SEMANTIC_SIMILARITY_THRESHOLD
            )[0]

            for neighbour_index in neighbours:
                if not visited[neighbour_index]:
                    visited[neighbour_index] = True
                    stack.append(int(neighbour_index))

        cluster_sizes.append(cluster_size)

    largest_cluster = max(cluster_sizes)
    return float(
        np.clip(
            1.0 - largest_cluster / number_of_answers,
            0.0,
            1.0,
        )
    )


def score_answer_disagreement(record, features):
    answer_embeddings = features["answer_embeddings"]

    if len(answer_embeddings) == 0:
        return 1.0

    consistency = (
        mean_pairwise_similarity(answer_embeddings) + 1.0
    ) / 2.0

    return float(
        np.clip(1.0 - consistency, 0.0, 1.0)
    )


def score_question_aligned_entropy(record, features):
    question_embedding = features["question_embedding"]
    answer_embeddings = features["answer_embeddings"]

    if len(answer_embeddings) == 0:
        return 1.0

    qa_similarities = answer_embeddings @ question_embedding
    qa_alignment = float(
        np.mean((qa_similarities + 1.0) / 2.0)
    )

    answer_consistency = (
        mean_pairwise_similarity(answer_embeddings) + 1.0
    ) / 2.0

    reliability = qa_alignment * answer_consistency
    return float(
        np.clip(1.0 - reliability, 0.0, 1.0)
    )


METHODS = {
    "Random baseline": score_random,
    "AFD frequency": score_afd_frequency,
    "Semantic AFD": score_semantic_afd,
    "Answer disagreement": score_answer_disagreement,
    "Question-aligned entropy": score_question_aligned_entropy,
}


def selective_metrics(records, uncertainty_scores):
    evaluation_df = pd.DataFrame(records).copy()
    evaluation_df["uncertainty"] = uncertainty_scores

    evaluation_df = evaluation_df.sort_values(
        "uncertainty",
        ascending=True,
        kind="mergesort",
    ).reset_index(drop=True)

    output = {}

    for coverage in SELECTIVE_COVERAGES:
        accepted_count = max(
            1,
            int(math.ceil(len(evaluation_df) * coverage)),
        )
        accepted_df = evaluation_df.iloc[:accepted_count]
        percentage = int(round(coverage * 100))

        output[
            f"Accepted samples @{percentage}%"
        ] = int(len(accepted_df))
        output[
            f"Accepted ROUGE-L @{percentage}%"
        ] = float(accepted_df["rougeL"].mean())
        output[
            f"Accepted METEOR @{percentage}%"
        ] = float(accepted_df["meteor"].mean())
        output[
            f"Accepted failure rate @{percentage}%"
        ] = float(accepted_df["failure"].mean())

    return output


def evaluate_output_json(json_path):
    print("=" * 80)
    print("Evaluating:", json_path)
    print("=" * 80)

    metadata, records = load_output_payload(json_path)

    if not output_is_complete(metadata, records):
        print(
            f"Skipping incomplete output: "
            f"{len(records)}/{metadata.get('eval_size')}"
        )
        return None

    records = sorted(
        records,
        key=lambda item: int(item.get("index", 0)),
    )

    model_id = metadata.get(
        "model_id",
        records[0].get("model_id", "unknown"),
    )
    model_key = metadata.get(
        "model_key",
        records[0].get("model_key", "unknown"),
    )
    batch_size = metadata.get("batch_size", "unknown")
    k_sampled = metadata.get(
        "k_sampled_answers",
        "unknown",
    )

    references = [
        record.get("ground_truth", "")
        for record in records
    ]
    hypotheses = [
        record.get("greedy_prediction", "")
        for record in records
    ]

    generation_metrics = get_nlp_mettics(
        references=references,
        hypotheses=hypotheses,
    )

    print("Generation metrics:", generation_metrics)

    scored_records = []

    for record in tqdm(
        records,
        desc="Per-sample metrics",
    ):
        item = dict(record)
        item.update(
            compute_single_metrics(
                prediction=item.get(
                    "greedy_prediction",
                    "",
                ),
                reference=item.get(
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

    # Build semantic text list once for this model.
    semantic_texts = []
    semantic_ranges = []

    for record in scored_records:
        start_index = len(semantic_texts)
        sampled_answers = record.get(
            "sampled_answers",
            [],
        )

        semantic_texts.append(
            record.get("question", "")
        )
        semantic_texts.extend(sampled_answers)
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

    for start_index, answer_count in semantic_ranges:
        question_embedding = all_embeddings[start_index]
        answer_embeddings = all_embeddings[
            start_index + 1:
            start_index + 1 + answer_count
        ]

        semantic_features.append(
            {
                "question_embedding": question_embedding,
                "answer_embeddings": answer_embeddings,
            }
        )

    method_scores = {}
    summary_rows = []
    scored_rows = []

    for method_name, method_function in METHODS.items():
        print("Running method:", method_name)

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
        method_scores[method_name] = scores

        summary_row = {
            "Model": model_key,
            "Model ID": model_id,
            "Method": method_name,
            "Dataset": DATASET_NAME,
            "Split": metadata.get(
                "split",
                SPLIT_NAME,
            ),
            "Evaluation samples": len(scored_records),
            "Batch size": batch_size,
            "K sampled answers": k_sampled,
            "BLEU-1": generation_metrics["BLEU-1"],
            "BLEU-2": generation_metrics["BLEU-2"],
            "ROUGE-L": generation_metrics["ROUGE-L"],
            "METEOR": generation_metrics["METEOR"],
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
        summary_rows.append(summary_row)

        for record, score in zip(
            scored_records,
            scores,
        ):
            scored_rows.append(
                {
                    "index": record.get("index"),
                    "question": record.get("question"),
                    "ground_truth": record.get("ground_truth"),
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
                    "rougeL": record.get("rougeL"),
                    "meteor": record.get("meteor"),
                    "failure": record.get("failure"),
                    "Method": method_name,
                    "uncertainty": float(score),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    scored_df = pd.DataFrame(scored_rows)

    summary_df = summary_df.sort_values(
        "Failure AUPRC",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    base_name = os.path.splitext(
        os.path.basename(json_path)
    )[0]

    summary_path = os.path.join(
        OUTPUT_DIR,
        f"afd_summary_{base_name}.csv",
    )
    scored_path = os.path.join(
        OUTPUT_DIR,
        f"afd_scored_records_{base_name}.csv",
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )
    scored_df.to_csv(
        scored_path,
        index=False,
    )

    print("Saved summary:", summary_path)
    print("Saved scored records:", scored_path)

    return summary_df


all_summary = []

for json_file in current_json_files:
    summary = evaluate_output_json(json_file)
    if summary is not None:
        all_summary.append(summary)

if not all_summary:
    raise RuntimeError(
        "No complete model outputs were available for AFD evaluation."
    )

final_summary_df = pd.concat(
    all_summary,
    ignore_index=True,
)

final_summary_df = final_summary_df.sort_values(
    ["Model", "Failure AUPRC"],
    ascending=[True, False],
    na_position="last",
).reset_index(drop=True)

for column in final_summary_df.columns:
    if final_summary_df[column].dtype.kind in "fc":
        final_summary_df[column] = final_summary_df[column].round(4)

final_summary_path = os.path.join(
    OUTPUT_DIR,
    "final_pathvqa_three_models_afd_summary.csv",
)

final_summary_df.to_csv(
    final_summary_path,
    index=False,
)

print("=" * 80)
print("Final AFD summary saved to:")
print(final_summary_path)
print("=" * 80)

display(final_summary_df)


# %%
# ============================================================
# CELL 7 - Clean final table and optional experiment archive
# ============================================================


pd.set_option("display.max_columns", None)
pd.set_option("display.width", 2400)
pd.set_option("display.max_colwidth", 60)


clean_report_columns = [
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

available_columns = [
    column
    for column in clean_report_columns
    if column in final_summary_df.columns
]

clean_summary_df = final_summary_df[
    available_columns
].copy()

clean_summary_df = clean_summary_df.sort_values(
    ["Model", "Failure AUPRC"],
    ascending=[True, False],
    na_position="last",
).reset_index(drop=True)

numeric_columns = clean_summary_df.select_dtypes(
    include=[np.number]
).columns

clean_summary_df.loc[
    :,
    numeric_columns,
] = clean_summary_df.loc[
    :,
    numeric_columns,
].round(4)

clean_summary_path = os.path.join(
    OUTPUT_DIR,
    "clean_final_pathvqa_afd_table.csv",
)

clean_summary_df.to_csv(
    clean_summary_path,
    index=False,
)

print("Clean table saved to:")
print(clean_summary_path)
display(clean_summary_df)


# -----------------------------
# Optional archive
# -----------------------------

ARCHIVE_RUN_NUMBER = ""

if ARCHIVE_RUN_NUMBER.strip():
    archive_run = ARCHIVE_RUN_NUMBER.strip()

    if not archive_run.isdigit():
        raise ValueError(
            "ARCHIVE_RUN_NUMBER must be a positive integer."
        )

    archive_dir = os.path.join(
        OUTPUT_DIR,
        "experiment_archive",
        f"pathvqa_run_{int(archive_run)}",
    )
    os.makedirs(archive_dir, exist_ok=True)

    archived_files = []

    for source_path in all_model_output_files:
        if os.path.isfile(source_path):
            destination = os.path.join(
                archive_dir,
                os.path.basename(source_path),
            )
            with open(
                source_path,
                "rb",
            ) as source_file:
                with open(
                    destination,
                    "wb",
                ) as destination_file:
                    destination_file.write(
                        source_file.read()
                    )
            archived_files.append(destination)

    archive_summary_path = os.path.join(
        archive_dir,
        "final_summary.csv",
    )
    clean_summary_df.to_csv(
        archive_summary_path,
        index=False,
    )

    manifest = {
        "run_number": int(archive_run),
        "dataset": DATASET_NAME,
        "split": SPLIT_NAME,
        "models": [
            config["model_key"]
            for config in MODEL_CONFIGS
        ],
        "inference_files": archived_files,
        "summary_file": archive_summary_path,
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

    print("Archive saved to:", archive_dir)
