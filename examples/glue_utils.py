import datasets
import numpy as np
import transformers
from datasets import load_dataset
from transformers.data.processors import glue as hf_glue


_glue_output_modes = hf_glue.glue_output_modes

_STSB_MIN = 0
_STSB_MAX = 5
_STSB_NUM_BINS = 5 * (_STSB_MAX - _STSB_MIN)

transformers.logging.set_verbosity_error()
datasets.logging.set_verbosity_error()


def _get_task_config(task: str, split: str) -> tuple[str, str, dict]:
    if task == "sst-2":
        dataset_task = "sst2"
    elif task == "sts-b":
        dataset_task = "stsb"
    elif task == "mnli-mm":
        dataset_task = "mnli"
        if split != "train":
            split = (
                "validation_mismatched" if split == "validation" else "test_mismatched"
            )
    elif task == "mnli":
        dataset_task = "mnli"
        if split != "train":
            split = "validation_matched" if split == "validation" else "test_matched"
    else:
        dataset_task = task

    if task in ["cola", "sst-2", "sst2"]:
        text_keys = {"text_a": "sentence", "text_b": None}
    elif task in ["mnli", "mnli-mm", "qnli", "wnli"]:
        text_keys = {"text_a": "premise", "text_b": "hypothesis"}
    elif task in ["qqp"]:
        text_keys = {"text_a": "question1", "text_b": "question2"}
    elif task in ["mrpc"]:
        text_keys = {"text_a": "sentence1", "text_b": "sentence2"}
    elif task in ["sts-b", "stsb"]:
        text_keys = {"text_a": "sentence1", "text_b": "sentence2"}
    elif task in ["rte"]:
        text_keys = {"text_a": "sentence1", "text_b": "sentence2"}
    else:
        raise ValueError(f"Task {task} not supported")

    return dataset_task, split, text_keys


def process_glue_dataset(
    examples,
    tokenizer: transformers.AutoTokenizer,
    max_length: int,
    task: str,
    text_keys: dict,
) -> dict:
    text_a = examples[text_keys["text_a"]]
    text_b = None if text_keys["text_b"] is None else examples[text_keys["text_b"]]

    # tokenize
    tokenized = tokenizer(
        text_a,
        text_pair=text_b,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )

    if "label" in examples:
        if _glue_output_modes[task] == "classification":
            tokenized["labels"] = examples["label"]
        elif _glue_output_modes[task] == "regression":
            if task in ["sts-b", "stsb"]:  # bin if needed
                stsb_bins = np.linspace(_STSB_MIN, _STSB_MAX, num=_STSB_NUM_BINS + 1)
                stsb_bins = stsb_bins[1:-1]
                labels = [np.digitize(label, stsb_bins) for label in examples["label"]]
                tokenized["labels"] = labels
            else:
                tokenized["labels"] = examples["label"]

    return tokenized


def load_glue_dataset(
    task: str,
    split: str,
    tokenizer: transformers.AutoTokenizer,
    max_length: int,
    batch_size: int = 1000,
) -> datasets.Dataset:
    dataset_task, dataset_split, text_keys = _get_task_config(task, split)

    # load
    dataset = load_dataset("nyu-mll/glue", dataset_task, split=dataset_split)

    # process
    tokenized_dataset = dataset.map(
        lambda examples: process_glue_dataset(
            examples, tokenizer, max_length, task, text_keys
        ),
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset.column_names,
    )

    return tokenized_dataset
