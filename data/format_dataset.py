"""
data/format_dataset.py
----------------------
Loads the raw JSON dataset, formats each entry into ChatML-style messages,
and produces a reproducible 90/10 train/validation split.

Kaggle equivalent paths:
  DATA_PATH  -> /kaggle/input/datasets/harshithswamy20/markmap-1500/markmap_dataset_1500.json
  OUTPUT_DIR -> /kaggle/working/data/
"""

import json
import os
from datasets import Dataset

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# Path to the raw JSON dataset file.
# Kaggle equivalent: /kaggle/input/datasets/harshithswamy20/markmap-1500/markmap_dataset_1500.json
DATA_PATH = "./data/markmap_dataset_1500.json"

SYSTEM_PROMPT = (
    "You are an expert technical knowledge synthesizer. "
    "Convert unstructured technical study notes into a strict, hierarchical Markmap mindmap."
)

TRAIN_VAL_SEED = 42
VAL_SPLIT = 0.1  # 90 % train / 10 % validation


def load_and_format(data_path: str = DATA_PATH) -> Dataset:
    """
    Reads the raw JSON file and converts every entry into a ChatML
    ``messages`` dict with system / user / assistant turns.

    Args:
        data_path: Path to the JSON dataset file.

    Returns:
        A HuggingFace Dataset with a single ``messages`` column.
    """
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    formatted_data = []
    for entry in raw_data:
        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": entry["input_summary_notes"]},
            {"role": "assistant", "content": entry["target_markmap"]},
        ]
        formatted_data.append({"messages": messages})

    return Dataset.from_list(formatted_data)


def split_dataset(full_dataset: Dataset, test_size: float = VAL_SPLIT, seed: int = TRAIN_VAL_SEED):
    """
    Performs a reproducible stratified split.

    Args:
        full_dataset: The fully formatted HuggingFace Dataset.
        test_size:    Fraction of data reserved for validation (default 0.10).
        seed:         Random seed for reproducibility (default 42).

    Returns:
        Tuple of (train_dataset, eval_dataset).
    """
    split = full_dataset.train_test_split(test_size=test_size, seed=seed)
    return split["train"], split["test"]


if __name__ == "__main__":
    print(f"[+] Loading dataset from: {DATA_PATH}")
    full_ds = load_and_format()
    train_ds, eval_ds = split_dataset(full_ds)

    print(f"[✓] Full dataset size : {len(full_ds)}")
    print(f"[✓] Train split size  : {len(train_ds)}")
    print(f"[✓] Val   split size  : {len(eval_ds)}")
