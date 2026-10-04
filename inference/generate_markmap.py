"""
inference/generate_markmap.py
------------------------------
Loads the fine-tuned QLoRA adapter (attached to the 4-bit quantised base
model) and runs three inference scenarios:

  1. Quick sanity-check on a built-in Transformers test note.
  2. Evaluation on 2 random held-out samples from the validation split.
  3. Out-of-domain generalisation test on a history/economics note.

Kaggle equivalent paths:
  ADAPTER_PATH -> /kaggle/input/<your-notebook-name>/qwen3b-markmap-qlora/final_adapter
  DATA_PATH    -> /kaggle/input/datasets/harshithswamy20/markmap-1500/markmap_dataset_1500.json
"""

import os
import sys
import json
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ---------------------------------------------------------------------------
# Allow imports from project root
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.format_dataset import load_and_format, split_dataset

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# Path to the saved LoRA adapter directory (contains adapter_model.safetensors etc.)
# Kaggle equivalent: /kaggle/input/<notebook-name>/qwen3b-markmap-qlora/final_adapter
ADAPTER_PATH = "./models/qwen3b-markmap-qlora/final_adapter"

# Path to the raw dataset (needed only for the held-out eval scenario).
# Kaggle equivalent: /kaggle/input/datasets/harshithswamy20/markmap-1500/markmap_dataset_1500.json
DATA_PATH = "./data/markmap_dataset_1500.json"

SYSTEM_PROMPT = (
    "You are an expert technical knowledge synthesizer. "
    "Convert unstructured technical study notes into a strict, hierarchical Markmap mindmap."
)

# Generation hyper-parameters
GEN_KWARGS = dict(
    temperature=0.1,       # Low temperature for strict, deterministic formatting
    repetition_penalty=1.1,
)


# ---------------------------------------------------------------------------
# MODEL LOADER
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(adapter_path: str = ADAPTER_PATH):
    """
    Loads the base model in 4-bit NF4 quantisation and attaches the LoRA adapter.

    Returns:
        (model, tokenizer) ready for inference.
    """
    # Load tokeniser from the adapter directory (it was saved there during training)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    # 4-bit quantisation config — identical to training setup
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    return model, tokenizer


# ---------------------------------------------------------------------------
# INFERENCE HELPER
# ---------------------------------------------------------------------------

def generate(model, tokenizer, messages: list, max_new_tokens: int = 512) -> str:
    """
    Applies the chat template, runs generation, and returns the decoded output.

    Args:
        model:          The PEFT/LoRA model.
        tokenizer:      The corresponding tokenizer.
        messages:       List of {role, content} dicts (system + user only).
        max_new_tokens: Max tokens to generate.

    Returns:
        The generated text (assistant response only).
    """
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            **GEN_KWARGS,
        )

    input_length = inputs.input_ids.shape[1]
    return tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# SCENARIO 1 — Built-in sanity check
# ---------------------------------------------------------------------------

TRANSFORMERS_NOTE = """
Transformers revolutionized NLP by replacing RNNs with the self-attention mechanism.
Unlike RNNs which process tokens sequentially, transformers process all tokens in parallel, enabling massive scalability.
Key components include:
1. Multi-head attention: Allows the model to focus on different parts of the sequence simultaneously.
2. Positional encoding: Injects sequence order information since the model lacks inherent recurrence.
3. Feed-forward networks: Applies non-linear transformations to each position independently.
"""


def run_sanity_check(model, tokenizer):
    """Scenario 1: fixed Transformers note — quick sanity check."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": TRANSFORMERS_NOTE},
    ]
    print("\n🧠 [Scenario 1] Generating Markmap from Transformers note...\n")
    result = generate(model, tokenizer, messages, max_new_tokens=512)
    print(result)


# ---------------------------------------------------------------------------
# SCENARIO 2 — Held-out validation samples
# ---------------------------------------------------------------------------

def run_eval_samples(model, tokenizer, data_path: str = DATA_PATH, n_samples: int = 2):
    """
    Scenario 2: randomly draw n_samples from the unseen 10 % validation split
    and compare generated output to the ground-truth target.
    """
    full_dataset = load_and_format(data_path)
    _, eval_dataset = split_dataset(full_dataset)
    random_samples = random.sample(list(eval_dataset), n_samples)

    for i, sample in enumerate(random_samples, 1):
        input_notes   = sample["messages"][1]["content"]
        target_markmap = sample["messages"][2]["content"]

        # Pass only system + user turns to the model
        eval_messages = sample["messages"][:2]

        print(f"\n{'='*50}\n🔍 TEST SAMPLE {i}\n{'='*50}")
        print(f"\n[RAW INPUT NOTES (Truncated)]:\n{input_notes[:250]}...\n")

        generated = generate(model, tokenizer, eval_messages, max_new_tokens=768)

        print(f"🤖 [GENERATED MARKMAP]:\n{generated}\n")
        print(f"🎯 [ACTUAL TARGET FROM DATASET]:\n{target_markmap}\n")


# ---------------------------------------------------------------------------
# SCENARIO 3 — Out-of-domain generalisation
# ---------------------------------------------------------------------------

ECONOMICS_NOTE = """
The evolution of money transitioned through distinct phases to solve scaling issues in human trade.
It began with the barter system, which suffered from the 'double coincidence of wants'—both parties
had to want exactly what the other had simultaneously. To solve this, societies adopted commodity money
like salt, cowrie shells, and eventually precious metals. While commodities had intrinsic value, their
physical weight, divisibility issues, and supply fluctuations (e.g. a sudden silver mine discovery
causing inflation) created severe system constraints.

This led to the architecture of representative money, where paper receipts were backed by gold reserves
stored in bank vaults. However, the Gold Standard restricted central banks from adjusting the money
supply during economic crises like the Great Depression. Consequently, governments shifted to Fiat money,
which is completely unbacked by physical commodities and derives its value entirely from government decree
and social trust.

The core paradox of fiat money is that while it enables dynamic monetary policy and massive scalability
through digital banking ledgers, over-reliance on printing money can trigger hyperinflation. Today, edge
cases like decentralized cryptocurrencies (Bitcoin) and Central Bank Digital Currencies (CBDCs) attempt
to solve fiat's centralization flaws using blockchain consensus algorithms, though they struggle with
network latency, high energy costs, and transaction throughput limitations.
"""


def run_ood_test(model, tokenizer):
    """Scenario 3: out-of-domain history/economics note."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": ECONOMICS_NOTE},
    ]
    print("\n🧠 [Scenario 3] Generating Economics Markmap...\n")
    result = generate(model, tokenizer, messages, max_new_tokens=768)
    print(result)


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer(ADAPTER_PATH)

    run_sanity_check(model, tokenizer)
    run_eval_samples(model, tokenizer, DATA_PATH)
    run_ood_test(model, tokenizer)
