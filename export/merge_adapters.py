"""
export/merge_adapters.py
------------------------
Fuses a LoRA adapter back into the bfloat16 base model weights and saves
the result as a standalone, portable model (no PEFT dependency needed at
inference time).

Strategy: load the base model to CPU RAM first to avoid VRAM overflow
during the weight merge, then serialize with safe_serialization=True.

Kaggle equivalent paths:
  ADAPTER_PATH     -> /kaggle/working/qwen3b-markmap-qlora/final_adapter
  MERGED_OUTPUT_DIR -> /kaggle/working/qwen3b-markmap-merged
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# Path to the trained LoRA adapter directory.
# Kaggle equivalent: /kaggle/working/qwen3b-markmap-qlora/final_adapter
ADAPTER_PATH = "./models/qwen3b-markmap-qlora/final_adapter"

# Destination directory for the merged, standalone model.
# Kaggle equivalent: /kaggle/working/qwen3b-markmap-merged
MERGED_OUTPUT_DIR = "./models/qwen3b-markmap-merged"


# ---------------------------------------------------------------------------
# MERGE ROUTINE
# ---------------------------------------------------------------------------

def merge_and_save(
    base_model_id: str = BASE_MODEL_ID,
    adapter_path: str  = ADAPTER_PATH,
    output_dir: str    = MERGED_OUTPUT_DIR,
) -> None:
    """
    Fuses the LoRA adapter weights into the bfloat16 base model and saves
    the resulting standalone model to ``output_dir``.

    Args:
        base_model_id: HuggingFace Hub model ID for the base model.
        adapter_path:  Local path to the saved LoRA adapter directory.
        output_dir:    Where to write the merged model shards.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load the base model in 16-bit (CRITICAL: no BitsAndBytes config here!)
    #    Load to CPU RAM first to avoid VRAM overflow during the merge step.
    print("Loading base model in bfloat16 (CPU)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",   # Avoids VRAM overflow — move to GPU only after merge
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

    # 2. Attach the LoRA adapter
    print("Attaching LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    # 3. Merge weights and unload the adapter overhead
    print("Fusing weights together (this may take a minute)...")
    merged_model = model.merge_and_unload()

    # 4. Save the final standalone model
    print(f"Saving standalone model to {output_dir} ...")
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    print(f"✅ Success! Standalone model saved to: {output_dir}")


if __name__ == "__main__":
    merge_and_save()
