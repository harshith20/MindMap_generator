"""
training/train_qlora.py
-----------------------
QLoRA fine-tuning script for Qwen2.5-3B-Instruct on the Markmap dataset.
Includes:
  - 4-bit NF4 quantization via BitsAndBytes
  - LoRA adapter configuration targeting all projection layers
  - VRAM tracking callback
  - EarlyStopping to prevent overfitting
  - SFTTrainer with assistant-only loss

Kaggle equivalent paths:
  DATA_PATH  -> /kaggle/input/datasets/harshithswamy20/markmap-1500/markmap_dataset_1500.json
  OUTPUT_DIR -> /kaggle/working/qwen3b-markmap-qlora
"""

import os
import sys
import time
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ---------------------------------------------------------------------------
# Allow imports from the project root (e.g. data.format_dataset)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.format_dataset import load_and_format, split_dataset

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# Path to the raw dataset JSON file.
# Kaggle equivalent: /kaggle/input/datasets/harshithswamy20/markmap-1500/markmap_dataset_1500.json
DATA_PATH = "./data/markmap_dataset_1500.json"

# Directory where adapter checkpoints and the final adapter are saved.
# Kaggle equivalent: /kaggle/working/qwen3b-markmap-qlora
OUTPUT_DIR = "./models/qwen3b-markmap-qlora"

MAX_SEQ_LENGTH = 2048


# ---------------------------------------------------------------------------
# VRAM TRACKER CALLBACK
# ---------------------------------------------------------------------------

class VRAMTrackerCallback(TrainerCallback):
    """Logs VRAM usage and elapsed time at every logging step and at the end."""

    def __init__(self):
        super().__init__()
        self.gb_divisor = 1024 ** 3

    def on_train_begin(self, args, state, control, **kwargs):
        """Initializes global timers and resets CUDA memory peak counters."""
        self.start_time = time.time()
        self.last_log_time = time.time()
        # Reset hardware peak tracking to zero for the entire run
        torch.cuda.reset_peak_memory_stats()
        print("\n🚀 [Training Started] Global VRAM & execution timer initialized.\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Prints regular interval checkpoints every logging_steps."""
        current_time = time.time()
        time_elapsed = current_time - self.last_log_time
        self.last_log_time = current_time

        mem_allocated = torch.cuda.memory_allocated() / self.gb_divisor
        mem_reserved  = torch.cuda.memory_reserved()  / self.gb_divisor

        print(f"\n--- 📊 Step {state.global_step}/{state.max_steps} Stats ---")
        print(f"⏱️ Time for last {args.logging_steps} steps: {time_elapsed:.2f}s")
        print(f"💾 Active VRAM:   {mem_allocated:.2f} GB")
        print(f"📦 Reserved VRAM: {mem_reserved:.2f} GB")
        print(f"📉 Current Loss:  {logs.get('loss', 'N/A')}")
        print("------------------------------------------")

    def on_train_end(self, args, state, control, **kwargs):
        """Prints the final holistic summary once all epochs finish."""
        total_time_seconds = time.time() - self.start_time
        total_minutes      = total_time_seconds / 60
        global_peak_vram   = torch.cuda.max_memory_allocated() / self.gb_divisor

        # Grab the last recorded training loss from the state history
        final_loss = state.log_history[-1].get("loss", "N/A") if state.log_history else "N/A"

        print("\n==========================================")
        print("🎯 TRAINING COMPLETE: FINAL SUMMARY 🎯")
        print("==========================================")
        print(f"⏱️ Total Wall-Clock Time: {total_minutes:.2f} minutes")
        print(f"🚀 Total Steps Processed: {state.global_step}")
        print(f"🔥 Global Peak VRAM:      {global_peak_vram:.2f} GB (Max memory hit across entire run)")
        print(f"📉 Final Training Loss:   {final_loss}")
        print("==========================================\n")


# ---------------------------------------------------------------------------
# MAIN TRAINING ROUTINE
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 1. DATASET PREPARATION ---
    print(f"[+] Loading dataset from: {DATA_PATH}")
    full_dataset  = load_and_format(DATA_PATH)
    train_dataset, eval_dataset = split_dataset(full_dataset)
    print(f"[✓] Train: {len(train_dataset)} samples | Val: {len(eval_dataset)} samples")

    # --- 2. TOKENIZER & QUANTIZATION CONFIG ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token     = tokenizer.eos_token
    tokenizer.padding_side  = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    # --- 3. QLORA CONFIGURATION ---
    peft_config = LoraConfig(
        r=16,               # Rank; reduced from 32 to prevent overfitting
        lora_alpha=32,      # Scaled down to match the new rank
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.1,   # Increased from 0.05 for stronger regularization
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # --- 4. TRAINING ARGUMENTS ---
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        max_length=MAX_SEQ_LENGTH,
        eval_strategy="steps",
        assistant_only_loss=True,        # Only compute loss on assistant turns
        loss_type="nll",
        num_train_epochs=1,              # Fits within a single Kaggle T4 session (~6-7 h)
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=2,
        eval_steps=25,                   # Synced with save_steps to enable Early Stopping
        save_strategy="steps",
        save_steps=25,                   # Synced with eval_steps to enable Early Stopping
        load_best_model_at_end=True,     # Required for EarlyStoppingCallback
        save_total_limit=2,
        logging_steps=10,
        learning_rate=5e-5,              # Lowered from 2e-4 for stable, generalised convergence
        weight_decay=0.01,
        warmup_steps=25,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",
    )

    # --- 5. TRAINER INITIALISATION & RUN ---
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        args=training_args,
        callbacks=[
            VRAMTrackerCallback(),
            EarlyStoppingCallback(early_stopping_patience=3),  # Halts if eval loss stagnates for 75 steps
        ],
    )

    print("\n[+] Starting QLoRA Fine-Tuning...")
    trainer.train()

    # --- 6. SAVE FINAL ADAPTER ---
    adapter_save_path = os.path.join(OUTPUT_DIR, "final_adapter")
    trainer.model.save_pretrained(adapter_save_path)
    tokenizer.save_pretrained(adapter_save_path)
    print(f"[✓] LoRA adapter successfully saved to {adapter_save_path}")


if __name__ == "__main__":
    main()
