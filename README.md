# markmap-synthesizer

> **QLoRA fine-tuned Qwen2.5-3B-Instruct** that converts raw, unstructured technical study notes into hierarchical [Markmap](https://markmap.js.org/) mind-maps in a single inference call.

---

## 🔗 Hugging Face Links

| Artefact | Link |
|---|---|
| Merged bfloat16 model | [harshith20/qwen2.5-3b-markmap-synthesizer](https://huggingface.co/harshith20/qwen2.5-3b-markmap-synthesizer) |
| Raw LoRA adapter | [harshith20/qwen2.5-3b-markmap-synthesizer/tree/main/lora-adapter](https://huggingface.co/harshith20/qwen2.5-3b-markmap-synthesizer/tree/main/lora-adapter) |

---

## 🏗️ Repository Structure

```
markmap-synthesizer/
├── README.md                   ← You are here
├── requirements.txt            ← Python dependencies
├── data/
│   └── format_dataset.py       ← Dataset formatting & 90/10 train/val split
├── training/
│   └── train_qlora.py          ← QLoRA training loop (VRAM callback + EarlyStopping)
├── inference/
│   └── generate_markmap.py     ← Inference script (3 test scenarios)
└── export/
    ├── merge_adapters.py       ← Fuses LoRA weights into bfloat16 base model
    └── upload_to_hf.py         ← Uploads merged model & raw adapter to HF Hub
```

> **Note:** Trained model artefacts are written to `./models/` (created at runtime, not committed to git).  
> Place your dataset at `./data/markmap_dataset_1500.json` before running any script.

---

## 🧠 Architecture

### Base Model
**Qwen/Qwen2.5-3B-Instruct** — a 3 B-parameter instruction-tuned causal LM chosen for its strong zero-shot instruction following and its ability to fit within Kaggle's 16 GB T4 VRAM budget under 4-bit quantisation.

### Fine-Tuning Method — QLoRA
| Setting | Value |
|---|---|
| Quantisation | 4-bit NF4 (bfloat16 compute) |
| LoRA rank `r` | 16 |
| LoRA alpha | 32 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| LoRA dropout | 0.1 |
| Trainable params | ~30 M / 3.1 B (≈ 0.96 %) |

### Training Setup
| Hyper-parameter | Value |
|---|---|
| Epochs | 1 |
| Batch size | 2 × 4 grad-accum steps (effective batch = 8) |
| Learning rate | 5 × 10⁻⁵ (cosine decay, 25 warmup steps) |
| Max sequence length | 2 048 tokens |
| Optimiser | `paged_adamw_8bit` |
| Loss | NLL on assistant turns only (`assistant_only_loss=True`) |
| Early stopping patience | 3 eval checkpoints (75 steps) |
| Hardware | Kaggle NVIDIA Tesla T4 (16 GB VRAM) |
| Wall-clock time | ~6 h 35 m |
| Peak VRAM | 10.49 GB |

### Training Results

| Step | Train Loss | Val Loss | Token Accuracy |
|---:|---:|---:|---:|
| 25 | 0.402 | 0.314 | 92.3 % |
| 50 | 0.178 | 0.159 | 96.0 % |
| 100 | 0.142 | 0.133 | 96.5 % |
| 150 | 0.123 | 0.129 | 96.6 % |
| 167 | 0.110 | 0.129 | 96.6 % |

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place the dataset
```
data/
└── markmap_dataset_1500.json   ← download from Kaggle: harshithswamy20/markmap-1500
```

### 3. Fine-tune
```bash
python training/train_qlora.py
# Adapter saved to: ./models/qwen3b-markmap-qlora/final_adapter
```

### 4. Run inference
```bash
python inference/generate_markmap.py
# Runs 3 scenarios: sanity check, held-out eval, and out-of-domain test
```

### 5. Merge LoRA into base weights
```bash
python export/merge_adapters.py
# Standalone model saved to: ./models/qwen3b-markmap-merged
```

### 6. Upload to Hugging Face Hub
```bash
export HF_TOKEN="hf_..."
python export/upload_to_hf.py
```

---

## 📋 Path Reference (Kaggle → Local)

| Original Kaggle Path | Local Equivalent |
|---|---|
| `/kaggle/input/datasets/harshithswamy20/markmap-1500/markmap_dataset_1500.json` | `./data/markmap_dataset_1500.json` |
| `/kaggle/working/qwen3b-markmap-qlora` | `./models/qwen3b-markmap-qlora` |
| `/kaggle/working/qwen3b-markmap-qlora/final_adapter` | `./models/qwen3b-markmap-qlora/final_adapter` |
| `/kaggle/working/qwen3b-markmap-merged` | `./models/qwen3b-markmap-merged` |

---

## 📄 License
MIT
