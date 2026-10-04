"""
export/upload_to_hf.py
----------------------
Uploads model artefacts to the Hugging Face Hub:

  • Step A — Merged bfloat16 model  → repo root
  • Step B — Raw LoRA adapter files → <repo>/lora-adapter/ subfolder

Authentication:
  On Kaggle: reads HF_TOKEN from Kaggle Secrets (UserSecretsClient).
  Locally  : reads the HF_TOKEN environment variable.

Kaggle equivalent paths:
  LOCAL_MERGED_PATH -> /kaggle/working/qwen3b-markmap-merged
  ADAPTER_PATH      -> /kaggle/working/qwen3b-markmap-qlora/final_adapter
"""

import os
from huggingface_hub import HfApi, login

# ---------------------------------------------------------------------------
# CONFIGURATION — edit these before running
# ---------------------------------------------------------------------------

# Your HuggingFace Hub repository ID (username/repo-name).
HF_REPO_ID = "harshith20/qwen2.5-3b-markmap-synthesizer"

# Local path to the merged standalone model.
# Kaggle equivalent: /kaggle/working/qwen3b-markmap-merged
LOCAL_MERGED_PATH = "./models/qwen3b-markmap-merged"

# Local path to the raw LoRA adapter files.
# Kaggle equivalent: /kaggle/working/qwen3b-markmap-qlora/final_adapter
ADAPTER_PATH = "./models/qwen3b-markmap-qlora/final_adapter"


# ---------------------------------------------------------------------------
# AUTHENTICATION
# ---------------------------------------------------------------------------

def authenticate() -> None:
    """
    Logs in to the Hugging Face Hub.

    Priority:
      1. Kaggle Secrets (UserSecretsClient) — used when running on Kaggle.
      2. HF_TOKEN environment variable      — used locally.
    """
    hf_token = None

    try:
        from kaggle_secrets import UserSecretsClient  # Only available on Kaggle
        hf_token = UserSecretsClient().get_secret("HF_TOKEN")
        print("[Auth] Using Kaggle Secrets for HF_TOKEN.")
    except Exception:
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            print("[Auth] Using HF_TOKEN environment variable.")
        else:
            raise EnvironmentError(
                "No HF_TOKEN found. Set the HF_TOKEN environment variable or "
                "add it to Kaggle Secrets before running this script."
            )

    login(token=hf_token)


# ---------------------------------------------------------------------------
# UPLOAD ROUTINES
# ---------------------------------------------------------------------------

def upload_merged_model(
    repo_id: str       = HF_REPO_ID,
    local_path: str    = LOCAL_MERGED_PATH,
) -> None:
    """
    Uploads the entire merged model directory to the repo root.

    Args:
        repo_id:    HuggingFace Hub repo ID (e.g. ``username/repo-name``).
        local_path: Local directory containing merged model shards.
    """
    api = HfApi()

    print(f"Creating repository {repo_id} (if it doesn't exist)...")
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)

    print(f"Uploading merged model to https://huggingface.co/{repo_id} ...")
    api.upload_folder(
        folder_path=local_path,
        repo_id=repo_id,
        repo_type="model",
    )
    print("✅ Merged model uploaded successfully!\n")


def upload_lora_adapter(
    repo_id: str      = HF_REPO_ID,
    adapter_path: str = ADAPTER_PATH,
) -> None:
    """
    Uploads the raw LoRA adapter files to the ``lora-adapter/`` subfolder
    in the repo, keeping them isolated from the merged weights.

    Args:
        repo_id:      HuggingFace Hub repo ID.
        adapter_path: Local directory containing the LoRA adapter files.
    """
    api = HfApi()

    print(f"Uploading raw LoRA adapters to https://huggingface.co/{repo_id}/tree/main/lora-adapter ...")
    api.upload_folder(
        folder_path=adapter_path,
        repo_id=repo_id,
        path_in_repo="lora-adapter",  # Isolates the adapters from the merged weights
        repo_type="model",
    )
    print("✅ LoRA adapters uploaded successfully!\n")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    authenticate()
    upload_merged_model()
    upload_lora_adapter()
