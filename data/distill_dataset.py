"""
data/distill_dataset.py
-----------------------
Reverse-data-distillation pipeline that generates the Markmap training dataset
using the Gemini API. Runs in two phases for every topic:

  Phase 2 — Generates a structured Markmap mindmap (the TARGET label).
  Phase 3 — Synthesises 500-700 word Obsidian-style study notes from the
             Markmap (the INPUT to the fine-tuned model).

Features:
  • Multi-key rotation: hot-swaps to the next GEMINI_API_KEY_* when a 429 /
    RESOURCE_EXHAUSTED error hits the 1,000 RPD free-tier limit.
  • Auto-resume: if a partial output file exists it is loaded and the pipeline
    continues from where it left off — safe to re-run after a crash.
  • Atomic saves: each completed pair is written via a .tmp rename so a mid-save
    crash never corrupts the dataset.
  • Style injection: three randomised note-writing styles keep the training
    distribution diverse and prevent the model from learning a single template.

Environment variables (set in .env or export before running):
  GEMINI_API_KEY_1   Primary Gemini API key
  GEMINI_API_KEY_2   Secondary key (optional, used after key 1 hits quota)
  GEMINI_API_KEY_3   Tertiary key  (optional, used after key 2 hits quota)

Kaggle equivalent paths (for reference):
  INPUT_TOPICS_PATH  -> /kaggle/input/datasets/<user>/phase1-deduplicated-topics/phase1_deduplicated_topics.json
  PREVIOUS_RUN_FILE  -> /kaggle/input/datasets/<user>/markmap-training-1500/markmap_dataset_1500.json
  OUTPUT_FILE        -> /kaggle/working/markmap_dataset_1500.json
"""

import os
import json
import time
import re
import random
import shutil
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# Load API keys from environment — never hardcode keys in source files.
# Add keys to a local .env file (which is gitignored) and source it,
# or export them in your shell before running this script.
_raw_keys = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
]
API_KEYS = [k for k in _raw_keys if k]  # Drop any unset / None entries

if not API_KEYS:
    raise EnvironmentError(
        "No Gemini API keys found. Set at least GEMINI_API_KEY_1 in your "
        "environment (or in a .env file) before running this script."
    )

current_key_index = 0
client = genai.Client(api_key=API_KEYS[current_key_index])

MODEL_ID    = "gemini-2.5-flash-lite"
SAMPLE_SIZE = 1500

# Path to the Phase-1 deduplicated topics JSON file.
# Kaggle equivalent: /kaggle/input/datasets/<user>/phase1-deduplicated-topics/...
INPUT_TOPICS_PATH = "./data/phase1_deduplicated_topics.json"

# If a previous partial run exists, copy it here first so we can resume.
# Kaggle equivalent: /kaggle/input/datasets/<user>/markmap-training-1500/markmap_dataset_1500.json
PREVIOUS_RUN_FILE = ""  # Leave empty if there is no previous partial run to resume from.

# Destination for the generated dataset (progressively updated).
# Kaggle equivalent: /kaggle/working/markmap_dataset_1500.json
OUTPUT_FILE = "./data/markmap_dataset_1500.json"


# ---------------------------------------------------------------------------
# API KEY ROTATION
# ---------------------------------------------------------------------------

def switch_api_key() -> None:
    """Hot-swaps the active Gemini client when the 1,000 RPD limit is hit."""
    global current_key_index, client
    current_key_index += 1

    if current_key_index >= len(API_KEYS):
        print("\n[!] FATAL: All provided API keys have exhausted their daily quotas.")
        exit()

    print(f"\n[!] Daily Quota Hit. Hot-swapping to API Key {current_key_index + 1}...")
    client = genai.Client(api_key=API_KEYS[current_key_index])
    time.sleep(3)  # Brief buffer before resuming


# ---------------------------------------------------------------------------
# DATA I/O UTILITIES
# ---------------------------------------------------------------------------

def load_phase1_topics(filepath: str) -> list:
    """
    Loads the Phase-1 topic list from a JSON file or directory.

    Args:
        filepath: Path to the JSON file, or to a directory containing exactly
                  one JSON file.

    Returns:
        List of topic dicts, or empty list if the file is missing.
    """
    if not os.path.exists(filepath):
        print(f"Error: Path does not exist: {filepath}")
        return []
    if os.path.isdir(filepath):
        json_files = [f for f in os.listdir(filepath) if f.endswith(".json")]
        if not json_files:
            return []
        filepath = os.path.join(filepath, json_files[0])
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_dataset_safely(dataset: list, filename: str = OUTPUT_FILE) -> None:
    """
    Atomic write: saves to a .tmp file first then renames — crash-proof.

    Args:
        dataset:  List of completed (topic, markmap, notes) dicts.
        filename: Destination JSON path.
    """
    temp_filename = filename + ".tmp"
    with open(temp_filename, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    os.replace(temp_filename, filename)  # Instant, crash-proof rename


# ---------------------------------------------------------------------------
# MARKMAP VALIDATION & CLEANING
# ---------------------------------------------------------------------------

def clean_markmap_output(raw_text: str) -> str:
    """
    Sanitises LLM output to prevent Obsidian parsing crashes.

    Removes accidental code fences, non-breaking spaces, tabs, and stray
    block-quote characters injected by the model.

    Args:
        raw_text: Raw string returned by the Gemini API.

    Returns:
        Cleaned Markmap string.
    """
    if not raw_text:
        return ""
    text = re.sub(r"^```[a-zA-Z]*\n", "", raw_text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\n```$", "", text, flags=re.MULTILINE)
    text = text.replace("\u00a0", " ").replace("\t", "  ")
    text = text.replace("> -", "-").replace(">-", "-")
    # Fix "- \nText" bug: removes the erroneous newline after a bullet dash
    text = re.sub(r"-\s*\n\s*", "- ", text)
    return text.strip()


def validate_markmap_syntax(markmap_str: str) -> tuple[bool, str]:
    """
    Validates that the generated Markmap has the expected structure.

    Rules:
      - First non-empty line must be a Level-1 heading (``# ``).
      - Must contain exactly 4 Level-2 headings (``## ``).

    Args:
        markmap_str: The cleaned Markmap string.

    Returns:
        (is_valid, reason) where reason is ``"Valid"`` or an error description.
    """
    lines = [line for line in markmap_str.strip().split("\n") if line.strip()]
    if not lines or not lines[0].startswith("# "):
        return False, "Missing root # Heading"
    h2_count = sum(1 for line in lines if line.startswith("## "))
    if h2_count != 4:
        return False, f"Expected 4 Level-2 branches, found {h2_count}"
    return True, "Valid"


# ---------------------------------------------------------------------------
# PHASE 2 — MARKMAP GENERATION
# ---------------------------------------------------------------------------

def generate_phase2_markmap(topic_name: str, retries: int = 3) -> str | None:
    """
    Calls Gemini to generate a structured, 4-branch Markmap for a topic.

    Args:
        topic_name: The highly specific domain topic to map.
        retries:    Number of retry attempts on non-quota errors (e.g. 503).

    Returns:
        Cleaned Markmap string, or None on persistent failure.
    """
    prompt = f"""Act as a Principal Researcher. Map the highly specific domain of '{topic_name}' into a hierarchical Markmap mindmap using clean Markdown lists.

STRUCTURAL REQUIREMENTS (EXACTLY 4 MAIN CATEGORIES):
Use these exact 4 Level-2 headings (##):
## Core Intuition & Foundational Questions
## Mechanics, Architecture & Evolution
## System Constraints & Core Logic
- IF quantitative: Include bullets for formulas and explicit parameter definitions. Use standard inline LaTeX ($x$).
- IF qualitative: Detail systemic bottlenecks, resource caps, or structural limits.
## Edge Cases, Paradoxes & Real-World Impact

HIERARCHY, SYNTAX & COMPRESSION RULES (CRITICAL):
1. ROOT COMPRESSION: Do NOT just copy the full topic name. Compress the topic into a short, punchy title (Max 4-6 words) for Line 1: # [Short Title]
2. Exactly 4 Level-2 branches (##) as defined above.
3. Under each Level-2 branch, provide 3 to 6 bullet items (- ).
4. BULLET FLUIDITY & COMPRESSION: Keep nodes concise (6-12 words). Write them as natural-sounding, complete thoughts or crisp statements rather than abrupt, fragmented nouns.
5. SPECIFIC KEYWORDS: Anchor on specific technical terminology so the summary writer has exact concepts to expand upon.
6. Sub-bullets MUST nest using exactly 2 standard spaces (  - ). STRICTLY DO NOT use tabs or blockquotes (>).
7. Output ONLY the raw Markdown list. Do NOT use ``` code fences or introductory text."""

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.3),
            )
            return clean_markmap_output(response.text)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                switch_api_key()
                return generate_phase2_markmap(topic_name, retries)  # Retry with new key
            print(f"  [Phase 2 Error on attempt {attempt + 1}]: {e}")
            time.sleep(3)  # Wait before retry on transient 503 errors
    return None


# ---------------------------------------------------------------------------
# PHASE 3 — STUDY NOTES GENERATION (WITH STYLE INJECTION)
# ---------------------------------------------------------------------------

# Three distinct writing styles are randomly sampled to keep the training
# distribution diverse and prevent the model from learning a single template.
STYLE_VARIATIONS = [
    """STYLE INJECTION: The Socratic Synthesizer
- LAYOUT: Flowing, inquisitive narrative paragraphs mixed with targeted bullet points.
- SPECIFIC TONE: Deeply inquisitive. Continuously ask rhetorical questions (e.g., "But why structure it this way?", "What is the hidden tradeoff here?") before explaining the concepts.
- THE ENDING: Conclude with a 'Critical Synthesis' paragraph that questions the long-term viability, limits, or hidden paradoxes of the system.""",

    """STYLE INJECTION: The Skeptical Braindump
- LAYOUT: Fast-paced, slightly raw shorthand. Use heavily nested bullets, abrupt transitions, and punchy insights.
- SPECIFIC TONE: Pragmatic and paranoid. Interrogate every constraint and formula, focusing entirely on where the architecture breaks under pressure.
- THE ENDING: Conclude with a 'Lingering Questions' block that aggressively challenges the edge cases and unanswered problems.""",

    """STYLE INJECTION: The First-Principles Interrogator
- LAYOUT: Balanced deep explanatory text with crisp technical breakdowns. Invent organic, subject-specific headers (do not use generic templates).
- SPECIFIC TONE: Strip concepts to their core by starting sections with a hard question (e.g., "What fundamental bottleneck does this actually solve?").
- THE ENDING: Conclude with a 'Summary & Open Problems' section that critically evaluates the system's tradeoffs and questions future constraints.""",
]


def generate_phase3_notes(markmap_code: str, retries: int = 3) -> str | None:
    """
    Calls Gemini to synthesise Obsidian-style study notes from a Markmap.

    A random style is injected into the prompt each time to produce diverse
    writing patterns across the training corpus.

    Args:
        markmap_code: The validated Markmap string produced by Phase 2.
        retries:      Number of retry attempts on non-quota errors.

    Returns:
        Raw study-notes string, or None on persistent failure.
    """
    selected_style = random.choice(STYLE_VARIATIONS)

    prompt = f"""Review the following hierarchical Markmap outline:
{markmap_code}

Act as a pragmatic expert compiling clear, personal study notes in Obsidian based entirely on this outline.
Generate 500-700 words of comprehensive summary notes that capture 100% of the outline's information.

DATA DISTILLATION CONSTRAINTS (CRITICAL FOR REVERSE EXTRACTION):
- INFORMATION COMPLETENESS: You must explicitly mention and explain *every single bullet point, formula, parameter, and edge case* present in the outline.
- NO ORPHANED NODES: If a reader only had your summary notes, they must be able to perfectly reconstruct the exact original Markmap without needing any outside knowledge. Do not over-summarize or drop specific terminology.

GLOBAL TONE & VOCABULARY CONSTRAINTS:
- Write in a clear, direct, and professional tone regardless of the domain.
- STRICTLY AVOID "ultra-sophisticated" academic jargon, flowery adjectives, or pretentious phrasing.
- Use simple, declarative sentences. Explain the concepts exactly as an experienced practitioner would to a capable peer on a whiteboard.

{selected_style}

ADAPTIVE OBSIDIAN FORMATTING REQUIREMENTS:
1. Critical Opening: Start by introducing the core intuition through an active question (answering the 'why' behind the concept rather than just defining it).
2. Organic Headers: DO NOT use generic template headers like "Mechanisms" or "Edge Cases". Invent 2-4 organic, subject-specific headers based on the actual technical domain.
3. Dynamic Mechanics & Interrogation:
   - IF the outline contains math/variables: Use standard LaTeX for inline math ($x$) and define variables clearly. Interrogate the formulas—explain *why* the variables relate the way they do instead of just listing them.
   - IF the outline is qualitative: Explain the structural logic and system dynamics in plain English, explicitly questioning the constraints (e.g., "Why is this the limit?").
4. Grounding: Do not mention 'Markmap', 'mindmap', or 'outline'. Write this strictly as a standalone technical document."""

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.7),
            )
            return response.text.strip()
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                switch_api_key()
                return generate_phase3_notes(markmap_code, retries)
            print(f"  [Phase 3 Error on attempt {attempt + 1}]: {e}")
            time.sleep(3)
    return None


# ---------------------------------------------------------------------------
# PIPELINE ORCHESTRATOR
# ---------------------------------------------------------------------------

def run_markmap_pipeline(
    input_path: str  = INPUT_TOPICS_PATH,
    output_file: str = OUTPUT_FILE,
) -> None:
    """
    Orchestrates the full Phase 2 → Phase 3 data-generation pipeline.

    Supports auto-resume: if ``output_file`` already contains N completed
    pairs, generation starts from index N.

    Args:
        input_path:  Path to the Phase-1 deduplicated topics JSON file (or dir).
        output_file: Path where the growing dataset is progressively saved.
    """
    topics = load_phase1_topics(input_path)
    if not topics:
        return

    test_topics = topics[:SAMPLE_SIZE]
    dataset: list = []
    start_index   = 0

    # --- AUTO-RESUME LOGIC ---
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                dataset = json.load(f)
                start_index = len(dataset)
                print(f"[!] Found existing dataset with {start_index} items. Resuming...")
        except json.JSONDecodeError:
            print("[!] Warning: Existing file could not be read. Starting from 0.")
            dataset = []

    if start_index >= len(test_topics):
        print("\nAll samples are already complete!")
        return

    print(f"--- Running Markmap Pipeline from index {start_index} to {len(test_topics)} ---")

    for i in range(start_index, len(test_topics)):
        item        = test_topics[i]
        topic_title = item.get("topic", "")
        print(f"\n[{i + 1}/{SAMPLE_SIZE}] Generating: {topic_title[:55]}...")

        # Step 1: Markmap Output (Phase 2)
        markmap_output = generate_phase2_markmap(topic_title)
        time.sleep(5.0)
        if not markmap_output:
            print("  -> Skipping due to Phase 2 failure.")
            continue

        is_valid, reason = validate_markmap_syntax(markmap_output)
        if not is_valid:
            print(f"  -> Markmap Validation Warning: {reason}")

        # Step 2: Obsidian Notes Output (Phase 3)
        obsidian_notes = generate_phase3_notes(markmap_output)
        time.sleep(5.0)
        if not obsidian_notes:
            print("  -> Skipping due to Phase 3 failure.")
            continue

        dataset.append({
            "topic":               topic_title,
            "target_markmap":      markmap_output,
            "input_summary_notes": obsidian_notes,
            "validation":          reason,
        })

        # Progressively save using the crash-proof atomic function
        save_dataset_safely(dataset, output_file)

    print(f"\nCompleted {len(dataset)} pairs saved to '{output_file}'")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # If you have a partial dataset from a previous run, set PREVIOUS_RUN_FILE
    # to its path so it is copied into OUTPUT_FILE before the pipeline starts.
    # Kaggle equivalent: shutil.copy(/kaggle/input/.../markmap_dataset_1500.json,
    #                                /kaggle/working/markmap_dataset_1500.json)
    if PREVIOUS_RUN_FILE and os.path.exists(PREVIOUS_RUN_FILE) and not os.path.exists(OUTPUT_FILE):
        print(f"Copying previous partial dataset from {PREVIOUS_RUN_FILE} ...")
        shutil.copy(PREVIOUS_RUN_FILE, OUTPUT_FILE)

    run_markmap_pipeline(INPUT_TOPICS_PATH, OUTPUT_FILE)
