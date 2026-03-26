"""Shared evaluation logic for shape-bias benchmarking.

This module contains all shared constants, helpers, and trial-running logic
used by both `scripts/run_remote.py` and `scripts/run_local.py`.
"""

from __future__ import annotations

import csv
import random
import time
from pathlib import Path

from PIL import Image

# ===========================================================================
# PATHS
# ===========================================================================
REPO_ROOT = Path(__file__).resolve().parent.parent
STIMULI_DIR = REPO_ROOT / "stimuli_pipe" / "stimuli_per_stl_packages"
RESULTS_DIR = REPO_ROOT / "results"
ENV_PATH = REPO_ROOT / ".env"

# ===========================================================================
# HYPERPARAMETERS
# ===========================================================================
MAX_RETRIES = 5               # retries per trial if answer is ambiguous/empty
MAX_TOKENS_REMOTE = 128       # max tokens for remote API calls (thinking disabled)
MAX_TOKENS_LOCAL = 128        # max tokens for local model generation
TEMPERATURE = 0.0             # sampling temperature (0.0 = greedy/deterministic)
RANDOM_SEED = 42              # default random seed for reproducibility
DEFAULT_STIM_SET = "stimuli_A_auto_contrast"
DEFAULT_DEVICE = "cuda"

# ===========================================================================
# WORD PAIRS — 5 pseudo (sudo) + 5 random, length-matched
# ===========================================================================
WORD_PAIRS = [
    # (sudo_word,    random_word,   length)
    ("shiple",       "afnafq",      6),
    ("clapher",      "ieyiccw",     7),
    ("plailass",     "orvufaig",    8),
    ("procation",    "qahftrxck",   9),
    ("adinefults",   "cgchqjjfgy",  10),
]

# ===========================================================================
# PROMPT TEMPLATE
# ===========================================================================
PROMPT_TEMPLATE = (
    "The first image is a {word}. "
    "Which of the following two images (A or B) is also a {word}? "
    "Answer with just 'A' or 'B'."
)

# ===========================================================================
# CSV OUTPUT FIELDS
# ===========================================================================
CSV_FIELDS = [
    "model", "model_name", "stim_id", "word", "word_type", "word_length",
    "ordering", "a_is", "b_is", "raw_text", "parsed_answer", "choice",
    "generation_time_s", "num_tokens_generated", "attempts",
]


# ---------------------------------------------------------------------------
# Word list
# ---------------------------------------------------------------------------
def load_words() -> list[dict]:
    """Load the curated length-matched word pairs as a flat list."""
    words = []
    for sudo, rand, length in WORD_PAIRS:
        words.append({"name": sudo,  "type": "sudo",   "length": length})
        words.append({"name": rand,  "type": "random", "length": length})
    return words


def make_prompt(word: str) -> str:
    return PROMPT_TEMPLATE.format(word=word)


# ---------------------------------------------------------------------------
# Stimuli loading
# ---------------------------------------------------------------------------
def load_stimuli(stim_set: str = DEFAULT_STIM_SET,
                 num_stimuli: int | None = None) -> list[dict]:
    stim_base = STIMULI_DIR / stim_set
    stim_dirs = sorted([d for d in stim_base.iterdir() if d.is_dir()],
                       key=lambda d: int(d.name))

    if num_stimuli is not None:
        stim_dirs = random.sample(stim_dirs, min(num_stimuli, len(stim_dirs)))

    stimuli = []
    for d in stim_dirs:
        stimuli.append({
            "stim_id": d.name,
            "reference": Image.open(d / "reference.png").convert("RGB"),
            "shape_match": Image.open(d / "shape_match.png").convert("RGB"),
            "texture_match": Image.open(d / "texture_match.png").convert("RGB"),
        })
    return stimuli


# ---------------------------------------------------------------------------
# Answer parsing with retry
# ---------------------------------------------------------------------------
def parse_answer(raw_text: str) -> str | None:
    upper = raw_text.upper()
    has_a = "A" in upper
    has_b = "B" in upper
    if has_a and has_b:
        return None
    if has_a:
        return "A"
    if has_b:
        return "B"
    return None


def run_with_retry(run_fn, images: list[Image.Image], prompt: str) -> dict:
    for attempt in range(1, MAX_RETRIES + 1):
        result = run_fn(images, prompt)
        answer = parse_answer(result["raw_text"])
        if answer is not None:
            result["parsed_answer"] = answer
            result["attempts"] = attempt
            return result
        print(f"    [retry {attempt}/{MAX_RETRIES}] ambiguous/empty: {result['raw_text']!r}")

    result["parsed_answer"] = None
    result["attempts"] = MAX_RETRIES
    return result


# ---------------------------------------------------------------------------
# Single trial (both orderings)
# ---------------------------------------------------------------------------
def run_trial(run_fn, stimulus: dict, word: str, word_type: str,
              word_length: int = 0) -> list[dict]:
    """Run one stimulus in both orderings. Returns two result dicts."""
    ref = stimulus["reference"]
    shape = stimulus["shape_match"]
    texture = stimulus["texture_match"]
    prompt = make_prompt(word)
    results = []

    # Ordering 1: A=shape, B=texture
    res = run_with_retry(run_fn, [ref, shape, texture], prompt)
    answer = res.get("parsed_answer")
    if answer == "A":
        choice = "shape"
    elif answer == "B":
        choice = "texture"
    else:
        choice = "unclear"
    results.append({
        **res,
        "stim_id": stimulus["stim_id"],
        "word": word,
        "word_type": word_type,
        "word_length": word_length,
        "ordering": "shape_first",
        "a_is": "shape",
        "b_is": "texture",
        "choice": choice,
    })

    # Ordering 2: A=texture, B=shape
    res = run_with_retry(run_fn, [ref, texture, shape], prompt)
    answer = res.get("parsed_answer")
    if answer == "A":
        choice = "texture"
    elif answer == "B":
        choice = "shape"
    else:
        choice = "unclear"
    results.append({
        **res,
        "stim_id": stimulus["stim_id"],
        "word": word,
        "word_type": word_type,
        "word_length": word_length,
        "ordering": "texture_first",
        "a_is": "texture",
        "b_is": "shape",
        "choice": choice,
    })

    return results


# ---------------------------------------------------------------------------
# Shared CLI argument setup
# ---------------------------------------------------------------------------
def add_common_args(parser) -> None:
    """Add arguments shared by both local and remote scripts."""
    parser.add_argument("--stim-set", default=DEFAULT_STIM_SET,
                        help=f"Stimulus set directory name (default: {DEFAULT_STIM_SET})")
    parser.add_argument("--num-stimuli", type=int, default=None,
                        help="Number of stimuli to sample (default: all)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output CSV path (default: results/<timestamp>.csv)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help=f"Random seed (default: {RANDOM_SEED})")


# ---------------------------------------------------------------------------
# CSV writing and summary
# ---------------------------------------------------------------------------
def write_results(all_results: list[dict], output_path: Path) -> None:
    """Write results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nResults saved to {output_path}")


def print_summary(all_results: list[dict], model_names: list[str]) -> None:
    """Print shape-bias summary tables."""
    print(f"\n{'='*70}")
    print(f"{'SUMMARY':^70}")
    print(f"{'='*70}")
    print(f"  {'Model':25s}  {'Shape':>6s}  {'Texture':>8s}  {'Unclear':>8s}  {'Shape %':>8s}")
    print(f"  {'-'*25}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}")

    for model_key in model_names:
        model_res = [r for r in all_results if r["model"] == model_key]
        s = sum(1 for r in model_res if r["choice"] == "shape")
        t = sum(1 for r in model_res if r["choice"] == "texture")
        u = sum(1 for r in model_res if r["choice"] == "unclear")
        pct = f"{s / (s + t) * 100:.0f}%" if (s + t) > 0 else "N/A"
        print(f"  {model_key:25s}  {s:>6d}  {t:>8d}  {u:>8d}  {pct:>8s}")

    # Breakdown by ordering
    print(f"\n  {'Model':25s}  {'Ordering':15s}  {'Shape':>6s}  {'Texture':>8s}  {'Shape %':>8s}")
    print(f"  {'-'*25}  {'-'*15}  {'-'*6}  {'-'*8}  {'-'*8}")
    for model_key in model_names:
        for ordering in ["shape_first", "texture_first"]:
            subset = [r for r in all_results
                      if r["model"] == model_key and r["ordering"] == ordering]
            s = sum(1 for r in subset if r["choice"] == "shape")
            t = sum(1 for r in subset if r["choice"] == "texture")
            pct = f"{s / (s + t) * 100:.0f}%" if (s + t) > 0 else "N/A"
            print(f"  {model_key:25s}  {ordering:15s}  {s:>6d}  {t:>8d}  {pct:>8s}")


def resolve_output_path(output_arg: str | None, prefix: str = "eval") -> Path:
    """Resolve output CSV path from CLI arg or generate timestamped default."""
    if output_arg is not None:
        return Path(output_arg)
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return RESULTS_DIR / f"{prefix}_{timestamp}.csv"
