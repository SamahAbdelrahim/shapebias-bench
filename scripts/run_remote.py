#!/usr/bin/env python3
"""Shape-bias evaluation for remote (API-based) VLMs.

Usage:
    # Run a single model
    python scripts/run_remote.py --models qwen3.5-9b

    # Run multiple models
    python scripts/run_remote.py --models qwen3.5-9b llama4-scout

    # Run all remote models
    python scripts/run_remote.py --models all

    # Limit stimuli count
    python scripts/run_remote.py --models qwen3.5-9b --num-stimuli 5

    # Specify output
    python scripts/run_remote.py --models qwen3.5-9b -o results/my_run.csv
"""

from __future__ import annotations

import argparse
import base64
import os
import random
import sys
import time
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation_pipe.eval_core import (
    ENV_PATH,
    MAX_TOKENS_REMOTE,
    TEMPERATURE,
    add_common_args,
    load_stimuli,
    load_words,
    print_summary,
    resolve_output_path,
    run_trial,
    write_results,
)

load_dotenv(ENV_PATH)

# ===========================================================================
# REMOTE MODEL REGISTRY
# ===========================================================================
REMOTE_MODELS = {
    "qwen3.5-9b":       {"provider": "huggingface",      "model_id": "Qwen/Qwen3.5-9B"},
    "qwen3.5-27b":      {"provider": "huggingface",      "model_id": "Qwen/Qwen3.5-27B"},
    "qwen3.5-35b-a3b":  {"provider": "huggingface",      "model_id": "Qwen/Qwen3.5-35B-A3B"},
    "qwen3.5-122b-a10b":{"provider": "huggingface",      "model_id": "Qwen/Qwen3.5-122B-A10B"},
    "llama4-scout":     {"provider": "huggingface-groq",  "model_id": "meta-llama/llama-4-scout-17b-16e-instruct"},
    # "llama4-maverick":{"provider": "huggingface-sambanova", "model_id": "Llama-4-Maverick-17B-128E-Instruct"},
}

PROVIDER_BASE_URLS = {
    "huggingface":           "https://router.huggingface.co/v1",
    "huggingface-groq":      "https://router.huggingface.co/groq/openai/v1",
    "huggingface-sambanova": "https://router.huggingface.co/sambanova/v1",
}


# ---------------------------------------------------------------------------
# Remote inference helpers
# ---------------------------------------------------------------------------
def image_to_base64_url(img: Image.Image, fmt: str = "PNG") -> str:
    buf = BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{b64}"


def build_messages(images: list[Image.Image], prompt: str) -> list[dict]:
    content = []
    for img in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": image_to_base64_url(img)},
        })
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def run_remote(model_name: str, images: list[Image.Image], prompt: str) -> dict:
    from openai import OpenAI

    cfg = REMOTE_MODELS[model_name]
    base_url = PROVIDER_BASE_URLS[cfg["provider"]]
    hf_token = os.environ.get("HUGGING_FACE") or os.environ.get("HF_API_TOKEN")

    client = OpenAI(api_key=hf_token, base_url=base_url)
    messages = build_messages(images, prompt)

    # Disable thinking mode for Qwen3.5 models to avoid wasting tokens.
    # Also bump max_tokens as a safety net — if thinking isn't fully
    # disabled by the provider, the thinking tokens eat into the budget.
    extra = {}
    max_tok = MAX_TOKENS_REMOTE
    if "qwen" in cfg["model_id"].lower():
        extra["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        max_tok = max(max_tok, 8192)

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=cfg["model_id"],
        messages=messages,
        max_tokens=max_tok,
        temperature=TEMPERATURE,
        **extra,
    )
    elapsed = time.perf_counter() - start

    choice = response.choices[0]
    raw_text = (choice.message.content or "").strip()
    tokens = response.usage.completion_tokens if response.usage else None

    return {
        "raw_text": raw_text,
        "generation_time_s": round(elapsed, 2),
        "model_name": cfg["model_id"],
        "num_tokens_generated": tokens,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run shape-bias evaluation (remote API models)")
    parser.add_argument("--models", nargs="+", required=True,
                        help="Model names to evaluate. Use 'all' for all remote models.")
    add_common_args(parser)
    args = parser.parse_args()

    random.seed(args.seed)

    # Resolve model list
    model_names = []
    for m in args.models:
        if m == "all":
            model_names.extend(REMOTE_MODELS.keys())
        else:
            if m not in REMOTE_MODELS:
                print(f"Error: unknown remote model '{m}'. Available: {list(REMOTE_MODELS.keys())}")
                sys.exit(1)
            model_names.append(m)

    # Load stimuli and words
    words = load_words()
    stimuli = load_stimuli(args.stim_set, args.num_stimuli)
    print(f"Models:  {model_names}")
    print(f"Stimuli: {len(stimuli)} from {args.stim_set}")
    print(f"Words:   {len(words)} ({len(words)//2} sudo + {len(words)//2} random)")
    print(f"Trials per model: {len(stimuli)} x {len(words)} x 2 orderings = {len(stimuli) * len(words) * 2}")
    print()

    output_path = resolve_output_path(args.output, prefix="remote")
    all_results = []

    for model_key in model_names:
        print(f"{'='*60}")
        print(f"Remote model: {model_key}")
        print(f"{'='*60}")

        def run_fn(images, prompt, _mk=model_key):
            return run_remote(_mk, images, prompt)

        for stim in stimuli:
            for w in words:
                word, word_type, word_length = w["name"], w["type"], w["length"]
                print(f"  Stimulus {stim['stim_id']:>3s} (word={word}, type={word_type}, len={word_length})")
                trial_results = run_trial(run_fn, stim, word, word_type, word_length)
                for r in trial_results:
                    r["model"] = model_key
                    print(f"    {r['ordering']:15s} -> {r['raw_text']!r:10s}  choice={r['choice']}")
                    all_results.append(r)

    write_results(all_results, output_path)
    print_summary(all_results, model_names)


if __name__ == "__main__":
    main()
