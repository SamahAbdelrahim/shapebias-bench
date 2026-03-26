#!/usr/bin/env python3
"""Shape-bias evaluation for local (GPU-based) VLMs.

Usage:
    # Run a single local model
    python scripts/run_local.py --models smolvlm --device cuda

    # Run multiple local models
    python scripts/run_local.py --models smolvlm internvl qwen3.5-0.8b

    # Run all registered local models
    python scripts/run_local.py --models all

    # Limit stimuli count
    python scripts/run_local.py --models smolvlm --num-stimuli 5

    # Specify output
    python scripts/run_local.py --models smolvlm -o results/local_run.csv
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation_pipe.eval_core import (
    DEFAULT_DEVICE,
    ENV_PATH,
    MAX_TOKENS_LOCAL,
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


# ---------------------------------------------------------------------------
# Local inference
# ---------------------------------------------------------------------------
def run_local(model, images: list[Image.Image], prompt: str) -> dict:
    from evaluation_pipe.models.base import ModelResponse
    resp: ModelResponse = model.generate(
        images=images, prompt=prompt,
        max_new_tokens=MAX_TOKENS_LOCAL, temperature=TEMPERATURE,
    )
    return {
        "raw_text": resp.raw_text,
        "generation_time_s": round(resp.generation_time_s, 2),
        "model_name": resp.model_name,
        "num_tokens_generated": resp.num_tokens_generated,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run shape-bias evaluation (local GPU models)")
    parser.add_argument("--models", nargs="+", required=True,
                        help="Model names to evaluate. Use 'all' for all registered local models.")
    parser.add_argument("--device", default=DEFAULT_DEVICE,
                        help=f"Device for local models (default: {DEFAULT_DEVICE})")
    add_common_args(parser)
    args = parser.parse_args()

    random.seed(args.seed)

    from evaluation_pipe.models import create_model, list_models

    # Resolve model list
    available = list_models()
    model_names = []
    for m in args.models:
        if m == "all":
            model_names.extend(available)
        else:
            if m not in available:
                print(f"Error: unknown local model '{m}'. Available: {available}")
                sys.exit(1)
            model_names.append(m)

    # Load stimuli and words
    words = load_words()
    stimuli = load_stimuli(args.stim_set, args.num_stimuli)
    print(f"Models:  {model_names}")
    print(f"Device:  {args.device}")
    print(f"Stimuli: {len(stimuli)} from {args.stim_set}")
    print(f"Words:   {len(words)} ({len(words)//2} sudo + {len(words)//2} random)")
    print(f"Trials per model: {len(stimuli)} x {len(words)} x 2 orderings = {len(stimuli) * len(words) * 2}")
    print()

    output_path = resolve_output_path(args.output, prefix="local")
    all_results = []

    for model_key in model_names:
        print(f"{'='*60}")
        print(f"Local model: {model_key}")
        print(f"{'='*60}")

        model = create_model(model_key, device=args.device)
        print(f"  Loaded: {model.name}")

        def run_fn(images, prompt, _m=model):
            return run_local(_m, images, prompt)

        for stim in stimuli:
            for w in words:
                word, word_type, word_length = w["name"], w["type"], w["length"]
                print(f"  Stimulus {stim['stim_id']:>3s} (word={word}, type={word_type}, len={word_length})")
                trial_results = run_trial(run_fn, stim, word, word_type, word_length)
                for r in trial_results:
                    r["model"] = model_key
                    print(f"    {r['ordering']:15s} -> {r['raw_text']!r:10s}  choice={r['choice']}")
                    all_results.append(r)

        model.unload()
        print(f"  Unloaded {model_key}")

    write_results(all_results, output_path)
    print_summary(all_results, model_names)


if __name__ == "__main__":
    main()
