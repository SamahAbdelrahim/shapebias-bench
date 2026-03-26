#!/bin/bash
set -euo pipefail

# ── Hardcoded configuration ──
MODELS="smolvlm qwen3-vl-4b" # space-separated model names to evaluate (use "all" for all registered models)
REPEATS=1 # number of passses through the data for each ordering
TEMPERATURE=0.0 # 0 is deterministic, higher values add more randomness
RESULTS_DIR="results" # directory to save results in
OUTPUT="${RESULTS_DIR}/local_eval.csv" # output file for results

# Fresh output file
rm -f "$OUTPUT"

# ── Run shape_first + texture_first (via --ordering both) ──
echo ""
echo "========================================================"
echo "  Run: ordering=both  models=$MODELS  repeats=$REPEATS  temp=$TEMPERATURE"
echo "========================================================"
echo ""
conda run --no-capture-output -n hackathon python scripts/run_local.py --models $MODELS --ordering both --repeats "$REPEATS" --temperature "$TEMPERATURE" -o "$OUTPUT"

# ── Run random ordering ──
echo ""
echo "========================================================"
echo "  Run: ordering=random  models=$MODELS  repeats=$REPEATS  temp=$TEMPERATURE"
echo "========================================================"
echo ""
conda run --no-capture-output -n hackathon python scripts/run_local.py --models $MODELS --ordering random --repeats "$REPEATS" --temperature "$TEMPERATURE" -o "$OUTPUT"

echo ""
echo "Done. Results: $OUTPUT"
