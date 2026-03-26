#!/bin/bash
set -euo pipefail

# ── Hardcoded configuration ──
MODEL="smolvlm" # the model to evaluate: options: smolvlm, openflamingo, mini_gpt4, or all models with "all"
REPEATS=1 # number of passses through the data for each ordering
TEMPERATURE=0.0 # 0 is deterministic, higher values add more randomness
RESULTS_DIR="results" # directory to save results in
OUTPUT="${RESULTS_DIR}/local_eval.csv" # output file for results

# Fresh output file
rm -f "$OUTPUT"

# ── Run all three orderings ──
for ORDERING in random shape_first texture_first; do
    echo ""
    echo "========================================================"
    echo "  Run: ordering=$ORDERING  model=$MODEL  repeats=$REPEATS  temp=$TEMPERATURE"
    echo "========================================================"
    echo ""
    conda run --no-capture-output -n hackathon python scripts/run_local.py --models "$MODEL" --ordering "$ORDERING" --repeats "$REPEATS" --temperature "$TEMPERATURE" -o "$OUTPUT"
done

echo ""
echo "Done. Results: $OUTPUT"
