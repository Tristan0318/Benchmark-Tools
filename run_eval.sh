#!/usr/bin/env bash
# run_eval.sh — SingleImage-NoReview on the original six-generator benchmark.
#
# --models is passed EXPLICITLY. The registry merges config/models_eval.json
# over detect.py's 10 built-ins, so omitting --models would also queue the
# paper-era built-in models — two of which (qvq-max-latest, grok-4-1-fast-reasoning) no longer exist
# at their vendor and would error on any image the April run happened to miss.
#
# Never add --no-resume: it overwrites records rather than skipping completed ones.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ~/.fraudbench_keys

CFG=config/models_eval.json
CONCURRENCY="${CONCURRENCY:-4}"

MODELS=(
    qwen3.8-max-0902
    qwen3.8-flash
    qwen3.8-27b
    qwen3.7-max-2026-06-08
    qwen3.7-plus-2026-05-26
    qwen3.7-flash-2026-07-15
    qwen3.5-omni-plus-2026-03-15
    qwen3.5-omni-flash-2026-03-15
    kimi-k3
    deepseek-v4.1-flash
    gpt-5.6-terra
    gpt-5.6-sol
    gpt-5.6-luna
    gemini-3.8-flash
    gemini-3.5-flash-lite
    grok-4.6
    gpt-6-astra
    gemini-3.1-pro-preview
)

GENERATORS=(
    gpt-image-2
    grok-imagine-image
    nano-banana-2
    qwen-image-2.0-pro
    qwen-image-edit-max
    wan2.7-image-pro
)

exec bash scripts/run_detect.sh \
    --models-config "$CFG" \
    --models "${MODELS[@]}" \
    --generators "${GENERATORS[@]}" \
    --concurrency "$CONCURRENCY" \
    "$@"
