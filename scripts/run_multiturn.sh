#!/usr/bin/env bash
# run_multiturn.sh — runner for the TRUE multi-turn condition (MultiTurn-*).
#
# Unlike run_detect.sh, which issues one API call per review folder, this runs
# detect_multiturn.py: one API call PER IMAGE inside a single conversation,
# with the model's own replies fed back as assistant turns.
#
# Usage
# -----
#   bash run_multiturn.sh                      # MultiTurn-NoReview
#   bash run_multiturn.sh --with-review        # MultiTurn-withReview
#
# Optional overrides
#   --concurrency N     conversations in flight per model (default 3)
#   --min-images N      skip review folders with < N images (default 2 — the
#                       single-image folders cannot be multi-turn and their
#                       result is already covered by SingleImage-*)
#   --running-verdict   also record a verdict after every turn
#   --models  A B C     restrict to a subset of detectors
#   --models-config P   JSON detector registry. Omit to use the detectors
#                       hard-coded in detect.py (same default as run_detect.sh).
#   --categories A B    restrict to a subset of categories
#   --python PATH       interpreter (default python3)
#
# Cost warning: an N-image review costs N API calls with a conversation that
# re-uploads every previous image each turn.  Run one category first.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(dirname "$SCRIPT_DIR")"
TOOLS_DIR="$(dirname "$EVAL_DIR")"
ROOT="$(dirname "$TOOLS_DIR")"
DETECT="$EVAL_DIR/tools/detect_multiturn.py"
PYTHON="${PYTHON:-python3}"

CONCURRENCY=3
MIN_IMAGES=2
WITH_REVIEW=false
RUNNING_VERDICT=false
MODELS=()
MODELS_CONFIG=""
CATEGORIES=()
GENERATORS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-review)     WITH_REVIEW=true; shift ;;
        --running-verdict) RUNNING_VERDICT=true; shift ;;
        --concurrency)     CONCURRENCY="$2"; shift 2 ;;
        --min-images)      MIN_IMAGES="$2"; shift 2 ;;
        --models-config)   MODELS_CONFIG="$2"; shift 2 ;;
        --python)          PYTHON="$2"; shift 2 ;;
        --models)          shift; while [[ $# -gt 0 && "$1" != --* ]]; do MODELS+=("$1"); shift; done ;;
        --categories)      shift; while [[ $# -gt 0 && "$1" != --* ]]; do CATEGORIES+=("$1"); shift; done ;;
        --generators)      shift; while [[ $# -gt 0 && "$1" != --* ]]; do GENERATORS+=("$1"); shift; done ;;
        *) echo "[error] unknown arg: $1"; exit 1 ;;
    esac
done

# No silent fallback to a default registry: without --models-config the run uses
# the detectors hard-coded in detect.py, exactly like run_detect.sh. Quietly
# substituting config/models.json here would mean forgetting the flag scores a
# different roster than intended, with nothing in the output to show it.

EXTRA_FLAGS=()
if $WITH_REVIEW; then
    MODE="MultiTurn-withReview"
    EXTRA_FLAGS+=(--with-review)
else
    MODE="MultiTurn-NoReview"
fi
$RUNNING_VERDICT && EXTRA_FLAGS+=(--running-verdict)
[[ -n "$MODELS_CONFIG" ]] && EXTRA_FLAGS+=(--models-config "$MODELS_CONFIG")
[[ ${#MODELS[@]} -gt 0 ]] && EXTRA_FLAGS+=(--models "${MODELS[@]}")
[[ ${#GENERATORS[@]} -gt 0 ]] && EXTRA_FLAGS+=(--generators "${GENERATORS[@]}")

if [[ ${#CATEGORIES[@]} -eq 0 ]]; then
    CATEGORIES=(
        "All Beauty" "Amazon Fashion" "Appliances" "Arts, Crafts & Sewing"
        "Automotive" "Baby Products" "Beauty & Personal Care" "Books"
        "CDs & Vinyl" "Cell Phones & Accessories" "Clothing, Shoes & Jewelry"
        "Delivery, Pickup & Dine-Out" "Electronics" "Grocery & Gourmet Food"
        "Handmade Products" "Health & Household" "Health & Personal Care"
        "Home & Kitchen" "Hotels & Accommodations" "Industrial & Scientific"
        "Magazine Subscriptions" "Musical Instruments" "Office Products"
        "Patio, Lawn & Garden" "Pet Supplies" "Sports & Outdoors"
        "Tools & Home Improvement" "Toys & Games" "Video Games"
    )
fi

total=${#CATEGORIES[@]}
echo "============================================================"
echo "  Mode         : $MODE  (true multi-turn — N calls per review)"
echo "  Min images   : $MIN_IMAGES"
echo "  Concurrency  : $CONCURRENCY"
echo "  Models cfg   : ${MODELS_CONFIG:-<detect.py built-ins>}"
echo "  Models       : ${MODELS[*]:-<all>}"
echo "  Generators   : ${GENERATORS[*]:-<every folder under DeepFake/>}"
echo "  Categories   : $total"
echo "  Script       : $DETECT"
echo "============================================================"
echo

i=0
for cat in "${CATEGORIES[@]}"; do
    i=$((i+1))
    cat_dir="$ROOT/$cat"
    out_dir="$cat_dir/Results/$MODE"

    if [[ ! -d "$cat_dir/DeepFake" || ! -d "$cat_dir/Negative" ]]; then
        echo "[$(date '+%F %T')] [$MODE] [$i/$total] $cat SKIP (missing DeepFake/ or Negative/)"
        echo
        continue
    fi

    echo "------------------------------------------------------------"
    echo "[$(date '+%F %T')] [$MODE] [$i/$total] $cat"
    echo "  out : $out_dir"
    echo "------------------------------------------------------------"

    "$PYTHON" "$DETECT" \
        --input "Negative=$cat_dir/Negative" \
        --input "DeepFake=$cat_dir/DeepFake" \
        --output "$out_dir" \
        --concurrency "$CONCURRENCY" \
        --min-images "$MIN_IMAGES" \
        "${EXTRA_FLAGS[@]}" \
        || echo "[$(date '+%F %T')] [$MODE] [$i/$total] $cat FAILED, continuing"

    echo
done

echo "[$(date '+%F %T')] [$MODE] all $total categories done"
