#!/usr/bin/env bash
# run_detect.sh — Unified runner for all six experiment modes.
#
# Combines run_SingleImage_NoReview.sh, run_SingleImage_withReview.sh,
# run_MultiImage_NoReview.sh, run_MultiImage_SingleTurn_NoReview.sh,
# run_MultiImage_withReview.sh, and run_MultiImage_SingleTurn_withReview.sh
# into a single script.  Flags are forwarded directly to detect.py.
#
# Usage
# -----
#   bash run_detect.sh                                       # SingleImage-NoReview
#   bash run_detect.sh --with-review                        # SingleImage-withReview
#   bash run_detect.sh --review-mode                        # MultiStep-NoReview
#   bash run_detect.sh --review-mode --single-turn          # MultiImage-NoReview
#   bash run_detect.sh --review-mode --with-review          # MultiStep-withReview
#   bash run_detect.sh --review-mode --single-turn --with-review  # MultiImage-withReview
#   bash run_detect.sh --multi-turn                         # MultiTurn-NoReview
#   bash run_detect.sh --multi-turn --with-review           # MultiTurn-withReview
#
# --multi-turn is the only mode that holds a real conversation, and it runs
# detect_multiturn.py rather than detect.py: one API call per image with the
# model's own replies fed back. The other five are all a single call each.
#
# Optional overrides
#   --concurrency N   API calls per model (default 4)
#   --python PATH     Python interpreter (default python3)
#   --categories A B  Run only these category folders instead of all 29

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(dirname "$SCRIPT_DIR")"
TOOLS_DIR="$(dirname "$EVAL_DIR")"
ROOT="$(dirname "$TOOLS_DIR")"
DETECT="$EVAL_DIR/tools/detect.py"
DETECT_MT="$EVAL_DIR/tools/detect_multiturn.py"
PYTHON="${PYTHON:-python3}"
CONCURRENCY=4

# ── Flag parsing ──────────────────────────────────────────────────────────────
REVIEW_MODE=false
SINGLE_TURN=false
MULTI_TURN=false
WITH_REVIEW=false
MIN_IMAGES=""
RUNNING_VERDICT=false
MODELS_CONFIG=""
MODELS=()
GENERATORS=()
ONLY_CATEGORIES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --review-mode)   REVIEW_MODE=true; shift ;;
        --single-turn)   SINGLE_TURN=true; shift ;;
        --multi-turn)    MULTI_TURN=true; shift ;;
        --running-verdict) RUNNING_VERDICT=true; shift ;;
        --min-images)    MIN_IMAGES="$2"; shift 2 ;;
        --with-review)   WITH_REVIEW=true; shift ;;
        --concurrency)   CONCURRENCY="$2"; shift 2 ;;
        --python)        PYTHON="$2"; shift 2 ;;
        --models-config) MODELS_CONFIG="$2"; shift 2 ;;
        --models)        shift; while [[ $# -gt 0 && "$1" != --* ]]; do MODELS+=("$1"); shift; done ;;
        --generators)    shift; while [[ $# -gt 0 && "$1" != --* ]]; do GENERATORS+=("$1"); shift; done ;;
        --categories)    shift; while [[ $# -gt 0 && "$1" != --* ]]; do ONLY_CATEGORIES+=("$1"); shift; done ;;
        *) echo "[error] unknown arg: $1"; exit 1 ;;
    esac
done

# Forwarded verbatim to detect.py. Without --models-config the run uses the 11
# detectors hard-coded in detect.py, which is what reproduces the paper.
PASS_ARGS=()
[[ -n "$MODELS_CONFIG" ]]    && PASS_ARGS+=(--models-config "$MODELS_CONFIG")
[[ ${#MODELS[@]} -gt 0 ]]     && PASS_ARGS+=(--models "${MODELS[@]}")
[[ ${#GENERATORS[@]} -gt 0 ]] && PASS_ARGS+=(--generators "${GENERATORS[@]}")

# ── Derive MODE name and extra flags ─────────────────────────────────────────
if $MULTI_TURN; then
    MODE_BASE="MultiTurn"
    EXTRA_FLAGS=""
    DETECT="$DETECT_MT"          # its own script; --review-mode is implicit there
    [[ -n "$MIN_IMAGES" ]] && EXTRA_FLAGS="$EXTRA_FLAGS --min-images $MIN_IMAGES"
    $RUNNING_VERDICT && EXTRA_FLAGS="$EXTRA_FLAGS --running-verdict"
elif $REVIEW_MODE; then
    MODE_BASE="MultiStep"
    EXTRA_FLAGS="--review-mode"
    if $SINGLE_TURN; then
        MODE_BASE="MultiImage"
        EXTRA_FLAGS="--review-mode --single-turn"
    fi
else
    MODE_BASE="SingleImage"
    EXTRA_FLAGS=""
fi

if $WITH_REVIEW; then
    MODE="${MODE_BASE}-withReview"
    EXTRA_FLAGS="$EXTRA_FLAGS --with-review"
else
    MODE="${MODE_BASE}-NoReview"
fi

# ── Categories ────────────────────────────────────────────────────────────────
CATEGORIES=(
    "All Beauty"
    "Amazon Fashion"
    "Appliances"
    "Arts, Crafts & Sewing"
    "Automotive"
    "Baby Products"
    "Beauty & Personal Care"
    "Books"
    "CDs & Vinyl"
    "Cell Phones & Accessories"
    "Clothing, Shoes & Jewelry"
    "Delivery, Pickup & Dine-Out"
    "Electronics"
    "Grocery & Gourmet Food"
    "Handmade Products"
    "Health & Household"
    "Health & Personal Care"
    "Home & Kitchen"
    "Hotels & Accommodations"
    "Industrial & Scientific"
    "Magazine Subscriptions"
    "Musical Instruments"
    "Office Products"
    "Patio, Lawn & Garden"
    "Pet Supplies"
    "Sports & Outdoors"
    "Tools & Home Improvement"
    "Toys & Games"
    "Video Games"
)

# A subset is validated against the list above so a typo fails loudly instead of
# silently running nothing.
if [[ ${#ONLY_CATEGORIES[@]} -gt 0 ]]; then
    SELECTED=()
    for want in "${ONLY_CATEGORIES[@]}"; do
        found=false
        for cat in "${CATEGORIES[@]}"; do
            if [[ "$cat" == "$want" ]]; then SELECTED+=("$cat"); found=true; break; fi
        done
        $found || { echo "[error] unknown category: $want"; exit 1; }
    done
    CATEGORIES=("${SELECTED[@]}")
fi

# ── Run ───────────────────────────────────────────────────────────────────────
total=${#CATEGORIES[@]}
echo "============================================================"
echo "  Mode        : $MODE"
echo "  Extra flags : ${EXTRA_FLAGS:-<none>}"
echo "  Concurrency : $CONCURRENCY"
echo "  Models cfg  : ${MODELS_CONFIG:-<detect.py built-ins>}"
echo "  Models      : ${MODELS[*]:-<all in registry>}"
echo "  Generators  : ${GENERATORS[*]:-<every folder under DeepFake/>}"
echo "  Categories  : $total"
echo "  Script      : $DETECT"
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

    # shellcheck disable=SC2086
    "$PYTHON" "$DETECT" \
        --input "Negative=$cat_dir/Negative" \
        --input "DeepFake=$cat_dir/DeepFake" \
        --output "$out_dir" \
        --concurrency "$CONCURRENCY" \
        $EXTRA_FLAGS "${PASS_ARGS[@]}" \
        || echo "[$(date '+%F %T')] [$MODE] [$i/$total] $cat FAILED, continuing"

    echo
done

echo "[$(date '+%F %T')] [$MODE] all $total categories done"
