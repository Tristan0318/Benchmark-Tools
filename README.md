# FraudBench — Evaluation Pipeline

This directory contains the inference and analysis scripts for the paper:

> **FraudBench: A Multimodal Benchmark for Detecting AI-Generated Fraudulent Refund Evidence**

---

## 1. Overview

The pipeline evaluates 30 multimodal large language models (MLLMs), 10 built into `detect.py` and 20 registered in `config/models_eval.json`, across six experiment conditions and two ablation studies.  All inference is performed via vendor APIs; no local GPU is required.

### 1.1 Experiment conditions

| Mode | Script | Flag(s) | API calls per review | Description |
|---|---|---|---|---|
| SingleImage-NoReview | `detect.py` | *(default)* | 1 per image | One image per call, no review text |
| SingleImage-withReview | `detect.py` | `--with-review` | 1 per image | One image per call, review injected |
| MultiImage-NoReview | `detect.py` | `--review-mode --single-turn` | **1** | All N images packed into one user message, one prompt at the end |
| MultiImage-withReview | `detect.py` | `--review-mode --single-turn --with-review` | **1** | Packed delivery with review injected |
| MultiStep-NoReview | `detect.py` | `--review-mode` | **1** | All N images in one user message, interleaved with "that was image i of N" continuation text |
| MultiStep-withReview | `detect.py` | `--review-mode --with-review` | **1** | Interleaved delivery with review injected |
| MultiTurn-NoReview | `detect_multiturn.py` | *(default)* | **N** | Genuine dialogue: one call per image, the model answers each one and its reply is fed back |
| MultiTurn-withReview | `detect_multiturn.py` | `--with-review` | **N** | Dialogue with image *i*'s customer comment injected at turn *i* |

> **MultiStep is not a conversation.** Its continuation prompts sit inside a
> *single* user message and the model never replies between images, so no
> dialogue state exists — it is one API call, exactly like MultiImage, and for
> the 34 % of review folders holding a single image the two prompts are
> byte-identical. `MultiTurn-*` is the genuinely multi-turn condition; use it
> whenever the claim is about sequential/interactive evidence submission.
>
> **Naming in the paper.** The paper's *single-image*, *multi-image* and
> *multi-step* settings correspond to `SingleImage-*`, `MultiImage-*` and
> `MultiTurn-*` here. `MultiStep-*` is the single-call interleaved variant
> described above and is not reported in the paper.

### 1.2 Ablation studies

| Study | Description |
|---|---|
| Prompt Sensitivity | Five prompt variants (Base / Merged / NoChk. / Gen. / Min.) on a 10 % stratified sample |
| Mismatch Review | Same images paired with a review from a **different** product category |

---

## 2. Requirements

```
python >= 3.10
pillow        # required for xAI HTTP-413 image downscale retry
pandas
openpyxl      # required for Excel export
flask         # required for the human evaluation interface
```

Install dependencies:

```bash
pip install pillow pandas openpyxl flask
```

---

## 3. API Keys

Set the following environment variables before running any script.  Models whose key is absent are automatically skipped.

```bash
export DASHSCOPE_API_KEY_1="..."   # qvq-max-latest, qwen3-vl-plus
export DASHSCOPE_API_KEY_2="..."   # qwen3.6-plus, kimi-k2.6, qwen3-vl-flash
export DASHSCOPE_API_KEY_3="..."   # qwen3.6-flash
export XAI_API_KEY="..."           # grok-4-1-fast-reasoning, grok-4.20-reasoning-latest
export GEMINI_API_KEY="..."        # gemini-3-flash
export OPENAI_API_KEY="..."        # gpt-5.4-mini
```

---

## 4. Script Reference

**`scripts/`** — Shell runners (entry points for reviewers)

| Script | Role |
|---|---|
| `scripts/run_detect.sh` | **Unified runner** for the six single-call conditions |
| `scripts/run_multiturn.sh` | Runner for the true multi-turn conditions (`MultiTurn-*`) |
| `scripts/run_ablation.sh` | **Unified runner** for both ablation studies (incl. index generation) |

**`tools/`** — Python inference and analysis scripts

| Script | Role |
|---|---|
| `tools/detect.py` | Unified detection script for the six single-call conditions |
| `tools/detect_multiturn.py` | True multi-turn detection — N sequential calls per review, replies fed back |
| `tools/model_registry.py` | Loads `config/models.json` so detectors can be added without editing `detect.py` |
| `tools/check_models.py` | Lists the model ids each provider currently serves, and probes configured detectors with one minimal call |
| `tools/detect_prompt_ablation.py` | Prompt ablation detection (reads from sample index) |
| `tools/detect_mismatch.py` | Mismatch-review detection (reads from mismatch index) |
| `tools/generate_sample_index.py` | Generates the stratified 10 % sample index for ablation |
| `tools/generate_mismatch_index.py` | Generates the cross-category mismatch index |
| `tools/compute_accuracy.py` | Computes macro-averaged metrics from main experiment results |
| `tools/compute_ablation_accuracy.py` | Computes macro-averaged metrics for prompt ablation variants |
| `tools/rebuild_summary.py` | Rebuilds `summary.json` from the per-model result files |
| `tools/read_results.py` | Loads all `summary.json` files into one table (used by the metric scripts) |

**`human_eval_interface/`** — Local web app for blind human evaluation

| File | Role |
|---|---|
| `human_eval_interface/app.py` | Flask server — serves images and records judgements |
| `human_eval_interface/catalog.py` | Scans dataset directories and builds the image registry |
| `human_eval_interface/sampler.py` | Balanced real/fake sampler with anti-recency logic |
| `human_eval_interface/store.py` | Atomic per-evaluator result storage with undo support |

---

## 5. Usage

### 5.1  Main Experiments

```bash
# SingleImage-NoReview (default)
bash scripts/run_detect.sh

# SingleImage-withReview
bash scripts/run_detect.sh --with-review

# MultiImage-NoReview
bash scripts/run_detect.sh --review-mode --single-turn

# MultiImage-withReview
bash scripts/run_detect.sh --review-mode --single-turn --with-review

# MultiStep-NoReview
bash scripts/run_detect.sh --review-mode

# MultiStep-withReview
bash scripts/run_detect.sh --review-mode --with-review

# Override concurrency (default 4)
bash scripts/run_detect.sh --concurrency 6
```

**MultiTurn** (true dialogue) has its own runner, because it costs N API calls
per review instead of 1:

```bash
# MultiTurn-NoReview   (skips 1-image folders by default: --min-images 2)
bash scripts/run_multiturn.sh

# MultiTurn-withReview
bash scripts/run_multiturn.sh --with-review

# Try one category and one model first — this condition is the expensive one
bash scripts/run_multiturn.sh --categories Electronics --models gpt-5.4-mini

# Also record a verdict after every turn (verdict trajectory)
bash scripts/run_multiturn.sh --running-verdict
```

Each `MultiTurn-*` record carries the full conversation: `num_turns`, and a
`turns[]` list with the prompt, the model's reply, latency, attempts and token
usage for every round trip.  All other fields match `detect.py`, so
`compute_accuracy.py` reads the results with no changes.

#### Confirming endpoints and model ids first

Vendor model ids move (a "GPT-6" or "Qwen 3.8" is served under an exact string
that is not guessable).  Resolve them from the API rather than from memory:

```bash
python tools/check_models.py --list                      # every provider
python tools/check_models.py --list --provider openai --grep gpt-6
python tools/check_models.py --list --provider dashscope --grep qwen3.8
```

Then fill the id into `config/models.json`, set `"disabled": false`, and verify
with one minimal real request per model (a 64x64 image, negligible cost):

```bash
python tools/check_models.py --probe --models-config config/models.json
```

A `FAIL` here is a wrong id, a wrong base URL, a model that refuses images, or a
key without access — all of which are much cheaper to find now than 60,000
calls into a sweep.

#### Adding a detector without touching `detect.py`

`config/models.json` is an editable detector registry.  Add an entry, then pass
it to either script:

```bash
bash scripts/run_detect.sh --review-mode --single-turn      # built-ins only
python tools/detect.py --models-config config/models.json --models new-model ...
bash scripts/run_multiturn.sh --models-config config/models.json
```

Omitting `--models-config` uses the 10 built-ins hard-coded in `detect.py`, so
the published runs stay reproducible.

Results are written to:

```
{category}/Results/{MODE}/summary.json
```

### 5.2  Ablation studies

```bash
# Run both ablation studies (default)
bash scripts/run_ablation.sh

# Prompt sensitivity only
bash scripts/run_ablation.sh --prompt

# Mismatch review only
bash scripts/run_ablation.sh --mismatch

# Force regeneration of indices
bash scripts/run_ablation.sh --regen-sample
bash scripts/run_ablation.sh --regen-mismatch

# Custom options
bash scripts/run_ablation.sh --concurrency 6 --sample-ratio 0.1 --sample-seed 42
```

Both studies are **resumable**: re-running skips already-completed `(image, model)` pairs.  Indices are generated automatically if they do not exist.

Results are written to:

```
PromptAblation/{variant}/{category}/summary.json
MismatchReview/{category}/summary.json
```

### 5.3  Human Evaluation Interface

A lightweight Flask web app for blind image-level human evaluation (Real vs. DeepFake).

```bash
# Start the server (default: dataset root inferred from script location)
python human_eval_interface/app.py

# Specify dataset root and/or custom results directory explicitly
python human_eval_interface/app.py \
    --root /path/to/dataset \
    --results-root human_eval_interface/Results

# Custom host/port (default: 127.0.0.1:5050)
python human_eval_interface/app.py --host 0.0.0.0 --port 8080
```

Open `http://127.0.0.1:5050/` in a browser.  Enter an evaluator name and select a category scope (a specific category or "All").  Images are served one at a time in a balanced real/fake order; press **F / ←** for Real, **J / →** for DeepFake, **U** to undo the last judgement.

Each session is **resumable**: restarting the server and re-entering the same name resumes from where it left off.

Results are written atomically to:

```
human_eval_interface/Results/{evaluator}/{scope}/
├── Negative.json
├── DeepFake/{gen_model}.json
├── summary.json          ← written on completion
└── _progress.json        ← incremental progress checkpoint
```

---

## 6. Output Structure

```
{category}/
└── Results/
    ├── SingleImage-NoReview/
    │   ├── Negative/<model>.json
    │   ├── DeepFake/<generator>/<model>.json
    │   └── summary.json               ← cross-model roll-up, one row per image
    ├── SingleImage-withReview/        ...
    ├── MultiStep-NoReview/            ...   (interleaved, 1 call)
    ├── MultiStep-withReview/          ...
    ├── MultiImage-NoReview/           ...   (packed, 1 call)
    ├── MultiImage-withReview/         ...
    ├── MultiTurn-NoReview/            ...   (true dialogue, N calls)
    └── MultiTurn-withReview/          ...

PromptAblation/
├── sample_index.json               ← fixed 10 % sample index, generated once
├── v1_baseline/{category}/summary.json
├── v2_merged_role/...
├── v3_no_artifacts/...
├── v4_generic_role/...
└── v5_minimal/...

MismatchReview/
├── mismatch_index.json             ← cross-category pairings index, generated once
├── {category}/summary.json
└── logs/{category}.log

human_eval_interface/Results/
└── {evaluator}/
    └── {scope}/
        ├── Negative.json
        ├── DeepFake/{gen_model}.json
        ├── summary.json            ← written on session completion
        └── _progress.json          ← incremental checkpoint (resumable)
```

Each MLLM `summary.json` contains a `rows` list with one entry per image/review,
each row holding a `verdicts` dict keyed by model name with fields:
`status`, `is_ai_modified`, `confidence`, `reason`, `error`.

---

## 7. Computing Metrics

```bash
# Main experiment results → accuracy_results.xlsx
python tools/compute_accuracy.py

# Prompt ablation results → ablation_results.xlsx
python tools/compute_ablation_accuracy.py
```

Both scripts compute **macro-averaged** metrics over product categories:
- **TNR** — True Negative Rate on real damaged images
- **TPR** — per-generator True Positive Rate on AI-modified images
- **F1** — binary F1 on the fake class (macro-averaged over categories)
- **Bal.Acc** — balanced accuracy = (TPR + TNR) / 2, pooled across generators
- **Conf.** — mean confidence score on **correctly classified** images only


---

## 8. Models

| Model | Provider | API key env var |
|---|---|---|
| `qvq-max-latest` | Alibaba DashScope | `DASHSCOPE_API_KEY_1` |
| `qwen3-vl-plus` | Alibaba DashScope | `DASHSCOPE_API_KEY_1` |
| `qwen3.6-plus` | Alibaba DashScope | `DASHSCOPE_API_KEY_2` |
| `kimi-k2.6` | Moonshot / DashScope (used) | `DASHSCOPE_API_KEY_2` |
| `qwen3-vl-flash` | Alibaba DashScope | `DASHSCOPE_API_KEY_2` |
| `qwen3.6-flash` | Alibaba DashScope | `DASHSCOPE_API_KEY_3` |
| `grok-4-1-fast-reasoning` | xAI | `XAI_API_KEY` |
| `grok-4.20-reasoning-latest` | xAI | `XAI_API_KEY` |
| `gemini-3-flash` | Google | `GEMINI_API_KEY` |
| `gpt-5.4-mini` | OpenAI | `OPENAI_API_KEY` |

These 10 are hard-coded in `detect.py`.  The other 20 MLLMs evaluated in the
paper are registered in `config/models_eval.json` and loaded with
`--models-config` rather than by editing `detect.py`, so the published runs
remain reproducible.

All reasoning/thinking parameters are left at vendor defaults (no overrides).
For xAI models, HTTP 413 (payload too large) triggers automatic image
downscaling through up to 10 progressive resolution tiers before the request
is abandoned.

---

## 9. Reproducibility Notes

- The stratified sample for ablation studies uses **ratio = 0.1, seed = 42**.
- The cross-category review assignment for mismatch review uses **seed = 42**.
- Passing `--regen-sample` or `--regen-mismatch` rebuilds indices from scratch; omitting these flags reuses existing indices so every run tests the exact same image set.
- All results are written atomically (temp-file + rename) and are safe to interrupt and resume.
