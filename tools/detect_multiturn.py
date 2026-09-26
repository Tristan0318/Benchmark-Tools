#!/usr/bin/env python3
"""
detect_multiturn.py — TRUE multi-turn (conversational) DeepFake detection.
==========================================================================

Why this script exists
----------------------
detect.py's "multi-step" mode (`--review-mode` without `--single-turn`) is NOT
a multi-turn conversation.  It packs every image AND every continuation prompt
into ONE user message and issues ONE API call:

    [system] [user: img1, "that was image 1 of N, do not answer yet",
                    img2, "that was image 2 of N, do not answer yet",
                    ..., imgN, FINAL_PROMPT]

The model never replies between images, so no dialogue state ever exists.  For
single-image reviews (34% of the dataset) the prompt is byte-identical to the
single-turn multi-image mode.

This script instead simulates the real-world interaction: a customer opens a
refund claim and uploads evidence photos one message at a time, and the model
answers after EACH image.  An N-image review becomes N sequential API calls
inside one conversation, with the model's own replies fed back as `assistant`
turns:

    call 1: [system][user: img1 + turn prompt]                  -> reply 1
    call 2: [system][user: img1 ...][assistant: reply 1]
                    [user: img2 + turn prompt]                  -> reply 2
    ...
    call N: full history + [user: imgN + final JSON prompt]     -> verdict

Modes
-----
    --with-review        inject the customer's review comment for image i at
                         turn i (the message it would have arrived with)
    --running-verdict    ask for a JSON verdict at EVERY turn, not just the
                         last one, so the verdict trajectory can be analysed
    --min-images N       skip review folders with fewer than N images
                         (use 2 to skip the single-image groups, whose result
                         is by definition identical to the single-image run)

Output
------
Identical schema to detect.py (same ResultStore, so compute_accuracy.py works
unchanged), plus per-record:

    num_turns          number of API round trips actually made
    turns[]            {index, image_rel, prompt, response, reasoning_content,
                        latency_ms, attempts, usage, verdict}
    verdict_trajectory list of per-turn verdicts (--running-verdict only)

Usage
-----
    export DASHSCOPE_API_KEY_1=...   # etc, see detect.py
    python tools/detect_multiturn.py \
        --input "Negative=../Negative" \
        --input "DeepFake=../DeepFake" \
        --output ./Results/MultiTurn-NoReview \
        --min-images 2 --concurrency 3
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import traceback
import urllib.error
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import detect as D
from model_registry import DEFAULT_CONFIG, load_models

# ──────────────────────────────────────────────────────────────────────────────
# Conversational prompts
# ──────────────────────────────────────────────────────────────────────────────
MT_SYSTEM = D.SYSTEM_PROMPT + (
    " You are reviewing a customer refund claim in which the customer uploads "
    "their evidence photos one message at a time. Answer each message as it "
    "arrives, and carry your earlier observations forward."
)

MT_TURN_FIRST = (
    "A customer has opened a refund claim for this product and is uploading "
    "their evidence photos one at a time. This is image {i} of {n}. "
    "Give a short forensic observation (2-3 sentences) of anything relevant to "
    "whether this image was produced or modified by AI — either through AI "
    "image-editing LLMs or AI image-generation LLMs. "
    "Do not commit to a final verdict yet; more images from this claim are coming."
)

MT_TURN_MIDDLE = (
    "Here is image {i} of {n} from the same refund claim. "
    "Taking your earlier observations into account, give a short forensic "
    "observation (2-3 sentences) of this image and say whether it changes your "
    "reading of the set so far. "
    "Do not commit to a final verdict yet; more images from this claim are coming."
)

MT_FINAL_HEAD = (
    "Here is image {n} of {n} — the final image from this refund claim. "
    "Considering this image together with every image you have seen and every "
    "observation you made earlier in this conversation, decide whether these {n} "
    "images have been produced or modified by AI — either through AI "
    "image-editing LLMs or through AI image-generation LLMs — or whether they "
    "are unmodified genuine camera photographs of real scenes."
)

# Reuse detect.py's schema block verbatim so the parsed fields match exactly.
MT_FINAL_TAIL = D._REVIEW_FINAL_TAIL

MT_RUNNING_TAIL = (
    "Then return your running assessment as a single JSON object on the last "
    "line — no markdown fences — with this exact schema:\n"
    "{\n"
    '  "is_ai_modified": <true or false>,   // your verdict given everything seen so far\n'
    '  "confidence":     <number between 0 and 1>,\n'
    '  "reason":         "<1-2 sentences>"\n'
    "}"
)


def _turn_prompt(idx: int, n: int, *, running_verdict: bool) -> str:
    if idx == n:
        return (D._render_review_prompt(MT_FINAL_HEAD, n=n) + "\n\n"
                + D._render_review_prompt(MT_FINAL_TAIL, n=n))
    template = MT_TURN_FIRST if idx == 1 else MT_TURN_MIDDLE
    text = D._render_review_prompt(template, i=idx, n=n)
    if running_verdict:
        text += "\n\n" + MT_RUNNING_TAIL
    return text


def _single_image_prompt(comment: str | None) -> str:
    """n == 1: identical to detect.py's single-image prompt (comparability)."""
    mid = D._comment_block(comment) if comment else "\n\n"
    return D._USER_HEAD + mid + D._USER_TAIL


# ──────────────────────────────────────────────────────────────────────────────
# Streaming helper that also captures usage
# ──────────────────────────────────────────────────────────────────────────────
def _consume_sse(resp) -> tuple[str, str, dict | None, str | None]:
    content, reasoning, usage, api_model = [], [], None, None
    try:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            if api_model is None and chunk.get("model"):
                api_model = chunk["model"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            if delta.get("content"):
                content.append(delta["content"])
            if delta.get("reasoning_content"):
                reasoning.append(delta["reasoning_content"])
    finally:
        resp.close()
    return "".join(content), "".join(reasoning), usage, api_model


# ──────────────────────────────────────────────────────────────────────────────
# One API round trip, given the conversation so far
# ──────────────────────────────────────────────────────────────────────────────
def _openai_turn(cfg: D.ModelCfg, image_paths: list[Path], prompts: list[str],
                 replies: list[str], *, downscale_level: int, final: bool) -> dict:
    """Send turns 1..len(prompts); replies holds the len(prompts)-1 prior answers."""
    merge_system = cfg.name.startswith(("qwen", "qvq"))
    messages: list[dict] = []
    if not merge_system:
        messages.append({"role": "system", "content": MT_SYSTEM})

    for i, (path, prompt) in enumerate(zip(image_paths, prompts)):
        data_uri, _ = D._encode_image(path, downscale_level)
        parts: list[dict] = [{"type": "image_url", "image_url": {"url": data_uri}}]
        parts.append({"type": "text", "text": prompt})
        if merge_system and i == 0:
            parts.insert(0, {"type": "text", "text": MT_SYSTEM})
        messages.append({"role": "user", "content": parts})
        if i < len(replies):
            messages.append({"role": "assistant", "content": replies[i]})

    payload: dict[str, Any] = {
        "model": cfg.model_id,
        "messages": messages,
        "stream": cfg.stream,
    }
    if cfg.stream:
        payload["stream_options"] = {"include_usage": True}
    if cfg.extra_body:
        payload.update(cfg.extra_body)

    url = f"{cfg.endpoint.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {D._provider_key(cfg)}",
               "Content-Type": "application/json"}
    status, result = D._http_post_json(url, payload, headers, stream=cfg.stream)

    if cfg.stream:
        content, reasoning, usage, api_model = _consume_sse(result)
        return {"content": content, "reasoning_content": reasoning or None,
                "usage": usage, "api_model": api_model}

    try:
        msg = result["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as e:
        raise D.HttpError(status, json.dumps(result)[:500]) from e
    return {"content": msg.get("content") or "",
            "reasoning_content": msg.get("reasoning_content") or None,
            "usage": result.get("usage"),
            "api_model": result.get("model")}


def _gemini_turn(cfg: D.ModelCfg, image_paths: list[Path], prompts: list[str],
                 replies: list[str], *, downscale_level: int, final: bool) -> dict:
    contents: list[dict] = []
    for i, (path, prompt) in enumerate(zip(image_paths, prompts)):
        if downscale_level > 0:
            data_uri, mime = D._encode_image(path, downscale_level)
            b64 = data_uri.split(",", 1)[1]
        else:
            mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
            b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        contents.append({"role": "user", "parts": [
            {"inlineData": {"mimeType": mime, "data": b64}},
            {"text": prompt},
        ]})
        if i < len(replies):
            contents.append({"role": "model", "parts": [{"text": replies[i]}]})

    gen_cfg: dict[str, Any] = {}
    if final:                                   # free text on intermediate turns
        gen_cfg["responseMimeType"] = "application/json"
    if cfg.gemini_thinking_level:
        gen_cfg["thinkingConfig"] = {"thinkingLevel": cfg.gemini_thinking_level}

    payload = {
        "systemInstruction": {"parts": [{"text": MT_SYSTEM}]},
        "contents": contents,
        "generationConfig": gen_cfg,
    }
    url = (f"{D.GEMINI_BASE}/models/{cfg.model_id}:generateContent"
           f"?key={urllib.parse.quote(D._provider_key(cfg))}")
    status, result = D._http_post_json(url, payload, {"Content-Type": "application/json"},
                                       stream=False)
    try:
        cand = result["candidates"][0]
    except (KeyError, IndexError, TypeError) as e:
        raise D.HttpError(status, json.dumps(result)[:500]) from e

    content, reasoning = [], []
    for p in (cand.get("content") or {}).get("parts", []):
        if "text" in p:
            (reasoning if p.get("thought") else content).append(p["text"])
    return {"content": "".join(content),
            "reasoning_content": "".join(reasoning) or None,
            "usage": result.get("usageMetadata"),
            "api_model": result.get("modelVersion")}


def _turn_with_retry(cfg: D.ModelCfg, image_paths: list[Path], prompts: list[str],
                     replies: list[str], *, final: bool,
                     downscale_level: int) -> tuple[dict | None, int, str | None, int]:
    """Returns (result, attempts, error, downscale_level_used)."""
    fn = _gemini_turn if cfg.provider == "gemini" else _openai_turn
    # Same escalation policy as detect.py's call_with_retry, reusing its
    # predicates rather than restating them: xAI's 413, OpenAI's 400 whose
    # message names the total image size, and DashScope's silent connection
    # reset are all the one condition "too much image data".
    max_downscale = len(D.DOWNSCALE_TIERS)
    total_budget  = D.MAX_RETRIES + max_downscale
    attempt = backoff_n = conn_drops = 0
    last_err: str | None = None

    while attempt < total_budget:
        attempt += 1
        try:
            res = fn(cfg, image_paths, prompts, replies,
                     downscale_level=downscale_level, final=final)
            return res, attempt, None, downscale_level
        except D.HttpError as e:
            last_err = f"HTTP {e.status}: {e.body[:300]}"
            if D._is_payload_too_large(e) and downscale_level < max_downscale:
                if not D._HAS_PIL:
                    return None, attempt, (f"{last_err}; install Pillow to enable "
                                           f"downscale retry"), downscale_level
                downscale_level += 1
                continue
            if D._is_transient_upstream_400(e):
                # DashScope's gateway giving up on fetching the data URIs. It
                # bites hardest here: every turn resends all images shown so
                # far, so a long conversation is exactly the shape that trips
                # it, and without this a review dies mid-dialogue.
                if D._HAS_PIL and downscale_level < max_downscale:
                    downscale_level += 1
                backoff_n += 1
            elif 400 <= e.status < 500 and e.status != 429:
                return None, attempt, last_err, downscale_level
            else:
                backoff_n += 1
        except urllib.error.URLError as e:
            last_err = f"URLError: {e}"
            if (D._is_connection_dropped(e)
                    and D._raw_payload_bytes(image_paths) >= D.CONN_DROP_SIZE_FLOOR):
                conn_drops += 1
                if conn_drops >= 2 and D._HAS_PIL and downscale_level < max_downscale:
                    downscale_level += 1
                    conn_drops = 0
                    continue
            backoff_n += 1
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            backoff_n += 1
        if attempt < total_budget:
            time.sleep(D.RETRY_BASE * (2 ** (backoff_n - 1)))
    return None, attempt, last_err, downscale_level


# ──────────────────────────────────────────────────────────────────────────────
# Whole conversation for one review folder
# ──────────────────────────────────────────────────────────────────────────────
def run_conversation(cfg: D.ModelCfg, job: D.ImageJob, *,
                     running_verdict: bool, with_review: bool) -> dict:
    record = D._new_record(job)
    n = job.num_images
    record["num_turns"] = 0
    record["turns"] = []

    prompts: list[str] = []
    replies: list[str] = []
    seen_comments: set[str] = set()
    downscale_level = 0
    total_attempts = 0
    t_conv = time.time()
    last_content = ""

    for idx in range(1, n + 1):
        if n == 1:
            prompt = _single_image_prompt(job.reviewer_comments[0] if with_review else None)
        else:
            prompt = _turn_prompt(idx, n, running_verdict=running_verdict)
            if with_review:
                comment = (job.reviewer_comments[idx - 1]
                           if idx - 1 < len(job.reviewer_comments) else None)
                # Each distinct comment is shown once, on the turn it first
                # applies. The dataset stores ONE review body per review and
                # repeats it per image, so in practice it lands on turn 1 --
                # matching both the real interaction (the customer states the
                # complaint when opening the claim, then uploads photos) and
                # _comment_block_combined, which collapses an all-identical list
                # to a single block for the single-call conditions. Re-stating it
                # every turn would leave the conversation carrying the same text
                # n times and confound multi-turn with comment repetition.
                if comment and comment not in seen_comments:
                    seen_comments.add(comment)
                    prompt += D._comment_block(comment).rstrip()

        prompts.append(prompt)
        final = (idx == n)

        t0 = time.time()
        res, attempts, err, downscale_level = _turn_with_retry(
            cfg, job.image_paths[:idx], prompts, replies,
            final=final, downscale_level=downscale_level,
        )
        latency = int((time.time() - t0) * 1000)
        total_attempts += attempts

        turn_rec: dict[str, Any] = {
            "index": idx,
            "image_rel": job.image_rels[idx - 1],
            "prompt": prompt,
            "latency_ms": latency,
            "attempts": attempts,
            "error": err,
            "response": None,
            "reasoning_content": None,
            "usage": None,
            "api_model": None,
            "verdict": None,
        }

        if res is None:
            turn_rec["error"] = err or "unknown error"
            record["turns"].append(turn_rec)
            record["num_turns"] = idx
            record["attempts"] = total_attempts
            record["latency_ms"] = int((time.time() - t_conv) * 1000)
            record["error"] = f"turn {idx}/{n} failed: {err}"
            return record

        content = res.get("content") or ""
        turn_rec["response"] = content
        turn_rec["reasoning_content"] = res.get("reasoning_content")
        turn_rec["usage"] = res.get("usage")
        turn_rec["api_model"] = res.get("api_model")
        if running_verdict and not final:
            parsed = D.parse_json_reply(content)
            if parsed:
                turn_rec["verdict"] = {
                    "is_ai_modified": parsed.get("is_ai_modified"),
                    "confidence": parsed.get("confidence"),
                    "reason": parsed.get("reason"),
                }

        record["turns"].append(turn_rec)
        replies.append(content)
        last_content = content

    record["num_turns"] = n
    record["attempts"] = total_attempts
    record["latency_ms"] = int((time.time() - t_conv) * 1000)
    record["raw_response"] = last_content
    record["reasoning_content"] = record["turns"][-1]["reasoning_content"]
    record["api_model"] = record["turns"][-1]["api_model"]
    if running_verdict:
        record["verdict_trajectory"] = [t["verdict"] for t in record["turns"][:-1]]

    parsed = D.parse_json_reply(last_content)
    if parsed is None:
        record["error"] = "failed to parse JSON from final turn"
        return record

    is_ai = parsed.get("is_ai_modified")
    if is_ai is None:
        is_ai = parsed.get("is_ai_generated")
    if isinstance(is_ai, str):
        is_ai = is_ai.strip().lower() in ("true", "yes", "1", "ai", "ai-modified", "ai-generated")
    record["is_ai_modified"] = bool(is_ai) if is_ai is not None else None

    conf = parsed.get("confidence")
    try:
        record["confidence"] = float(conf) if conf is not None else None
    except (TypeError, ValueError):
        record["confidence"] = None

    reason = parsed.get("reason") or parsed.get("reasoning") or parsed.get("explanation")
    if isinstance(reason, str):
        record["reason"] = reason.strip()
    elif isinstance(reason, list):
        record["reason"] = " ".join(
            str(x).strip() for x in reason if x is not None and str(x).strip()
        ) or None
    elif isinstance(reason, dict):
        record["reason"] = json.dumps(reason, ensure_ascii=False)

    per_img = (parsed.get("per_image") or parsed.get("per_image_analysis")
               or parsed.get("images"))
    if isinstance(per_img, list):
        record["per_image_analysis"] = per_img

    return record


# ──────────────────────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────────────────────
def run(jobs, models, output_dir, concurrency, resume, *,
        running_verdict: bool, with_review: bool) -> None:
    store = D.ResultStore(output_dir, models, jobs)
    tasks, skipped = [], 0
    for job in jobs:
        for cfg in models:
            if resume and store.already_done(job.bucket, cfg.name, job.portable_key):
                skipped += 1
                continue
            tasks.append((cfg, job))

    total = len(tasks)
    n_turns = sum(j.num_images for _, j in tasks)
    print(f"[plan] mode=true-multi-turn | {len(jobs)} reviews × {len(models)} models "
          f"= {len(jobs) * len(models)} conversations "
          f"(skipped {skipped} already-done, {total} to run, "
          f"~{n_turns} API round trips); concurrency = {concurrency * len(models)}",
          flush=True)
    if total == 0:
        store.flush_summary(jobs)
        print("[done] nothing to do.", flush=True)
        return

    pool = ThreadPoolExecutor(max_workers=max(1, concurrency * len(models)))
    futures = {pool.submit(run_conversation, cfg, job,
                           running_verdict=running_verdict,
                           with_review=with_review): (cfg, job)
               for cfg, job in tasks}

    done = ok = fail = 0
    t_start = time.time()
    try:
        for fut in as_completed(futures):
            cfg, job = futures[fut]
            try:
                record = fut.result()
            except Exception as e:
                record = D._new_record(job)
                record["error"] = f"worker crashed: {type(e).__name__}: {e}"
                traceback.print_exc()

            store.record(job.bucket, cfg.name, record)
            done += 1
            if record["error"] is None and record["is_ai_modified"] is not None:
                ok += 1
            else:
                fail += 1

            elapsed = time.time() - t_start
            rate = done / elapsed if elapsed else 0
            eta = (total - done) / rate if rate else 0
            flag = "?" if record["error"] else ("MOD" if record["is_ai_modified"] else "real")
            print(f"[{done:>{len(str(total))}}/{total}] ok={ok} fail={fail} "
                  f"eta={eta:>5.0f}s | {cfg.name:<28} "
                  f"{record.get('num_turns', 0)}turn "
                  f"{record['latency_ms']:>6}ms [{flag}] {job.display_rel}"
                  + (f" — {record['error'][:80]}" if record["error"] else ""),
                  flush=True)
            if done % 25 == 0:
                store.flush_summary(jobs)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        store.flush_summary(jobs)

    print(f"[done] {done} conversations in {time.time() - t_start:.1f}s "
          f"(ok={ok}, fail={fail}). Summary: {output_dir / 'summary.json'}", flush=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", action="append", required=True, help="LABEL=PATH or PATH. Repeatable.")
    p.add_argument("--output", required=True, help="Root directory for results.")
    p.add_argument("--models", nargs="+", default=None, help="Subset of detector names (default: all).")
    p.add_argument("--models-config", default=None,
                   help=f"JSON detector registry to merge in (e.g. {DEFAULT_CONFIG}).")
    p.add_argument("--concurrency", type=int, default=3, help="Max in-flight conversations per model.")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--with-review", action="store_true",
                   help="Inject image i's customer comment at turn i.")
    p.add_argument("--running-verdict", action="store_true",
                   help="Ask for a JSON verdict at every turn (verdict trajectory).")
    p.add_argument("--min-images", type=int, default=1,
                   help="Skip review folders with fewer than N images (use 2 to skip "
                        "single-image groups, which cannot be multi-turn).")
    p.add_argument("--generators", nargs="+", default=None,
                   help="Restrict DeepFake buckets to these generator folder names. "
                        "DeepFake/ now holds both the original six generators and the "
                        "ones added later, so evaluating the original benchmark needs "
                        "this filter. Default: every subfolder found.")
    p.add_argument("--limit", type=int, default=0, help="Process only the first N reviews.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    inputs = [D.parse_input_arg(s) for s in args.input]
    jobs = D.walk_inputs(inputs, review_mode=True, with_review=args.with_review)

    if args.generators:
        wanted = set(args.generators)
        found  = {j.generator for j in jobs if j.generator}
        unknown = wanted - found
        if unknown:
            print(f"[warn] --generators named folder(s) not present under DeepFake/: "
                  f"{', '.join(sorted(unknown))}", file=sys.stderr)
            print(f"       present: {', '.join(sorted(found))}", file=sys.stderr)
        before = len(jobs)
        jobs = [j for j in jobs if j.generator is None or j.generator in wanted]
        kept = {j.generator for j in jobs if j.generator}
        print(f"[info] --generators: kept {len(jobs)}/{before} reviews across "
              f"{len(kept)} generator(s): {', '.join(sorted(kept))}")

    if args.min_images > 1:
        before = len(jobs)
        jobs = [j for j in jobs if j.num_images >= args.min_images]
        print(f"[info] --min-images {args.min_images}: kept {len(jobs)}/{before} reviews.")
    if args.limit > 0:
        jobs = jobs[:args.limit]
    if not jobs:
        print("[error] no reviews found.", file=sys.stderr)
        return 1

    registry = load_models(args.models_config)
    by_name = {m.name: m for m in registry}
    if args.models:
        unknown = [n for n in args.models if n not in by_name]
        if unknown:
            print(f"[error] unknown model(s): {', '.join(unknown)}\n"
                  f"        available: {', '.join(sorted(by_name))}", file=sys.stderr)
            return 2
        selected = [by_name[n] for n in args.models]
    else:
        selected = registry

    missing = sorted({m.key_env for m in selected if not os.getenv(m.key_env)})
    if missing:
        print(f"[error] missing env var(s): {', '.join(missing)}", file=sys.stderr)
        return 2

    # Same coverage report detect.py prints: a bucket whose metadata names the
    # review body differently silently yields no comment at all, and the count
    # is the only place that shows up before the run is over.
    if args.with_review:
        no_comment   = [j for j in jobs if not any(c for c in j.reviewer_comments if c)]
        with_comment = len(jobs) - len(no_comment)
        print(f"[info] {len(jobs)} jobs total, {with_comment} have reviewer comment(s).",
              flush=True)
        if no_comment:
            print(f"[warn] {len(no_comment)} job(s) have no reviewer comment — "
                  f"will run without review context:", file=sys.stderr)
            for j in no_comment[:10]:
                print(f"  [{j.bucket}] {j.display_rel}", file=sys.stderr)
            if len(no_comment) > 10:
                print(f"  ... ({len(no_comment) - 10} more)", file=sys.stderr)

    run(jobs=jobs, models=selected,
        output_dir=Path(args.output).expanduser().resolve(),
        concurrency=max(1, args.concurrency), resume=not args.no_resume,
        running_verdict=args.running_verdict, with_review=args.with_review)
    return 0


if __name__ == "__main__":
    sys.exit(main())
