#!/usr/bin/env python3
"""
check_models.py — confirm endpoints and model ids before a run.
================================================================

Two jobs, both of which need only the API keys in the environment:

  --list     Ask each provider which models it currently serves and print the
             ids.  This is how you pin down an exact `model_id` (e.g. the real
             id behind "GPT-6" or "Qwen 3.8") instead of guessing.

  --probe    Send ONE minimal real request per configured detector — a 64x64
             image plus a one-word prompt — and report OK / HTTP error /
             latency.  Cheap, and it catches a wrong id, a wrong base URL, a
             model that rejects images, or a key without access, before a
             62,000-call sweep does.

             CAUTION: passing --probe only proves the endpoint ACCEPTED a
             request carrying an image.  A text-only model served behind a
             gateway that silently drops the image part answers just as
             happily.  Use --vision-check for the question that matters.

  --vision-check [PATH]
             Send a REAL photo and ask what is in it.  A model that cannot see
             the image cannot answer, so this separates "accepts images" from
             "actually reads images" — the distinction that decides whether a
             detector belongs in the roster at all.  Answers are printed side
             by side; models that see the image agree, blind ones do not.

Usage
-----
    export OPENAI_API_KEY=...  XAI_API_KEY=...  GEMINI_API_KEY=...
    export DASHSCOPE_API_KEY_1=... DASHSCOPE_API_KEY_2=... DASHSCOPE_API_KEY_3=...

    # what ids exist right now
    python tools/check_models.py --list
    python tools/check_models.py --list --grep gpt-6
    python tools/check_models.py --list --provider xai

    # do the configured detectors actually work?
    python tools/check_models.py --probe
    python tools/check_models.py --probe --models-config config/models.json

    # do they actually SEE the image?
    python tools/check_models.py --vision-check --models-config config/models.json
    python tools/check_models.py --vision-check /path/to/photo.jpg

Exit code is non-zero if any probe fails.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detect as D  # noqa: E402
from model_registry import DEFAULT_CONFIG, load_models  # noqa: E402

TIMEOUT = 180   # thinking-only models (QVQ) can take minutes on a cold call

def _targets(provider: str) -> list[tuple[str, str, str, str]]:
    """(label, list-URL, key env var, reply shape) pairs to query for a provider.

    DashScope is listed once PER KEY: Bailian issues workspace endpoints where
    each key is bound to its own host, so one global URL would only ever show
    one workspace's catalogue.
    """
    if provider == "dashscope":
        return [(f"dashscope[key{n}]", f"{D.dashscope_base(n)}/models",
                 f"DASHSCOPE_API_KEY_{n}", "openai") for n in (1, 2, 3)]
    if provider == "openai":
        return [("openai", f"{D.OPENAI_BASE}/models", "OPENAI_API_KEY", "openai")]
    if provider == "xai":
        return [("xai", f"{D.XAI_BASE}/models", "XAI_API_KEY", "openai")]
    if provider == "gemini":
        return [("gemini", f"{D.GEMINI_BASE}/models", "GEMINI_API_KEY", "gemini")]
    return []


PROVIDERS = ("dashscope", "gemini", "openai", "xai")

# 64x64 solid-grey JPEG, base64 — small enough that a probe costs almost nothing.
_TINY_JPEG_B64 = (
    "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0a"
    "HBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIy"
    "MjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCABAAEADASIA"
    "AhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQA"
    "AAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3"
    "ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWm"
    "p6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEA"
    "AwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSEx"
    "BhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElK"
    "U1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3"
    "uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD3+iii"
    "gD//2Q=="
)


def _default_vision_image() -> Path | None:
    """A real product photo from the dataset, used by --probe and --vision-check."""
    root = Path(__file__).resolve().parent.parent.parent.parent
    first = root / "All Beauty" / "Positive" / "Review_003" / "Image_003_01.jpg"
    if first.exists():
        return first
    for cat in sorted(p for p in root.iterdir() if p.is_dir()):
        pos = cat / "Positive"
        if not pos.is_dir():
            continue
        for rev in sorted(p for p in pos.iterdir() if p.is_dir()):
            for f in sorted(rev.iterdir()):
                if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    return f
    return None


def _get(url: str, headers: dict) -> dict:
    req = urllib.request.Request(url, method="GET", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as e:
        raise D.HttpError(e.code, e.read().decode("utf-8", errors="replace")) from None


def list_target(label: str, url: str, key_env: str, shape: str, grep: str | None) -> int:
    key = os.getenv(key_env)
    if not key:
        print(f"[{label}] SKIP — {key_env} is not set")
        return 0
    print(f"[{label}] {url}")

    try:
        if shape == "gemini":
            data = _get(f"{url}?key={urllib.parse.quote(key)}&pageSize=200", {})
            ids = [m.get("name", "").removeprefix("models/") for m in data.get("models", [])]
        else:
            data = _get(url, {"Authorization": f"Bearer {key}"})
            ids = [m.get("id", "") for m in data.get("data", [])]
    except D.HttpError as e:
        print(f"    ERROR {e.status}: {e.body[:300]}")
        return 0
    except Exception as e:
        print(f"    ERROR {type(e).__name__}: {e}")
        return 0

    ids = sorted(i for i in ids if i)
    if grep:
        ids = [i for i in ids if grep.lower() in i.lower()]
    print(f"    {len(ids)} model(s)" + (f" matching {grep!r}" if grep else ""))
    for i in ids:
        print(f"      {i}")
    return len(ids)


VISION_PROMPT = (
    "Look at the image. Reply with ONLY two lowercase words separated by a "
    "space: the main object, then its dominant colour. No punctuation, no "
    "explanation."
)


def probe_model(cfg: D.ModelCfg, image_b64: str | None = None,
                prompt: str | None = None) -> dict:
    """One minimal real request.

    Default: a 64x64 grey JPEG + "reply ok" — checks the endpoint, id and key.
    With `image_b64`/`prompt`: a real photo and a question about its content —
    checks that the model actually receives and reads the pixels.
    """
    t0 = time.time()
    try:
        key = D._provider_key(cfg)
    except EnvironmentError as e:
        return {"cfg": cfg, "ok": False, "ms": 0, "err": str(e)}

    # Default to a REAL dataset photo, not the synthetic 64x64 grey JPEG.
    # That tiny image produced errors in both directions: GLM-5.3-Flash rejects
    # it outright ("image input format / parsing error") even though it handles real photos
    # fine, and text-only models sail through it because a gateway that drops
    # the image still returns 200.  A real photo plus a question about its
    # content avoids both.
    if image_b64 is None:
        img = _default_vision_image()
        if img is not None:
            image_b64 = base64.b64encode(img.read_bytes()).decode("ascii")
            prompt = prompt or VISION_Q
        else:
            image_b64 = _TINY_JPEG_B64
    prompt = prompt or "Reply with the single word: ok"
    try:
        if cfg.provider == "gemini":
            payload = {
                "contents": [{"role": "user", "parts": [
                    {"inlineData": {"mimeType": "image/jpeg", "data": image_b64}},
                    {"text": prompt}]}],
            }
            url = (f"{D.GEMINI_BASE}/models/{cfg.model_id}:generateContent"
                   f"?key={urllib.parse.quote(key)}")
            _, result = D._http_post_json(url, payload, {"Content-Type": "application/json"},
                                          stream=False, timeout=TIMEOUT)
            parts = (result["candidates"][0].get("content") or {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)
            api_model = result.get("modelVersion")
        else:
            payload = {
                "model": cfg.model_id,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    {"type": "text", "text": prompt}]}],
                "stream": cfg.stream,
            }
            if cfg.stream:
                payload["stream_options"] = {"include_usage": True}
            if cfg.extra_body:
                payload.update(cfg.extra_body)
            url = f"{cfg.endpoint.rstrip('/')}/chat/completions"
            status, result = D._http_post_json(
                url, payload,
                {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                stream=cfg.stream, timeout=TIMEOUT)
            if cfg.stream:
                text, _, api_model = D._consume_sse(result)
            else:
                text = result["choices"][0]["message"].get("content") or ""
                api_model = result.get("model")
        ms = int((time.time() - t0) * 1000)
        return {"cfg": cfg, "ok": True, "ms": ms, "err": None,
                "api_model": api_model, "reply": (text or "").strip()[:80]}
    except D.HttpError as e:
        return {"cfg": cfg, "ok": False, "ms": int((time.time() - t0) * 1000),
                "err": f"HTTP {e.status}: {e.body[:200]}"}
    except Exception as e:
        return {"cfg": cfg, "ok": False, "ms": int((time.time() - t0) * 1000),
                "err": f"{type(e).__name__}: {e}"}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="List the model ids each provider serves.")
    p.add_argument("--probe", action="store_true", help="Send one minimal real request per detector.")
    p.add_argument("--vision-check", nargs="?", const="", default=None, metavar="IMAGE",
                   help="Send a REAL photo and ask what is in it, proving the model "
                        "actually reads pixels rather than merely accepting the request. "
                        "Optional path; defaults to a dataset photo.")
    p.add_argument("--provider", choices=PROVIDERS, default=None,
                   help="--list: restrict to one provider.")
    p.add_argument("--grep", default=None, help="--list: substring filter on the model id.")
    p.add_argument("--models-config", default=None,
                   help=f"--probe: detector registry to test (e.g. {DEFAULT_CONFIG}).")
    p.add_argument("--models", nargs="+", default=None, help="--probe: subset of detector names.")
    p.add_argument("--timeout", type=int, default=TIMEOUT,
                   help=f"--probe: per-request timeout in seconds (default {TIMEOUT}).")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    globals()["TIMEOUT"] = args.timeout

    if not args.list and not args.probe and args.vision_check is None:
        p.print_help()
        return 1

    if args.list:
        providers = [args.provider] if args.provider else list(PROVIDERS)
        for prov in providers:
            for label, url, key_env, shape in _targets(prov):
                list_target(label, url, key_env, shape, args.grep)
                print()

    rc = 0
    if args.vision_check is not None:
        return _run_vision_check(args)

    if False:
        img = Path(args.vision_check) if args.vision_check else _default_vision_image()
        if img is None or not img.exists():
            print(f"[error] vision-check image not found: {img}", file=sys.stderr)
            return 2
        image_b64 = base64.b64encode(img.read_bytes()).decode("ascii")
        registry = load_models(args.models_config)
        by_name = {m.name: m for m in registry}
        selected = ([by_name[n] for n in args.models] if args.models
                    else registry)
        print(f"Vision check on {img}")
        print(f"Prompt: {VISION_PROMPT}\n")
        rows = []
        with ThreadPoolExecutor(max_workers=min(6, len(selected))) as pool:
            futures = [pool.submit(probe_model, c, image_b64, VISION_PROMPT)
                       for c in selected]
            for fut in as_completed(futures):
                r = fut.result()
                rows.append(r)
        for r in sorted(rows, key=lambda x: (not x["ok"], x["cfg"].name)):
            if r["ok"]:
                print(f"  {r['cfg'].name:<30} {r['ms']:>6}ms  -> {r['reply']!r}")
            else:
                rc = 1
                print(f"  {r['cfg'].name:<30} {r['ms']:>6}ms  FAIL {r['err'][:110]}")
        print("\nAnswers that agree on the object are seeing the image; an answer that "
              "is generic, refuses, or contradicts the others is a model that is NOT "
              "reading the pixels — exclude it from the roster.")

    if args.probe:
        registry = load_models(args.models_config)
        by_name = {m.name: m for m in registry}
        if args.models:
            unknown = [n for n in args.models if n not in by_name]
            if unknown:
                print(f"[error] unknown model(s): {', '.join(unknown)}", file=sys.stderr)
                return 2
            selected = [by_name[n] for n in args.models]
        else:
            selected = registry

        print(f"Probing {len(selected)} detector(s) — one tiny image each\n")
        with ThreadPoolExecutor(max_workers=min(8, len(selected))) as pool:
            futures = [pool.submit(probe_model, c) for c in selected]
            for fut in as_completed(futures):
                r = fut.result()
                cfg = r["cfg"]
                if r["ok"]:
                    resolved = r.get("api_model")
                    drift = ""
                    if resolved and resolved != cfg.model_id:
                        drift = f"   [alias resolves to: {resolved}]"
                    print(f"  OK   {cfg.name:<28} {r['ms']:>6}ms  {cfg.model_id}  "
                          f"-> {r.get('reply', '')!r}{drift}")
                else:
                    rc = 1
                    print(f"  FAIL {cfg.name:<28} {r['ms']:>6}ms  {cfg.model_id}\n"
                          f"       {r['err']}")
    return rc




# ──────────────────────────────────────────────────────────────────────────────
# Two-image differential vision check
# ──────────────────────────────────────────────────────────────────────────────
# A single image proves nothing: a text-only model served behind a gateway that
# silently DROPS the image part still answers the question, plausibly and
# confidently.  Verified on this deployment — qwen3-8b and deepseek-r1, both
# text-only, replied "ok" to an image-bearing request.
#
# So send TWO visually unrelated photos and ask the same question about each.
# A model that reads pixels gives two different, image-appropriate answers.
# A blind model cannot: it repeats itself or guesses, and its two answers do not
# track the images.
VISION_PAIR = [
    ("A", "All Beauty/Positive/Review_003/Image_003_01.jpg",
     {"palette", "paint", "makeup", "colour", "color", "brush", "cake", "pan", "cosmetic"}),
    ("B", "Amazon Fashion/Positive/Review_003/Image_003_01.jpg",
     {"sunglass", "glasses", "eyewear", "shade", "spectacle", "lens"}),
]

VISION_Q = ("Look at this photo and reply with ONLY two or three lowercase words "
            "naming the single main object. No punctuation, no explanation.")


def _run_vision_check(args) -> int:
    root = Path(__file__).resolve().parent.parent.parent.parent
    images = []
    for tag, rel, keys in VISION_PAIR:
        f = Path(args.vision_check) if args.vision_check else root / rel
        if not f.exists():
            print(f"[error] vision image missing: {f}", file=sys.stderr)
            return 2
        images.append((tag, base64.b64encode(f.read_bytes()).decode("ascii"), keys))
        if args.vision_check:
            break

    registry = load_models(args.models_config)
    by_name = {m.name: m for m in registry}
    if args.models:
        unknown = [n for n in args.models if n not in by_name]
        if unknown:
            print(f"[error] unknown model(s): {', '.join(unknown)}", file=sys.stderr)
            return 2
        selected = [by_name[n] for n in args.models]
    else:
        selected = registry

    print(f"Differential vision check — {len(selected)} model(s), "
          f"{len(images)} image(s) each\n")

    def one(cfg):
        out = {"cfg": cfg, "answers": {}, "errors": {}}
        for tag, b64, _ in images:
            r = probe_model(cfg, b64, VISION_Q)
            if r["ok"]:
                out["answers"][tag] = (r.get("reply") or "").lower().strip()
            else:
                out["errors"][tag] = r["err"]
        return out

    rows = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for fut in as_completed([pool.submit(one, c) for c in selected]):
            rows.append(fut.result())

    sees, blind, failed = [], [], []
    for r in rows:
        name = r["cfg"].name
        if r["errors"]:
            failed.append((name, list(r["errors"].values())[0]))
            continue
        a = r["answers"].get("A", "")
        b = r["answers"].get("B", "")
        hit_a = any(k in a for _, _, keys in [images[0]] for k in keys)
        hit_b = len(images) > 1 and any(k in b for k in images[1][2])
        differs = a != b
        if hit_a and (hit_b or len(images) == 1) and differs:
            sees.append((name, a, b))
        else:
            blind.append((name, a, b, hit_a, hit_b, differs))

    print(f"── SEES THE IMAGE ({len(sees)}) ──")
    for n, a, b in sorted(sees):
        print(f"  {n:<32} A={a[:28]!r}  B={b[:28]!r}")
    print(f"\n── DOES NOT READ PIXELS ({len(blind)}) ──")
    for n, a, b, ha, hb, d in sorted(blind):
        why = []
        if not d:
            why.append("identical answers")
        if not ha:
            why.append("A wrong")
        if not hb:
            why.append("B wrong")
        print(f"  {n:<32} A={a[:24]!r} B={b[:24]!r}  ({', '.join(why)})")
    if failed:
        print(f"\n── REQUEST FAILED ({len(failed)}) ──")
        for n, e in sorted(failed):
            print(f"  {n:<32} {e[:100]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
