#!/usr/bin/env python3
"""
detect.py — Unified DeepFake detection script.
=========================================================
Covers all six experiment variants:

  --review-mode               group images by review folder (multi-image)
  --single-turn               pack all images into one user message
                              (MultiImage); omit for multi-step (MultiStep)
  --with-review               inject the product review text into the prompt
                              (withReview experiments); omit for NoReview

Six experiment modes:
  (1) SingleImage-NoReview    (default, no flags)
  (2) SingleImage-withReview  --with-review
  (3) MultiStep-NoReview      --review-mode
  (4) MultiImage-NoReview     --review-mode --single-turn
  (5) MultiStep-withReview    --review-mode --with-review
  (6) MultiImage-withReview   --review-mode --single-turn --with-review

Models
------
  Alibaba Bailian / DashScope (OpenAI-compatible, 3 keys, 7 models total)
    KEY 1 (heaviest):
      * qvq-max-latest                 (thinking-only VL)
      * qwen3-vl-plus                  (hybrid, thinking default off)
    KEY 2:
      * qwen3.6-plus                   (hybrid, thinking default on)
      * kimi-k2.6                      (hybrid, thinking default off)
      * qwen3-vl-flash                 (hybrid, thinking default off)
    KEY 3:
      * qwen3.6-flash                  (hybrid, thinking default on)
  xAI
    * grok-4-1-fast-reasoning          (reasoning-only variant)
    * grok-4.20-reasoning-latest       (reasoning-only variant)
  Google
    * gemini-3-flash-preview           (thinking_level default = high)
  OpenAI
    * gpt-5.4-mini                     (reasoning_effort default = none)

Environment
-----------
    export DASHSCOPE_API_KEY_1="..."
    export DASHSCOPE_API_KEY_2="..."
    export DASHSCOPE_API_KEY_3="..."
    export XAI_API_KEY="..."
    export GEMINI_API_KEY="..."
    export OPENAI_API_KEY="..."

Usage
-----
    python detect.py \\
        --input Negative=../Negative \\
        --input DeepFake=../DeepFake \\
        --output ./Results/SingleImage-NoReview \\
        --concurrency 4

    python detect.py \\
        --input Negative=../Negative \\
        --input DeepFake=../DeepFake \\
        --output ./Results/MultiImage-withReview \\
        --review-mode --single-turn --with-review \\
        --concurrency 4
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

try:
    from PIL import Image as _PILImage   # type: ignore
    _HAS_PIL = True
except ImportError:
    _PILImage = None                      # type: ignore
    _HAS_PIL = False

# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────
# Base URLs are env-overridable because a key is not always usable against the
# public host: Alibaba Bailian issues WORKSPACE endpoints where each API key is
# bound to its own host (and region), so the base URL has to travel with the key
# rather than being one global constant.  Defaults reproduce the paper runs.
DASHSCOPE_BASE = os.getenv("DASHSCOPE_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
XAI_BASE       = os.getenv("XAI_BASE_URL",   "https://api.x.ai/v1")
OPENAI_BASE    = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
GEMINI_BASE    = os.getenv("GEMINI_BASE_URL",
                           "https://generativelanguage.googleapis.com/v1beta")


def dashscope_base(n: int) -> str:
    """OpenAI-compatible base URL for DashScope key slot `n` (1-3).

    Reads DASHSCOPE_BASE_{n} so a per-key workspace host can be supplied,
    falling back to the shared/public endpoint.
    """
    return os.getenv(f"DASHSCOPE_BASE_{n}", DASHSCOPE_BASE)


def dashscope_native_base(n: int) -> str:
    """Native (non-OpenAI-compatible) DashScope base for key slot `n`."""
    fallback = DASHSCOPE_BASE.replace("/compatible-mode/v1", "/api/v1")
    return os.getenv(f"DASHSCOPE_NATIVE_{n}", fallback)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

MAX_RETRIES     = 3
RETRY_BASE      = 3
REQUEST_TIMEOUT = 300

# Progressive downscale tiers, tried in order when a provider rejects a request
# for carrying too much image data. Originally added for xAI's HTTP 413; the
# same ceiling is hit on OpenAI, which reports it as a 400 whose message names
# the total image size. Multi-image reviews are where it bites: the largest
# generator (wan2.7-image-pro, ~6.4 MB median) puts a 7-image review over 70 MB
# once base64 inflates it.
DOWNSCALE_TIERS: list[tuple[int, int]] = [
    (2560, 95),
    (2048, 92),
    (1792, 90),
    (1536, 88),
    (1280, 85),
    (1024, 82),
    ( 896, 80),
    ( 768, 78),
    ( 640, 75),
    ( 512, 72),
]
XAI_DOWNSCALE_TIERS = DOWNSCALE_TIERS   # backwards-compatible alias


# ──────────────────────────────────────────────────────────────────────────────
# Model configuration
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class ModelCfg:
    name:     str
    provider: str
    model_id: str
    key_env:  str
    endpoint: str = ""
    stream:   bool = False
    extra_body: dict = field(default_factory=dict)
    gemini_thinking_level: str | None = None


MODELS: list[ModelCfg] = [
    # ── KEY 1 ────────────────────────────────────────────────────────────────
    ModelCfg(name="qvq-max-latest",   provider="dashscope", model_id="qvq-max-latest",
             key_env="DASHSCOPE_API_KEY_1", endpoint=dashscope_base(1), stream=True),
    ModelCfg(name="qwen3-vl-plus",    provider="dashscope", model_id="qwen3-vl-plus",
             key_env="DASHSCOPE_API_KEY_1", endpoint=dashscope_base(1), stream=False),
    # ── KEY 2 ────────────────────────────────────────────────────────────────
    ModelCfg(name="qwen3.6-plus",     provider="dashscope", model_id="qwen3.6-plus",
             key_env="DASHSCOPE_API_KEY_2", endpoint=dashscope_base(2), stream=True),
    ModelCfg(name="kimi-k2.6",        provider="dashscope", model_id="kimi-k2.6",
             key_env="DASHSCOPE_API_KEY_2", endpoint=dashscope_base(2), stream=False),
    ModelCfg(name="qwen3-vl-flash",   provider="dashscope", model_id="qwen3-vl-flash",
             key_env="DASHSCOPE_API_KEY_2", endpoint=dashscope_base(2), stream=False),
    # ── KEY 3 ────────────────────────────────────────────────────────────────
    ModelCfg(name="qwen3.6-flash",    provider="dashscope", model_id="qwen3.6-flash",
             key_env="DASHSCOPE_API_KEY_3", endpoint=dashscope_base(3), stream=True),
    # ── Non-DashScope ────────────────────────────────────────────────────────
    ModelCfg(name="grok-4-1-fast-reasoning", provider="xai",
             model_id="grok-4-1-fast-reasoning",
             key_env="XAI_API_KEY", endpoint=XAI_BASE, stream=False),
    ModelCfg(name="grok-4.20-reasoning-latest", provider="xai",
             model_id="grok-4.20-reasoning-latest",
             key_env="XAI_API_KEY", endpoint=XAI_BASE, stream=False),
    ModelCfg(name="gemini-3-flash",   provider="gemini", model_id="gemini-3-flash-preview",
             key_env="GEMINI_API_KEY"),
    ModelCfg(name="gpt-5.4-mini",     provider="openai", model_id="gpt-5.4-mini",
             key_env="OPENAI_API_KEY", endpoint=OPENAI_BASE, stream=False),
]
MODEL_BY_NAME = {m.name: m for m in MODELS}


# ──────────────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a forensic image analyst specialising in detecting images that "
    "have been produced or modified by AI — either through AI image-editing LLMs "
    "or through AI image-generation LLMs. You look for characteristic "
    "traces such as local patch inconsistencies inside an otherwise plausible "
    "scene, object insertion or removal seams, mismatched shadows or reflections "
    "on specific objects, blurred or resampled regions around modified areas, "
    "copy-paste texture repetition, broken or re-drawn text and logos, "
    "anatomical or geometric implausibilities, over-smoothed textures, and "
    "diffusion-style artefacts. You never refuse the task and you always "
    "return valid JSON."
)

# Split head/tail so a reviewer comment can be inserted between them.
# Without comment:  HEAD + "\n\n" + TAIL  (identical to no-review prompt)
# With comment:     HEAD + _comment_block(comment) + TAIL
_USER_HEAD = (
    "Analyse the provided image and decide whether it has been produced or "
    "modified by AI — either through AI image-editing LLMs or "
    "through AI image-generation LLMs — or whether it is an unmodified "
    "genuine camera photograph of a real scene."
)
_USER_TAIL = (
    "Return ONLY a single JSON object — no markdown fences, no prose outside "
    "the object — with this exact schema:\n"
    "{\n"
    '  "is_ai_modified": <true or false>,\n'
    '  "confidence":     <number between 0 and 1>,\n'
    '  "reason":         "<1-3 concise sentences naming the specific visual '
    'evidence you relied on>"\n'
    "}"
)
USER_PROMPT = _USER_HEAD + "\n\n" + _USER_TAIL

REVIEW_CONTINUATION_PROMPT = (
    "That was image {i} of {n} from this review. "
    "Do not answer yet — I will show you the next image."
)

_REVIEW_FINAL_HEAD = (
    "That was image {n} of {n} — the final image from this review. "
    "Considering all {n} images together as the set provided by this review, "
    "decide whether these {n} images have been produced or modified by AI — "
    "either through AI image-editing LLMs or through AI image-generation "
    "LLMs — or whether they are unmodified genuine camera photographs of "
    "real scenes."
)
_REVIEW_FINAL_TAIL = (
    "Return ONLY a single JSON object — no markdown fences, no prose outside "
    "the object — with this exact schema:\n"
    "{\n"
    '  "is_ai_modified": <true or false>,             '
    "// overall verdict for the set\n"
    '  "confidence":     <number between 0 and 1>,    '
    "// overall confidence\n"
    '  "reason":         "<1-3 sentences explaining the combined verdict '
    'over all {n} images>",\n'
    '  "per_image": [                                 '
    "// one entry per image, in order\n"
    "    {\n"
    '      "index":          <1-indexed position, 1..{n}>,\n'
    '      "is_ai_modified": <true or false>,\n'
    '      "confidence":     <number between 0 and 1>,\n'
    '      "notes":          "<1-2 sentences on visual evidence from THIS '
    'image>"\n'
    "    },\n"
    "    ...\n"
    "  ]\n"
    "}"
)
REVIEW_FINAL_PROMPT = _REVIEW_FINAL_HEAD + "\n\n" + _REVIEW_FINAL_TAIL

_BATCH_HEAD = (
    "The {n} images above all come from the same product listing. "
    "Examine all {n} images together and decide whether they have been "
    "produced or modified by AI — either through AI image-editing LLMs or "
    "through AI image-generation LLMs — or whether they are unmodified "
    "genuine camera photographs of a real product."
)
_BATCH_TAIL = (
    "Return ONLY a single JSON object — no markdown fences, no prose outside "
    "the object — with this exact schema:\n"
    "{\n"
    '  "is_ai_modified": <true or false>,             '
    "// overall verdict for the set\n"
    '  "confidence":     <number between 0 and 1>,    '
    "// overall confidence\n"
    '  "reason":         "<1-3 sentences explaining the combined verdict '
    'over all {n} images>",\n'
    '  "per_image": [                                 '
    "// one entry per image, in order\n"
    "    {\n"
    '      "index":          <1-indexed position, 1..{n}>,\n'
    '      "is_ai_modified": <true or false>,\n'
    '      "confidence":     <number between 0 and 1>,\n'
    '      "notes":          "<1-2 sentences on visual evidence from THIS '
    'image>"\n'
    "    },\n"
    "    ...\n"
    "  ]\n"
    "}"
)


# ── Review-comment injection ──────────────────────────────────────────────────
def _comment_block(comment: str) -> str:
    return (
        f'\n\nThe customer who purchased this product on Amazon left the following review:\n'
        f'"{comment}"\n'
        f'Consider this customer review when making your assessment.\n\n'
    )


def _comment_block_combined(comments: list[str | None]) -> str:
    """Combined comment block for multi-image final prompt."""
    non_null = [(i + 1, c) for i, c in enumerate(comments) if c]
    if not non_null:
        return "\n\n"
    unique_texts = {c for _, c in non_null}
    if len(unique_texts) == 1:
        return _comment_block(next(iter(unique_texts)))
    lines = "\n".join(f'  Image {i}: "{c}"' for i, c in non_null)
    return (
        "\n\nThe customer who purchased this product on Amazon left the following review "
        "comments about the images in this set:\n"
        f"{lines}\n"
        "Consider these customer review comments when making your assessment.\n\n"
    )


def _render_review_prompt(template: str, *, i: int | None = None, n: int) -> str:
    out = template.replace("{n}", str(n))
    if i is not None:
        out = out.replace("{i}", str(i))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def image_to_data_uri(path: Path) -> tuple[str, str]:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    b64  = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}", mime


def _downscaled_data_uri(path: Path, max_dim: int, quality: int) -> tuple[str, str]:
    if not _HAS_PIL:
        raise RuntimeError(
            "Pillow is required to retry xAI requests on HTTP 413. "
            "Install it with: pip install Pillow"
        )
    with _PILImage.open(path) as im:
        im.load()
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        w, h = im.size
        long_edge = max(w, h)
        if long_edge > max_dim:
            scale = max_dim / long_edge
            im = im.resize(
                (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                _PILImage.LANCZOS,
            )
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}", "image/jpeg"


def _encode_image(path: Path, downscale_level: int) -> tuple[str, str]:
    if downscale_level <= 0:
        return image_to_data_uri(path)
    tier = DOWNSCALE_TIERS[min(downscale_level - 1, len(XAI_DOWNSCALE_TIERS) - 1)]
    return _downscaled_data_uri(path, *tier)


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_reply(raw: str) -> dict | None:
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lstrip().lower().startswith("json"):
            s = s.split("\n", 1)[1] if "\n" in s else s[4:]
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    m = _JSON_OBJ_RE.search(s)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _provider_key(cfg: ModelCfg) -> str:
    k = os.getenv(cfg.key_env)
    if not k:
        raise EnvironmentError(
            f"{cfg.key_env} is not set (required for model '{cfg.name}')."
        )
    return k


# ──────────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ──────────────────────────────────────────────────────────────────────────────
class HttpError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:500]}")
        self.status = status
        self.body   = body


def _http_post_json(url, payload, headers, *, stream, timeout=REQUEST_TIMEOUT):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req  = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise HttpError(e.code, body) from None
    if stream:
        return resp.status, resp
    try:
        body = resp.read().decode("utf-8", errors="replace")
        return resp.status, json.loads(body)
    finally:
        resp.close()


# ──────────────────────────────────────────────────────────────────────────────
# Message builders
# ──────────────────────────────────────────────────────────────────────────────
def _build_openai_messages(
    image_paths: list[Path],
    *,
    merge_system: bool,
    reviewer_comments: list[str | None] | None = None,
    downscale_level: int = 0,
    single_turn: bool = False,
) -> list[dict]:
    """Build chat-completion messages.

    Single-image (n=1):
        [ image ] [ USER_PROMPT (± comment) ]

    Multi-step (n>1, single_turn=False):
        [ img_1 ] [ "That was image 1 of N …" ]  …
        [ img_N ] [ REVIEW_FINAL_PROMPT (± combined comment) ]

    Single-turn multi-image (n>1, single_turn=True):
        [ img_1 ] … [ img_N ] [ BATCH_USER_PROMPT (± combined comment) ]
    """
    n        = len(image_paths)
    if n == 0:
        raise ValueError("_build_openai_messages: image_paths must not be empty")
    comments = list(reviewer_comments) if reviewer_comments else [None] * n
    if len(comments) < n:
        comments += [None] * (n - len(comments))

    parts: list[dict] = []

    if n == 1:
        data_uri, _ = _encode_image(image_paths[0], downscale_level)
        parts.append({"type": "image_url", "image_url": {"url": data_uri}})
        comment = comments[0]
        mid = _comment_block(comment) if comment else "\n\n"
        parts.append({"type": "text", "text": _USER_HEAD + mid + _USER_TAIL})

    elif single_turn:
        for path in image_paths:
            data_uri, _ = _encode_image(path, downscale_level)
            parts.append({"type": "image_url", "image_url": {"url": data_uri}})
        head = _render_review_prompt(_BATCH_HEAD, n=n)
        tail = _render_review_prompt(_BATCH_TAIL, n=n)
        parts.append({"type": "text", "text": head + _comment_block_combined(comments) + tail})

    else:
        for idx, path in enumerate(image_paths, start=1):
            data_uri, _ = _encode_image(path, downscale_level)
            parts.append({"type": "image_url", "image_url": {"url": data_uri}})
            if idx < n:
                parts.append({"type": "text",
                               "text": _render_review_prompt(REVIEW_CONTINUATION_PROMPT, i=idx, n=n)})
            else:
                head = _render_review_prompt(_REVIEW_FINAL_HEAD, n=n)
                tail = _render_review_prompt(_REVIEW_FINAL_TAIL, n=n)
                parts.append({"type": "text",
                               "text": head + _comment_block_combined(comments) + tail})

    if merge_system:
        parts.insert(0, {"type": "text", "text": SYSTEM_PROMPT})
        return [{"role": "user", "content": parts}]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": parts},
    ]


def _consume_sse(resp) -> tuple[str, str, str | None]:
    """Returns (content, reasoning, api_model).

    `api_model` is the model string the API echoes back.  Vendors resolve a
    floating alias ("qwen3.6-flash", anything ending "-latest" or "-preview")
    to the dated snapshot that actually served the request, so recording it is
    the only way a run stays version-identifiable after the alias moves on.
    """
    content   = []
    reasoning = []
    api_model = None
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
            if api_model is None and chunk.get("model"):
                api_model = chunk["model"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            c = delta.get("content")
            if c:
                content.append(c)
            r = delta.get("reasoning_content")
            if r:
                reasoning.append(r)
    finally:
        resp.close()
    return "".join(content), "".join(reasoning), api_model


def call_openai_compat(
    cfg: ModelCfg,
    image_paths: list[Path],
    reviewer_comments: list[str | None] | None = None,
    *,
    downscale_level: int = 0,
    single_turn: bool = False,
) -> dict:
    api_key = _provider_key(cfg)
    url     = f"{cfg.endpoint.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    payload: dict[str, Any] = {
        "model":    cfg.model_id,
        "messages": _build_openai_messages(
            image_paths,
            merge_system=cfg.name.startswith(("qwen", "qvq")),
            reviewer_comments=reviewer_comments,
            downscale_level=downscale_level,
            single_turn=single_turn,
        ),
        "stream": cfg.stream,
    }
    if cfg.stream:
        payload["stream_options"] = {"include_usage": True}
    if cfg.extra_body:
        payload.update(cfg.extra_body)

    status, result = _http_post_json(url, payload, headers, stream=cfg.stream)

    if cfg.stream:
        content, reasoning, api_model = _consume_sse(result)
        return {"content": content, "reasoning_content": reasoning or None,
                "raw_response": content, "api_model": api_model}

    try:
        msg = result["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as e:
        raise HttpError(status, json.dumps(result)[:500]) from e
    return {"content": msg.get("content") or "",
            "reasoning_content": msg.get("reasoning_content") or None,
            "raw_response": msg.get("content") or "",
            "api_model": result.get("model")}


def call_gemini(
    cfg: ModelCfg,
    image_paths: list[Path],
    reviewer_comments: list[str | None] | None = None,
    *,
    downscale_level: int = 0,
    single_turn: bool = False,
) -> dict:
    api_key = _provider_key(cfg)
    url = (
        f"{GEMINI_BASE}/models/{cfg.model_id}:generateContent"
        f"?key={urllib.parse.quote(api_key)}"
    )
    n        = len(image_paths)
    if n == 0:
        raise ValueError("call_gemini: image_paths must not be empty")
    comments = list(reviewer_comments) if reviewer_comments else [None] * n
    if len(comments) < n:
        comments += [None] * (n - len(comments))

    parts: list[dict] = []

    if n == 1:
        _uri, mime = _encode_image(image_paths[0], downscale_level)
        b64  = _uri.split(",", 1)[1]
        parts.append({"inlineData": {"mimeType": mime, "data": b64}})
        comment = comments[0]
        mid = _comment_block(comment) if comment else "\n\n"
        parts.append({"text": _USER_HEAD + mid + _USER_TAIL})

    elif single_turn:
        for path in image_paths:
            _uri, mime = _encode_image(path, downscale_level)
            b64  = _uri.split(",", 1)[1]
            parts.append({"inlineData": {"mimeType": mime, "data": b64}})
        head = _render_review_prompt(_BATCH_HEAD, n=n)
        tail = _render_review_prompt(_BATCH_TAIL, n=n)
        parts.append({"text": head + _comment_block_combined(comments) + tail})

    else:
        for idx, path in enumerate(image_paths, start=1):
            _uri, mime = _encode_image(path, downscale_level)
            b64  = _uri.split(",", 1)[1]
            parts.append({"inlineData": {"mimeType": mime, "data": b64}})
            if idx < n:
                parts.append({"text": _render_review_prompt(REVIEW_CONTINUATION_PROMPT, i=idx, n=n)})
            else:
                head = _render_review_prompt(_REVIEW_FINAL_HEAD, n=n)
                tail = _render_review_prompt(_REVIEW_FINAL_TAIL, n=n)
                parts.append({"text": head + _comment_block_combined(comments) + tail})

    gen_cfg: dict[str, Any] = {"responseMimeType": "application/json"}
    if cfg.gemini_thinking_level:
        gen_cfg["thinkingConfig"] = {"thinkingLevel": cfg.gemini_thinking_level}

    payload: dict[str, Any] = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": gen_cfg,
    }
    headers = {"Content-Type": "application/json"}
    status, result = _http_post_json(url, payload, headers, stream=False)

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    try:
        cand = result["candidates"][0]
    except (KeyError, IndexError, TypeError) as e:
        raise HttpError(status, json.dumps(result)[:500]) from e
    for p in (cand.get("content") or {}).get("parts", []):
        if "text" in p:
            if p.get("thought"):
                reasoning_parts.append(p["text"])
            else:
                content_parts.append(p["text"])
    content = "".join(content_parts)
    return {"content": content, "reasoning_content": "".join(reasoning_parts) or None,
            "raw_response": content,
            "api_model": result.get("modelVersion")}


# ──────────────────────────────────────────────────────────────────────────────
# Retry wrapper
# ──────────────────────────────────────────────────────────────────────────────
# Providers signal "too much image data" differently: xAI returns 413, OpenAI a
# 400 whose message names the total image size, Gemini and DashScope their own
# wording. Matching on the message as well as the status keeps one retry path
# for all of them instead of a per-vendor special case.
_TOO_LARGE_RE = re.compile(
    r"image size|too large|exceeds?\s+the\s+maximum|payload.{0,20}(large|limit)"
    r"|request entity too large|maximum.{0,20}size",
    re.IGNORECASE,
)


def _is_payload_too_large(e: "HttpError") -> bool:
    if e.status == 413:
        return True
    return e.status == 400 and bool(_TOO_LARGE_RE.search(e.body or ""))


# DashScope answers some multi-image requests with a 400 that is not a client
# error at all: its gateway fetches each data URI itself, and when several large
# ones arrive together it gives up part way. "Download multimodal file timed out"
# is plainly transient; "The provided URL does not appear to be valid" is the
# same failure reported as a parse error -- the URIs are well-formed base64 that
# the same model accepts one at a time. A blanket "4xx means stop" drops both
# without a single retry, which is how a whole conversation dies at turn 5 of 8.
_DASHSCOPE_FETCH_RE = re.compile(
    r"download .{0,30}file .{0,20}timed out"
    r"|provided url does not appear to be valid"
    r"|failed to (download|fetch) .{0,30}(image|file)",
    re.IGNORECASE,
)


def _is_transient_upstream_400(e: "HttpError") -> bool:
    return e.status == 400 and bool(_DASHSCOPE_FETCH_RE.search(e.body or ""))


# DashScope does not answer an oversized request with a status at all: it resets
# the connection while the body is still being uploaded, which surfaces as
# ECONNRESET / broken pipe. The exception alone cannot be told apart from a
# transient network blip, so this only gates the downscale path -- the guards in
# call_with_retry decide when to act on it.
_CONN_DROP_RE = re.compile(
    r"connection reset|broken pipe|connection aborted|errno 104|errno 32",
    re.IGNORECASE,
)

# Raw bytes above which a dropped connection is read as "too much image data"
# rather than a flaky link. base64 inflates by ~4/3, so this sits far below any
# provider ceiling yet far above an ordinary one- or two-image request.
CONN_DROP_SIZE_FLOOR = 10 * 1024 * 1024


def _is_connection_dropped(e: BaseException) -> bool:
    return bool(_CONN_DROP_RE.search(str(e)))


def _raw_payload_bytes(image_paths: list[Path]) -> int:
    total = 0
    for p in image_paths:
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return total


def call_with_retry(
    fn: Callable,
    cfg: ModelCfg,
    image_paths: list[Path],
    reviewer_comments: list[str | None],
    max_retries: int = MAX_RETRIES,
    *,
    single_turn: bool = False,
) -> tuple[dict | None, int, str | None]:
    """Retry wrapper with progressive image downscale on payload-size errors."""
    last_err: str | None = None
    downscale_level = 0
    conn_drops      = 0
    max_downscale   = len(DOWNSCALE_TIERS)
    total_budget    = max_retries + max_downscale
    attempt         = 0
    backoff_n       = 0
    while attempt < total_budget:
        attempt += 1
        try:
            res = fn(cfg, image_paths, reviewer_comments,
                     downscale_level=downscale_level, single_turn=single_turn)
            return res, attempt, None
        except HttpError as e:
            last_err = f"HTTP {e.status}: {e.body[:300]}"
            if _is_payload_too_large(e) and downscale_level < max_downscale:
                if not _HAS_PIL:
                    return None, attempt, (f"{last_err}; install Pillow to enable "
                                           f"downscale retry")
                downscale_level += 1
                continue
            if _is_transient_upstream_400(e):
                # Back off and, once PIL is there, also shrink: the gateway is
                # timing out fetching the images, so fewer bytes is the remedy.
                if _HAS_PIL and downscale_level < max_downscale:
                    downscale_level += 1
                backoff_n += 1
            elif 400 <= e.status < 500 and e.status != 429:
                return None, attempt, last_err
            else:
                backoff_n += 1
        except urllib.error.URLError as e:
            last_err = f"URLError: {e}"
            # A heavy request that keeps getting its connection dropped is the
            # unlabelled form of "too much image data" (DashScope). One retry at
            # the current size covers a genuine blip; a second drop steps the
            # images down a tier instead of burning the budget at a size the
            # provider will never accept.
            if (_is_connection_dropped(e)
                    and _raw_payload_bytes(image_paths) >= CONN_DROP_SIZE_FLOOR):
                conn_drops += 1
                if conn_drops >= 2 and _HAS_PIL and downscale_level < max_downscale:
                    downscale_level += 1
                    conn_drops = 0
                    continue
            backoff_n += 1
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            backoff_n += 1
        if attempt < total_budget:
            time.sleep(RETRY_BASE * (2 ** (backoff_n - 1)))
    return None, attempt, last_err


# ──────────────────────────────────────────────────────────────────────────────
# Reviewer comment loader  (only used when --with-review is set)
# ──────────────────────────────────────────────────────────────────────────────
def _load_reviewer_comments(
    label:       str | None,
    image_paths: list[Path],
    review_id:   str | None,
) -> list[str | None]:
    """Return one comment (or None) per image_path."""
    n = len(image_paths)
    if n == 0:
        return []

    def _num_from(s: str) -> str:
        parts = s.rsplit("_", 1)
        return parts[-1] if len(parts) == 2 else ""

    if label == "Negative":
        review_dir = image_paths[0].parent
        num = _num_from(review_id) if review_id else (
            _num_from(review_dir.name)
            if re.match(r"^Review_\d+$", review_dir.name) else ""
        )
        if not num:
            return [None] * n
        meta = review_dir / f"MetaReview_{num}.json"
        if not meta.exists():
            return [None] * n
        try:
            data  = json.loads(meta.read_text(encoding="utf-8"))
            # The three sources name the review body differently and each has
            # its own MetaReview schema: Amazon uses title + text, Trip.com uses
            # content, GrabFood uses review_text. Missing the last one made the
            # 63 GrabFood reviews silently run without review context — the
            # withReview condition degrading to NoReview for that category.
            title = (data.get("title") or "").strip()
            body  = (data.get("text")
                     or data.get("content")
                     or data.get("review_text")
                     or "").strip()
            parts = [p for p in (title, body) if p]
            text  = ". ".join(parts) or None
            return [text] * n
        except (json.JSONDecodeError, OSError):
            return [None] * n

    if label and (label == "DeepFake" or label.startswith("DeepFake/")):
        review_dir    = image_paths[0].parent
        deepfake_root = review_dir.parent.parent
        num = _num_from(review_id) if review_id else (
            _num_from(review_dir.name)
            if re.match(r"^Review_\d+$", review_dir.name) else ""
        )
        if not num:
            return [None] * n
        meta = deepfake_root / "Metadata" / f"Edit_{num}.json"
        if not meta.exists():
            return [None] * n
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
            comment_map: dict[str, str] = {
                r["image"]: r["reviewer_comment"]
                for r in data.get("per_image", [])
                if isinstance(r, dict) and r.get("image") and r.get("reviewer_comment")
            }
            return [comment_map.get(p.name) for p in image_paths]
        except (json.JSONDecodeError, OSError):
            return [None] * n

    return [None] * n


# ──────────────────────────────────────────────────────────────────────────────
# Input discovery
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class ImageJob:
    image_paths:        list[Path]
    image_rels:         list[str]
    label:              str | None
    bucket:             str
    generator:          str | None
    reviewer_comments:  list[str | None] = field(default_factory=list)
    review_id:          str | None = None
    review_path:        Path | None = None

    @property
    def image_path(self) -> Path:
        return self.image_paths[0]

    @property
    def image_rel(self) -> str:
        return self.image_rels[0]

    @property
    def num_images(self) -> int:
        return len(self.image_paths)

    @property
    def is_review(self) -> bool:
        return self.review_id is not None

    @property
    def key(self) -> str:
        if self.review_path is not None:
            return str(self.review_path)
        return str(self.image_paths[0])

    @property
    def portable_key(self) -> str:
        """Identity that survives moving the dataset to another machine.

        `key` is an absolute path, so results written on one machine can never
        be resumed on another — records written elsewhere carry
        that machine's absolute paths, which match nothing here, and a re-run
        would append a second record per image instead of skipping. review_id
        (review mode) and image_rel (single-image mode) are relative to the
        bucket and unique within it, so they match across machines.
        """
        return self.review_id or self.image_rel

    @property
    def display_rel(self) -> str:
        if self.review_id is not None:
            return f"{self.review_id} ({self.num_images} imgs)"
        return self.image_rels[0]


def parse_input_arg(arg: str) -> tuple[str | None, Path]:
    if "=" in arg:
        label, path = arg.split("=", 1)
        label = label.strip() or None
    else:
        label = None
        path  = arg
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Input path does not exist: {p}")
    return label, p


def _bucket_for(label: str | None, rel_to_root: str) -> tuple[str, str | None, str]:
    parts = Path(rel_to_root).parts
    if label == "DeepFake" and len(parts) > 1:
        generator = parts[0]
        bucket    = f"DeepFake/{generator}"
        rel       = str(Path(*parts[1:]))
        return bucket, generator, rel
    if label:
        return label, None, rel_to_root
    return "unlabeled", None, rel_to_root


@dataclass
class _ImgInfo:
    abs_path:      Path
    root:          Path
    rel_to_root:   str
    label:         str | None
    bucket:        str
    generator:     str | None
    rel_to_bucket: str


def _collect_images(inputs: list[tuple[str | None, Path]]) -> list[_ImgInfo]:
    out: list[_ImgInfo] = []
    seen: set[Path] = set()
    for label, root in inputs:
        if root.is_file():
            if root.suffix.lower() in IMAGE_EXTS and root not in seen:
                bucket, gen, rel = _bucket_for(label, root.name)
                out.append(_ImgInfo(abs_path=root, root=root.parent,
                                    rel_to_root=root.name, label=label,
                                    bucket=bucket, generator=gen, rel_to_bucket=rel))
                seen.add(root)
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS or p in seen:
                continue
            rel_to_root = str(p.relative_to(root))
            bucket, gen, rel = _bucket_for(label, rel_to_root)
            out.append(_ImgInfo(abs_path=p, root=root, rel_to_root=rel_to_root,
                                label=label, bucket=bucket, generator=gen,
                                rel_to_bucket=rel))
            seen.add(p)
    return out


def walk_inputs(
    inputs: list[tuple[str | None, Path]],
    *,
    review_mode: bool = False,
    with_review: bool = False,
) -> list[ImageJob]:
    imgs = _collect_images(inputs)

    if not review_mode:
        jobs = []
        for i in imgs:
            comments = (
                _load_reviewer_comments(i.label, [i.abs_path], None)
                if with_review else [None]
            )
            jobs.append(ImageJob(
                image_paths=[i.abs_path],
                image_rels=[i.rel_to_bucket],
                label=i.label,
                bucket=i.bucket,
                generator=i.generator,
                reviewer_comments=comments,
            ))
        return jobs

    # ── Review mode: group by (bucket, review folder) ───────────────────────
    groups: dict[tuple[str, str], list[_ImgInfo]] = {}
    flat_skipped: list[str] = []
    for i in imgs:
        parts = Path(i.rel_to_bucket).parts
        if len(parts) < 2:
            flat_skipped.append(str(i.abs_path))
            continue
        review_id = parts[0]
        groups.setdefault((i.bucket, review_id), []).append(i)

    if flat_skipped:
        print(f"[warn] --review-mode: skipped {len(flat_skipped)} flat image(s):",
              file=sys.stderr)
        for f in flat_skipped[:5]:
            print(f"  {f}", file=sys.stderr)
        if len(flat_skipped) > 5:
            print(f"  ... ({len(flat_skipped) - 5} more)", file=sys.stderr)

    jobs: list[ImageJob] = []
    for (bucket, review_id), items in sorted(groups.items()):
        items.sort(key=lambda x: x.abs_path)
        head        = items[0]
        review_path = head.abs_path.parent
        image_paths = [x.abs_path for x in items]
        comments    = (
            _load_reviewer_comments(head.label, image_paths, review_id)
            if with_review else [None] * len(image_paths)
        )
        jobs.append(ImageJob(
            image_paths=image_paths,
            image_rels=[x.rel_to_bucket for x in items],
            label=head.label,
            bucket=bucket,
            generator=head.generator,
            reviewer_comments=comments,
            review_id=review_id,
            review_path=review_path,
        ))
    return jobs


# ──────────────────────────────────────────────────────────────────────────────
# Output aggregator
# ──────────────────────────────────────────────────────────────────────────────
class ResultStore:
    def __init__(self, output_dir: Path, models: list[ModelCfg], jobs: list[ImageJob]):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._models = {m.name: m for m in models}
        self._locks:   dict[tuple[str, str], threading.Lock] = {}
        self._files:   dict[tuple[str, str], Path]           = {}
        self._bundles: dict[tuple[str, str], dict]           = {}
        buckets = sorted({j.bucket for j in jobs})
        for bucket in buckets:
            bucket_dir = output_dir / bucket
            bucket_dir.mkdir(parents=True, exist_ok=True)
            for m in models:
                key = (bucket, m.name)
                self._locks[key]   = threading.Lock()
                self._files[key]   = bucket_dir / f"{m.name}.json"
                self._bundles[key] = self._load_or_init(bucket, m)

    def _load_or_init(self, bucket: str, m: ModelCfg) -> dict:
        path = self._files[(bucket, m.name)]
        if path.exists():
            try:
                with path.open(encoding="utf-8") as f:
                    bundle = json.load(f)
                bundle.setdefault("results", [])
                return bundle
            except (json.JSONDecodeError, OSError):
                pass
        return {
            "model": m.name, "model_id": m.model_id, "provider": m.provider,
            "bucket": bucket,
            "config": {"stream": m.stream, "extra_body": m.extra_body,
                       "gemini_thinking_level": m.gemini_thinking_level},
            "started_at": now_iso(), "updated_at": now_iso(), "results": [],
        }

    @staticmethod
    def _record_key(r: dict) -> str:
        """Portable identity of a stored record — see ImageJob.portable_key.

        Falls back to the absolute key only for records predating those fields.
        """
        return (r.get("review_id") or r.get("image_rel")
                or r.get("key") or r.get("image_path") or "")

    def already_done(self, bucket: str, model_name: str, job_key: str) -> bool:
        bundle = self._bundles.get((bucket, model_name))
        if bundle is None:
            return False
        for r in bundle["results"]:
            if self._record_key(r) == job_key and r.get("error") is None:
                return True
        return False

    def record(self, bucket: str, model_name: str, record: dict) -> None:
        key  = (bucket, model_name)
        lock = self._locks[key]
        with lock:
            bundle  = self._bundles[key]
            replaced = False
            rkey    = self._record_key(record)
            for i, r in enumerate(bundle["results"]):
                if self._record_key(r) == rkey:
                    bundle["results"][i] = record
                    replaced = True
                    break
            if not replaced:
                bundle["results"].append(record)
            bundle["updated_at"] = now_iso()
            atomic_write_json(self._files[key], bundle)

    def build_summary(self, jobs: list[ImageJob]) -> dict:
        # Keyed by (bucket, record key), NOT by the record key alone: a job's
        # portable key is its review_id or image_rel, and those repeat verbatim
        # across buckets -- Negative/Review_002/Image_002_01.jpg and the same
        # relative path under each of the six generators. Without the bucket the
        # seven records of one model collapse onto one entry and whichever
        # bucket is written last is then reported for all of them.
        by_model: dict[str, dict[tuple[str, str], dict]] = {name: {} for name in self._models}
        for (bucket, model_name), bundle in self._bundles.items():
            for r in bundle["results"]:
                rk = self._record_key(r)
                if rk:
                    by_model[model_name][(bucket, rk)] = r
        rows = []
        any_review = any(j.is_review for j in jobs)
        for j in jobs:
            k = (j.bucket, j.portable_key)
            row: dict[str, Any] = {
                "key": j.key, "bucket": j.bucket, "label": j.label,
                "generator": j.generator, "num_images": j.num_images,
                "verdicts": {},
            }
            if j.is_review:
                row["review_id"]   = j.review_id
                row["review_path"] = str(j.review_path)
                row["image_rels"]  = list(j.image_rels)
            else:
                row["image_path"] = str(j.image_path)
                row["image_rel"]  = j.image_rel
            for name in sorted(self._models):
                r = by_model[name].get(k)
                if r is None:
                    row["verdicts"][name] = {"status": "missing"}
                else:
                    row["verdicts"][name] = {
                        "status":         "ok" if r.get("error") is None else "error",
                        "is_ai_modified": r.get("is_ai_modified"),
                        "confidence":     r.get("confidence"),
                        "reason":         r.get("reason"),
                        "error":          r.get("error"),
                    }
            rows.append(row)
        return {
            "generated_at": now_iso(),
            "mode":         "review" if any_review else "image",
            "num_jobs":     len(jobs),
            "num_images":   sum(j.num_images for j in jobs),
            "buckets":      sorted({j.bucket for j in jobs}),
            "models":       sorted(self._models.keys()),
            "rows":         rows,
        }

    @staticmethod
    def _merge_summary(path: Path, fresh: dict) -> dict:
        """Fold a previous summary.json into the one this run produced.

        build_summary() only sees the jobs of the current run, so a run narrowed
        by --models or --generators would otherwise overwrite summary.json with
        just its own slice and drop every row an earlier run wrote. The
        per-model JSONs under <bucket>/<model>.json are unaffected either way —
        only this roll-up needed protecting.

        Rows are merged by key and their `verdicts` dicts are unioned, with this
        run winning for models it actually scored. A "missing" verdict from this
        run never overwrites a real one already on disk.
        """
        if not path.exists():
            return fresh
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return fresh
        if not isinstance(old, dict) or old.get("mode") != fresh.get("mode"):
            return fresh          # different experiment shape — do not mix

        # Match rows the same portable way records are matched, so a summary
        # written on another machine still merges instead of duplicating.
        #
        # The bucket MUST be part of the identity here. Per-model records live in
        # one file per bucket, so review_id/image_rel alone is unique there — but
        # every row of every bucket shares this one summary, and in single-image
        # mode "Review_001/Image_001_01.jpg" names a different image under
        # Negative than under each DeepFake/<generator>. Without the bucket the
        # buckets overwrite each other and all but one disappear.
        def rid(r: dict) -> str:
            ident = (r.get("review_id") or r.get("image_rel") or r.get("key") or "")
            return f"{r.get('bucket') or ''}::{ident}"

        by_key: dict[str, dict] = {
            rid(r): r for r in old.get("rows", [])
            if isinstance(r, dict) and rid(r)
        }
        for r in fresh.get("rows", []):
            k = rid(r)
            prev = by_key.get(k)
            if prev is None:
                by_key[k] = r
                continue
            verdicts = dict(prev.get("verdicts") or {})
            for name, v in (r.get("verdicts") or {}).items():
                was = (verdicts.get(name) or {}).get("status")
                if v.get("status") == "missing" and was not in (None, "missing"):
                    continue      # keep the real verdict an earlier run recorded
                verdicts[name] = v
            row = dict(prev)
            row.update(r)
            row["verdicts"] = verdicts
            by_key[k] = row

        rows = list(by_key.values())
        out = dict(fresh)
        out["rows"]       = rows
        out["models"]     = sorted(set(old.get("models") or []) | set(fresh.get("models") or []))
        out["buckets"]    = sorted(set(old.get("buckets") or []) | set(fresh.get("buckets") or []))
        out["num_jobs"]   = len(rows)
        out["num_images"] = sum(int(r.get("num_images") or 1) for r in rows)
        return out

    def flush_summary(self, jobs: list[ImageJob]) -> Path:
        path = self.output_dir / "summary.json"
        atomic_write_json(path, self._merge_summary(path, self.build_summary(jobs)))
        return path


# ──────────────────────────────────────────────────────────────────────────────
# Per-task worker
# ──────────────────────────────────────────────────────────────────────────────
def _new_record(job: ImageJob) -> dict[str, Any]:
    return {
        "key":               job.key,
        "bucket":            job.bucket,
        "label":             job.label,
        "generator":         job.generator,
        "review_id":         job.review_id,
        "review_path":       str(job.review_path) if job.review_path else None,
        "num_images":        job.num_images,
        "image_paths":       [str(p) for p in job.image_paths],
        "image_rels":        list(job.image_rels),
        "image_path":        str(job.image_paths[0]),
        "image_rel":         job.image_rels[0],
        "reviewer_comments": job.reviewer_comments,
        "api_model":         None,   # snapshot the API says served the request
        "is_ai_modified":    None,
        "confidence":        None,
        "reason":            None,
        "per_image_analysis": None,
        "reasoning_content":  None,
        "raw_response":       None,
        "latency_ms":         0,
        "attempts":           0,
        "error":              None,
    }


def run_one(cfg: ModelCfg, job: ImageJob, *, single_turn: bool = False) -> dict:
    fn = call_gemini if cfg.provider == "gemini" else call_openai_compat
    t0 = time.time()
    raw, attempts, err = call_with_retry(
        fn, cfg, job.image_paths, job.reviewer_comments, single_turn=single_turn,
    )
    latency_ms = int((time.time() - t0) * 1000)

    record = _new_record(job)
    record["latency_ms"] = latency_ms
    record["attempts"]   = attempts
    record["error"]      = err
    if raw is None:
        return record

    record["raw_response"]      = raw.get("raw_response")
    record["reasoning_content"] = raw.get("reasoning_content")
    record["api_model"]         = raw.get("api_model")

    parsed = parse_json_reply(raw.get("content") or "")
    if parsed is None:
        record["error"] = (
            (err + "; " if err else "") + "failed to parse JSON from model output"
        )
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
# Main orchestration
# ──────────────────────────────────────────────────────────────────────────────
def run(jobs, models, output_dir, concurrency, resume, single_turn=False):
    store = ResultStore(output_dir, models, jobs)
    tasks: list[tuple[ModelCfg, ImageJob]] = []
    skipped = 0
    for job in jobs:
        for cfg in models:
            if resume and store.already_done(job.bucket, cfg.name, job.portable_key):
                skipped += 1
                continue
            tasks.append((cfg, job))

    total      = len(tasks)
    n_images   = sum(j.num_images for j in jobs)
    any_review = any(j.is_review for j in jobs)
    unit       = "reviews" if any_review else "images"
    mode_tag   = "single-turn-multi-image" if single_turn else ("multi-step" if any_review else "single-image")
    print(
        f"[plan] mode={mode_tag} | {len(jobs)} {unit} ({n_images} images total) × "
        f"{len(models)} models = {len(jobs) * len(models)} pairs "
        f"(skipped {skipped} already-done, {total} to run); "
        f"concurrency = up to {concurrency * len(models)} parallel calls",
        flush=True,
    )
    if total == 0:
        store.flush_summary(jobs)
        print("[done] nothing to do.", flush=True)
        return

    pool = ThreadPoolExecutor(max_workers=max(1, concurrency * len(models)))
    futures = {}
    for cfg, job in tasks:
        fut = pool.submit(run_one, cfg, job, single_turn=single_turn)
        futures[fut] = (cfg, job)

    done = ok = fail = 0
    t_start = time.time()
    try:
        for fut in as_completed(futures):
            cfg, job = futures[fut]
            try:
                record = fut.result()
            except Exception as e:
                record = _new_record(job)
                record["error"] = f"worker crashed: {type(e).__name__}: {e}"
                traceback.print_exc()

            store.record(job.bucket, cfg.name, record)
            done += 1
            if record["error"] is None and record["is_ai_modified"] is not None:
                ok += 1
            else:
                fail += 1

            elapsed = time.time() - t_start
            rate    = done / elapsed if elapsed else 0
            eta     = (total - done) / rate if rate else 0
            verdict = record["is_ai_modified"]
            flag    = "?" if record["error"] else ("MOD" if verdict else "real")
            print(
                f"[{done:>{len(str(total))}}/{total}] "
                f"ok={ok} fail={fail} "
                f"eta={eta:>5.0f}s | {cfg.name:<28} "
                f"{record['latency_ms']:>6}ms "
                f"[{flag}] {job.display_rel}"
                + (f" — {record['error'][:80]}" if record["error"] else ""),
                flush=True,
            )
            if done % 25 == 0:
                store.flush_summary(jobs)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        store.flush_summary(jobs)

    print(
        f"[done] {done} calls completed in {time.time() - t_start:.1f}s "
        f"(ok={ok}, fail={fail}). Summary: {output_dir / 'summary.json'}",
        flush=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", action="append", required=True,
                   help="LABEL=PATH or PATH. Repeatable.")
    p.add_argument("--output", required=True,
                   help="Root directory for results.")
    p.add_argument("--models", nargs="+", default=None,
                   help="Subset of models to run (default: all). Names come from the "
                        "built-in list, or from --models-config when given.")
    p.add_argument("--models-config", default=None,
                   help="JSON detector registry merged over the built-ins "
                        "(see ../config/models.json). Omit to reproduce the paper runs.")
    p.add_argument("--concurrency", type=int, default=3,
                   help="Max in-flight images (default 3).")
    p.add_argument("--no-resume", action="store_true",
                   help="Disable resuming from existing output JSONs.")
    p.add_argument("--review-mode", action="store_true",
                   help="Group images by review folder into a single multi-image call.")
    p.add_argument("--single-turn", action="store_true",
                   help="(Requires --review-mode) Pack all images in one user message "
                        "instead of multi-step continuation prompts.")
    p.add_argument("--with-review", action="store_true",
                   help="Inject the product review text into the prompt "
                        "(WithReview experiments). Omit for NoReview experiments.")
    p.add_argument("--generators", nargs="+", default=None,
                   help="Restrict DeepFake buckets to these generator folder names. "
                        "The DeepFake/ directory now holds both the original six "
                        "generators and the ones added later, so evaluating the "
                        "original benchmark needs this filter — otherwise every "
                        "generator present is scored. Default: every subfolder found.")
    p.add_argument("--limit", type=int, default=0,
                   help="Process only the first N jobs (smoke test).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args   = parse_args(argv if argv is not None else sys.argv[1:])
    inputs = [parse_input_arg(s) for s in args.input]
    jobs   = walk_inputs(inputs, review_mode=args.review_mode, with_review=args.with_review)

    if args.generators:
        wanted = set(args.generators)
        found  = {j.generator for j in jobs if j.generator}
        unknown = wanted - found
        if unknown:
            print(f"[warn] --generators named folder(s) not present under DeepFake/: "
                  f"{', '.join(sorted(unknown))}", file=sys.stderr)
            print(f"       present: {', '.join(sorted(found))}", file=sys.stderr)
        before = len(jobs)
        # Keep every non-DeepFake job (the Negative bucket has no generator).
        jobs = [j for j in jobs if j.generator is None or j.generator in wanted]
        kept = {j.generator for j in jobs if j.generator}
        print(f"[info] --generators: kept {len(jobs)}/{before} jobs across "
              f"{len(kept)} generator(s): {', '.join(sorted(kept))}")

    if args.limit > 0:
        jobs = jobs[:args.limit]
    if not jobs:
        print("[error] no images found.", file=sys.stderr)
        return 1

    if args.models_config:
        from model_registry import load_models          # local import: optional dependency
        registry = load_models(args.models_config)
    else:
        registry = list(MODELS)
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

    missing  = sorted({m.key_env for m in selected if not os.getenv(m.key_env)})
    if missing:
        print(f"[error] missing env var(s): {', '.join(missing)}", file=sys.stderr)
        return 2

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

    run(
        jobs=jobs,
        models=selected,
        output_dir=Path(args.output).expanduser().resolve(),
        concurrency=max(1, args.concurrency),
        resume=not args.no_resume,
        single_turn=args.single_turn,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
