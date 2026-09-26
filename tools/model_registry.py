#!/usr/bin/env python3
"""
model_registry.py — external, editable detector registry.

detect.py hard-codes 10 of the detectors used for the paper.  This module lets a
JSON file add new detectors (or override existing ones) WITHOUT touching
detect.py, so the published results stay reproducible.

Config format (config/models.json):

    {
      "detectors": [
        {
          "name":      "gpt-6",              // key used in output filenames
          "provider":  "openai",             // openai | dashscope | xai | gemini
          "model_id":  "gpt-6",              // exact API model id
          "key_env":   "OPENAI_API_KEY",
          "endpoint":  "openai",             // alias or full base URL; optional
          "stream":    false,
          "extra_body": {"reasoning_effort": "medium"},
          "gemini_thinking_level": null,     // gemini only
          "disabled":  false
        }
      ]
    }

Any field except `name` may be omitted.  An entry whose `name` already exists
in detect.MODELS replaces it in place; a new `name` is appended.
`"disabled": true` removes a detector from the registry.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import detect as D

ENDPOINT_ALIASES = {
    "dashscope":  D.DASHSCOPE_BASE,
    # Bailian workspace endpoints are per-key: each API key is bound to its own
    # host, so a DashScope entry must name the slot matching its key_env.
    "dashscope1": D.dashscope_base(1),
    "dashscope2": D.dashscope_base(2),
    "dashscope3": D.dashscope_base(3),
    "xai":        D.XAI_BASE,
    "openai":     D.OPENAI_BASE,
    "gemini":     D.GEMINI_BASE,
}

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env(value: str, *, where: str) -> str:
    """Expand ${VAR} references so hosts/keys can live outside the repo."""
    missing: list[str] = []

    def sub(m: re.Match) -> str:
        val = os.getenv(m.group(1))
        if val is None:
            missing.append(m.group(1))
            return m.group(0)
        return val

    out = _ENV_REF.sub(sub, value)
    if missing:
        raise ValueError(f"{where}: environment variable(s) not set: {', '.join(missing)}")
    return out

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "models.json"


def _resolve_endpoint(entry: dict[str, Any]) -> str:
    ep = entry.get("endpoint") or entry.get("provider", "")
    if ep in ENDPOINT_ALIASES:
        return ENDPOINT_ALIASES[ep]
    return _expand_env(ep, where=f"detector {entry.get('name')!r} endpoint")


def _to_cfg(entry: dict[str, Any]) -> D.ModelCfg:
    name = entry.get("name")
    if not name:
        raise ValueError(f"registry entry missing 'name': {entry}")
    provider = entry.get("provider")
    if provider not in ("openai", "dashscope", "xai", "gemini"):
        raise ValueError(f"{name}: provider must be openai|dashscope|xai|gemini, got {provider!r}")
    return D.ModelCfg(
        name=name,
        provider=provider,
        model_id=entry.get("model_id") or name,
        key_env=entry.get("key_env") or "",
        endpoint=_resolve_endpoint(entry),
        stream=bool(entry.get("stream", False)),
        extra_body=dict(entry.get("extra_body") or {}),
        gemini_thinking_level=entry.get("gemini_thinking_level"),
    )


def load_models(config_path: str | Path | None = None,
                *, base: list[D.ModelCfg] | None = None) -> list[D.ModelCfg]:
    """Return detect.MODELS merged with the JSON registry at `config_path`."""
    models = list(D.MODELS if base is None else base)
    if config_path is None:
        return models

    path = Path(config_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"models config not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("detectors", []) if isinstance(data, dict) else data

    index = {m.name: i for i, m in enumerate(models)}
    for entry in entries:
        name = entry.get("name")
        if entry.get("disabled"):
            if name in index:
                models[index[name]] = None          # type: ignore[call-overload]
            continue
        cfg = _to_cfg(entry)
        if name in index:
            models[index[name]] = cfg
        else:
            index[name] = len(models)
            models.append(cfg)

    return [m for m in models if m is not None]
