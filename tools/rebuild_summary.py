#!/usr/bin/env python3
"""
rebuild_summary.py — regenerate summary.json from the per-model result files.

summary.json is only a roll-up: every verdict it holds also lives in
<bucket>/<model>.json, which detect.py writes one record at a time. So a
summary that was lost, truncated or written by a buggy merge can always be
rebuilt from those files without re-running a single API call.

Usage
-----
    python tools/rebuild_summary.py <results-dir> [...]        # rebuild in place
    python tools/rebuild_summary.py --all                      # every Results/<mode>/
    python tools/rebuild_summary.py --all --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect import atomic_write_json, now_iso  # noqa: E402

DATASET = Path(__file__).resolve().parent.parent.parent.parent


def rebuild(results_dir: Path) -> dict:
    """Read every <bucket>/<model>.json under results_dir and rebuild the roll-up."""
    rows: dict[str, dict] = {}
    models: set[str] = set()
    buckets: set[str] = set()
    any_review = False

    for f in sorted(results_dir.rglob("*.json")):
        if f.name == "summary.json":
            continue
        try:
            bundle = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        model = bundle.get("model") or f.stem
        bucket = bundle.get("bucket") or str(f.parent.relative_to(results_dir))
        models.add(model)
        buckets.add(bucket)

        for r in bundle.get("results", []):
            if not isinstance(r, dict):
                continue
            ident = r.get("review_id") or r.get("image_rel") or r.get("key") or ""
            if not ident:
                continue
            key = f"{bucket}::{ident}"
            row = rows.get(key)
            if row is None:
                row = {
                    "key": r.get("key") or r.get("image_path") or "",
                    "bucket": bucket,
                    "label": r.get("label"),
                    "generator": r.get("generator"),
                    "num_images": r.get("num_images") or 1,
                    "verdicts": {},
                }
                if r.get("review_id"):
                    any_review = True
                    row["review_id"] = r["review_id"]
                    row["review_path"] = r.get("review_path")
                    row["image_rels"] = r.get("image_rels")
                else:
                    row["image_path"] = r.get("image_path")
                    row["image_rel"] = r.get("image_rel")
                rows[key] = row
            row["verdicts"][model] = {
                "status": "ok" if r.get("error") is None else "error",
                "is_ai_modified": r.get("is_ai_modified"),
                "confidence": r.get("confidence"),
                "reason": r.get("reason"),
                "error": r.get("error"),
            }

    # A model that never scored a given row is reported as missing, matching
    # what build_summary() produces on a fresh run.
    for row in rows.values():
        for m in models:
            row["verdicts"].setdefault(m, {"status": "missing"})

    out = list(rows.values())
    return {
        "generated_at": now_iso(),
        "mode": "review" if any_review else "image",
        "num_jobs": len(out),
        "num_images": sum(int(r.get("num_images") or 1) for r in out),
        "buckets": sorted(buckets),
        "models": sorted(models),
        "rows": out,
        "rebuilt_from": "per-model result files",
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="*", type=Path)
    p.add_argument("--all", action="store_true",
                   help="Rebuild every <Category>/Results/<mode>/ under the dataset root.")
    p.add_argument("--dataset", type=Path, default=DATASET)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv if argv is not None else sys.argv[1:])

    targets = list(a.dirs)
    if a.all:
        targets += sorted(
            d for cat in a.dataset.iterdir() if (cat / "Results").is_dir()
            for d in (cat / "Results").iterdir() if d.is_dir()
        )
    if not targets:
        p.error("give one or more results dirs, or --all")

    for d in targets:
        if not d.is_dir():
            print(f"[skip] not a directory: {d}", file=sys.stderr)
            continue
        s = rebuild(d)
        path = d / "summary.json"
        was = "-"
        if path.exists():
            try:
                was = str(len(json.loads(path.read_text(encoding="utf-8")).get("rows", [])))
            except (json.JSONDecodeError, OSError):
                pass
        rel = d.relative_to(a.dataset) if str(d).startswith(str(a.dataset)) else d
        print(f"  {str(rel):<56} rows {was:>5} -> {s['num_jobs']:<5} "
              f"buckets={len(s['buckets'])} models={len(s['models'])}")
        if not a.dry_run:
            atomic_write_json(path, s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
