#!/usr/bin/env python3
"""
read_results.py — Load detection results into a pandas DataFrame.

Reads summary.json files from every category × mode combination and
flattens per-model verdicts into columns.

Adaptation issue
----------------
Hotel and GrabFood results were generated on a different machine, so their
`key` / `image_path` / `review_path` fields contain stale absolute paths:
  Hotel    → /path/to/raw-data/Trip/Final/...
  GrabFood → /path/to/raw-data/Grab/Final/...
This script ignores those stale paths and derives review_id reliably from
the portable `image_rel` / `image_rels` / `review_id` fields instead.

Output DataFrame columns
------------------------
  platform      amazon | hotel | grabfood
  category      top-level folder (e.g. "All Beauty", "Hotels & Accommodations")
  mode          SingleImage-NoReview | SingleImage-withReview |
                MultiStep-NoReview   | MultiStep-withReview  |
                MultiImage-NoReview  | MultiImage-withReview |
                MultiTurn-NoReview   | MultiTurn-withReview
  bucket        Negative | DeepFake/gpt-image-2 | ...
  label         Negative | DeepFake | None
  generator     None | gpt-image-2 | ...
  review_id     Review_001 | ...
  image_rel     (single-image modes) e.g. Review_001/Image_001_01.jpg
  num_images    int
  <model>_ok           bool   — True if status == "ok" and no error
  <model>_verdict      bool | None  — is_ai_modified
  <model>_confidence   float | None

Usage (module)
--------------
  from read_results import load_dataframe
  df = load_dataframe()
  df = load_dataframe(platforms=["hotel", "grabfood"])
  df = load_dataframe(modes=["SingleImage-NoReview"])

Usage (CLI)
-----------
  python read_results.py                         # print shape + head
  python read_results.py --output results.csv    # save to CSV
  python read_results.py --platform hotel --mode SingleImage-NoReview
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

# ── Dataset root ──────────────────────────────────────────────────────────────
# Walk up from this file until a directory that holds the category folders
# (each with a Results/ subfolder) is found; otherwise fall back to the parent.
_HERE = Path(__file__).resolve().parent


def _find_dataset_root(start: Path) -> Path:
    for cand in (start, *start.parents):
        if any((d / "Results").is_dir() for d in cand.iterdir() if d.is_dir()):
            return cand
    return start.parent


DATASET = _find_dataset_root(_HERE)

# ── Platform → category dirs ──────────────────────────────────────────────────
AMAZON_DIRS = [
    "All Beauty", "Amazon Fashion", "Appliances", "Arts, Crafts & Sewing",
    "Automotive", "Baby Products", "Beauty & Personal Care", "Books",
    "CDs & Vinyl", "Cell Phones & Accessories", "Clothing, Shoes & Jewelry",
    "Electronics", "Grocery & Gourmet Food", "Handmade Products",
    "Health & Household", "Health & Personal Care", "Home & Kitchen",
    "Industrial & Scientific", "Magazine Subscriptions", "Musical Instruments",
    "Office Products", "Patio, Lawn & Garden", "Pet Supplies",
    "Sports & Outdoors", "Tools & Home Improvement", "Toys & Games",
    "Video Games",
]
HOTEL_DIRS    = ["Hotels & Accommodations"]
GRABFOOD_DIRS = ["Delivery, Pickup & Dine-Out"]

MODES = [
    "SingleImage-NoReview",
    "SingleImage-withReview",
    "MultiStep-NoReview",
    "MultiStep-withReview",
    "MultiImage-NoReview",
    "MultiImage-withReview",
    "MultiTurn-NoReview",
    "MultiTurn-withReview",
]


# ── review_id extraction ──────────────────────────────────────────────────────
def _review_id_from_row(row: dict) -> str:
    """Extract review_id reliably without touching stale absolute paths."""
    # review mode: field is already present
    if row.get("review_id"):
        return str(row["review_id"])
    # single-image mode: first component of image_rel
    rel = row.get("image_rel") or (
        row["image_rels"][0] if row.get("image_rels") else ""
    )
    if rel:
        return Path(rel).parts[0]  # "Review_001/Image_001_01.jpg" → "Review_001"
    return ""


# ── Core iterator ─────────────────────────────────────────────────────────────
def iter_rows(
    dataset_root: Path = DATASET,
    *,
    platforms: list[str] | None = None,
    modes:     list[str] | None = None,
) -> Iterator[dict]:
    """Yield one flat dict per (job, mode, category) from every summary.json."""
    want_platforms = set(platforms or ["amazon", "hotel", "grabfood"])
    want_modes     = set(modes or MODES)

    dir_map: list[tuple[str, str]] = (
        [(d, "amazon")   for d in AMAZON_DIRS]
        + [(d, "hotel")    for d in HOTEL_DIRS]
        + [(d, "grabfood") for d in GRABFOOD_DIRS]
    )

    for category, platform in dir_map:
        if platform not in want_platforms:
            continue
        results_root = dataset_root / category / "Results"
        if not results_root.is_dir():
            continue

        for mode in MODES:
            if mode not in want_modes:
                continue
            summary_path = results_root / mode / "summary.json"
            if not summary_path.exists():
                continue

            try:
                with summary_path.open(encoding="utf-8") as f:
                    summary = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"[warn] {summary_path}: {e}", file=sys.stderr)
                continue

            models = summary.get("models") or []

            for row in summary.get("rows", []):
                flat: dict = {
                    "platform":  platform,
                    "category":  category,
                    "mode":      mode,
                    "bucket":    row.get("bucket", ""),
                    "label":     row.get("label"),
                    "generator": row.get("generator"),
                    "review_id": _review_id_from_row(row),
                    "image_rel": row.get("image_rel", ""),
                    "num_images": row.get("num_images", 0),
                }

                verdicts = row.get("verdicts") or {}
                for model in models:
                    v = verdicts.get(model) or {}
                    # `_ran` separates "this model was never given this image"
                    # from "it ran and failed". Both leave `_ok` False, but only
                    # the second belongs in a denominator: rosters evaluated at
                    # different times share one summary, so a row can predate a
                    # model entirely.
                    flat[f"{model}_ran"]        = bool(v) and v.get("status") != "missing"
                    flat[f"{model}_ok"]         = v.get("status") == "ok" and v.get("error") is None
                    flat[f"{model}_verdict"]    = v.get("is_ai_modified")
                    flat[f"{model}_confidence"] = v.get("confidence")

                yield flat


# ── Public API ────────────────────────────────────────────────────────────────
def load_dataframe(
    dataset_root: Path | str = DATASET,
    *,
    platforms: list[str] | None = None,
    modes:     list[str] | None = None,
):
    """Return a pandas DataFrame of all detection results.

    Parameters
    ----------
    dataset_root : FraudBench root directory (auto-detected by default)
    platforms    : subset of ["amazon", "hotel", "grabfood"]  (default: all)
    modes        : subset of MODES list above                 (default: all 4)
    """
    if not _HAS_PANDAS:
        raise ImportError("pandas is required: pip install pandas")

    rows = list(iter_rows(Path(dataset_root), platforms=platforms, modes=modes))
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["num_images"] = df["num_images"].astype(int)

    # Cast per-model columns to appropriate types
    for col in df.columns:
        if col.endswith("_confidence"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif col.endswith("_ok") or col.endswith("_ran"):
            df[col] = df[col].fillna(False).astype(bool)

    return df


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--platform", choices=["amazon", "hotel", "grabfood"],
                   action="append", dest="platforms", metavar="PLATFORM")
    p.add_argument("--mode", choices=MODES,
                   action="append", dest="modes", metavar="MODE")
    p.add_argument("--output", metavar="FILE",
                   help="Save to CSV (e.g. results.csv).")
    p.add_argument("--dataset", default=str(DATASET))
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    root = Path(args.dataset).expanduser().resolve()

    if not _HAS_PANDAS:
        # Fallback: plain-text summary without pandas
        total = 0
        for row in iter_rows(root, platforms=args.platforms, modes=args.modes):
            print(row)
            total += 1
        print(f"\nTotal rows: {total}", file=sys.stderr)
        return

    df = load_dataframe(root, platforms=args.platforms, modes=args.modes)

    if args.output:
        out = Path(args.output)
        df.to_csv(out, index=False, encoding="utf-8")
        print(f"Saved {len(df)} rows → {out}", file=sys.stderr)
        return

    print(f"Shape: {df.shape}")
    print()
    print("Rows by platform / mode:")
    print(df.groupby(["platform", "mode"]).size().to_string())
    print()
    print("Rows by platform / label:")
    print(df.groupby(["platform", "label"]).size().to_string())
    print()
    model_cols = [c for c in df.columns if c.endswith("_verdict")]
    print(f"Models detected: {[c[:-8] for c in model_cols]}")
    print()
    print(df[["platform", "category", "mode", "label", "generator", "review_id", "num_images"]].head(10).to_string())


if __name__ == "__main__":
    main()
