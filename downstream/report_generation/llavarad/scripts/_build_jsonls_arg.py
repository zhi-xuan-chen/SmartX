"""Helper for analyze_all.sh — turn leaderboard rows into a --jsonls arg.

Outputs two lines on stdout (or just one, controlled by --print):
    line 1: <model_key>:<jsonl_path>,<model_key>:<jsonl_path>,...
    line 2: <reference_model_key>

If --reference_model is not present in the leaderboard, fall back to picking
its best checkpoint from <results_dir>/llavarad_<key>_<dataset>_eval/checkpoint_comparison.csv
by --fallback_metric (default F1-RadGraph).
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys


# Display name (leaderboard `method` column) -> folder key under results/.
METHOD_TO_KEY = {
    "ARK Base":       "ark_base",
    "ARK Large":      "ark_large",
    "BiomedCLIP":     "biomedclip",
    "CheSS":          "chess",
    "CheXzero":       "chexzero",
    "GLoRIA":         "gloria",
    "MaCO":           "maco",
    "RadDINO":        "rad_dino",
    "topk_v4 (AR)":   "topk_v4_last_ar",
    "topk_v5 (AR)":   "topk_v5_last_ar",
    "topk_v14 (AR)":  "topk_v14_last_ar",
    "topk_v19 (AR)":  "topk_v19_last_ar",
    "topk_v20 (AR)":  "topk_v20_last_ar",
    "topk_v21 (AR)":  "topk_v21_last_ar",
    "topk_v23 (AR)":  "topk_v23_last_ar",
    "topk_v109 (AR)": "topk_v109_last_ar",
    "topk_v109 (LP)": "topk_v109_last_lp",
}


def folder_for(key: str, dataset: str) -> str:
    return f"llavarad_{key}_{dataset}_eval"


def eval_dir_path(results_dir: str, key: str, dataset: str) -> str:
    return os.path.join(results_dir, folder_for(key, dataset))


def jsonl_path(results_dir: str, key: str, dataset: str, ckpt: str) -> str:
    base = eval_dir_path(results_dir, key, dataset)
    if ckpt == "single":
        return os.path.join(base, "test_0.jsonl")
    return os.path.join(base, ckpt, "test_0.jsonl")


def eval_main_path(results_dir: str, key: str, dataset: str, ckpt: str) -> str:
    base = eval_dir_path(results_dir, key, dataset)
    if ckpt == "single":
        return os.path.join(base, "eval", "main.csv")
    return os.path.join(base, ckpt, "eval", "main.csv")


def parse_topk_versions(raw: str) -> set[str] | None:
    """Parse 'v19 v14,v21' -> {'v19','v14','v21'}; empty string -> None (no filter)."""
    if not raw.strip():
        return None
    parts = re.split(r"[\s,]+", raw.strip())
    return {p for p in parts if p}


def topk_version(method: str) -> str | None:
    m = re.match(r"topk_(v\d+)", method)
    return m.group(1) if m else None


def read_leaderboard(path: str, dataset: str) -> dict[str, str]:
    """Return {method_display_name: best_ckpt} for the given dataset."""
    out: dict[str, str] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["dataset"] != dataset:
                continue
            out[row["method"]] = row["best_ckpt"]
    return out


def pick_best_ckpt_from_comparison(
    results_dir: str, key: str, dataset: str, metric: str,
) -> str | None:
    comp = os.path.join(results_dir, folder_for(key, dataset), "checkpoint_comparison.csv")
    if not os.path.exists(comp):
        return None
    best_ckpt, best_val = None, -float("inf")
    with open(comp, newline="") as f:
        for row in csv.DictReader(f):
            try:
                v = float(row[metric])
            except (KeyError, ValueError):
                continue
            if v > best_val:
                best_val = v
                best_ckpt = row["checkpoint"]
    return best_ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leaderboard", required=True)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--reference_model", required=True,
                    help="Display name as it appears in leaderboard `method` column.")
    ap.add_argument(
        "--topk_versions", default="",
        help="Comma/space-separated topk versions to keep (e.g. v19,v14,v21,v23). "
             "Other topk_* methods are excluded; non-topk baselines are always kept. "
             "Empty = include all topk.",
    )
    ap.add_argument("--fallback_metric", default="F1-RadGraph")
    ap.add_argument("--print", choices=["jsonls", "ref_key", "both"], default="both")
    args = ap.parse_args()

    lb = read_leaderboard(args.leaderboard, args.dataset)
    allowed_topk = parse_topk_versions(args.topk_versions)

    # Make sure the reference model is in the candidate set.
    if args.reference_model not in lb:
        ref_key = METHOD_TO_KEY.get(args.reference_model)
        if ref_key is None:
            sys.exit(f"reference_model {args.reference_model!r} not in METHOD_TO_KEY")
        ckpt = pick_best_ckpt_from_comparison(
            args.results_dir, ref_key, args.dataset, args.fallback_metric,
        )
        if ckpt is None:
            sys.exit(
                f"reference_model {args.reference_model!r} not in leaderboard for "
                f"dataset={args.dataset!r} and no checkpoint_comparison.csv fallback"
            )
        print(
            f"[build_jsonls_arg] fallback: {args.reference_model} "
            f"({args.dataset}) -> {ckpt} (by {args.fallback_metric})",
            file=sys.stderr,
        )
        lb[args.reference_model] = ckpt

    # Build entries; skip missing paths with a warning.
    entries: list[str] = []
    ref_key_out = ""
    for method, ckpt in lb.items():
        ver = topk_version(method)
        if ver is not None and allowed_topk is not None and ver not in allowed_topk:
            print(
                f"[build_jsonls_arg] skip topk {method!r} (not in --topk_versions)",
                file=sys.stderr,
            )
            continue
        key = METHOD_TO_KEY.get(method)
        if key is None:
            print(f"[build_jsonls_arg] WARN: no key mapping for {method!r}; skipping", file=sys.stderr)
            continue
        path = jsonl_path(args.results_dir, key, args.dataset, ckpt)
        if not os.path.exists(path):
            print(f"[build_jsonls_arg] WARN: missing {path}; skipping", file=sys.stderr)
            continue
        entries.append(f"{key}:{path}")
        if method == args.reference_model:
            ref_key_out = key

    if not ref_key_out:
        sys.exit(f"reference_model {args.reference_model!r} jsonl not found on disk")

    jsonls_arg = ",".join(entries)
    if args.print == "jsonls":
        print(jsonls_arg)
    elif args.print == "ref_key":
        print(ref_key_out)
    else:
        print(jsonls_arg)
        print(ref_key_out)


if __name__ == "__main__":
    main()
