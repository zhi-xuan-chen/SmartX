"""Regenerate results/leaderboard_best_ckpt.csv by scanning the results dir.

Rules (encoded in METHODS below):
  - baselines (ARK / BiomedCLIP / CheSS / CheXzero / GLoRIA / MaCO): single ckpt
    -> read metrics from <eval_dir>/eval/main.csv
  - RadDINO: force checkpoint-18500 when it exists (per request); else single
  - topk_*: scan <eval_dir>/checkpoint-*/eval/main.csv directly (more complete
    and always up-to-date vs the cached checkpoint_comparison.csv); pick best
    by *composite* score — mean rank (1=best, descending) over
        BLEU-4, ROUGE-L, Micro-F1-14+, Macro-F1-14+, F1-RadGraph
    -> tie-break by F1-RadGraph (higher wins).

Output columns: method, dataset, best_ckpt, BLEU-4, ROUGE-L,
                Micro-F1-14+, Macro-F1-14+, F1-RadGraph
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd


COMPOSITE_METRICS = ["BLEU-4", "ROUGE-L", "Micro-F1-14+", "Macro-F1-14+", "F1-RadGraph"]
OUTPUT_METRICS = COMPOSITE_METRICS  # same set goes into the leaderboard csv
TIEBREAK_METRIC = "F1-RadGraph"

# (display_name, folder_key, strategy)
# strategy:
#   "single"                -> root test_0.jsonl + root eval/main.csv
#   "topk"                  -> rank ckpts in checkpoint_comparison.csv
#   {dataset: strategy_str} -> per-dataset override; entries can be
#                              "single" / "topk" / "checkpoint-XXXXX"
METHODS = [
    ("ARK Base",       "ark_base",          "single"),
    ("ARK Large",      "ark_large",         "single"),
    ("BiomedCLIP",     "biomedclip",        "single"),
    ("CheSS",          "chess",             "single"),
    ("CheXzero",       "chexzero",          "single"),
    ("GLoRIA",         "gloria",            "single"),
    ("MaCO",           "maco",              "single"),
    ("RadDINO",        "rad_dino",          {"mimic": "checkpoint-18500", "iu": "single"}),
    ("topk_v4 (AR)",   "topk_v4_last_ar",   "topk"),
    ("topk_v5 (AR)",   "topk_v5_last_ar",   "topk"),
    ("topk_v14 (AR)",  "topk_v14_last_ar",  "topk"),
    ("topk_v19 (AR)",  "topk_v19_last_ar",  "topk"),
    ("topk_v20 (AR)",  "topk_v20_last_ar",  "topk"),
    ("topk_v21 (AR)",  "topk_v21_last_ar",  "topk"),
    ("topk_v23 (AR)",  "topk_v23_last_ar",  "topk"),
    ("topk_v109 (AR)", "topk_v109_last_ar", "topk"),
    ("topk_v109 (LP)", "topk_v109_last_lp", "topk"),
]

DATASETS = ["iu", "mimic", "chexpert"]


def eval_dir(results_dir: str, key: str, dataset: str) -> str:
    return os.path.join(results_dir, f"llavarad_{key}_{dataset}_eval")


def read_metrics_from_main(eval_main_csv: str) -> dict[str, float]:
    if not os.path.exists(eval_main_csv):
        return {}
    df = pd.read_csv(eval_main_csv, index_col=0)
    if "median" not in df.index:
        return {}
    row = df.loc["median"]
    return {m: float(row[m]) for m in OUTPUT_METRICS if m in row.index}


def scan_topk_ckpts(base_dir: str) -> pd.DataFrame:
    """Walk <base_dir>/checkpoint-*/eval/main.csv and collect median metrics."""
    rows = []
    for ckpt_dir in sorted(glob.glob(os.path.join(base_dir, "checkpoint-*"))):
        if not os.path.isdir(ckpt_dir):
            continue
        metrics = read_metrics_from_main(os.path.join(ckpt_dir, "eval", "main.csv"))
        if not metrics:
            continue
        rows.append({"checkpoint": os.path.basename(ckpt_dir), **metrics})
    return pd.DataFrame(rows)


def pick_topk_best_ckpt(base_dir: str) -> str | None:
    """Mean-rank across COMPOSITE_METRICS, lower = better; tie-break by TIEBREAK_METRIC."""
    df = scan_topk_ckpts(base_dir)
    if df.empty:
        return None
    have = [m for m in COMPOSITE_METRICS if m in df.columns]
    if not have:
        return None
    ranks = df[have].rank(ascending=False, method="min")
    df = df.assign(_mean_rank=ranks.mean(axis=1))
    sort_cols = ["_mean_rank"]
    sort_asc = [True]
    if TIEBREAK_METRIC in df.columns:
        sort_cols.append(TIEBREAK_METRIC)
        sort_asc.append(False)
    df = df.sort_values(by=sort_cols, ascending=sort_asc)
    return str(df.iloc[0]["checkpoint"])


def resolve(strategy, dataset, base_dir):
    """Return (best_ckpt_label, eval_main_csv_path)."""
    s = strategy.get(dataset, "single") if isinstance(strategy, dict) else strategy

    if s == "single":
        return "single", os.path.join(base_dir, "eval", "main.csv")

    if s == "topk":
        best = pick_topk_best_ckpt(base_dir)
        if best is None:
            return None, None
        return best, os.path.join(base_dir, best, "eval", "main.csv")

    if s.startswith("checkpoint-"):
        ckpt_eval = os.path.join(base_dir, s, "eval", "main.csv")
        if os.path.exists(ckpt_eval):
            return s, ckpt_eval
        # Fall back to single if the explicit ckpt isn't present.
        fallback = os.path.join(base_dir, "eval", "main.csv")
        if os.path.exists(fallback):
            print(f"[build_leaderboard] WARN: {base_dir}/{s} missing, "
                  f"falling back to single", file=sys.stderr)
            return "single", fallback
        return None, None

    raise ValueError(f"unknown strategy: {s!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for method, key, strategy in METHODS:
        for dataset in DATASETS:
            base = eval_dir(args.results_dir, key, dataset)
            if not os.path.isdir(base):
                print(f"[build_leaderboard] skip {method}/{dataset}: dir missing", file=sys.stderr)
                continue
            ckpt, eval_main = resolve(strategy, dataset, base)
            if ckpt is None:
                print(f"[build_leaderboard] skip {method}/{dataset}: no usable ckpt", file=sys.stderr)
                continue
            metrics = read_metrics_from_main(eval_main)
            if not metrics:
                print(f"[build_leaderboard] WARN: {method}/{dataset} ckpt={ckpt} "
                      f"no metrics in {eval_main}", file=sys.stderr)
            rows.append({
                "method": method,
                "dataset": dataset,
                "best_ckpt": ckpt,
                **{m: metrics.get(m, "") for m in OUTPUT_METRICS},
            })

    df = pd.DataFrame(rows, columns=["method", "dataset", "best_ckpt"] + OUTPUT_METRICS)
    df = df.sort_values(by=["method", "dataset"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[build_leaderboard] saved {len(df)} rows -> {args.out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
