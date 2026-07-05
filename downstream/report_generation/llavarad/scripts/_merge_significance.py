"""Concatenate per-dataset significance CSVs into one file with a `dataset` column.

Supports:
  - flat layout:  <output_root>/pairwise_significance_<dataset>.csv
  - legacy layout: <output_root>/<dataset>/pairwise_significance.csv
"""

from __future__ import annotations

import argparse
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", required=True,
                    help="Dir that contains <dataset>/pairwise_significance.csv subdirs.")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--out_name", default="pairwise_significance_all.csv")
    args = ap.parse_args()

    dfs: list[pd.DataFrame] = []
    for d in args.datasets:
        candidates = [
            os.path.join(args.output_root, f"pairwise_significance_{d}.csv"),
            os.path.join(args.output_root, d, "pairwise_significance.csv"),
        ]
        p = next((c for c in candidates if os.path.exists(c)), None)
        if p is None:
            print(f"[merge] skip missing {d} (tried: {', '.join(candidates)})")
            continue
        df = pd.read_csv(p)
        if "dataset" not in df.columns:
            df.insert(0, "dataset", d)
        dfs.append(df)

    if not dfs:
        print("[merge] nothing to merge")
        return

    out = pd.concat(dfs, ignore_index=True)
    out_path = os.path.join(args.output_root, args.out_name)
    out.to_csv(out_path, index=False)
    print(f"[merge] saved {len(out)} rows -> {out_path}")
    print(out.to_string())


if __name__ == "__main__":
    main()
