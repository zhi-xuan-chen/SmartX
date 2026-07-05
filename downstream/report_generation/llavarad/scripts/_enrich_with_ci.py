"""Enrich pairwise_significance_all.csv with per-model point estimate + 95% CI.

Reads the consolidated ``metrics_ci_all.csv`` (produced by
``analyze_significance.py``, recomputed from the prediction jsonls + cached
CheXBERT/RadGraph labels — it NEVER reads any per-folder ``main.csv``) and joins
the point/CI columns onto ``model_a`` / ``model_b`` of the pairwise table.

Because the metric-CI table and the pairwise ``score_a`` / ``score_b`` are
computed by the same code on the same ``common_ids``, ``score_a`` and the joined
``score_a_point`` should be identical. A divergence beyond ``--tol`` is flagged
(it would mean a genuine bug — wrong model/metric mapping, scale error, etc.).
"""

from __future__ import annotations

import argparse
import os

import pandas as pd


def load_ci_lookup(metrics_ci_csv: str) -> dict[tuple, dict]:
    """Return {(dataset, model, metric): {point, ci_l, ci_h}}."""
    df = pd.read_csv(metrics_ci_csv)
    lookup: dict[tuple, dict] = {}
    for _, r in df.iterrows():
        lookup[(r["dataset"], r["model"], r["metric"])] = {
            "point": r["point"], "ci_l": r["ci_l"], "ci_h": r["ci_h"],
        }
    return lookup


def _attach(df: pd.DataFrame, lookup: dict[tuple, dict], model_col: str, prefix: str) -> pd.DataFrame:
    pts, los, his = [], [], []
    for _, row in df.iterrows():
        v = lookup.get((row["dataset"], row[model_col], row["metric"]))
        if v is None:
            pts.append(pd.NA); los.append(pd.NA); his.append(pd.NA)
        else:
            pts.append(v["point"]); los.append(v["ci_l"]); his.append(v["ci_h"])
    df[f"{prefix}_point"] = pts
    df[f"{prefix}_ci_l"] = los
    df[f"{prefix}_ci_h"] = his
    return df


def _consistency_check(df: pd.DataFrame, tol: float) -> None:
    """Warn when the pairwise score and the joined point estimate disagree."""
    for score_col, point_col, model_col in (
        ("score_a", "score_a_point", "model_a"),
        ("score_b", "score_b_point", "model_b"),
    ):
        if score_col not in df.columns or point_col not in df.columns:
            continue
        a = pd.to_numeric(df[score_col], errors="coerce")
        b = pd.to_numeric(df[point_col], errors="coerce")
        diff = (a - b).abs()
        bad = df[diff > tol]
        for _, row in bad.iterrows():
            print(
                f"[enrich_ci][WARN] {row['dataset']}/{row[model_col]}/{row['metric']}: "
                f"{score_col}={row[score_col]} vs {point_col}={row[point_col]} "
                f"(|diff|>{tol})"
            )


def enrich_pairwise(pw: pd.DataFrame, lookup: dict[tuple, dict], tol: float = 1e-6) -> pd.DataFrame:
    pw = pw.copy()
    pw = _attach(pw, lookup, "model_a", "score_a")
    pw = _attach(pw, lookup, "model_b", "score_b")

    _consistency_check(pw, tol)

    # Place the new columns right after score_a / score_b for readability.
    cols = list(pw.columns)
    for anchor, new_cols in (
        ("score_a", ["score_a_point", "score_a_ci_l", "score_a_ci_h"]),
        ("score_b", ["score_b_point", "score_b_ci_l", "score_b_ci_h"]),
    ):
        for c in new_cols:
            if c in cols:
                cols.remove(c)
        idx = cols.index(anchor) + 1
        for c in reversed(new_cols):
            cols.insert(idx, c)
    return pw[cols]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--metrics_ci_name", default="metrics_ci_all.csv")
    ap.add_argument("--pairwise_name", default="pairwise_significance_all.csv")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="Max |score - point| before flagging an inconsistency.")
    args = ap.parse_args()

    metrics_ci_path = os.path.join(args.output_root, args.metrics_ci_name)
    pairwise_path = os.path.join(args.output_root, args.pairwise_name)

    if not os.path.exists(metrics_ci_path):
        print(f"[enrich_ci] missing {metrics_ci_path}; abort")
        return
    if not os.path.exists(pairwise_path):
        print(f"[enrich_ci] missing {pairwise_path}; abort")
        return

    lookup = load_ci_lookup(metrics_ci_path)
    if not lookup:
        print("[enrich_ci] empty metrics_ci table; abort")
        return

    pw = pd.read_csv(pairwise_path)
    if pw.empty:
        print(f"[enrich_ci] pairwise file empty: {pairwise_path}")
        return

    pw = enrich_pairwise(pw, lookup, tol=args.tol)
    pw.to_csv(pairwise_path, index=False)
    print(f"[enrich_ci] enriched {len(pw)} rows -> {pairwise_path}")


if __name__ == "__main__":
    main()
