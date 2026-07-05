"""Standalone analysis for the pre-training ablation (MIL-private only).

Why a dedicated script (instead of analyze_all.sh):
  * the ablation checkpoints are split across disks (jhcnas3 + jhcnas5), which the
    single-BASE_DIR pipeline can't express;
  * we only need the MIL in-domain numbers for the ablation figure (panel a +
    panel b), not the external / DeLong / threshold-significance machinery.

Estimand (identical to bootstrap.py, so numbers are comparable to the main
results): per region the macro-mean over the kept diseases for each metric, then
an equal-weight average over the 10 regions. The macro class set (fixed_valid) is
frozen on each model/region's full sample. CIs come from a study-level bootstrap;
because every model is row-aligned within a region and shares the per-region
resample seed, the same draws also give *paired* differences vs the chosen config
(with BH-FDR over the model x metric family).

Outputs (CSV, into --output_dir):
  ablation_mil_mean.csv          one row per model: value + 95% CI for each metric
  ablation_mil_region.csv        per (model, region) point estimates (panel c)
  ablation_mil_significance.csv  diff vs chosen config: delta, p_value, p_value_fdr

Usage:
  python analyze_ablation.py                       # defaults below
  python analyze_ablation.py --n_bootstrap 1000 --n_workers 10
"""
import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy import stats

from utils import load_results, included_classes
from bootstrap import compute_all_metrics, METRICS

# Wilcoxon parameters taken verbatim from stat_test.paired_bootstrap_wilcoxon_test
# (the project default significance method) so the method label is identical.
_WILCOXON_KW = dict(zero_method="wilcox", alternative="two-sided")

# ---------------------------------------------------------------------------
# Configuration. Analysis is DECOUPLED from figure layout: here we only list the
# unique models to score and the paired comparisons to test. Panel/x/chosen
# placement lives in plots/codes/ablation/build_ablation_csv.py (a model such as
# r09 plays different roles in different panels, so layout can't live here).
# ---------------------------------------------------------------------------
NAS3 = "${SMARTX_OUTPUT_ROOT}/experiments/MIL_LP"
NAS5 = "${SMARTX_OUTPUT_ROOT}/experiments/MIL_LP"

# unique model -> (disk, human label)
MODELS = {
    "topk_a2_r04_last":   (NAS3, "rho=0.4"),
    "topk_v21_last":      (NAS5, "rho=0.8"),
    "topk_a2_r09_last":   (NAS3, "rho=0.9"),    # base model / curriculum-Full / sweep-chosen
    "topk_a2_r095_last":  (NAS3, "rho=0.95"),
    "topk_a2_r099_last":  (NAS3, "rho=0.99"),
    "topk_a2_r10_last":   (NAS3, "rho=1.0"),
    "topk_a1_no_s1_last":  (NAS3, "-Stage1"),
    "topk_a1_no_s12_last": (NAS3, "-Stage1+2"),
    "topk_a1_no_s3_last":  (NAS3, "-Stage3"),
    "topk_v19_last":       (NAS5, "Large (v19)"),
    "topk_a3_chest_only_last": (NAS3, "Chest-only"),
    "topk_a4_scale_25_last":   (NAS3, "25% data"),
}

# paired wilcoxon comparisons: (test_model, reference_model). diff = reference - test.
COMPARISONS = [
    ("topk_a2_r04_last",   "topk_a2_r09_last"),  # panel b: sweep vs chosen rho=0.9
    ("topk_v21_last",      "topk_a2_r09_last"),
    ("topk_a2_r095_last",  "topk_a2_r09_last"),
    ("topk_a2_r099_last",  "topk_a2_r09_last"),
    ("topk_a2_r10_last",   "topk_a2_r09_last"),
    ("topk_a1_no_s1_last",  "topk_a2_r09_last"),  # panel a: -Stage1 vs Full (=r09)
    ("topk_a1_no_s12_last", "topk_a2_r09_last"),  # panel a: -Stage1+2 vs Full (=r09)
    ("topk_a1_no_s3_last",  "topk_a2_r09_last"),  # panel a: -Stage3 vs Full (=r09)
    ("topk_a2_r09_last",   "topk_v19_last"),      # panel e: Base (r09) vs Large (v19)
    ("topk_a4_scale_25_last", "topk_a2_r09_last"),  # panel c: 25% data vs 100% (=r09)
    ("topk_a3_chest_only_last", "topk_a2_r09_last"),  # panel e: chest-only vs multi-region
]

# config dir whose name encodes (batch size, lr, mil aggregation)
DEFAULT_CFG = "bs4096_lr0.001_clam"

REGIONS = ["Abdominal", "Ankle", "Cervical_spine", "Chest", "Foot",
           "Knee", "Lumbar_spine", "Metacarpal", "Pelvis", "Wrist"]

# Stable per-region seed (independent of model) so the resample indices are the
# SAME across models within a region -> valid paired differences.
REGION_SEED = {r: i + 12345 for i, r in enumerate(REGIONS)}


def latest_csv(base, region, model, cfg):
    c = os.path.join(base, f"private_a_{region}", model, cfg)
    if not os.path.isdir(c):
        return None
    vs = [v for v in os.listdir(c) if v.startswith("version_")]
    if not vs:
        return None
    v = sorted(vs, key=lambda s: int(s.split("_")[1]))[-1]
    p = os.path.join(c, v, "test_results.csv")
    return p if os.path.isfile(p) else None


def _rm_worker(args):
    """One (region, model): returns region macro-means.
    point  : (n_metrics,) full-sample estimate
    boot   : (n_boot, n_metrics) study-level bootstrap draws
    """
    region, name, base, cfg, n_boot = args
    csv = latest_csv(base, region, name, cfg)
    if csv is None:
        return region, name, None, None
    y_t, y_p, y_b, _ = load_results(csv)
    keep = included_classes(f"private_a_{region}", y_t.shape[1])
    pt = compute_all_metrics(y_t, y_p, y_b)
    fv = {k: pt["valid"][k] & keep for k in METRICS}
    pt = compute_all_metrics(y_t, y_p, y_b, fixed_valid=fv)
    point = np.array([pt["mean"][k] for k in METRICS])

    n = y_t.shape[0]
    rng = np.random.RandomState(REGION_SEED[region])  # identical seq for every model
    boot = np.full((n_boot, len(METRICS)), np.nan)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        res = compute_all_metrics(y_t[idx], y_p[idx], y_b[idx], fixed_valid=fv)
        for ki, k in enumerate(METRICS):
            boot[b, ki] = res["mean"][k]
    return region, name, point, boot


def wilcoxon_p(diffs):
    """p-value mirrored verbatim from stat_test.paired_bootstrap_wilcoxon_test:
    Wilcoxon signed-rank on the bootstrap diff distribution (same scipy call /
    params), so the method label matches the main-results pipeline exactly. We
    apply it to the *paired macro-over-regions* diff distribution built in Pass 1
    (region-paired via the shared REGION_SEED), rather than calling the
    single-metric canonical wrapper 5x per comparison (mathematically identical,
    ~10x faster)."""
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return np.nan
    try:
        return float(stats.wilcoxon(d, **_WILCOXON_KW).pvalue)
    except ValueError:
        return np.nan


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values (same family handling as analyze_all)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * n / (rank + 1))
        adj[i] = prev
    return adj


def main():
    ap = argparse.ArgumentParser(description="Ablation-only MIL analysis (cross-disk).")
    ap.add_argument("--cfg", default=DEFAULT_CFG)
    ap.add_argument("--n_bootstrap", type=int, default=1000)
    ap.add_argument("--n_workers", type=int, default=10)
    ap.add_argument("--output_dir", default=os.path.join(os.path.dirname(__file__),
                                                         "..", "analysis_results", "ablation"))
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    names = list(MODELS)
    n_boot = args.n_bootstrap

    # collect per (region, model)
    point_sum = {nm: np.zeros(len(METRICS)) for nm in names}
    point_cnt = {nm: 0 for nm in names}
    boot_sum = {nm: np.zeros((n_boot, len(METRICS))) for nm in names}
    boot_cnt = {nm: 0 for nm in names}
    region_point: dict[str, dict[str, dict[str, float]]] = {nm: {} for nm in names}

    # ===== Pass 1: per-model value + 95% CI (marginal study-level bootstrap) =====
    jobs = [(r, nm, MODELS[nm][0], args.cfg, n_boot) for r in REGIONS for nm in names]
    with ProcessPoolExecutor(max_workers=args.n_workers) as exc:
        for f in as_completed([exc.submit(_rm_worker, j) for j in jobs]):
            region, nm, point, boot = f.result()
            if point is None:
                print(f"MISSING: {nm} / {region}")
                continue
            region_point[nm][region] = {k: float(point[ki]) for ki, k in enumerate(METRICS)}
            point_sum[nm] += point
            point_cnt[nm] += 1
            boot_sum[nm] += boot
            boot_cnt[nm] += 1

    point = {nm: point_sum[nm] / point_cnt[nm] for nm in names if point_cnt[nm]}
    boot = {nm: boot_sum[nm] / boot_cnt[nm] for nm in names if boot_cnt[nm]}

    rows = []
    for nm in names:
        if nm not in point:
            continue
        row = {"model": nm, "label": MODELS[nm][1]}
        for ki, k in enumerate(METRICS):
            row[f"{k}_value"] = point[nm][ki]
            row[f"{k}_ci_lo"] = np.percentile(boot[nm][:, ki], 2.5)
            row[f"{k}_ci_hi"] = np.percentile(boot[nm][:, ki], 97.5)
        rows.append(row)
    mean_df = pd.DataFrame(rows)
    mean_path = os.path.join(args.output_dir, "ablation_mil_mean.csv")
    mean_df.to_csv(mean_path, index=False)

    region_rows = []
    for nm in names:
        for region, mv in region_point[nm].items():
            row = {"model": nm, "region": region, "label": MODELS[nm][1]}
            for k in METRICS:
                row[f"{k}_value"] = mv[k]
            region_rows.append(row)
    region_path = os.path.join(args.output_dir, "ablation_mil_region.csv")
    pd.DataFrame(region_rows).to_csv(region_path, index=False)

    # ===== Pass 2: paired significance via canonical bootstrap_wilcoxon =====
    # boot[ref] and boot[test] were drawn on the SAME per-region resamples
    # (shared REGION_SEED), so their difference is the paired macro-over-regions
    # diff distribution. wilcoxon_p() applies the canonical wilcoxon test to it.
    sig_rows = []
    for (tm, rm) in COMPARISONS:
        if tm not in boot or rm not in boot:
            print(f"SIG SKIP (missing model): {tm} vs {rm}")
            continue
        for ki, m in enumerate(METRICS):
            d = boot[rm][:, ki] - boot[tm][:, ki]          # reference - test
            sig_rows.append({
                "test_model": tm, "test_label": MODELS[tm][1],
                "reference_model": rm, "reference_label": MODELS[rm][1],
                "metric": m,
                "delta_ref_minus_test": float(point[rm][ki] - point[tm][ki]),
                "p_value": wilcoxon_p(d), "method": "bootstrap_wilcoxon",
            })
    sig_df = pd.DataFrame(sig_rows)
    if not sig_df.empty:
        ok = sig_df["p_value"].notna()
        sig_df.loc[ok, "p_value_fdr"] = bh_fdr(sig_df.loc[ok, "p_value"].values)
    sig_path = os.path.join(args.output_dir, "ablation_mil_significance.csv")
    sig_df.to_csv(sig_path, index=False)

    # ---- console summary ----
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    show = ["label"] + [f"{m}_value" for m in METRICS]
    print(f"\nN_BOOT={n_boot}, macro over {len(REGIONS)} regions; "
          f"significance = canonical bootstrap_wilcoxon\n")
    print(mean_df[show].to_string(index=False))
    print("\n-- significance (FDR) --")
    print(sig_df[["test_label", "reference_label", "metric",
                  "delta_ref_minus_test", "p_value", "p_value_fdr"]].to_string(index=False))
    print(f"\nsaved: {mean_path}\nsaved: {region_path}\nsaved: {sig_path}")


if __name__ == "__main__":
    main()
