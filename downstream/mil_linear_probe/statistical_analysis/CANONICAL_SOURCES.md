# Statistical-analysis outputs — canonical sources

`analyze_all.sh` runs six analyses. Several CSVs report overlapping quantities
by different methods. To avoid reporting inconsistent numbers, use the
**canonical** source below for each quantity; the others are intermediate.

## Descriptive (point estimate + confidence interval)

| Quantity | Canonical file | Method |
|---|---|---|
| Macro-mean AUC / F1 / AP / MCC / b-acc + CI | `results_mean.csv` | percentile bootstrap, n=1000; macro over a class set **fixed on the full sample** |
| Per-disease F1 / AP / MCC / b-acc + CI | `results_per_disease.csv` | **stratified** percentile bootstrap (positives/negatives resampled separately); `ci_reliable=False` flags classes with `n_pos < MIN_POS` |
| Per-disease **AUC** + CI | `auc_per_disease.csv` | **DeLong** analytic CI (gold standard for a single AUC) |

Note: `results_per_disease.csv` also carries a per-disease AUC CI for
convenience, but the **DeLong** value in `auc_per_disease.csv` is canonical for
per-disease AUC. `auc_mean.csv` holds the macro-mean AUC **point only** (no CI);
its CI lives in `results_mean.csv`. The macro-mean AUC point in `results_mean.csv`
and `delong_results.csv` (`auc_a`) is identical by construction (verified: max
abs diff = 0).

## Inference (p-value, effect-size CI of a difference)

| Comparison | Canonical file | Method |
|---|---|---|
| AUC, per-disease | `delong_results.csv` (`is_mean=False`) | DeLong test |
| AUC, macro-mean | `delong_results.csv` (`is_mean=True`) | paired permutation (p) + paired bootstrap (diff CI) |
| F1 / MCC / b-acc, per-disease & macro-mean | `threshold_results.csv` | paired permutation (p) + alignment-safe paired bootstrap (diff CI) |

Both inference files keep the raw `p_value` (+ `significant`) and add
`p_value_fdr` / `significant_fdr` (Benjamini–Hochberg within `dataset[, metric],
is_mean`). Comparisons are **our-vs-all baselines**; `is_best_baseline` flags the
strongest baseline (by mean AUC) for optional "vs best" display — it does not
affect the FDR family.

## Excluding diseases per region

To drop specific diseases from a region's analysis, edit `REGION_EXCLUDE_DISEASES`
at the top of `utils.py` — keyed by region name, listing disease **names**:

```python
REGION_EXCLUDE_DISEASES = {
    "Chest": ["脊柱侧凸"],   # reason: ...
    "Foot":  ["骨化"],
}
```

- Default `{}` = no change.
- Applies **consistently everywhere**: macro mean (point + CI), per-disease rows,
  significance p-values, and the BH-FDR family all cover only the kept diseases.
  Per-disease results for the *other* diseases are unaffected.
- **Order inside each list is irrelevant** (it's treated as a set); duplicates are
  fine too. What matters is that each string **exactly matches** a disease name in
  `REGION_DISEASES[region]` (in `../data_utils.py`) — a name that isn't in that
  region's list (e.g. a typo) is **silently ignored**, so double-check spelling.
- Choose exclusions by an a-priori clinical/objective rule (anatomically
  irrelevant, too few positives), **not** by looking at the scores (that would be
  a winner's-curse selection). Note the reason in a comment.
- Assumes the `class_idx` order in `test_results.csv` matches
  `REGION_DISEASES[region]` (true for the current pipeline).

A softer alternative: keep all diseases but filter by the `n_pos` / `ci_reliable`
columns at table-generation time. That only hides rows; it does **not** recompute
the macro mean or FDR (those already include the diseases). Use exclusion above if
you want them out of the aggregates.
