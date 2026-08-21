"""Precompute per-report CheXBERT labels and F1-RadGraph rewards for jsonl preds.

Outputs are JSON files keyed by ``{model_name: {report_id: ...}}`` consumed by
``analyze_significance.py``:

  ``chexbert_labels.json``:
      Per (model, report_id):
        ``{"pred": [int*14], "ref": [int*14]}``
      where each int is the raw CheXBERT class
      (0=blank, 1=positive, 2=negative, 3=uncertain) following the order of
      ``CheXbert_CONDITIONS`` in ``rrg_eval.factuality_utils``.

      Storing the **raw 0-3 labels** lets ``analyze_significance.py`` derive
      any of the 8 F1 variants (Micro/Macro × 14/5 × uncertain-as-pos/neg) at
      permutation time without re-running CheXBERT.

  ``radgraph_rewards.json``:
      Per (model, report_id): ``float`` (F1-RadGraph partial reward).

Running this script requires GPU + the CheXBERT and RadGraph model weights
already wired up in your ``rrg_eval`` env (same prerequisites as
``run.py``). The actual ML inference happens inside
``rrg_eval.chexbert.CheXbert`` and ``rrg_eval.f1radgraph.F1RadGraphv2``.

Example::

    python compute_clinical_labels.py \\
        --jsonls smartx_large:preds_smartx_large.jsonl,\\
chess:preds_chess.jsonl,\\
biomedclip:preds_biomedclip.jsonl \\
        --output_dir analysis_results/v_stat \\
        --skip_radgraph     # skip RadGraph if you only need CheXBERT
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

# Make the rrg_eval package importable (it lives under llava/eval/rrg_eval/)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "llava" / "eval" / "rrg_eval"))


def load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def extract_id(row: dict, idx_fallback: int) -> str:
    key = row.get("id")
    if key is None:
        key = row.get("question_id", row.get("index", idx_fallback))
    return str(key)


def run_chexbert(preds: list[str], refs: list[str]) -> tuple[list[list[int]], list[list[int]]]:
    """Returns (pred_labels, ref_labels), each shape (N, 14), raw 0-3 ints."""
    from rrg_eval.chexbert import CheXbert  # type: ignore

    model = CheXbert()
    outputs = model(preds + refs)  # list of 14 lists of length 2N
    # outputs[c][i] is the class (0-3) for condition c on sample i (preds first, refs second).
    n = len(preds)
    # Transpose to (2N, 14)
    by_sample = list(map(list, zip(*outputs)))  # shape (2N, 14)
    pred_labels = by_sample[:n]
    ref_labels = by_sample[n:]
    return pred_labels, ref_labels


def run_radgraph(preds: list[str], refs: list[str]) -> list[float]:
    """Returns per-report F1-RadGraph partial reward (float)."""
    from rrg_eval.f1radgraph import F1RadGraphv2  # type: ignore

    scorer = F1RadGraphv2(reward_level="partial", batch_size=1)
    _, reward_list, *_ = scorer(hyps=preds, refs=refs)
    # At reward_level='partial' each entry is a (f1, precision, recall) tuple
    # (only 'all' restructures into separate lists). We keep the f1 component,
    # matching run.py's radgraph() which also indexes r[0].
    return [float(r[0]) for r in reward_list]


def parse_jsonls_arg(arg: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for piece in arg.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if ":" not in piece:
            raise ValueError(f"--jsonls entry must be 'name:path', got {piece!r}")
        name, path = piece.split(":", 1)
        out[name.strip()] = path.strip()
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Precompute per-report CheXBERT labels + F1-RadGraph rewards"
    )
    parser.add_argument("--jsonls", type=str, required=True,
                        help="Comma-separated 'name:path' entries.")
    parser.add_argument("--output_dir", type=str, default="analysis_results")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Optional dataset tag; output files become "
                             "chexbert_labels_<dataset>.json / radgraph_rewards_<dataset>.json.")
    parser.add_argument("--skip_chexbert", action="store_true")
    parser.add_argument("--skip_radgraph", action="store_true")
    args = parser.parse_args()

    if args.skip_chexbert and args.skip_radgraph:
        parser.error("Nothing to compute: both --skip_chexbert and --skip_radgraph set.")

    model_files = parse_jsonls_arg(args.jsonls)
    os.makedirs(args.output_dir, exist_ok=True)

    chexbert_out: dict[str, dict[str, dict]] = {}
    radgraph_out: dict[str, dict[str, float]] = {}

    for model, path in model_files.items():
        print(f"\n=== {model}  ({path}) ===")
        rows = load_jsonl(path)
        ids = [extract_id(r, i) for i, r in enumerate(rows)]
        preds = [r["prediction"] for r in rows]
        refs = [r["reference"] for r in rows]
        print(f"  loaded {len(rows)} reports")

        if not args.skip_chexbert:
            print(f"  [CheXBERT] running on {len(preds)} reports...")
            pred_labels, ref_labels = run_chexbert(preds, refs)
            chexbert_out[model] = {
                ids[i]: {"pred": pred_labels[i], "ref": ref_labels[i]}
                for i in range(len(ids))
            }
            print(f"  [CheXBERT] done")

        if not args.skip_radgraph:
            print(f"  [RadGraph] running on {len(preds)} reports...")
            rewards = run_radgraph(preds, refs)
            radgraph_out[model] = {ids[i]: rewards[i] for i in range(len(ids))}
            print(f"  [RadGraph] done")

    suffix = f"_{args.dataset}" if args.dataset else ""
    if not args.skip_chexbert:
        path = os.path.join(args.output_dir, f"chexbert_labels{suffix}.json")
        with open(path, "w") as f:
            json.dump(chexbert_out, f)
        print(f"\nSaved CheXBERT labels -> {path}")
    if not args.skip_radgraph:
        path = os.path.join(args.output_dir, f"radgraph_rewards{suffix}.json")
        with open(path, "w") as f:
            json.dump(radgraph_out, f)
        print(f"Saved RadGraph rewards -> {path}")


if __name__ == "__main__":
    main()
