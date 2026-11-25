"""Shared, source-verified report metric results (standard library only)."""
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile

EXPECTED = ["BLEU-4", "ROUGE-L", "F1-RadGraph", "Micro-F1-14+", "Macro-F1-14+", "BERTScore", "GREEN"]


def atomic_write(path, text):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        f.write(text)
        name = f.name
    os.replace(name, path)


class Summary:
    def __init__(self, filepath, output_dir):
        raw = Path(filepath).read_bytes()
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        if not rows or not all(isinstance(r.get("prediction"), str) and isinstance(r.get("reference"), str) for r in rows):
            raise ValueError("Expected nonempty prediction/reference string pairs.")
        self.predictions = [r["prediction"] for r in rows]
        self.references = [r["reference"] for r in rows]
        self.root = Path(output_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "metrics_manifest.json"
        fingerprint = hashlib.sha256(raw).hexdigest()
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
            if self.data["prediction_sha256"] != fingerprint:
                raise ValueError("Prediction file fingerprint mismatch; use a separate output directory.")
        else:
            self.data = dict(prediction_sha256=fingerprint, n_total=len(rows), metrics={})
        for name in EXPECTED:
            self.data["metrics"].setdefault(name, dict(score=None, status="missing", error="", n_valid=None, n_invalid=None))
        self.save()

    def update(self, name, score=None, status="ok", error="", n_valid=None, n_invalid=None):
        if score is None and status == "ok":
            status, error = "failed", "Missing metric value"
        if score is not None:
            score = float(score)
            if not math.isfinite(score):
                score, status, error = None, "failed", "Non-finite metric value"
        self.data["metrics"][name] = dict(score=score, status=status, error=error,
                                        n_valid=n_valid, n_invalid=n_invalid)
        self.save()

    def save(self):
        atomic_write(self.path, json.dumps(self.data, indent=2, allow_nan=False))
        out = io.StringIO()
        fields = ["metric", "score", "scale", "status", "n_total", "n_valid", "n_invalid", "prediction_sha256", "error"]
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for name, result in self.data["metrics"].items():
            writer.writerow(dict(metric=name, **result, scale="0-100" if name.startswith("BLEU") else "baseline-rescaled F1" if name == "BERTScore" else "0-1",
                                 n_total=self.data["n_total"], prediction_sha256=self.data["prediction_sha256"]))
        atomic_write(self.root / "metrics.csv", out.getvalue())
