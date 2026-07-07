"""Zero-shot metric helpers.

The metric implementation is shared with the public linear-probe task so label
handling stays consistent across downstream classification evaluations.
"""

from downstream.linear_probe.metrics import compute_classification_metrics

__all__ = ["compute_classification_metrics"]
