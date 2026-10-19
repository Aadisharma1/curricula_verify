"""Regression testing helpers.

Two honest regression gates:
1. Determinism: same seed + same config -> byte-identical predictions.
2. Golden set: `scripts/freeze_golden.py` records the model's predictions and
   metrics for a fixed eval set after a real run; tests compare against that
   snapshot. Until it is frozen, tests skip with an explicit reason (they do
   not invent thresholds).
"""

from __future__ import annotations

import json
from pathlib import Path


def load_golden(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def determinism_check(model_a, model_b, records: list[dict]) -> bool:
    """Two freshly trained models on identical seed/data must agree exactly."""
    import numpy as np

    pa, pb = model_a.predict_proba(records), model_b.predict_proba(records)
    return bool(np.array_equal(pa, pb))
