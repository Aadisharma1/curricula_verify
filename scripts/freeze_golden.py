"""Freeze a golden regression snapshot from an ACTUAL trained model.

Run after a successful training run on this machine:
    python scripts/freeze_golden.py

Writes outputs/golden.json with the model's measured F1 on a fixed eval set.
tests/test_regression.py then enforces it. The thresholds come from the real
run — they are never invented.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from curricula_verify.evaluation.suite import evaluate
from curricula_verify.features.verifier import VerifierModel
from curricula_verify.generators.generators import generate_level
from curricula_verify.taxonomy import Taxonomy


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    taxonomy = Taxonomy.from_file(root / "configs" / "taxonomy.json")
    seed = 7
    n_train, n_eval = 120, 60
    train = generate_level(3, n_train, taxonomy, seed=seed)
    eval_set = generate_level(3, n_eval, taxonomy, seed=seed)

    model = VerifierModel(tfidf_max_features=2000).fit(train)
    metrics = evaluate(eval_set, model.predict_proba(eval_set))

    out = root / "outputs" / "golden.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    frozen = {
        "seed": seed,
        "n_eval": n_eval,
        "measured_f1": metrics["f1"],
        "thresholds": {"min_f1": round(metrics["f1"] - 0.05, 4)},
    }
    out.write_text(json.dumps(frozen, indent=2))
    print(f"frozen golden snapshot -> {out}")
    print(f"measured f1={metrics['f1']:.4f}; frozen min_f1={frozen['thresholds']['min_f1']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
