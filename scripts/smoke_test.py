"""Smoke test: full curriculum at tiny scale. Prints ACTUAL final metrics."""

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from curricula_verify.pipeline import run_curriculum

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    cfg = {
        "seed": 7,
        "level_sizes": {"1": 60, "2": 60, "3": 60, "4": 60},
        "eval_fraction": 0.3,
        "gate_threshold_f1": 0.5,
        "model": {"tfidf_max_features": 2000, "C": 1.0},
        "trajectories": {"enabled": True, "max_augment_per_level": 60},
    }
    t0 = time.monotonic()
    results = run_curriculum(cfg, root / "configs" / "taxonomy.json",
                             root / "outputs" / "smoke")
    elapsed = time.monotonic() - t0

    print("\n=== CURRICULAVERIFY SMOKE TEST (actual results of this run) ===")
    for step in results["progression"]:
        print(f"  level {step['level']}: f1={step['f1']:.3f} gate={'PASS' if step['gate_passed'] else 'FAIL'}")
    fm = results["final_metrics"]
    print(f"  final: accuracy={fm['accuracy']:.3f} precision={fm['precision']:.3f} "
          f"recall={fm['recall']:.3f} f1={fm['f1']:.3f} roc_auc={fm['roc_auc']:.3f}")
    tr = results["trajectories"]
    print(f"  self-corrections: {tr.get('n_corrections', 0)} | f1 {tr.get('f1_before', 0):.3f} "
          f"-> {tr.get('f1_after', 0):.3f} (delta {tr.get('delta_f1', 0):+.3f})")
    print(f"  elapsed: {elapsed:.1f}s | outputs: outputs/smoke/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
