"""Full curriculum run using configs/curriculum.json.

Run:  python scripts/train_full.py
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from curricula_verify.pipeline import run_curriculum

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "configs" / "curriculum.json").read_text(encoding="utf-8"))
    results = run_curriculum(cfg, root / "configs" / "taxonomy.json", root / "outputs" / "full")
    print("\n=== CURRICULAVERIFY FULL RUN (actual results) ===")
    print(json.dumps(results["final_metrics"], indent=2))
    print(json.dumps(results["trajectories"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
