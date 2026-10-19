"""Print dataset statistics per curriculum level (counts, balance, shapes).

Run:  python scripts/dataset_stats.py [--n-per-level 200]
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from curricula_verify.generators.generators import generate_all_levels
from curricula_verify.taxonomy import Taxonomy


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-per-level", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    taxonomy = Taxonomy.from_file(root / "configs" / "taxonomy.json")
    levels = generate_all_levels({l: args.n_per_level for l in (1, 2, 3, 4)},
                                 taxonomy, args.seed)

    print("=== CURRICULAVERIFY DATASET STATS (actual generator output) ===")
    for lvl in sorted(levels):
        records = levels[lvl]
        labels = Counter(r["label"] for r in records)
        categories = Counter(c for r in records for c in r["categories"])
        turns = [sum(1 for t in r["conversation"] if t["role"] == "user") for r in records]
        chars = [len(r["rationale"]) for r in records]
        stats = {
            "level": lvl,
            "n_records": len(records),
            "labels": dict(sorted(labels.items())),
            "safe_fraction": round(labels["safe"] / len(records), 3),
            "categories": dict(sorted(categories.items(), key=lambda kv: -kv[1])),
            "avg_user_turns": round(sum(turns) / len(turns), 2),
            "max_user_turns": max(turns),
            "avg_rationale_chars": round(sum(chars) / len(chars), 1),
        }
        print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
