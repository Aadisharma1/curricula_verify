"""Generate the synthetic dataset files (optional CLI convenience).

Run:  python scripts/generate_data.py [--out data/]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from curricula_verify.generators.generators import generate_all_levels
from curricula_verify.taxonomy import Taxonomy


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-per-level", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    taxonomy = Taxonomy.from_file(root / "configs" / "taxonomy.json")
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)

    sizes = {lvl: args.n_per_level for lvl in (1, 2, 3, 4)}
    levels = generate_all_levels(sizes, taxonomy, args.seed)
    for lvl, records in levels.items():
        path = out / f"level_{lvl}.json"
        path.write_text(json.dumps(records, indent=2))
        print(f"wrote {len(records):>5} records -> {path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
