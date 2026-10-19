"""End-to-end curriculum pipeline orchestration."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from curricula_verify.evaluation.suite import confusion_report
from curricula_verify.generators.generators import generate_all_levels
from curricula_verify.scheduler.curriculum import CurriculumScheduler
from curricula_verify.taxonomy import Taxonomy
from curricula_verify.trajectories.self_correction import (
    apply_corrections,
    generate_self_corrections,
)

logger = logging.getLogger(__name__)


def _split(records: list[dict], eval_fraction: float, seed: int):
    import random

    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    n_eval = max(1, int(len(shuffled) * eval_fraction))
    return shuffled[n_eval:], shuffled[:n_eval]


def run_curriculum(cfg: dict, taxonomy_path: str | Path, out_dir: str | Path) -> dict:
    """cfg: parsed configs/curriculum.json. Returns actual measured results."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    taxonomy = Taxonomy.from_file(taxonomy_path)

    sizes = {int(k): v for k, v in cfg["level_sizes"].items()}
    all_levels = generate_all_levels(sizes, taxonomy, cfg["seed"])

    levels_data: dict[int, dict[str, list]] = {}
    eval_all: list[dict] = []
    for lvl, records in all_levels.items():
        train, ev = _split(records, cfg["eval_fraction"], cfg["seed"] + lvl)
        levels_data[lvl] = {"train": train, "eval": ev}
        eval_all += ev

    model_kwargs = {"tfidf_max_features": cfg["model"]["tfidf_max_features"],
                    "C": cfg["model"]["C"]}
    scheduler = CurriculumScheduler(gate_threshold_f1=cfg["gate_threshold_f1"])
    model, progression, final_metrics = scheduler.run(levels_data, model_kwargs)

    corrections, traj_report = [], {"enabled": cfg["trajectories"]["enabled"]}
    if cfg["trajectories"]["enabled"]:
        before = final_metrics["f1"]
        corrections = generate_self_corrections(
            model, [r for lvl in levels_data.values() for r in lvl["train"]],
            cfg["trajectories"]["max_augment_per_level"],
        )
        model_after, metrics_after = apply_corrections(
            model_kwargs, [r for lvl in levels_data.values() for r in lvl["train"]],
            corrections, eval_all,
        )
        traj_report.update({
            "n_corrections": len(corrections),
            "f1_before": before,
            "f1_after": metrics_after["f1"],
            "delta_f1": metrics_after["f1"] - before,
        })
        model = model_after
        final_metrics = metrics_after

    proba = model.predict_proba(eval_all)
    confusion_report(eval_all, proba, out / "evaluation")
    results = {
        "progression": progression,
        "final_metrics": final_metrics,
        "trajectories": traj_report,
        "eval_size": len(eval_all),
        "seed": cfg["seed"],
    }
    model.save(out / "verifier.joblib")
    (out / "results.json").write_text(json.dumps(results, indent=2))
    logger.info("curriculum complete: %s", json.dumps(results["final_metrics"], default=str)[:200])
    return results
