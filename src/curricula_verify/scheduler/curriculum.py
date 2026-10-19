"""Curriculum scheduler: progressive levels with an F1 gate between levels."""

from __future__ import annotations

import logging

from curricula_verify.features.verifier import VerifierModel
from curricula_verify.evaluation.suite import evaluate

logger = logging.getLogger(__name__)


class CurriculumScheduler:
    def __init__(self, gate_threshold_f1: float = 0.6):
        if not 0 < gate_threshold_f1 <= 1:
            raise ValueError("gate_threshold_f1 must be in (0, 1]")
        self.gate_threshold_f1 = gate_threshold_f1

    def gate(self, f1: float) -> bool:
        """Advance to the next level only if current F1 clears the gate."""
        return f1 >= self.gate_threshold_f1

    def run(self, levels_data: dict[int, dict[str, list]], model_kwargs: dict | None = None):
        """levels_data: {level: {"train": [...], "eval": [...]}}.

        Returns (model, progression_log, final_eval_metrics).
        """
        model_kwargs = model_kwargs or {}
        model = VerifierModel(**model_kwargs)
        progression = []
        cumulative_train: list[dict] = []
        final_metrics = {}

        for level in sorted(levels_data):
            train = levels_data[level]["train"]
            ev = levels_data[level]["eval"]
            cumulative_train += train
            model = VerifierModel(**model_kwargs).fit(cumulative_train)
            proba = model.predict_proba(ev)
            metrics = evaluate(ev, proba)
            passed = self.gate(metrics["f1"])
            progression.append({
                "level": level,
                "train_size": len(cumulative_train),
                "f1": metrics["f1"],
                "gate_passed": passed,
            })
            logger.info("level %s: f1=%.3f gate=%s", level, metrics["f1"], passed)
            final_metrics = metrics
            if not passed:
                logger.warning("gate failed at level %s; stopping curriculum", level)
                break

        return model, progression, final_metrics
