"""Self-correction trajectory augmentation.

For verifier mistakes on training data, synthesize corrected training records:
the erroneous prediction becomes a labeled counter-example with an explicit
corrected rationale. The measured effect (before/after metrics) is reported by
the pipeline — it may be positive, zero, or negative; nothing is assumed.
"""

from __future__ import annotations

import copy


def generate_self_corrections(model, records: list[dict], max_augment: int) -> list[dict]:
    """Find model errors on `records` and produce corrected copies."""
    if not records:
        return []
    proba = model.predict_proba(records)
    preds = (proba >= 0.5).astype(int)
    corrections = []
    for r, p, pred in zip(records, proba, preds):
        truth = 1 if r["label"] == "unsafe" else 0
        if pred == truth:
            continue
        fixed = copy.deepcopy(r)
        fixed["id"] = r["id"] + "-selfcorrection"
        fixed["label"] = r["label"]
        direction = "false_negative" if truth == 1 else "false_positive"
        fixed["rationale"] = (
            f"Self-correction trajectory ({direction}): model scored "
            f"p_unsafe={p:.3f} but ground truth is {r['label']}. Corrected."
        )
        corrections.append(fixed)
        if len(corrections) >= max_augment:
            break
    return corrections


def apply_corrections(model_kwargs: dict, base_train: list[dict], corrections: list[dict],
                      eval_records: list[dict]):
    """Train with corrections appended; return (model, metrics dict)."""
    from curricula_verify.features.verifier import VerifierModel
    from curricula_verify.evaluation.suite import evaluate

    model = VerifierModel(**model_kwargs).fit(base_train + corrections)
    proba = model.predict_proba(eval_records)
    return model, evaluate(eval_records, proba)
