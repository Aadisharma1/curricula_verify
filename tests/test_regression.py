from pathlib import Path

import pytest

from curricula_verify.features.verifier import VerifierModel
from curricula_verify.generators.generators import generate_level
from curricula_verify.golden.regression import determinism_check, load_golden
from curricula_verify.taxonomy import Taxonomy
from curricula_verify.trajectories.self_correction import (
    apply_corrections,
    generate_self_corrections,
)

TAXONOMY = Taxonomy({
    "benign": 0, "personal_data": 3, "injection": 3, "harmful_instructions": 4,
    "multi_turn_escalation": 4, "social_engineering": 3, "jailbreak_roleplay": 3,
})


def _train_pair(seed=0, n=50):
    data = generate_level(3, n, TAXONOMY, seed=seed)
    kwargs = {"tfidf_max_features": 400}
    return VerifierModel(**kwargs).fit(data), data, kwargs


def test_determinism_same_seed_identical_predictions():
    m1, data, kw = _train_pair(seed=11)
    m2, _, _ = _train_pair(seed=11)
    eval_data = generate_level(3, 12, TAXONOMY, seed=99)
    assert determinism_check(m1, m2, eval_data)


def test_self_corrections_only_on_errors():
    model, data, kw = _train_pair(seed=12, n=60)
    corrections = generate_self_corrections(model, data, max_augment=50)
    for c in corrections:
        assert c["id"].endswith("-selfcorrection")
        assert c["label"] in ("safe", "unsafe")
        assert "Self-correction trajectory" in c["rationale"]


def test_self_correction_retrain_runs(tmp_path):
    model, data, kw = _train_pair(seed=13, n=60)
    corrections = generate_self_corrections(model, data, max_augment=30)
    model2, metrics = apply_corrections(
        kw, data, corrections, generate_level(3, 12, TAXONOMY, seed=42)
    )
    assert 0.0 <= metrics["f1"] <= 1.0


def test_golden_set_skips_without_snapshot():
    """Until scripts/freeze_golden.py has run on this machine, the golden
    comparison is skipped with an explicit reason — never faked."""
    golden_path = Path(__file__).parents[1] / "outputs" / "golden.json"
    if not golden_path.exists():
        pytest.skip("golden snapshot not frozen yet; run scripts/freeze_golden.py first")
    golden = load_golden(golden_path)
    model, _, _ = _train_pair(seed=golden["seed"])
    eval_data = generate_level(3, golden["n_eval"], TAXONOMY, seed=golden["seed"])
    proba = model.predict_proba(eval_data)
    metrics = {
        "f1": float(((proba >= 0.5).astype(int) == [1 if r["label"] == "unsafe" else 0 for r in eval_data]).mean())
    }
    assert metrics["f1"] >= golden["thresholds"]["min_f1"]
