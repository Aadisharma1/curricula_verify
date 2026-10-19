import numpy as np
import pytest

from curricula_verify.evaluation.suite import confusion_report, evaluate
from curricula_verify.features.verifier import VerifierModel, conversation_text, numeric_features
from curricula_verify.generators.generators import generate_level
from curricula_verify.scheduler.curriculum import CurriculumScheduler
from curricula_verify.taxonomy import Taxonomy

TAXONOMY = Taxonomy({
    "benign": 0, "personal_data": 3, "injection": 3, "harmful_instructions": 4,
    "multi_turn_escalation": 4, "social_engineering": 3, "jailbreak_roleplay": 3,
})


def _dataset(n=60, seed=0, level=1):
    return generate_level(level, n, TAXONOMY, seed=seed)


def test_conversation_text_and_numeric_features():
    records = _dataset(8, seed=1)
    texts = [conversation_text(r) for r in records]
    assert all(t.strip() for t in texts)
    num = numeric_features(records)
    assert num.shape == (8, 4)
    assert (num[:, 0] >= 1).all()


def test_verifier_train_predict_and_perfect_sep():
    train = _dataset(60, seed=2)
    model = VerifierModel(tfidf_max_features=500).fit(train)
    proba = model.predict_proba(train)
    preds = (proba >= 0.5).astype(int)
    y = np.array([1 if r["label"] == "unsafe" else 0 for r in train])
    assert (preds == y).mean() > 0.8


def test_verifier_unfitted_raises():
    model = VerifierModel()
    with pytest.raises(RuntimeError):
        model.predict_proba(_dataset(4, seed=0))


def test_verifier_single_class_raises():
    safe_only = [r for r in _dataset(10, seed=3) if r["label"] == "safe"]
    with pytest.raises(ValueError):
        VerifierModel().fit(safe_only)


def test_save_load_roundtrip(tmp_path):
    model = VerifierModel(tfidf_max_features=300).fit(_dataset(40, seed=4))
    p = tmp_path / "m.joblib"
    model.save(str(p))
    loaded = VerifierModel.load(str(p))
    records = _dataset(10, seed=5)
    assert np.array_equal(model.predict_proba(records), loaded.predict_proba(records))


def test_evaluate_metrics_and_fp_fn_lists():
    records = _dataset(20, seed=6)
    y = np.array([1 if r["label"] == "unsafe" else 0 for r in records])
    proba = y * 0.9 + (1 - y) * 0.1
    res = evaluate(records, proba)
    assert res["accuracy"] > 0.9
    assert len(res["false_positives"]) <= 1
    assert len(res["false_negatives"]) <= 1
    per_cat = res["per_category"]
    assert "benign" in per_cat and per_cat["benign"]["n"] > 0


def test_evaluate_fp_fn_previews_populate():
    """Force wrong predictions so the FP/FN example lists (and their text
    previews) are actually exercised."""
    records = _dataset(12, seed=6)
    y = np.array([1 if r["label"] == "unsafe" else 0 for r in records])
    proba = (1 - y) * 0.9 + y * 0.1
    res = evaluate(records, proba)
    assert len(res["false_positives"]) == int((y == 0).sum())
    assert len(res["false_negatives"]) == int((y == 1).sum())
    for item in res["false_positives"] + res["false_negatives"]:
        assert isinstance(item["text"], str) and item["text"]


def test_confusion_report_files(tmp_path):
    records = _dataset(20, seed=7)
    y = np.array([1 if r["label"] == "unsafe" else 0 for r in records])
    proba = y * 0.9 + (1 - y) * 0.1
    report = confusion_report(records, proba, tmp_path)
    assert sum(sum(row) for row in report["matrix"]) == 20
    assert (tmp_path / "confusion_matrix.json").exists()
    assert (tmp_path / "confusion_matrix.png").exists()


def test_scheduler_gate_and_progression():
    sched = CurriculumScheduler(gate_threshold_f1=0.5)
    assert sched.gate(0.6) and not sched.gate(0.4)
    with pytest.raises(ValueError):
        CurriculumScheduler(gate_threshold_f1=0)

    levels = {}
    for lvl in (1, 2, 3):
        data = generate_level(lvl, 40, TAXONOMY, seed=10 + lvl)
        levels[lvl] = {"train": data[:24], "eval": data[24:]}
    model, progression, final_metrics = sched.run(levels, {"tfidf_max_features": 400})
    assert [p["level"] for p in progression] == [1, 2, 3]
    assert final_metrics, "final metrics must be populated"


def test_scheduler_stops_at_first_gate_failure():
    """When a level's eval is adversarially mislabeled, F1 collapses and the
    curriculum must refuse to advance further."""
    levels = {}
    for lvl in (1, 2, 3):
        data = generate_level(lvl, 40, TAXONOMY, seed=20 + lvl)
        train, ev = data[:24], data[24:]
        if lvl == 1:
            ev = [{**r, "label": "unsafe" if r["label"] == "safe" else "safe"} for r in ev]
        levels[lvl] = {"train": train, "eval": ev}

    sched = CurriculumScheduler(gate_threshold_f1=0.5)
    _, progression, _ = sched.run(levels, {"tfidf_max_features": 400})
    assert [p["level"] for p in progression] == [1], (
        "curriculum must stop at the failed gate, not continue to level 2"
    )
    assert progression[0]["gate_passed"] is False


def test_generator_level_balance_is_exact():
    """Each level is built n - n//2 safe + the rest unsafe: exactly balanced
    for even n."""
    for lvl in (1, 2, 3, 4):
        records = generate_level(lvl, 20, TAXONOMY, seed=50 + lvl)
        safe = sum(1 for r in records if r["label"] == "safe")
        assert safe == 10
        assert len(records) == 20


def test_level4_escalation_phrase_present():
    records = generate_level(4, 10, TAXONOMY, seed=60)
    unsafe = [r for r in records if r["label"] == "unsafe"]
    assert unsafe
    for r in unsafe:
        texts = [t["content"] for t in r["conversation"]]
        assert any("ignore all previous" in t for t in texts), (
            "level-4 escalation must include the injection pivot turn"
        )
