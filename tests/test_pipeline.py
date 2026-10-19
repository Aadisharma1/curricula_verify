import json

from curricula_verify.pipeline import run_curriculum

CFG = {
    "seed": 7,
    "level_sizes": {"1": 40, "2": 40, "3": 40, "4": 40},
    "eval_fraction": 0.3,
    "gate_threshold_f1": 0.5,
    "model": {"tfidf_max_features": 600, "C": 1.0},
    "trajectories": {"enabled": True, "max_augment_per_level": 50},
}


def test_pipeline_end_to_end(tmp_path):
    taxonomy = str(_write_taxonomy(tmp_path))
    results = run_curriculum(CFG, taxonomy, tmp_path / "out")
    assert [p["level"] for p in results["progression"]] == [1, 2, 3, 4]
    for m in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert m in results["final_metrics"]
    assert results["trajectories"]["enabled"] is True
    assert "delta_f1" in results["trajectories"]
    assert (tmp_path / "out" / "verifier.joblib").exists()
    assert (tmp_path / "out" / "results.json").exists()
    assert (tmp_path / "out" / "evaluation" / "confusion_matrix.png").exists()


def _write_taxonomy(tmp_path):
    p = tmp_path / "taxonomy.json"
    p.write_text(json.dumps({
        "categories": {"benign": 0, "personal_data": 3, "injection": 3,
                       "harmful_instructions": 4, "multi_turn_escalation": 4,
                       "social_engineering": 3, "jailbreak_roleplay": 3},
        "severity_scale": {"min": 0, "max": 4},
    }))
    return p
