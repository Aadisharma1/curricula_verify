import pytest

from curricula_verify.taxonomy import Taxonomy
from curricula_verify.generators.generators import generate_all_levels, generate_level

TAXONOMY = Taxonomy({
    "benign": 0, "personal_data": 3, "social_engineering": 3, "injection": 3,
    "jailbreak_roleplay": 3, "harmful_instructions": 4, "multi_turn_escalation": 4,
})


def test_taxonomy_validation():
    with pytest.raises(ValueError):
        Taxonomy({})
    with pytest.raises(ValueError):
        Taxonomy({"bad": -1})
    assert TAXONOMY.severity("benign") == 0
    assert TAXONOMY.unsafe_categories() == [
        "personal_data", "social_engineering", "injection",
        "jailbreak_roleplay", "harmful_instructions", "multi_turn_escalation",
    ]


@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_level_schema_and_balance(level):
    records = generate_level(level, 20, TAXONOMY, seed=100 + level)
    assert len(records) == 20
    labels = [r["label"] for r in records]
    assert set(labels) == {"safe", "unsafe"}
    for r in records:
        assert set(r) == {"id", "level", "conversation", "label", "categories", "severity", "rationale"}
        assert r["level"] == level
        assert r["conversation"], "conversation must not be empty"
        for turn in r["conversation"]:
            assert turn["role"] in ("user", "assistant") and turn["content"]
        expected_sev = max(TAXONOMY.severity(c) for c in r["categories"])
        assert r["severity"] == expected_sev


def test_seed_determinism():
    a = generate_level(3, 12, TAXONOMY, seed=5)
    b = generate_level(3, 12, TAXONOMY, seed=5)
    assert a == b
    c = generate_level(3, 12, TAXONOMY, seed=6)
    assert a != c


def test_level4_is_multi_turn():
    records = generate_level(4, 10, TAXONOMY, seed=3)
    unsafe = [r for r in records if r["label"] == "unsafe"]
    assert unsafe, "level 4 must include unsafe conversations"
    for r in unsafe:
        assert len(r["conversation"]) >= 4
        assert "multi_turn_escalation" in r["categories"]


def test_generate_all_levels_sizes():
    sizes = {1: 10, 2: 10, 3: 10, 4: 10}
    levels = generate_all_levels(sizes, TAXONOMY, seed=1)
    assert sorted(levels) == [1, 2, 3, 4]
    assert all(len(v) == 10 for v in levels.values())


def test_unknown_level_rejected():
    with pytest.raises(ValueError):
        generate_level(5, 10, TAXONOMY, seed=0)
