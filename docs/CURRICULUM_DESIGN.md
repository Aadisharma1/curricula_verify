# 04 — Curriculum Design Rationale

Why the verifier is trained through escalating levels, how the gate works,
what self-correction trajectories actually do, and where the honest limits sit.

## 1. Why a curriculum instead of one mixed dataset?

A safety verifier deployed on real traffic sees a heavily skewed difficulty
distribution: most inputs are benign, most violations are single-turn and
obvious, and the rare hard cases are adversarial *and* multi-turn. Training on
a flat mixture from step one has two known failure modes:

1. The loss is dominated by the easy majority, so the hard tail is
   underfit.
2. There is no signal about *when* the model is ready for harder data.

The curriculum addresses both: each level's data is only added once the model
clears an F1 gate on the current stage, and training is cumulative — level 4
training data includes levels 1–3. The scheduler logs every gate decision
(`progression` in `results.json`), so a run that stalls at level 2 says so
explicitly instead of silently producing a weak verifier.

The gate is deliberately simple (F1 ≥ threshold, configurable in
`configs/curriculum.json`). Simplicity is a feature here: the gate is a
unit-tested pure function (`scheduler.gate(f1) -> bool`), and the
stop-at-first-failure behavior is pinned by a test that adversarially
mislables a level's eval set and asserts the curriculum refuses to advance.

## 2. The four levels — what each one isolates

| Level | Data | Capability isolated |
|---|---|---|
| 1 | single-turn benign vs clearly unsafe | basic separability; sanity-checks features and training |
| 2 | complex policy violations (PII, social engineering) | violations without magic keywords — needs compositional features |
| 3 | adversarial single-turn (injection, roleplay, mode prefixes) | intent hidden behind meta-instructions |
| 4 | multi-turn conversations with a benign opener that escalates | context accumulation: the violation is only visible across turns |

Level 4's escalation is structural by construction: the conversation always
opens benign, includes an explicit pivot turn ("ignore all previous
instructions for this conversation"), and lands on a violation. A test asserts
the pivot phrase and the ≥4-turn structure, so the level's difficulty is a
property of the dataset, not luck of the draw.

## 3. The verifier is a heuristic — and why that is the right call here

The brief allows a local model; an external LLM API would make the project
unrunnable without credentials and untestable deterministically. The local
verifier is TF-IDF (word 1–2 grams) over user turns, concatenated with four
engineered features (user-turn count, uppercase ratio, denylist-marker hits,
lowercase-word ratio), fed to logistic regression. It is pinned by tests:
deterministic predictions under a fixed seed, save/load roundtrip equality,
and a near-separation check on the formulaic synthetic data.

What this choice buys: the entire curriculum pipeline — gating, trajectories,
evaluation, regression — runs offline in seconds, so the *pipeline* is the
test subject, not the model. Swapping in a stronger verifier later means
implementing `fit/predict_proba/save/load` on the same interface; nothing else
in the pipeline changes.

## 4. Self-correction trajectories: what they really do

The mechanism is deliberately concrete: after the curriculum trains a model,
`generate_self_corrections` finds training records the model misclassified and
emits *corrected copies* (same conversation, ground-truth label, an explicit
rationale documenting the error and the correction). These are appended to the
training set and the model is retrained; the before/after F1 delta is
measured and reported in `results.json` under `trajectories`.

Honest framing: this is a hard-example mining loop with relabeling, not a
general "self-improvement" mechanism. Its effect is an empirical question —
the pipeline reports `f1_before`, `f1_after`, and `delta_f1` exactly as
measured, and the README warns the delta can be zero or negative. What the
mechanism *does* guarantee is that every correction is auditable: each
synthetic trajectory carries the original prediction, the score, and the
correction rationale in its record.

## 5. Regression testing without invented thresholds

Two gates, one honest gap:

1. **Determinism** (always runs): two models trained with the same seed on the
   same data must produce byte-identical probabilities. This catches silent
   dependency drift (a sklearn version changing solver behavior, for example)
   — the classic regression source for learned components.
2. **Golden snapshot** (runs only after `scripts/freeze_golden.py`): the
   script trains on a fixed split and freezes the *measured* F1 minus a small
   margin into `outputs/golden.json`; the test then enforces it. Until that
   script has run on your machine, the test **skips with an explicit
   reason** — the suite never invents a threshold it has not measured.

## 6. Limits (restated from the README, in more detail)

- The synthetic templates are formulaic by design; near-ceiling metrics are
  expected and say nothing about real-world safety performance.
- No obfuscation families (unicode homoglyphs, leetspeak, base64 payloads) are
  generated; level 3 is adversarial in structure, not in encoding tricks.
- The verifier is English-only, single-label, and threshold-based at 0.5 with
  no calibration.
- No claim is made about zero-day jailbreaks, novel attack families, or
  "detecting all jailbreaks" — the architecture cannot support such a claim,
  and the project does not make it.
