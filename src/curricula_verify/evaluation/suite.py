"""Evaluation suite: metrics, confusion matrix, FP/FN analysis, per-category."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from curricula_verify.features.verifier import conversation_text


def evaluate(records: list[dict], proba: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.array([1 if r["label"] == "unsafe" else 0 for r in records])
    y_pred = (proba >= threshold).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    out["roc_auc"] = float(roc_auc_score(y_true, proba)) if len(set(y_true.tolist())) == 2 else float("nan")

    fp_idx = np.where((y_pred == 1) & (y_true == 0))[0]
    fn_idx = np.where((y_pred == 0) & (y_true == 1))[0]
    out["false_positives"] = [
        {"id": records[i]["id"], "text": _preview(records[i]), "p_unsafe": float(proba[i])}
        for i in fp_idx
    ]
    out["false_negatives"] = [
        {"id": records[i]["id"], "text": _preview(records[i]), "p_unsafe": float(proba[i])}
        for i in fn_idx
    ]
    out["per_category"] = _per_category(records, y_pred)
    return out


def _preview(record: dict) -> str:
    return conversation_text(record)[:120]


def _per_category(records: list[dict], y_pred: np.ndarray) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for r, pred in zip(records, y_pred):
        for cat in r["categories"]:
            slot = out.setdefault(cat, {"n": 0, "predicted_unsafe": 0})
            slot["n"] += 1
            slot["predicted_unsafe"] += int(pred)
    return out


def confusion_report(records: list[dict], proba: np.ndarray, out_dir: str | Path,
                     threshold: float = 0.5) -> dict:
    y_true = np.array([1 if r["label"] == "unsafe" else 0 for r in records])
    y_pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = {
        "labels": ["safe", "unsafe"],
        "matrix": cm.tolist(),
        "threshold": threshold,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "confusion_matrix.json").write_text(json.dumps(report, indent=2))

    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    ax.set_xticks([0, 1], ["pred safe", "pred unsafe"])
    ax.set_yticks([0, 1], ["true safe", "true unsafe"])
    ax.set_title("Verifier confusion matrix")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    return report
