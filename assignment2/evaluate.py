"""Comprehensive evaluation metrics, confusion matrix, and ROC curve computation.

Follows the required assignment conventions:
- Class 1: Boys (label 0, positive class for binary metrics)
- Class 2: Girls (label 1, negative class for binary metrics)
- Confusion Matrix:
    TP: Actual Boys, Predicted Boys
    FN: Actual Boys, Predicted Girls
    FP: Actual Girls, Predicted Boys
    TN: Actual Girls, Predicted Girls
- Metrics:
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Sensitivity (Recall) = TP / (TP + FN)
    Precision = TP / (TP + FP)
    Specificity = TN / (TN + FP)
    F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    ROC AUC computed from continuous probability estimates or decision scores.
"""
from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import auc, roc_curve

from assignment2.config import Config

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Rigorous evaluation metrics container."""
    model_name: str
    tp: int
    tn: int
    fp: int
    fn: int
    accuracy: float
    sensitivity: float      # Recall of Boys (Class 1)
    precision: float        # Precision of Boys (Class 1)
    specificity: float      # Recall of Girls (Class 2)
    f1_score: float         # F1 Score of Boys (Class 1)
    girls_precision: float  # Precision of Girls (Class 2)
    girls_recall: float     # Recall of Girls (Class 2)
    girls_f1: float         # F1 Score of Girls (Class 2)
    macro_f1: float
    roc_auc: float
    confusion_matrix: list[list[int]]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Perform division handling zero-division conditions gracefully."""
    if denominator == 0:
        return default
    return float(numerator / denominator)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob_boys: np.ndarray | None,
    model_name: str = "Classifier",
) -> tuple[EvaluationMetrics, np.ndarray, np.ndarray]:
    """Calculate all required metrics from ground truth and model predictions.

    Args:
        y_true: Ground truth binary labels (0 for Boys, 1 for Girls).
        y_pred: Predicted binary labels (0 for Boys, 1 for Girls).
        y_prob_boys: Predicted probability of being Boys (Class 1 positive class).
        model_name: Identifier for the model.

    Returns:
        EvaluationMetrics dataclass, fpr array, tpr array.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    # Class 1: Boys (0), Class 2: Girls (1)
    tp = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 0) & (y_pred == 1)))
    fp = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 1) & (y_pred == 1)))

    total = tp + tn + fp + fn
    accuracy = safe_divide(tp + tn, total)
    sensitivity = safe_divide(tp, tp + fn)       # Boys Recall
    precision = safe_divide(tp, tp + fp)         # Boys Precision
    specificity = safe_divide(tn, tn + fp)       # Girls Recall
    f1_score = safe_divide(2.0 * precision * sensitivity, precision + sensitivity)

    # Girls metrics
    girls_recall = specificity
    girls_precision = safe_divide(tn, tn + fn)
    girls_f1 = safe_divide(2.0 * girls_precision * girls_recall, girls_precision + girls_recall)

    macro_f1 = float((f1_score + girls_f1) / 2.0)

    # ROC / AUC calculation
    # Positive class for ROC is Boys (0)
    if y_prob_boys is not None and len(np.unique(y_true)) > 1:
        # Convert binary ground truth: 1 for Boys, 0 for Girls
        y_true_binary = (y_true == 0).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_prob_boys)
        roc_auc = float(auc(fpr, tpr))
    else:
        fpr, tpr = np.array([0.0, 1.0]), np.array([0.0, 1.0])
        roc_auc = 0.50

    cm = [[tp, fn], [fp, tn]]

    metrics = EvaluationMetrics(
        model_name=model_name,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        accuracy=accuracy,
        sensitivity=sensitivity,
        precision=precision,
        specificity=specificity,
        f1_score=f1_score,
        girls_precision=girls_precision,
        girls_recall=girls_recall,
        girls_f1=girls_f1,
        macro_f1=macro_f1,
        roc_auc=roc_auc,
        confusion_matrix=cm,
    )
    return metrics, fpr, tpr


def save_metrics_artifacts(
    metrics_list: list[EvaluationMetrics],
    cfg: Config,
) -> None:
    """Save metrics artifacts as JSON, CSV, and formatted Markdown table."""
    cfg.ensure_directories()

    # 1. Save JSON
    json_path = cfg.metrics_dir / "model_comparison.json"
    data_dict = [asdict(m) for m in metrics_list]
    json_path.write_text(json.dumps(data_dict, indent=2), encoding="utf-8")

    # 2. Save CSV
    csv_path = cfg.metrics_dir / "model_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Model",
            "Accuracy",
            "Sensitivity",
            "Precision",
            "Specificity",
            "F1 Score",
            "Girls Precision",
            "Girls Recall",
            "Girls F1",
            "Macro F1",
            "ROC AUC",
            "TP",
            "TN",
            "FP",
            "FN",
        ])
        for m in metrics_list:
            writer.writerow([
                m.model_name,
                f"{m.accuracy:.4f}",
                f"{m.sensitivity:.4f}",
                f"{m.precision:.4f}",
                f"{m.specificity:.4f}",
                f"{m.f1_score:.4f}",
                f"{m.girls_precision:.4f}",
                f"{m.girls_recall:.4f}",
                f"{m.girls_f1:.4f}",
                f"{m.macro_f1:.4f}",
                f"{m.roc_auc:.4f}",
                m.tp,
                m.tn,
                m.fp,
                m.fn,
            ])

    # 3. Save Markdown
    md_path = cfg.metrics_dir / "model_comparison.md"
    md_rows = []
    for m in metrics_list:
        md_rows.append(
            f"| **{m.model_name}** | {m.accuracy * 100:.2f}% | {m.sensitivity * 100:.2f}% | "
            f"{m.precision * 100:.2f}% | {m.specificity * 100:.2f}% | {m.f1_score:.4f} | {m.roc_auc:.4f} | "
            f"TP={m.tp}, TN={m.tn}, FP={m.fp}, FN={m.fn} |"
        )

    md_content = f"""# Machine Learning Model Evaluation & Comparison

Course: EC24313 Artificial Intelligence and Machine Learning  
Assignment 2: Voice-Based Gender Classification  
Evaluation Protocol: Speaker-Independent Hold-Out Test Set (Zero Speaker Overlap)

## Model Comparison Table

| Classifier | Accuracy | Sensitivity (Recall) | Precision | Specificity | F1 Score | ROC AUC | Confusion Matrix |
|---|---|---|---|---|---|---|---|
{chr(10).join(md_rows)}

### Evaluation Convention
- **Class 1 (Positive):** Boys (label 0)
- **Class 2 (Negative):** Girls (label 1)
- **Sensitivity:** Fraction of true boy recordings correctly detected ($TP / (TP + FN)$).
- **Specificity:** Fraction of true girl recordings correctly detected ($TN / (TN + FP)$).
- **Precision:** Fraction of boy predictions that are actually boys ($TP / (TP + FP)$).
- **F1 Score:** Harmonic mean of precision and sensitivity.
- **ROC AUC:** Area under Receiver Operating Characteristic curve.
"""
    md_path.write_text(md_content, encoding="utf-8")
    logger.info("Saved evaluation artifacts in %s", cfg.metrics_dir)

