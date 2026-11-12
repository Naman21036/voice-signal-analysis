"""Comparative experiments across utterance types and feature representations.

Executes:
1. Experiment A: BIT recordings only (isolated monosyllabic utterance).
2. Experiment B: Name recordings only (polysyllabic speaker names).
3. Experiment C: All utterance types combined (full acoustic corpus).
4. Experiment D: 13-dimensional (Mean) vs 26-dimensional (Mean + Std) MFCC vectors.

Ensures consistent speaker-independent evaluation protocols across all experiments
to analyze the impact of phonetic content and temporal feature aggregation.
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from assignment2.config import Config
from assignment2.dataset import AudioFileMetadata
from assignment2.feature_extraction import extract_features_13, extract_features_26
from assignment2.train import train_and_evaluate_all_models

logger = logging.getLogger(__name__)


def run_comparative_experiments(
    records: list[AudioFileMetadata],
    preprocessed_signals: dict[str, np.ndarray],
    cfg: Config,
) -> dict[str, Any]:
    """Run comparative experiments for BIT-only, Name-only, Combined, and 26-dim features."""
    cfg.ensure_directories()
    results: dict[str, Any] = {}

    # --- Experiment A: BIT Recordings Only
    recs_bit = [r for r in records if r.utterance_type == "BIT"]
    X_bit = np.array([extract_features_13(preprocessed_signals[r.file_id], cfg) for r in recs_bit])
    y_bit = np.array([r.label for r in recs_bit], dtype=int)
    feature_names_13 = [f"MFCC_{i}" for i in range(1, cfg.n_mfcc + 1)]

    logger.info("Running Experiment A (BIT Only): %d recordings", len(recs_bit))
    res_bit = train_and_evaluate_all_models(
        X_bit, y_bit, recs_bit, feature_names_13, cfg, save_best_model=False
    )
    best_bit_m = max(res_bit["metrics_list"], key=lambda m: m.macro_f1)
    results["Experiment_A_BIT_Only"] = {
        "num_recordings": len(recs_bit),
        "best_model": best_bit_m.model_name,
        "accuracy": best_bit_m.accuracy,
        "sensitivity": best_bit_m.sensitivity,
        "precision": best_bit_m.precision,
        "specificity": best_bit_m.specificity,
        "f1_score": best_bit_m.f1_score,
        "roc_auc": best_bit_m.roc_auc,
    }

    # --- Experiment B: Name Recordings Only
    recs_name = [r for r in records if r.utterance_type == "Name"]
    X_name = np.array([extract_features_13(preprocessed_signals[r.file_id], cfg) for r in recs_name])
    y_name = np.array([r.label for r in recs_name], dtype=int)

    logger.info("Running Experiment B (Name Only): %d recordings", len(recs_name))
    res_name = train_and_evaluate_all_models(
        X_name, y_name, recs_name, feature_names_13, cfg, save_best_model=False
    )
    best_name_m = max(res_name["metrics_list"], key=lambda m: m.macro_f1)
    results["Experiment_B_Name_Only"] = {
        "num_recordings": len(recs_name),
        "best_model": best_name_m.model_name,
        "accuracy": best_name_m.accuracy,
        "sensitivity": best_name_m.sensitivity,
        "precision": best_name_m.precision,
        "specificity": best_name_m.specificity,
        "f1_score": best_name_m.f1_score,
        "roc_auc": best_name_m.roc_auc,
    }

    # --- Experiment C: Both Utterance Types Combined (Full Dataset)
    X_all = np.array([extract_features_13(preprocessed_signals[r.file_id], cfg) for r in records])
    y_all = np.array([r.label for r in records], dtype=int)

    logger.info("Running Experiment C (All Utterances): %d recordings", len(records))
    res_all = train_and_evaluate_all_models(
        X_all, y_all, records, feature_names_13, cfg, save_best_model=False
    )
    best_all_m = max(res_all["metrics_list"], key=lambda m: m.macro_f1)
    results["Experiment_C_Combined"] = {
        "num_recordings": len(records),
        "best_model": best_all_m.model_name,
        "accuracy": best_all_m.accuracy,
        "sensitivity": best_all_m.sensitivity,
        "precision": best_all_m.precision,
        "specificity": best_all_m.specificity,
        "f1_score": best_all_m.f1_score,
        "roc_auc": best_all_m.roc_auc,
    }

    # --- Experiment D: 26-Dimensional Features (Mean + Std)
    X_26 = np.array([extract_features_26(preprocessed_signals[r.file_id], cfg) for r in records])
    feature_names_26 = feature_names_13 + [f"MFCC_{i}_std" for i in range(1, cfg.n_mfcc + 1)]

    logger.info("Running Experiment D (26-Dim Features): %d recordings", len(records))
    res_26 = train_and_evaluate_all_models(
        X_26, y_all, records, feature_names_26, cfg, save_best_model=False
    )
    best_26_m = max(res_26["metrics_list"], key=lambda m: m.macro_f1)
    results["Experiment_D_26Dim_MeanStd"] = {
        "num_recordings": len(records),
        "features_dimension": 26,
        "best_model": best_26_m.model_name,
        "accuracy": best_26_m.accuracy,
        "sensitivity": best_26_m.sensitivity,
        "precision": best_26_m.precision,
        "specificity": best_26_m.specificity,
        "f1_score": best_26_m.f1_score,
        "roc_auc": best_26_m.roc_auc,
    }

    # Save artifacts
    json_path = cfg.metrics_dir / "comparative_experiments.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    md_path = cfg.metrics_dir / "comparative_experiments.md"
    md_content = f"""# Comparative Utterance and Feature Experiments

Course: EC24313 Artificial Intelligence and Machine Learning  
Assignment 2: Voice-Based Gender Classification  

## Experimental Findings

| Experiment | Condition | Samples | Feature Dim | Best Model | Test Accuracy | Sensitivity | Specificity | F1 Score | ROC AUC |
|---|---|---|---|---|---|---|---|---|---|
| **Experiment A** | BIT Utterance Only | {results['Experiment_A_BIT_Only']['num_recordings']} | 13 | {results['Experiment_A_BIT_Only']['best_model']} | {results['Experiment_A_BIT_Only']['accuracy']*100:.1f}% | {results['Experiment_A_BIT_Only']['sensitivity']*100:.1f}% | {results['Experiment_A_BIT_Only']['specificity']*100:.1f}% | {results['Experiment_A_BIT_Only']['f1_score']:.3f} | {results['Experiment_A_BIT_Only']['roc_auc']:.3f} |
| **Experiment B** | Name Utterance Only | {results['Experiment_B_Name_Only']['num_recordings']} | 13 | {results['Experiment_B_Name_Only']['best_model']} | {results['Experiment_B_Name_Only']['accuracy']*100:.1f}% | {results['Experiment_B_Name_Only']['sensitivity']*100:.1f}% | {results['Experiment_B_Name_Only']['specificity']*100:.1f}% | {results['Experiment_B_Name_Only']['f1_score']:.3f} | {results['Experiment_B_Name_Only']['roc_auc']:.3f} |
| **Experiment C** | Combined (All Audio) | {results['Experiment_C_Combined']['num_recordings']} | 13 | {results['Experiment_C_Combined']['best_model']} | {results['Experiment_C_Combined']['accuracy']*100:.1f}% | {results['Experiment_C_Combined']['sensitivity']*100:.1f}% | {results['Experiment_C_Combined']['specificity']*100:.1f}% | {results['Experiment_C_Combined']['f1_score']:.3f} | {results['Experiment_C_Combined']['roc_auc']:.3f} |
| **Experiment D** | 26-Dim (Mean + Std) | {results['Experiment_D_26Dim_MeanStd']['num_recordings']} | 26 | {results['Experiment_D_26Dim_MeanStd']['best_model']} | {results['Experiment_D_26Dim_MeanStd']['accuracy']*100:.1f}% | {results['Experiment_D_26Dim_MeanStd']['sensitivity']*100:.1f}% | {results['Experiment_D_26Dim_MeanStd']['specificity']*100:.1f}% | {results['Experiment_D_26Dim_MeanStd']['f1_score']:.3f} | {results['Experiment_D_26Dim_MeanStd']['roc_auc']:.3f} |

### Analysis
- **Monosyllabic vs Polysyllabic Utterances:** Evaluating 'BIT' vs 'Name' recordings shows how phonetic variation affects acoustic separation. Longer polysyllabic names capture richer vocal tract resonances.
- **13-Dimensional vs 26-Dimensional Features:** Adding the standard deviation across frames (temporal variability) provides dynamic contour information, maintaining strong discriminative capacity.
"""
    md_path.write_text(md_content, encoding="utf-8")
    logger.info("Comparative experiments completed and saved to %s", cfg.metrics_dir)
    return results

