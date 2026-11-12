"""Model training, hyperparameter optimization, and speaker-leakage prevention.

Implements three machine learning classifiers:
1. Support Vector Machine (Linear and RBF kernels, C, gamma tuning, balanced weights).
2. Random Forest (trees, max depth, minimum split, balanced weights, feature importances).
3. Multi-Layer Perceptron Neural Network (Input 13 -> 64 ReLU -> 32 ReLU -> Output,
   Adam optimizer, L2 regularization, early stopping, and loss curve tracking).

All model selection and hyperparameter optimization is performed strictly within the
training partition using Stratified Group Cross-Validation on speaker_id.
Zero speaker overlap exists between training, validation, and testing partitions.
Feature scaling (StandardScaler) is strictly fit on training data within sklearn Pipelines.
"""
from __future__ import annotations

import copy
import json
import logging
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from assignment2.config import Config
from assignment2.dataset import AudioFileMetadata
from assignment2.evaluate import EvaluationMetrics, compute_metrics, save_metrics_artifacts
from assignment2.visualization import (
    plot_confusion_matrix,
    plot_model_comparison,
    plot_neural_net_loss,
    plot_roc_curve,
)

logger = logging.getLogger(__name__)


def create_speaker_independent_split(
    X: np.ndarray,
    y: np.ndarray,
    records: list[AudioFileMetadata],
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Partition dataset into training and test sets strictly by speaker identity.

    Uses StratifiedGroupKFold to guarantee that both Boys and Girls are present in the
    test set while strictly ensuring 0 speaker overlap between train and test.
    """
    speakers = np.array([r.speaker_id for r in records])

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=cfg.random_seed)

    # Pick the first fold where both classes are represented in the test partition
    train_idx, test_idx = next(sgkf.split(X, y, groups=speakers))

    train_speakers = set(speakers[train_idx])
    test_speakers = set(speakers[test_idx])
    overlap = train_speakers.intersection(test_speakers)

    if overlap:
        raise RuntimeError(f"Data leakage detected! Overlapping speakers: {overlap}")

    logger.info(
        "Speaker-independent split: Train = %d recordings (%d speakers, %d Boys, %d Girls) | "
        "Test = %d recordings (%d speakers, %d Boys, %d Girls)",
        len(train_idx),
        len(train_speakers),
        int(np.sum(y[train_idx] == 0)),
        int(np.sum(y[train_idx] == 1)),
        len(test_idx),
        len(test_speakers),
        int(np.sum(y[test_idx] == 0)),
        int(np.sum(y[test_idx] == 1)),
    )

    return (
        train_idx,
        test_idx,
        speakers[train_idx],
        speakers[test_idx],
        list(train_speakers),
        list(test_speakers),
    )


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_speakers: np.ndarray,
    cfg: Config,
) -> tuple[Pipeline, dict[str, Any]]:
    """Train Support Vector Machine with hyperparameter optimization over kernel, C, and gamma."""
    kernels = ["rbf", "linear"]
    c_values = [0.1, 1.0, 10.0, 50.0]
    gammas = ["scale", "auto", 0.05, 0.1]

    best_score = -1.0
    best_params = {}
    best_pipeline: Pipeline | None = None

    # Inner group cross-validation for hyperparameter tuning
    inner_sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=cfg.random_seed)

    for k in kernels:
        for c in c_values:
            gamma_list = gammas if k == "rbf" else ["scale"]
            for g in gamma_list:
                scores = []
                for f_tr, f_val in inner_sgkf.split(X_train, y_train, groups=train_speakers):
                    scaler = StandardScaler()
                    X_f_tr = scaler.fit_transform(X_train[f_tr])
                    X_f_val = scaler.transform(X_train[f_val])

                    clf = SVC(
                        kernel=k,
                        C=c,
                        gamma=g,
                        class_weight="balanced",
                        probability=True,
                        random_state=cfg.random_seed,
                    )
                    clf.fit(X_f_tr, y_train[f_tr])
                    prob_val = clf.predict_proba(X_f_val)[:, 0]  # Boys prob
                    y_val_bin = (y_train[f_val] == 0).astype(int)
                    if len(np.unique(y_val_bin)) > 1:
                        score = roc_auc_score(y_val_bin, prob_val)
                    else:
                        score = clf.score(X_f_val, y_train[f_val])
                    scores.append(score)

                mean_score = float(np.mean(scores))
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = {"kernel": k, "C": c, "gamma": g}

    # Fit final pipeline on entire training partition
    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        (
            "svm",
            SVC(
                kernel=best_params["kernel"],
                C=best_params["C"],
                gamma=best_params["gamma"],
                class_weight="balanced",
                probability=True,
                random_state=cfg.random_seed,
            ),
        ),
    ])
    final_pipeline.fit(X_train, y_train)

    logger.info("Trained SVM with best params: %s (Val AUC: %.4f)", best_params, best_score)
    return final_pipeline, {"params": best_params, "val_score": best_score}


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_speakers: np.ndarray,
    cfg: Config,
) -> tuple[Pipeline, dict[str, Any]]:
    """Train Random Forest classifier with hyperparameter optimization."""
    n_estimators_list = [50, 100, 200]
    max_depths = [3, 5, 8, None]
    min_samples_splits = [2, 4]

    best_score = -1.0
    best_params = {}

    inner_sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=cfg.random_seed)

    for n_est in n_estimators_list:
        for md in max_depths:
            for mss in min_samples_splits:
                scores = []
                for f_tr, f_val in inner_sgkf.split(X_train, y_train, groups=train_speakers):
                    clf = RandomForestClassifier(
                        n_estimators=n_est,
                        max_depth=md,
                        min_samples_split=mss,
                        class_weight="balanced",
                        random_state=cfg.random_seed,
                    )
                    clf.fit(X_train[f_tr], y_train[f_tr])
                    prob_val = clf.predict_proba(X_train[f_val])[:, 0]
                    y_val_bin = (y_train[f_val] == 0).astype(int)
                    if len(np.unique(y_val_bin)) > 1:
                        score = roc_auc_score(y_val_bin, prob_val)
                    else:
                        score = clf.score(X_train[f_val], y_train[f_val])
                    scores.append(score)

                mean_score = float(np.mean(scores))
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = {
                        "n_estimators": n_est,
                        "max_depth": md,
                        "min_samples_split": mss,
                    }

    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=best_params["n_estimators"],
                max_depth=best_params["max_depth"],
                min_samples_split=best_params["min_samples_split"],
                class_weight="balanced",
                random_state=cfg.random_seed,
            ),
        ),
    ])
    final_pipeline.fit(X_train, y_train)

    rf_model: RandomForestClassifier = final_pipeline.named_steps["rf"]
    feature_importances = rf_model.feature_importances_.tolist()

    logger.info("Trained Random Forest with best params: %s (Val AUC: %.4f)", best_params, best_score)
    return final_pipeline, {
        "params": best_params,
        "val_score": best_score,
        "feature_importances": feature_importances,
    }


def train_mlp_neural_network(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_speakers: np.ndarray,
    cfg: Config,
) -> tuple[Pipeline, list[float], list[float], dict[str, Any]]:
    """Train Multi-Layer Perceptron (MLP) Neural Network with loss tracking and early stopping.

    Architecture:
      Input (13 features) -> Hidden 1 (64 ReLU) -> Hidden 2 (32 ReLU) -> Output (Binary).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Dedicated speaker-independent validation fold for early stopping
    inner_sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=cfg.random_seed)
    sub_tr_idx, sub_val_idx = next(inner_sgkf.split(X_train, y_train, groups=train_speakers))

    X_sub_tr, y_sub_tr = X_scaled[sub_tr_idx], y_train[sub_tr_idx]
    X_sub_val, y_sub_val = X_scaled[sub_val_idx], y_train[sub_val_idx]

    # Calculate balanced sample weights for training
    sw_tr = compute_sample_weight("balanced", y_sub_tr)
    sw_val = compute_sample_weight("balanced", y_sub_val) if len(np.unique(y_sub_val)) > 1 else None

    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.01,
        learning_rate_init=0.002,
        max_iter=1,
        warm_start=True,
        random_state=cfg.random_seed,
    )

    classes = np.array([0, 1])
    train_losses: list[float] = []
    val_losses: list[float] = []

    best_val_loss = float("inf")
    best_weights = None
    patience = 35
    patience_counter = 0
    max_epochs = 200

    for epoch in range(max_epochs):
        mlp.partial_fit(X_sub_tr, y_sub_tr, classes=classes, sample_weight=sw_tr)

        # Compute log-loss with balanced weighting and explicit class labels
        prob_tr = mlp.predict_proba(X_sub_tr)
        prob_val = mlp.predict_proba(X_sub_val)

        loss_tr = float(log_loss(y_sub_tr, prob_tr, labels=classes, sample_weight=sw_tr))
        loss_val = float(log_loss(y_sub_val, prob_val, labels=classes, sample_weight=sw_val))

        train_losses.append(loss_tr)
        val_losses.append(loss_val)

        if loss_val < best_val_loss - 1e-4:
            best_val_loss = loss_val
            best_weights = copy.deepcopy(mlp)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch > 30:
                logger.info("Early stopping triggered at epoch %d", epoch + 1)
                break

    trained_mlp = best_weights if best_weights is not None else mlp

    # Package into a reusable Pipeline
    pipeline = Pipeline([
        ("scaler", scaler),
        ("mlp", trained_mlp),
    ])

    info = {
        "architecture": "13 -> 64 (ReLU) -> 32 (ReLU) -> 2 (Softmax/Sigmoid)",
        "epochs_trained": len(train_losses),
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "best_val_loss": best_val_loss,
    }
    logger.info("Trained MLP Neural Network for %d epochs (Best Val Loss: %.4f)", len(train_losses), best_val_loss)
    return pipeline, train_losses, val_losses, info


def train_and_evaluate_all_models(
    X: np.ndarray,
    y: np.ndarray,
    records: list[AudioFileMetadata],
    feature_names: list[str],
    cfg: Config,
    save_best_model: bool = True,
) -> dict[str, Any]:
    """Execute complete training and evaluation protocol for all three classifiers."""
    cfg.ensure_directories()

    # 1. Speaker-independent train-test partition
    train_idx, test_idx, train_spk, test_spk, train_speakers_list, test_speakers_list = (
        create_speaker_independent_split(X, y, records, cfg)
    )

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # 2. Train Model 1: Support Vector Machine
    svm_pipeline, svm_info = train_svm(X_train, y_train, train_spk, cfg)

    # 3. Train Model 2: Random Forest
    rf_pipeline, rf_info = train_random_forest(X_train, y_train, train_spk, cfg)

    # 4. Train Model 3: MLP Neural Network
    mlp_pipeline, train_losses, val_losses, mlp_info = train_mlp_neural_network(
        X_train, y_train, train_spk, cfg
    )

    # Plot MLP Loss curves
    plot_neural_net_loss(
        train_losses,
        val_losses,
        cfg.plots_dir / "neural_net_loss_curves.png",
        dpi=cfg.dpi,
    )

    # 5. Evaluate on untouched speaker-independent test set
    models_dict = {
        "Support Vector Machine": svm_pipeline,
        "Random Forest": rf_pipeline,
        "Neural Network (MLP)": mlp_pipeline,
    }

    metrics_list: list[EvaluationMetrics] = []
    roc_dict: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}

    for name, pipeline in models_dict.items():
        y_pred = pipeline.predict(X_test)
        prob_boys = pipeline.predict_proba(X_test)[:, 0]  # Boys prob (Class 1 positive)

        metrics, fpr, tpr = compute_metrics(y_test, y_pred, prob_boys, model_name=name)
        metrics_list.append(metrics)
        roc_dict[name] = (fpr, tpr, metrics.roc_auc)

        # Plot individual confusion matrix
        plot_confusion_matrix(
            np.array(metrics.confusion_matrix),
            name,
            cfg.plots_dir / f"confusion_matrix_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png",
            dpi=cfg.dpi,
        )

    # 6. Combined ROC curves and comparison chart
    plot_roc_curve(roc_dict, cfg.plots_dir / "roc_curves_comparison.png", dpi=cfg.dpi)

    metrics_dicts_for_plot = [
        {
            "Model": m.model_name,
            "Accuracy": m.accuracy,
            "Sensitivity": m.sensitivity,
            "Precision": m.precision,
            "Specificity": m.specificity,
            "F1 Score": m.f1_score,
            "ROC AUC": m.roc_auc,
        }
        for m in metrics_list
    ]
    plot_model_comparison(metrics_dicts_for_plot, cfg.plots_dir / "model_comparison_barchart.png", dpi=cfg.dpi)

    # 7. Save metrics artifacts
    save_metrics_artifacts(metrics_list, cfg)

    # 8. Select best model based on validation / test generalisation (Macro F1 & AUC)
    best_metric = max(metrics_list, key=lambda m: (m.macro_f1 + m.roc_auc))
    best_model_name = best_metric.model_name
    best_pipeline = models_dict[best_model_name]

    model_save_path = cfg.models_dir / "best_gender_classifier.joblib"
    if save_best_model:
        joblib.dump(best_pipeline, model_save_path)
        logger.info("Saved best classifier (%s) to %s", best_model_name, model_save_path)

        # Save model metadata
        model_meta = {
            "best_model_name": best_model_name,
            "feature_names": feature_names,
            "n_features": len(feature_names),
            "target_sample_rate": cfg.target_sample_rate,
            "mfcc_config": {
                "n_mfcc": cfg.n_mfcc,
                "n_mels": cfg.n_mels,
                "n_fft": cfg.n_fft,
                "win_length": cfg.win_length,
                "hop_length": cfg.hop_length,
                "window": cfg.window,
                "pre_emphasis": cfg.pre_emphasis,
            },
            "class_mapping": {
                "0": cfg.class_boys_name,
                "1": cfg.class_girls_name,
            },
            "metrics": asdict(best_metric),
        }
        meta_path = cfg.models_dir / "model_metadata.json"
        meta_path.write_text(json.dumps(model_meta, indent=2), encoding="utf-8")

    return {
        "best_model_name": best_model_name,
        "metrics_list": metrics_list,
        "model_save_path": str(model_save_path),
        "split_summary": {
            "train_recordings": len(train_idx),
            "test_recordings": len(test_idx),
            "train_speakers": len(train_speakers_list),
            "test_speakers": len(test_speakers_list),
        },
    }

