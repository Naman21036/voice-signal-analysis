"""Automated tests for evaluation metrics, confusion matrix, ROC/AUC, and prediction."""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from assignment2.config import Config
from assignment2.evaluate import compute_metrics, safe_divide
from assignment2.predict import predict_audio_file


def test_safe_divide():
    assert safe_divide(10, 2) == 5.0
    assert safe_divide(10, 0, default=0.0) == 0.0


def test_compute_metrics_exact_values():
    # Boys = 0 (Positive class), Girls = 1 (Negative class)
    # y_true: 4 Boys, 2 Girls
    y_true = np.array([0, 0, 0, 0, 1, 1])
    # y_pred: 3 True Boys (TP=3), 1 False Girl (FN=1), 1 True Girl (TN=1), 1 False Boy (FP=1)
    y_pred = np.array([0, 0, 0, 1, 1, 0])
    prob_boys = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.6])

    metrics, fpr, tpr = compute_metrics(y_true, y_pred, prob_boys, "TestModel")

    assert metrics.tp == 3
    assert metrics.fn == 1
    assert metrics.fp == 1
    assert metrics.tn == 1

    # Accuracy = (3 + 1) / 6 = 4/6 = 0.6667
    assert np.isclose(metrics.accuracy, 4.0 / 6.0)
    # Sensitivity (Boys Recall) = TP / (TP + FN) = 3 / 4 = 0.75
    assert np.isclose(metrics.sensitivity, 0.75)
    # Precision (Boys Precision) = TP / (TP + FP) = 3 / 4 = 0.75
    assert np.isclose(metrics.precision, 0.75)
    # Specificity (Girls Recall) = TN / (TN + FP) = 1 / 2 = 0.50
    assert np.isclose(metrics.specificity, 0.50)
    # F1 Score = 2 * (0.75 * 0.75) / (0.75 + 0.75) = 0.75
    assert np.isclose(metrics.f1_score, 0.75)
    # ROC AUC should be valid between 0.0 and 1.0
    assert 0.0 <= metrics.roc_auc <= 1.0


def test_predict_interface(tmp_path):
    cfg = Config()
    # Create temporary WAV tone
    wav_path = tmp_path / "test_voice.wav"
    fs = 16000
    t = np.linspace(0, 1.0, fs)
    sig = 0.5 * np.sin(2 * np.pi * 350 * t)
    sf.write(str(wav_path), sig, fs, subtype="PCM_16")

    # Run prediction
    res = predict_audio_file(wav_path, cfg=cfg)

    assert "predicted_class_id" in res
    assert res["predicted_class_id"] in [1, 2]
    assert res["predicted_gender"] in ["Boys", "Girls"]
    assert "probabilities" in res
    assert len(res["mfcc_features"]) == 13
    assert "MFCC_1" in res["mfcc_features"]
    assert "MFCC_13" in res["mfcc_features"]

