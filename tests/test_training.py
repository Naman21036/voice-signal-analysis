"""Automated tests for model training and speaker leakage prevention."""
from pathlib import Path

import joblib
import numpy as np
import pytest

from assignment2.config import Config
from assignment2.dataset import AudioFileMetadata
from assignment2.train import (
    create_speaker_independent_split,
    train_mlp_neural_network,
    train_random_forest,
    train_svm,
)


def test_zero_speaker_leakage_split():
    cfg = Config()
    # Mock records with multiple recordings per speaker
    records = []
    for spk_id in range(20):
        gender = "Boys" if spk_id < 16 else "Girls"
        label = 0 if gender == "Boys" else 1
        for rec_num in range(2):  # 2 recordings per speaker
            records.append(
                AudioFileMetadata(
                    file_id=f"REC_{spk_id}_{rec_num}",
                    filename=f"Spk_{spk_id}_utt_{rec_num}.wav",
                    gender=gender,
                    label=label,
                    duration=2.0,
                    sample_rate=16000,
                    channels=1,
                    num_samples=32000,
                    peak_amplitude=0.8,
                    rms_amplitude=0.2,
                    audio_path="",
                    integrity_status="VALID",
                    sha256_hash=f"hash_{spk_id}_{rec_num}",
                    speaker_id=f"Speaker_{spk_id}",
                    utterance_type="BIT" if rec_num == 0 else "Name",
                )
            )

    X = np.random.randn(len(records), 13)
    y = np.array([r.label for r in records])

    train_idx, test_idx, tr_spk, te_spk, tr_list, te_list = (
        create_speaker_independent_split(X, y, records, cfg)
    )

    # Verification: Zero speaker overlap!
    overlap = set(tr_list).intersection(set(te_list))
    assert len(overlap) == 0, f"Speaker leakage detected: {overlap}"
    assert len(test_idx) > 0
    assert len(train_idx) > 0


def test_svm_training_and_serialization(tmp_path):
    cfg = Config()
    X = np.random.randn(40, 13)
    y = np.array([0] * 32 + [1] * 8)
    spk = np.array([f"Spk_{i}" for i in range(40)])

    pipe, info = train_svm(X, y, spk, cfg)
    preds = pipe.predict(X)
    probs = pipe.predict_proba(X)

    assert preds.shape == (40,)
    assert probs.shape == (40, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)

    # Test serialization
    model_file = tmp_path / "model.joblib"
    joblib.dump(pipe, model_file)
    loaded = joblib.load(model_file)
    assert np.allclose(pipe.predict_proba(X), loaded.predict_proba(X))


def test_random_forest_training():
    cfg = Config()
    X = np.random.randn(40, 13)
    y = np.array([0] * 32 + [1] * 8)
    spk = np.array([f"Spk_{i}" for i in range(40)])

    pipe, info = train_random_forest(X, y, spk, cfg)
    assert len(info["feature_importances"]) == 13
    assert abs(sum(info["feature_importances"]) - 1.0) < 1e-4


def test_mlp_training_and_loss_convergence():
    cfg = Config()
    X = np.random.randn(50, 13)
    y = np.array([0] * 40 + [1] * 10)
    spk = np.array([f"Spk_{i}" for i in range(50)])

    pipe, train_losses, val_losses, info = train_mlp_neural_network(X, y, spk, cfg)
    assert len(train_losses) > 0
    assert len(val_losses) == len(train_losses)
    # Final loss should be lower than initial loss
    assert train_losses[-1] <= train_losses[0]

