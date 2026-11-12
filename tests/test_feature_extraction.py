"""Automated tests for 13-coefficient MFCC extraction and feature aggregation."""
import numpy as np
import pytest

from assignment2.config import Config
from assignment2.feature_extraction import (
    apply_preemphasis,
    build_mel_filterbank,
    extract_features_13,
    extract_features_26,
    extract_mfcc_frames,
    hz_to_mel,
    mel_to_hz,
)


def test_mel_conversion_bijective():
    hz_values = np.array([100.0, 500.0, 1000.0, 4000.0, 8000.0])
    mels = hz_to_mel(hz_values)
    hz_reconstructed = mel_to_hz(mels)
    assert np.allclose(hz_values, hz_reconstructed, atol=1e-5)


def test_build_mel_filterbank_shape():
    fbank = build_mel_filterbank(n_mels=26, n_fft=512, fs=16000)
    assert fbank.shape == (26, 257)
    # Filterbank energies should be non-negative
    assert (fbank >= 0.0).all()
    # Peak of triangular filters should be at or near 1.0
    assert np.max(fbank) <= 1.0


def test_apply_preemphasis():
    sig = np.array([1.0, 2.0, 3.0, 4.0])
    alpha = 0.97
    emp = apply_preemphasis(sig, alpha)
    assert emp[0] == 1.0
    assert np.isclose(emp[1], 2.0 - 0.97 * 1.0)


def test_extract_mfcc_frames_dimensions():
    cfg = Config()
    t = np.linspace(0, 1.0, 16000)
    tone = np.sin(2 * np.pi * 300 * t)

    frames = extract_mfcc_frames(tone, cfg)
    # Exactly 13 coefficients per frame
    assert frames.ndim == 2
    assert frames.shape[1] == 13
    assert frames.shape[0] > 0
    assert np.isfinite(frames).all()


def test_extract_features_13_exact_shape():
    cfg = Config()
    t = np.linspace(0, 1.5, int(1.5 * 16000))
    speech_sim = np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 440 * t)

    feat13 = extract_features_13(speech_sim, cfg)
    assert feat13.shape == (13,)
    assert np.isfinite(feat13).all()
    assert not np.isnan(feat13).any()


def test_extract_features_26_exact_shape():
    cfg = Config()
    t = np.linspace(0, 1.5, int(1.5 * 16000))
    speech_sim = np.sin(2 * np.pi * 220 * t)

    feat26 = extract_features_26(speech_sim, cfg)
    assert feat26.shape == (26,)
    assert np.isfinite(feat26).all()

