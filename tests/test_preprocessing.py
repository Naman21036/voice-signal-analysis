"""Automated tests for audio preprocessing (resampling, stereo downmixing, silence boundaries)."""
import numpy as np
import pytest

from assignment2.config import Config
from assignment2.preprocessing import (
    detect_speech_boundaries,
    preprocess_signal,
    resample_audio,
    to_mono_phase_safe,
)


def test_to_mono_phase_safe_in_phase():
    # 2 identical in-phase channels
    t = np.linspace(0, 1.0, 16000)
    sig = np.sin(2 * np.pi * 300 * t)
    stereo = np.stack([sig, sig], axis=1)
    mono = to_mono_phase_safe(stereo)
    assert mono.shape == (16000,)
    assert np.allclose(mono, sig, atol=1e-5)


def test_to_mono_phase_safe_opposite_polarity():
    # Inverted polarity: (L + R) / 2 would cancel completely to 0!
    t = np.linspace(0, 1.0, 16000)
    sig = np.sin(2 * np.pi * 300 * t)
    stereo = np.stack([sig, -sig], axis=1)
    mono = to_mono_phase_safe(stereo)
    # Must NOT cancel to zero
    assert mono.shape == (16000,)
    assert np.max(np.abs(mono)) > 0.5


def test_resample_audio_exact_lengths():
    # 48 kHz -> 16 kHz (factor 3 downsampling)
    sig48 = np.sin(2 * np.pi * 400 * np.linspace(0, 1.0, 48000))
    res16 = resample_audio(sig48, 48000, 16000)
    assert res16.shape == (16000,)

    # 44.1 kHz -> 16 kHz
    sig44 = np.sin(2 * np.pi * 400 * np.linspace(0, 1.0, 44100))
    res44_16 = resample_audio(sig44, 44100, 16000)
    assert res44_16.shape == (16000,)


def test_detect_speech_boundaries_with_padding():
    fs = 16000
    # 0.5s silence + 1.0s tone + 0.5s silence = 2.0s
    t = np.linspace(0, 1.0, 16000)
    tone = 0.5 * np.sin(2 * np.pi * 400 * t)
    silence = np.zeros(8000)
    audio = np.concatenate([silence, tone, silence])

    start_idx, end_idx = detect_speech_boundaries(audio, fs=fs, edge_pad_ms=100.0)
    # Speech starts at 0.5s (8000 samples). With 100ms padding (1600 samples), start <= 8000
    assert start_idx <= 8000
    assert end_idx >= 24000


def test_preprocess_signal_full():
    cfg = Config()
    fs = 48000
    audio = np.random.randn(fs * 2, 2) * 0.1
    processed, info = preprocess_signal(audio, fs, cfg, apply_trimming=True)
    assert processed.ndim == 1
    assert np.isfinite(processed).all()
    assert np.max(np.abs(processed)) <= 0.95

