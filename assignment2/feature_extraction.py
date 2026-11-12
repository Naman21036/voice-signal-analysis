"""Acoustic feature extraction: Mel-Frequency Cepstral Coefficients (MFCC).

Extracts 13 MFCC coefficients per frame according to standard speech processing:
pre-emphasis filter, framing with 25 ms Hann window (400 samples) and 10 ms hop (160 samples),
512-point FFT power spectrum, 26-channel Mel triangular filterbank, log-energy compression,
and orthonormal Type-II Discrete Cosine Transform (DCT-II).

Performs temporal mean aggregation to yield the mandatory 13-dimensional fixed-size
feature vector (MFCC_1 to MFCC_13) and exports structured feature CSV datasets.
Also provides a 26-dimensional mean+std aggregation for comparative experimentation.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import scipy.fftpack as fftpack

from assignment2.config import Config
from assignment2.dataset import AudioFileMetadata

logger = logging.getLogger(__name__)


def apply_preemphasis(signal: np.ndarray, alpha: float = 0.97) -> np.ndarray:
    """Apply first-order high-pass pre-emphasis filter: y[n] = x[n] - alpha * x[n-1].

    Compensates for the negative spectral tilt of speech (-6 dB/octave) caused by
    glottal airflow dynamics, emphasizing higher formant frequencies.
    """
    if len(signal) == 0:
        return signal
    if alpha <= 0.0:
        return signal
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])


def hz_to_mel(hz: float | np.ndarray) -> float | np.ndarray:
    """Convert acoustic frequency in Hertz to perceived pitch in Mels."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float | np.ndarray) -> float | np.ndarray:
    """Convert pitch in Mels back to acoustic frequency in Hertz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_mel_filterbank(
    n_mels: int = 26,
    n_fft: int = 512,
    fs: int = 16000,
    f_min: float = 0.0,
    f_max: float = 8000.0,
) -> np.ndarray:
    """Construct a bank of triangular Mel filters spaced linearly on the Mel scale."""
    low_mel = hz_to_mel(f_min)
    high_mel = hz_to_mel(f_max)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / fs).astype(int)

    num_bins = int(n_fft // 2 + 1)
    fbank = np.zeros((n_mels, num_bins), dtype=np.float64)

    for m in range(1, n_mels + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]

        # Rising slope
        if f_m > f_m_minus:
            for k in range(f_m_minus, f_m):
                if k < num_bins:
                    fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)

        # Falling slope
        if f_m_plus > f_m:
            for k in range(f_m, f_m_plus):
                if k < num_bins:
                    fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    return fbank


def extract_mfcc_frames(
    audio: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    """Compute frame-level MFCC coefficients.

    Returns:
        np.ndarray of shape (num_frames, n_mfcc) with 13 coefficients per frame.
    """
    if len(audio) == 0:
        return np.zeros((0, cfg.n_mfcc), dtype=np.float64)

    # 1. Pre-emphasis
    emphasized = apply_preemphasis(audio, cfg.pre_emphasis)

    # 2. Framing and Hann windowing
    win_len = cfg.win_length
    hop_len = cfg.hop_length
    n_samples = len(emphasized)

    if n_samples < win_len:
        emphasized = np.pad(emphasized, (0, win_len - n_samples))
        n_samples = len(emphasized)

    num_frames = 1 + int(np.floor((n_samples - win_len) / hop_len))
    indices = (
        np.tile(np.arange(0, win_len), (num_frames, 1))
        + np.tile(np.arange(0, num_frames * hop_len, hop_len), (win_len, 1)).T
    )
    frames = emphasized[indices.astype(np.int32, copy=False)]

    window = np.hanning(win_len)
    windowed_frames = frames * window

    # 3. FFT and Power Spectrum
    mag_spectrum = np.abs(np.fft.rfft(windowed_frames, cfg.n_fft))
    power_spectrum = (1.0 / cfg.n_fft) * (mag_spectrum**2)

    # 4. Mel Filterbank Energy
    fbank = build_mel_filterbank(
        n_mels=cfg.n_mels,
        n_fft=cfg.n_fft,
        fs=cfg.target_sample_rate,
        f_min=cfg.f_min,
        f_max=cfg.f_max,
    )
    mel_energies = np.dot(power_spectrum, fbank.T)
    mel_energies = np.where(mel_energies == 0, np.finfo(float).eps, mel_energies)

    # 5. Log compression
    log_mel_energies = np.log(mel_energies)

    # 6. Discrete Cosine Transform (DCT-II, orthonormal)
    mfcc = fftpack.dct(log_mel_energies, type=2, axis=1, norm="ortho")

    # Retain the exact first 13 coefficients
    return mfcc[:, : cfg.n_mfcc]


def extract_features_13(audio: np.ndarray, cfg: Config) -> np.ndarray:
    """Primary 13-dimensional feature representation: temporal mean of each MFCC coefficient."""
    frames = extract_mfcc_frames(audio, cfg)
    if frames.shape[0] == 0:
        return np.zeros(cfg.n_mfcc, dtype=np.float64)
    return np.mean(frames, axis=0)


def extract_features_26(audio: np.ndarray, cfg: Config) -> np.ndarray:
    """Optional 26-dimensional feature representation: temporal mean and standard deviation."""
    frames = extract_mfcc_frames(audio, cfg)
    if frames.shape[0] == 0:
        return np.zeros(cfg.n_mfcc * 2, dtype=np.float64)
    mean_vec = np.mean(frames, axis=0)
    std_vec = np.std(frames, axis=0)
    return np.concatenate([mean_vec, std_vec])


def extract_dataset_features(
    records: list[AudioFileMetadata],
    preprocessed_signals: dict[str, np.ndarray],
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract 13-dimensional MFCC vectors for all dataset recordings and export CSVs.

    Returns:
        X (np.ndarray): Shape (N, 13)
        y (np.ndarray): Shape (N,)
        feature_names (list[str]): ['MFCC_1', ..., 'MFCC_13']
    """
    cfg.ensure_directories()
    feature_names_13 = [f"MFCC_{i}" for i in range(1, cfg.n_mfcc + 1)]
    feature_names_26 = feature_names_13 + [f"MFCC_{i}_std" for i in range(1, cfg.n_mfcc + 1)]

    rows_13: list[dict[str, Any]] = []
    rows_26: list[dict[str, Any]] = []
    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for r in records:
        signal = preprocessed_signals[r.file_id]
        feat_13 = extract_features_13(signal, cfg)
        feat_26 = extract_features_26(signal, cfg)

        # Validation: finite check
        if np.isnan(feat_13).any() or np.isinf(feat_13).any():
            raise ValueError(f"Extracted invalid non-finite MFCC values for file {r.filename}")

        X_list.append(feat_13)
        y_list.append(r.label)

        # 13-dim row
        base_meta = {
            "file_id": r.file_id,
            "filename": r.filename,
            "speaker_id": r.speaker_id,
            "gender": r.gender,
            "label": r.label,
            "utterance_type": r.utterance_type,
        }
        row_13 = dict(base_meta)
        for name, val in zip(feature_names_13, feat_13):
            row_13[name] = float(val)
        rows_13.append(row_13)

        # 26-dim row
        row_26 = dict(base_meta)
        for name, val in zip(feature_names_26, feat_26):
            row_26[name] = float(val)
        rows_26.append(row_26)

    # Save CSVs
    csv_13_path = cfg.features_dir / "mfcc_features_13.csv"
    with open(csv_13_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_13[0].keys()))
        writer.writeheader()
        writer.writerows(rows_13)
    logger.info("Saved primary 13-dim feature CSV: %s (%d records)", csv_13_path, len(rows_13))

    csv_26_path = cfg.features_dir / "mfcc_features_26.csv"
    with open(csv_26_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_26[0].keys()))
        writer.writeheader()
        writer.writerows(rows_26)
    logger.info("Saved optional 26-dim feature CSV: %s (%d records)", csv_26_path, len(rows_26))

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.int64)

    return X, y, feature_names_13

