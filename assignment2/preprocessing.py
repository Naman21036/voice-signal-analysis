"""Audio preprocessing pipeline for Assignment 2 Gender Classification.

Implements reliable WAV loading, anti-aliased polyphase resampling to 16 kHz,
phase-cancellation-safe stereo-to-mono downmixing, adaptive silence/speech boundary
detection with generous context padding to preserve unvoiced consonants and weak
speech, and export of standardized 16-bit PCM processed WAV recordings.
"""
from __future__ import annotations

import logging
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
import scipy.signal as sps
import soundfile as sf

from assignment2.config import Config
from assignment2.dataset import AudioFileMetadata

logger = logging.getLogger(__name__)


def to_mono_phase_safe(audio: np.ndarray) -> np.ndarray:
    """Convert multi-channel audio to mono without destructive phase cancellation.

    If channels are out-of-phase (e.g. inverted polarity), a direct channel average
    (L + R)/2 cancels speech energy. This function checks channel correlation;
    if strongly negative, it subtracts channels (or retains the higher-energy channel)
    to guarantee signal preservation.
    """
    if audio.ndim == 1:
        return audio
    if audio.shape[1] == 1:
        return audio[:, 0]

    ch1 = audio[:, 0]
    ch2 = audio[:, 1]
    norm1 = np.linalg.norm(ch1)
    norm2 = np.linalg.norm(ch2)
    denom = (norm1 * norm2) + 1e-12
    corr = np.dot(ch1, ch2) / denom

    if corr < -0.2:
        # Negative correlation indicates phase reversal / cancellation risk
        mono = (ch1 - ch2) / np.sqrt(2.0)
    else:
        # In-phase or multi-microphone setup: standard arithmetic mean
        mono = np.mean(audio, axis=1)

    return mono


def resample_audio(
    audio: np.ndarray, orig_fs: int, target_fs: int = 16000
) -> np.ndarray:
    """Resample 1D audio using polyphase anti-aliasing filter (scipy.signal.resample_poly)."""
    if orig_fs == target_fs:
        return audio
    common_divisor = gcd(orig_fs, target_fs)
    up = target_fs // common_divisor
    down = orig_fs // common_divisor
    return sps.resample_poly(audio, up, down, axis=0)


def detect_speech_boundaries(
    audio: np.ndarray,
    fs: int = 16000,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    edge_pad_ms: float = 100.0,
) -> tuple[int, int]:
    """Identify speech start and end indices using robust frame energy with hysteresis.

    Preserves weak speech components by using a conservative lower threshold
    and adding safety edge context padding before the first and after the last active frame.
    """
    frame_len = max(1, int(round(frame_ms * fs / 1000.0)))
    hop_len = max(1, int(round(hop_ms * fs / 1000.0)))
    pad_samples = int(round(edge_pad_ms * fs / 1000.0))

    if len(audio) < frame_len:
        return 0, len(audio)

    # Frame energy in dB
    num_frames = 1 + (len(audio) - frame_len) // hop_len
    idx = np.arange(frame_len)[None, :] + hop_len * np.arange(num_frames)[:, None]
    frames = audio[idx]
    power = np.mean(frames**2, axis=1)
    eps = 1e-12
    rms_db = 10.0 * np.log10(power + eps)

    # Robust noise floor and speech level estimation
    valid_frames = rms_db[rms_db > -80.0]
    if len(valid_frames) == 0:
        return 0, len(audio)

    noise_floor = float(np.percentile(valid_frames, 15.0))
    speech_peak = float(np.percentile(valid_frames, 95.0))

    # If SNR is too low, do not attempt aggressive trimming
    if (speech_peak - noise_floor) < 10.0:
        return 0, len(audio)

    # Hysteresis threshold
    threshold = noise_floor + 0.3 * (speech_peak - noise_floor)
    active_mask = rms_db >= threshold

    if not np.any(active_mask):
        return 0, len(audio)

    first_active_frame = int(np.argmax(active_mask))
    last_active_frame = int(len(active_mask) - 1 - np.argmax(active_mask[::-1]))

    start_sample = max(0, first_active_frame * hop_len - pad_samples)
    end_sample = min(len(audio), (last_active_frame * hop_len + frame_len) + pad_samples)

    # Safety check: if trimmed duration is less than 0.25s, return full audio
    if (end_sample - start_sample) < int(0.25 * fs):
        return 0, len(audio)

    return start_sample, end_sample


def preprocess_signal(
    raw_audio: np.ndarray,
    orig_fs: int,
    cfg: Config,
    apply_trimming: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Complete preprocessing for a raw audio buffer:

    1. Convert stereo to mono with phase cancellation protection.
    2. Resample to standard target sampling rate (16 kHz).
    3. Detect speech boundaries and conservatively trim inactive margins with padding.
    4. Normalize peak amplitude to prevent clipping and ensure consistent dynamic range.
    """
    mono = to_mono_phase_safe(raw_audio)
    resampled = resample_audio(mono, orig_fs, cfg.target_sample_rate)

    trim_info = {"trimmed": False, "orig_samples": len(resampled)}
    if apply_trimming:
        start_idx, end_idx = detect_speech_boundaries(
            resampled,
            fs=cfg.target_sample_rate,
            frame_ms=cfg.frame_ms,
            hop_ms=cfg.hop_ms,
            edge_pad_ms=cfg.edge_pad_ms,
        )
        if start_idx > 0 or end_idx < len(resampled):
            processed = resampled[start_idx:end_idx]
            trim_info["trimmed"] = True
            trim_info["start_s"] = start_idx / cfg.target_sample_rate
            trim_info["end_s"] = end_idx / cfg.target_sample_rate
            trim_info["removed_s"] = (len(resampled) - len(processed)) / cfg.target_sample_rate
        else:
            processed = resampled
    else:
        processed = resampled

    # Peak normalization to -1.0 dBFS (0.891) if active, avoiding clipping
    peak = float(np.max(np.abs(processed))) if processed.size > 0 else 0.0
    if peak > 1e-4:
        processed = processed / peak * 0.90

    trim_info["final_samples"] = len(processed)
    trim_info["final_duration_s"] = len(processed) / cfg.target_sample_rate
    return processed, trim_info


def preprocess_dataset(
    records: list[AudioFileMetadata],
    cfg: Config,
    save_processed_wav: bool = True,
) -> dict[str, np.ndarray]:
    """Preprocess all dataset recordings and optionally save standardized WAV files."""
    cfg.ensure_directories()
    processed_signals: dict[str, np.ndarray] = {}

    for r in records:
        data, fs = sf.read(r.audio_path, dtype="float64", always_2d=True)
        processed, info = preprocess_signal(data, fs, cfg, apply_trimming=True)
        processed_signals[r.file_id] = processed

        if save_processed_wav:
            out_filename = f"{r.file_id}_{r.gender}_{r.speaker_id}_{r.utterance_type}.wav"
            out_path = cfg.processed_audio_dir / out_filename
            sf.write(str(out_path), processed, cfg.target_sample_rate, subtype="PCM_16")

    logger.info(
        "Preprocessed %d recordings (Target FS: %d Hz). Saved to %s",
        len(processed_signals),
        cfg.target_sample_rate,
        cfg.processed_audio_dir if save_processed_wav else "memory",
    )
    return processed_signals

