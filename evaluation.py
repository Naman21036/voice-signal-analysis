"""Controlled evaluation of the silence detectors on SYNTHETIC signals with known ground truth.

These signals exist only to test the algorithms. They are never used in place of the
real assignment recordings.

Metrics (per scenario and detector):
  * frame-level precision / recall / F1 / false-positive rate / false-negative rate on a 10 ms grid
  * boundary error (ms) of the first start and last end of the detected activity vs. the truth
  * duration error after trimming (processed - ideal, seconds)
  * fraction of true speech samples that survive trimming (must be ~1.0 to be acceptable)

The improved detector reports confident speech and 'ambiguous' regions; for activity metrics both are
counted as active because both are preserved by default. The baseline has no ambiguous class.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps

import assignment1 as a

FS = 16000
GRID_MS = 10.0


def speech_like(duration: float, rms: float, rng: np.random.Generator, f0: float = 120.0, ramp_ms: float = 20.0,
                syllable_dip: float = 0.0) -> np.ndarray:
    """Harmonic complex with a spectral tilt, 4 Hz syllable modulation and raised-cosine ramps."""
    n = int(duration * FS)
    t = np.arange(n) / FS
    x = sum(np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 2 * np.pi)) / h for h in range(1, 25))
    x = x * (1 - syllable_dip * 0.5 * (1 + np.sin(2 * np.pi * 4 * t)))
    ramp = int(ramp_ms * FS / 1000)
    if ramp > 0 and n > 2 * ramp:
        w = 0.5 * (1 - np.cos(np.pi * np.arange(ramp) / ramp))
        x[:ramp] *= w
        x[-ramp:] *= w[::-1]
    return x / np.sqrt(np.mean(x ** 2)) * rms


def noise(n: int, rms: float, rng: np.random.Generator) -> np.ndarray:
    """Low-passed (4 kHz) Gaussian noise, a crude room-noise model."""
    b, a_ = sps.butter(2, 4000 / (FS / 2))
    x = sps.lfilter(b, a_, rng.standard_normal(n))
    return x / np.sqrt(np.mean(x ** 2)) * rms


def build(segments: list[tuple[float, float, float]], total: float, noise_rms: float, seed: int, ramp_ms: float = 20.0,
          dip: float = 0.0, stereo: str | None = None) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """segments = [(start_s, end_s, rms)]. Returns (samples (N, channels), truth [(start, end)])."""
    rng = np.random.default_rng(seed)
    n = int(total * FS)
    x = noise(n, noise_rms, rng) if noise_rms > 0 else np.zeros(n)
    for t0, t1, rms in segments:
        seg = speech_like(t1 - t0, rms, rng, ramp_ms=ramp_ms, syllable_dip=dip)
        i0 = int(t0 * FS)
        x[i0:i0 + seg.size] += seg
    if stereo == "opposite":
        # same content, polarity-inverted right channel: a plain channel average cancels it
        y = np.stack([x, -x], axis=1)
    elif stereo == "unequal":
        y = np.stack([x, 0.1 * x], axis=1)
    else:
        y = x[:, None]
    return y, [(t0, t1) for t0, t1, _ in segments]


def scenarios() -> dict[str, dict]:
    two = [(0.5, 1.1, 0.1), (1.6, 2.4, 0.1)]
    s = {}
    s["clean"] = build(two, 3.0, 1e-4, 1)
    for snr in (30, 20, 10):
        s[f"snr_{snr}dB"] = build(two, 3.0, 0.1 / 10 ** (snr / 20), 2)
    s["quiet_second_word_(-20dB)"] = build([(0.5, 1.1, 0.1), (1.6, 2.4, 0.01)], 3.0, 0.001, 3)
    s["pause_150ms"] = build([(0.5, 1.1, 0.1), (1.25, 2.0, 0.1)], 2.7, 1e-3, 4)
    s["pause_600ms"] = build([(0.5, 1.1, 0.1), (1.7, 2.4, 0.1)], 3.0, 1e-3, 5)
    s["little_silence_(speech_fills_file)"] = build([(0.1, 2.9, 0.1)], 3.0, 1e-3, 6, dip=0.8)
    s["near_guard_(SNR_16dB)"] = build(two, 3.0, 0.1 / 10 ** (16 / 20), 7)
    s["abrupt_onsets_(no_ramp)"] = build(two, 3.0, 1e-3, 8, ramp_ms=0)
    s["stereo_opposite_polarity"] = build(two, 3.0, 1e-3, 9, stereo="opposite")
    s["stereo_unequal_levels_(R=-20dB)"] = build(two, 3.0, 1e-3, 10, stereo="unequal")
    # transient click (15 ms) and a 300 ms noise burst inside the leading silence: the truth is still silence
    x, truth = build(two, 3.0, 1e-3, 11)
    x[int(0.2 * FS):int(0.215 * FS), 0] += 0.3
    s["click_in_leading_silence"] = (x, truth)
    x, truth = build(two, 3.0, 1e-3, 12)
    x[int(0.1 * FS):int(0.4 * FS), 0] += noise(int(0.3 * FS), 0.02, np.random.default_rng(13))
    s["noise_burst_300ms_(-34dBFS)_in_silence"] = (x, truth)
    return {k: {"x": v[0], "truth": v[1]} for k, v in s.items()}


def _mask(regions: list[tuple[float, float]], n_grid: int) -> np.ndarray:
    m = np.zeros(n_grid, dtype=bool)
    for t0, t1 in regions:
        m[int(round(t0 * 1000 / GRID_MS)):int(round(t1 * 1000 / GRID_MS))] = True
    return m


def frame_metrics(truth: np.ndarray, det: np.ndarray) -> dict:
    tp = int((truth & det).sum()); fp = int((~truth & det).sum()); fn = int((truth & ~det).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "false_positive_rate": fp / max(1, int((~truth).sum())), "false_negative_rate": fn / max(1, int(truth.sum()))}


def _mask_ms(regions, n):  # helper kept explicit for the 1 ms grid used above
    m = np.zeros(n, dtype=bool)
    for t0, t1 in regions:
        m[int(round(t0 * 1000)):int(round(t1 * 1000))] = True
    return m


def evaluate_one(x: np.ndarray, truth: list[tuple[float, float]], det: dict, cfg: a.Config) -> dict:
    total = x.shape[0] / FS
    n_grid = int(round(total * 1000 / GRID_MS))
    regions = [(r["start_s"], r["end_s"]) for r in det["active"]]
    out = frame_metrics(_mask(truth, n_grid), _mask(regions, n_grid))
    speech_only = [(r["start_s"], r["end_s"]) for r in det["speech"]]
    out["speech_only_recall"] = frame_metrics(_mask(truth, n_grid), _mask(speech_only, n_grid))["recall"]
    if regions:
        out["start_error_ms"] = (regions[0][0] - truth[0][0]) * 1000
        out["end_error_ms"] = (regions[-1][1] - truth[-1][1]) * 1000
    else:
        out["start_error_ms"] = out["end_error_ms"] = None
    proc, info = a.trim_silence(x, FS, det, cfg)
    pad = cfg.edge_pad_ms / 1000
    ideal = min(total, truth[-1][1] + pad) - max(0.0, truth[0][0] - pad)
    for g0, g1 in zip([t[1] for t in truth[:-1]], [t[0] for t in truth[1:]]):
        if (g1 - g0) * 1000 > cfg.max_internal_gap_ms:
            ideal -= (g1 - g0) - cfg.kept_gap_ms / 1000
    out["processed_duration_s"] = proc.shape[0] / FS
    out["ideal_duration_s"] = ideal
    out["duration_error_s"] = proc.shape[0] / FS - ideal
    # fraction of true speech samples inside the kept ranges
    n_ms = int(round(total * 1000)) + 1  # 1 ms grid
    kept = _mask_ms([tuple(k) for k in info["kept_segments"]], n_ms)
    tm = _mask_ms(truth, n_ms)
    out["speech_preserved_fraction"] = float((kept & tm).sum() / max(1, tm.sum()))
    out["detected_speech_s"] = float(sum(r["duration_s"] for r in det["speech"]))
    out["detected_ambiguous_s"] = float(sum(r["duration_s"] for r in det.get("ambiguous", [])))
    return out


def run_synthetic_evaluation(cfg: a.Config) -> dict:
    results = {"sampling_frequency_hz": FS, "grid_ms": GRID_MS, "scenarios": {}, "note":
               "Synthetic signals with known ground truth; used only to test the algorithms."}
    for name, sc in scenarios().items():
        x, truth = sc["x"], sc["truth"]
        base = evaluate_one(x, truth, a.detect_silence_baseline(x, FS, cfg), cfg)
        new = evaluate_one(x, truth, a.detect_silence(x, FS, cfg), cfg)
        results["scenarios"][name] = {"truth_segments_s": truth, "baseline": base, "improved": new}
    return results


def summary_rows(ev: dict) -> list[dict]:
    rows = []
    for name, r in ev["scenarios"].items():
        rows.append({"scenario": name, **{f"{d}_{m}": r[d][m] for d in ("baseline", "improved")
                                          for m in ("f1", "speech_preserved_fraction", "start_error_ms", "end_error_ms",
                                                    "duration_error_s")}})
    return rows
