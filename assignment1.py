#!/usr/bin/env python3
"""EC24313 Artificial Intelligence and Machine Learning - Assignment 1.

Analyses two voice recordings (time-domain waveform, silence detection,
conservative silence trimming, spectrogram) and writes plots, processed WAV
files, analysis_results.json and analysis_report.md.

Usage:  python assignment1.py [--audio-dir audio] [--output-dir outputs]
All paths are relative to this file's directory unless absolute.

Two detectors are implemented:
  * detect_silence_baseline - the original single-threshold percentile detector
    (kept unchanged so that old and new results can be compared);
  * detect_silence          - the improved hysteresis detector (default).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy import signal  # noqa: E402

ROOT = Path(__file__).resolve().parent
EPS = 1e-12
DPI = 250

# Fallback identification when audio/manifest.json is absent:
# key -> (filename keywords, declared utterance)
DEFAULT_RECORDINGS = {
    "bit": (("bit",), "BIT"),
    "name": (("name", "naman_gupta", "gupta"), "Naman Gupta"),
}


# --------------------------------------------------------------------------- config
@dataclass
class Config:
    # --- framing (both detectors)
    frame_ms: float = 25.0           # analysis frame length
    hop_ms: float = 10.0             # hop between frames (60 % overlap)
    min_silence_ms: float = 100.0    # shorter low-energy dips are bridged (pauses inside speech)
    min_speech_ms: float = 30.0      # shorter active runs are discarded as transients
    # --- baseline detector (single threshold, percentile noise estimate)
    noise_percentile: float = 10.0
    speech_percentile: float = 95.0
    floor_margin_db: float = 12.0    # threshold >= noise floor + this
    speech_range_db: float = 35.0    # threshold >= speech level - this
    # --- improved detector (hysteresis, robust noise estimate)
    noise_exclude_db: float = -90.0  # frames below this (digital zero) are excluded from the noise estimate
    noise_fraction: float = 0.1      # noise floor = median of the quietest 10 % of the remaining frames
    low_margin_db: float = 8.0       # low (hold) threshold  = noise + this
    high_margin_db: float = 20.0     # high (onset) threshold = noise + this, capped below the speech level
    level_gap_db: float = 10.0       # high threshold is kept at least this far below the speech level
    min_snr_db: float = 15.0         # below this level-noise separation nothing is declared silent
    min_core_ms: float = 30.0        # time above the high threshold needed to call a run confident speech
    max_extension_ms: float = 150.0  # confident speech extends at most this far beyond its high-threshold core
                                     # (soft onsets/decays); further low-level activity is 'ambiguous'
    trim_ambiguous: bool = False     # False (default): ambiguous regions are preserved; True: treated as removable
    # --- trimming
    edge_pad_ms: float = 100.0       # context kept before the first / after the last active region
    max_internal_gap_ms: float = 400.0  # confident-silence gaps between two speech regions longer than this are shortened
    kept_gap_ms: float = 200.0       # length such a gap is shortened to
    fade_ms: float = 5.0             # fade at every cut to avoid clicks
    # --- spectrogram
    spec_nfft: int = 1024
    spec_hop: int = 128
    spec_range_db: float = 80.0
    spec_max_hz: float = 8000.0


# --------------------------------------------------------------------------- loading / identity
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def discover_recordings(audio_dir: Path) -> dict[str, dict]:
    """Return {key: {path, declared_utterance, identity_source}}.

    Uses audio_dir/manifest.json when present (explicit file -> utterance mapping),
    otherwise falls back to filename keywords. Searches subdirectories.
    """
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    manifest = audio_dir / "manifest.json"
    found: dict[str, dict] = {}
    if manifest.is_file():
        entries = json.loads(manifest.read_text(encoding="utf-8"))["recordings"]
        for key, e in entries.items():
            p = audio_dir / e["file"]
            if not p.is_file():
                raise FileNotFoundError(f"Manifest lists '{e['file']}' for '{key}' but it does not exist in {audio_dir}")
            found[key] = {"path": p, "declared_utterance": e["declared_utterance"], "identity_source": "manifest.json"}
        return found
    files = sorted(p for p in audio_dir.rglob("*") if p.suffix.lower() in {".wav", ".flac", ".ogg"})
    for key, (keywords, phrase) in DEFAULT_RECORDINGS.items():
        matches = [p for p in files if any(k in p.stem.lower() for k in keywords)]
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected exactly one '{key}' recording in {audio_dir}, found {len(matches)}")
        found[key] = {"path": matches[0], "declared_utterance": phrase, "identity_source": "filename keyword"}
    return found


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Return float64 samples shaped (n_samples, n_channels) and the sampling frequency."""
    data, fs = sf.read(str(path), dtype="float64", always_2d=True)
    return data, int(fs)


def to_mono(data: np.ndarray) -> np.ndarray:
    """Channel average (used for plotting spectrograms and the baseline detector only)."""
    return data if data.ndim == 1 else data.mean(axis=1)


def get_metadata(path: Path, data: np.ndarray, fs: int) -> dict:
    info = sf.info(str(path))
    return {
        "source_file": path.name,
        "format": info.format, "subtype": info.subtype,
        "sampling_frequency_hz": fs,
        "sampling_frequency_source": "WAV header (fmt chunk) read via soundfile.info()",
        "channels": data.shape[1],
        "num_samples": data.shape[0],
        "duration_s": data.shape[0] / fs if fs else 0.0,
    }


def signal_stats(data: np.ndarray) -> dict:
    if data.size == 0:
        return {"peak_amplitude": 0.0, "rms_amplitude": 0.0, "peak_db_fs": None, "rms_db_fs": None}
    peak = float(np.max(np.abs(data)))
    rms = float(np.sqrt(np.mean(data ** 2)))
    return {"peak_amplitude": peak, "rms_amplitude": rms,
            "peak_db_fs": float(20 * np.log10(peak + EPS)), "rms_db_fs": float(20 * np.log10(rms + EPS))}


# --------------------------------------------------------------------------- frame analysis
def frame_rms_db(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    """Short-time RMS in dB full scale.

    x may be 1-D or (samples, channels); energy is pooled over all channels, so
    out-of-phase channels cannot cancel as they would in a channel average.
    Signals shorter than one frame are zero-padded to give a single frame.
    """
    if x.size == 0:
        return np.array([])
    if x.shape[0] < frame:
        x = np.pad(x, [(0, frame - x.shape[0])] + [(0, 0)] * (x.ndim - 1))
    n_frames = 1 + (x.shape[0] - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n_frames)[:, None]
    seg = x[idx]  # (n_frames, frame) or (n_frames, frame, channels)
    power = np.mean(seg ** 2, axis=tuple(range(1, seg.ndim)))
    return 10 * np.log10(power + EPS)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end_exclusive) of consecutive True runs."""
    if mask.size == 0:
        return []
    d = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _region(t0: float, t1: float, kind: str) -> dict:
    return {"start_s": float(t0), "end_s": float(t1), "duration_s": float(t1 - t0), "type": kind}


def _empty_result(method: str, cfg: Config, frame: int, hop: int) -> dict:
    return {"method": method, "frame_ms": cfg.frame_ms, "hop_ms": cfg.hop_ms, "frame_samples": frame,
            "hop_samples": hop, "min_silence_ms": cfg.min_silence_ms, "min_speech_ms": cfg.min_speech_ms,
            "silence": [], "speech": [], "ambiguous": [], "active": [], "flags": [],
            "threshold_db": None, "low_threshold_db": None, "high_threshold_db": None,
            "noise_floor_db": None, "speech_level_db": None}


def _frame_params(fs: int, cfg: Config) -> tuple[int, int]:
    return max(1, int(round(cfg.frame_ms * fs / 1000))), max(1, int(round(cfg.hop_ms * fs / 1000)))


# --------------------------------------------------------------------------- baseline detector (unchanged behaviour)
def detect_silence_baseline(x: np.ndarray, fs: int, cfg: Config) -> dict:
    """Original detector: threshold = max(P10 + margin, P95 - range) on the mono mix.

    Kept for comparison. Frame i is attributed to its centre time.
    """
    x = to_mono(x)
    duration = x.size / fs if fs else 0.0
    frame, hop = _frame_params(fs, cfg)
    result = _empty_result("baseline: single percentile threshold on mono mix", cfg, frame, hop)
    db = frame_rms_db(x, frame, hop) if x.size else np.array([])
    if db.size == 0 or np.max(np.abs(x)) < 1e-9:
        if duration > 0:
            result["silence"] = [_region(0.0, duration, "entire recording")]
        return result
    floor = float(np.percentile(db, cfg.noise_percentile))
    level = float(np.percentile(db, cfg.speech_percentile))
    thr = max(floor + cfg.floor_margin_db, level - cfg.speech_range_db)
    active = db > thr
    min_sil = max(1, int(round(cfg.min_silence_ms / cfg.hop_ms)))
    min_sp = max(1, int(round(cfg.min_speech_ms / cfg.hop_ms)))
    for s, e in _runs(~active):
        if e - s < min_sil and s > 0 and e < active.size:
            active[s:e] = True
    for s, e in _runs(active):
        if e - s < min_sp:
            active[s:e] = False

    def to_time(i0: int, i1: int) -> tuple[float, float]:
        t0 = 0.0 if i0 == 0 else (i0 * hop + frame / 2) / fs
        t1 = duration if i1 >= active.size else (i1 * hop + frame / 2) / fs
        return float(t0), float(min(t1, duration))

    n = active.size
    for s, e in _runs(~active):
        t0, t1 = to_time(s, e)
        kind = "leading" if s == 0 else "trailing" if e == n else "internal"
        if t1 - t0 >= cfg.min_silence_ms / 1000 or kind != "internal":
            result["silence"].append(_region(t0, t1, kind))
    result["speech"] = [_region(*to_time(s, e), "speech") for s, e in _runs(active)]
    result["active"] = list(result["speech"])
    result.update(threshold_db=float(thr), noise_floor_db=floor, speech_level_db=level,
                  frame_rms_db=db.tolist(), frame_times_s=((np.arange(n) * hop + frame / 2) / fs).tolist())
    return result


# --------------------------------------------------------------------------- improved detector
def detect_silence(x: np.ndarray, fs: int, cfg: Config) -> dict:
    """Hysteresis energy detector with a robust noise estimate and an explicit 'ambiguous' class.

    1. Frame RMS (dB) pooled over all channels (25 ms frames, 10 ms hop).
    2. Noise floor = median of the quietest `noise_fraction` of frames, after excluding digital-zero
       frames (no information); spread = MAD of those frames. Speech level = 95th percentile.
    3. low  threshold = noise + low_margin_db     (activity is held above it)
       high threshold = noise + high_margin_db, kept >= level_gap_db below the speech level.
       The MAD of the quiet frames is reported as a stationarity check but does not set thresholds: on the
       real recordings it was inflated by non-stationary noise and pushed the threshold into soft speech.
    4. Frames above `low` form active runs; low dips shorter than min_silence_ms are bridged, runs
       shorter than min_speech_ms are discarded.
    5. Hysteresis: confident SPEECH needs at least min_core_ms of frames above `high`; it is then held
       through frames above `low`, but extends at most max_extension_ms beyond the high-threshold core
       (soft onsets and decays). Surviving low-level activity outside that (or in runs that never reach
       `high`) is AMBIGUOUS (breath, noise or weak speech) and is not trimmed away by default.
    6. If level - noise < min_snr_db no region is declared silent (everything non-zero is ambiguous).
    Frame i owns the span [centre - hop/2, centre + hop/2]; region boundaries use those spans.
    """
    x = np.asarray(x, dtype=np.float64)
    n_samples = x.shape[0]
    duration = n_samples / fs if fs else 0.0
    frame, hop = _frame_params(fs, cfg)
    result = _empty_result("hysteresis RMS energy, robust noise estimate, ambiguous class", cfg, frame, hop)
    result.update(min_core_ms=cfg.min_core_ms)
    db = frame_rms_db(x, frame, hop) if n_samples else np.array([])
    if db.size == 0 or np.max(np.abs(x)) < 1e-9:
        if duration > 0:
            result["silence"] = [_region(0.0, duration, "entire recording")]
        return result

    valid = db > cfg.noise_exclude_db
    dv = db[valid] if valid.sum() >= 5 else db
    k = max(3, int(round(cfg.noise_fraction * dv.size)))
    quiet = np.sort(dv)[:k]
    noise = float(np.median(quiet))
    mad = float(1.4826 * np.median(np.abs(quiet - noise)))
    level = float(np.percentile(dv, cfg.speech_percentile))
    low = noise + cfg.low_margin_db
    high = max(min(noise + cfg.high_margin_db, level - cfg.level_gap_db), low + 3.0)
    snr = level - noise
    if snr < cfg.min_snr_db:
        result["flags"].append(f"low separation between speech level and noise floor ({snr:.1f} dB < {cfg.min_snr_db} dB): "
                               "no region declared silent")
        low, high = cfg.noise_exclude_db, np.inf

    active = db > low
    n = active.size
    min_sil = max(1, int(round(cfg.min_silence_ms / cfg.hop_ms)))
    min_sp = max(1, int(round(cfg.min_speech_ms / cfg.hop_ms)))
    min_core = max(1, int(round(cfg.min_core_ms / cfg.hop_ms)))
    for s, e in _runs(~active):            # bridge short dips between active frames
        if e - s < min_sil and s > 0 and e < n:
            active[s:e] = True
    ext = max(0, int(round(cfg.max_extension_ms / cfg.hop_ms)))
    runs = []
    for s, e in _runs(active):
        if e - s < min_sp:                 # transient
            active[s:e] = False
            continue
        cores = [(a, b) for a, b in _runs(db[s:e] > high) if b - a >= min_core]
        if not cores:
            runs.append((s, e, "ambiguous"))
            continue
        sp0, sp1 = max(s, s + cores[0][0] - ext), min(e, s + cores[-1][1] + ext)
        if sp0 - s >= min_sp:              # weak lead-in beyond the allowed onset extension
            runs.append((s, sp0, "ambiguous"))
        else:
            sp0 = s
        tail = e - sp1
        runs.append((sp0, sp1 if tail >= min_sp else e, "speech"))
        if tail >= min_sp:
            runs.append((sp1, e, "ambiguous"))

    def span(i0: int, i1: int) -> tuple[float, float]:
        t0 = 0.0 if i0 == 0 else max(0.0, (i0 * hop + frame / 2 - hop / 2) / fs)
        t1 = duration if i1 >= n else min(duration, ((i1 - 1) * hop + frame / 2 + hop / 2) / fs)
        return float(t0), float(t1)

    for s, e, kind in runs:
        reg = _region(*span(s, e), kind)
        result["active"].append(reg)
        result["speech" if kind == "speech" else "ambiguous"].append(reg)
    # silence = complement of the active regions
    edges = [0.0] + [t for r in result["active"] for t in (r["start_s"], r["end_s"])] + [duration]
    for i in range(0, len(edges), 2):
        t0, t1 = edges[i], edges[i + 1]
        if t1 - t0 <= 1e-9:
            continue
        kind = "leading" if i == 0 else "trailing" if i == len(edges) - 2 else "internal"
        if not result["active"]:
            kind = "entire recording"
        result["silence"].append(_region(t0, t1, kind))
    flags = result["flags"]
    if result["ambiguous"]:
        flags.append(f"{len(result['ambiguous'])} ambiguous low-level region(s) kept: level is above the noise floor but "
                     "never reaches the speech-onset threshold")
    result.update(threshold_db=float(low), low_threshold_db=float(low),
                  high_threshold_db=None if not np.isfinite(high) else float(high),
                  noise_floor_db=noise, noise_mad_db=mad, speech_level_db=level, level_minus_noise_db=float(snr),
                  frame_rms_db=db.tolist(), frame_times_s=((np.arange(n) * hop + frame / 2) / fs).tolist())
    return result


# --------------------------------------------------------------------------- trimming
def trim_silence(data: np.ndarray, fs: int, detection: dict, cfg: Config) -> tuple[np.ndarray, dict]:
    """Conservative trimming; never modifies `data`.

    * Leading/trailing: cut to the first/last ACTIVE region (speech or ambiguous) minus/plus padding.
    * Internal: a gap is shortened only if it is longer than max_internal_gap_ms AND both neighbouring
      regions are confident speech. Gaps next to ambiguous regions are always preserved.
    * Every cut gets a short fade; joins are checked for discontinuities.
    """
    n = data.shape[0]
    active = detection.get("active") or [{**r, "type": "speech"} for r in detection.get("speech", [])]
    if cfg.trim_ambiguous:
        active = [r for r in active if r["type"] == "speech"]
    if not active or n == 0:
        return data.copy(), {"kept_segments": [[0.0, n / fs if fs else 0.0]], "shortened_gaps": [],
                             "preserved_gaps": [], "join_samples": [],
                             "note": "no speech detected; audio left unchanged"}
    pad = int(round(cfg.edge_pad_ms * fs / 1000))
    start = max(0, int(round(active[0]["start_s"] * fs)) - pad)
    end = min(n, int(round(active[-1]["end_s"] * fs)) + pad)
    if end <= start:
        return data.copy(), {"kept_segments": [[0.0, n / fs]], "shortened_gaps": [], "preserved_gaps": [],
                             "join_samples": [], "note": "invalid boundaries; audio left unchanged"}

    keep: list[list[int]] = [[start, end]]
    shortened, preserved = [], []
    half = int(round(cfg.kept_gap_ms * fs / 2000))
    for a, b in zip(active[:-1], active[1:]):
        g0, g1 = int(round(a["end_s"] * fs)), int(round(b["start_s"] * fs))
        rec = {"start_s": g0 / fs, "end_s": g1 / fs, "duration_s": (g1 - g0) / fs,
               "neighbours": [a["type"], b["type"]]}
        long_enough = (g1 - g0) * 1000 / fs > cfg.max_internal_gap_ms and g1 - g0 > 2 * half
        if long_enough and a["type"] == b["type"] == "speech":
            last = keep[-1]
            keep[-1] = [last[0], g0 + half]
            keep.append([g1 - half, last[1]])
            rec["new_duration_s"] = 2 * half / fs
            rec["removed_s"] = (g1 - g0 - 2 * half) / fs
            shortened.append(rec)
        else:
            rec["reason"] = ("gap not longer than max_internal_gap_ms" if not long_enough
                             else "neighbour is ambiguous low-level activity")
            preserved.append(rec)

    fade = int(round(cfg.fade_ms * fs / 1000))
    pieces, joins, total = [], [], 0
    for i, (s, e) in enumerate(keep):
        seg = data[s:e].copy()
        fade_in = i > 0 or s > 0
        fade_out = i < len(keep) - 1 or e < n
        if seg.shape[0] > 2 * fade > 0:
            if fade_in:
                seg[:fade] *= np.linspace(0, 1, fade)[:, None]
            if fade_out:
                seg[-fade:] *= np.linspace(1, 0, fade)[:, None]
        if i > 0:
            joins.append(total)
        total += seg.shape[0]
        pieces.append(seg)
    out = np.concatenate(pieces, axis=0)
    info = {"kept_segments": [[s / fs, e / fs] for s, e in keep], "shortened_gaps": shortened,
            "preserved_gaps": preserved, "join_samples": joins,
            "note": "gaps are shortened only between two confident-speech regions and only when longer than max_internal_gap_ms"}
    return out, info


def check_joins(proc: np.ndarray, joins: list[int]) -> dict:
    """Largest sample-to-sample step at each edit join versus the 99.9th percentile step of the whole file."""
    if proc.shape[0] < 2:
        return {"checked": 0, "ok": True}
    steps = np.abs(np.diff(proc, axis=0)).max(axis=1)
    ref = float(np.percentile(steps, 99.9))
    join_steps = [float(steps[j - 1]) for j in joins if 0 < j < proc.shape[0]]
    return {"checked": len(join_steps), "join_max_step": max(join_steps) if join_steps else 0.0,
            "reference_p99_9_step": ref, "ok": all(s <= max(ref, 1e-4) for s in join_steps)}


def duration_reduction(orig_samples: int, new_samples: int, fs: int) -> dict:
    o, p = orig_samples / fs, new_samples / fs
    return {"original_duration_s": o, "processed_duration_s": p, "removed_duration_s": o - p,
            "percent_removed": (o - p) / o * 100 if o > 0 else 0.0}


def validate_wav(path: Path, expected_fs: int, expected_channels: int, expected: np.ndarray,
                 original_peak: float) -> dict:
    """Re-open the saved file and compare it with the in-memory processed audio and the metadata."""
    data, fs = sf.read(str(path), dtype="float64", always_2d=True)
    lsb = 1 / 32768
    same_shape = data.shape == expected.shape
    checks = {"reopened": True, "nonempty": data.size > 0,
              "all_finite": bool(np.all(np.isfinite(data))) if data.size else False,
              "not_all_zero": bool(np.any(data != 0)),
              "sampling_frequency_preserved": fs == expected_fs,
              "channels_preserved": data.shape[1] == expected_channels,
              "num_samples_matches_report": data.shape[0] == expected.shape[0],
              "matches_processed_array_within_1lsb": bool(same_shape and np.max(np.abs(data - expected)) <= lsb * 1.01),
              "no_clipped_samples": bool(np.sum(np.abs(data) >= 0.9999) == 0) if data.size else False,
              "peak_not_above_original": bool(np.max(np.abs(data)) <= original_peak + lsb) if data.size else False}
    return {"file": str(path), "checks": checks, "passed": all(checks.values()),
            "num_samples": int(data.shape[0]), "sampling_frequency_hz": int(fs),
            "peak_amplitude": float(np.max(np.abs(data))) if data.size else 0.0}


# --------------------------------------------------------------------------- spectrogram
def compute_spectrogram(x: np.ndarray, fs: int, cfg: Config, nfft: int | None = None, hop: int | None = None):
    """STFT power spectral density (Hann window). nfft is reduced for very short signals."""
    if x.size < 8:
        x = np.pad(x, (0, 8 - x.size))
    nperseg = int(min(nfft or cfg.spec_nfft, x.size))
    h = max(1, min(hop or cfg.spec_hop, nperseg // 2))
    f, t, sxx = signal.spectrogram(x, fs=fs, window="hann", nperseg=nperseg, noverlap=nperseg - h,
                                   scaling="spectrum", mode="psd")
    return f, t, sxx


def spectrogram_reference(x: np.ndarray, fs: int, cfg: Config) -> float:
    """0 dB reference: peak spectrogram power of the given (original) recording."""
    return float(compute_spectrogram(to_mono(x), fs, cfg)[2].max()) + EPS


# --------------------------------------------------------------------------- plotting
def _style() -> None:
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12, "axes.grid": True,
                         "grid.alpha": 0.3, "figure.dpi": 100, "savefig.dpi": DPI})


def _plot_channels(ax, data: np.ndarray, fs: int, t0: float = 0.0, t1: float | None = None) -> None:
    i0 = int(t0 * fs)
    i1 = data.shape[0] if t1 is None else int(t1 * fs)
    t = np.arange(i0, i1) / fs
    colors = ["#1f77b4", "#d95f02"]
    for c in range(data.shape[1]):
        label = ("Left", "Right")[c] if data.shape[1] == 2 else (f"Channel {c + 1}" if data.shape[1] > 1 else "Mono")
        ax.plot(t, data[i0:i1, c], lw=0.6, color=colors[c % 2], alpha=0.85 if c == 0 else 0.6, label=label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (full scale = 1)")


def plot_waveform(data, fs, title, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    _plot_channels(ax, data, fs)
    ax.set_xlim(0, data.shape[0] / fs)
    ax.set_title(f"{title} - time-domain waveform (fs = {fs} Hz)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


_CLASS_COLORS = {"silence": "red", "speech": "green", "ambiguous": "orange"}


def _shade(ax, det, labels: bool = True) -> None:
    for kind, regs in (("silence", det["silence"]), ("speech", det["speech"]), ("ambiguous", det.get("ambiguous", []))):
        for i, r in enumerate(regs):
            ax.axvspan(r["start_s"], r["end_s"], color=_CLASS_COLORS[kind], alpha=0.2 if kind != "speech" else 0.1,
                       label=({"silence": "Silence", "speech": "Speech (confident)",
                               "ambiguous": "Ambiguous low-level (kept)"}[kind] if (i == 0 and labels) else None))


def _plot_rms(ax, det) -> None:
    if det.get("frame_times_s"):
        ax.plot(det["frame_times_s"], det["frame_rms_db"], color="k", lw=1, label="Frame RMS")
        if det.get("high_threshold_db") is not None:
            ax.axhline(det["high_threshold_db"], color="green", ls="--", label=f"High / onset ({det['high_threshold_db']:.1f} dB)")
        ax.axhline(det["threshold_db"], color="red", ls="--", label=f"Low / hold ({det['threshold_db']:.1f} dB)")
        ax.axhline(det["noise_floor_db"], color="gray", ls=":", label=f"Noise floor ({det['noise_floor_db']:.1f} dB)")
    ax.set_ylabel("Frame RMS (dBFS)")
    ax.set_ylim(-80, 0)


def plot_silence(data, fs, det, title, path: Path) -> None:
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 6.8), sharex=True, gridspec_kw={"height_ratios": [3, 1.7]})
    _plot_channels(ax, data, fs)
    _shade(ax, det)
    ax.set_title(f"{title} - silence detection (hysteresis detector)")
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    _plot_rms(ax2, det)
    ax2.legend(loc="lower right", ncol=2, fontsize=8)
    ax2.set_xlabel("Time (s)")
    ax2.set_xlim(0, data.shape[0] / fs)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_boundary_review(data, fs, det, title, path: Path, cfg: Config, ref_power: float) -> None:
    """Zoomed waveform / frame-RMS / spectrogram around the first and last speech boundary."""
    dur = data.shape[0] / fs
    core = det["speech"] or det["active"]
    if not core:
        return
    wins = [(0.0, min(dur, core[0]["start_s"] + 0.25)), (max(0.0, core[-1]["end_s"] - 0.25), dur)]
    f, t, sxx = compute_spectrogram(to_mono(data), fs, cfg)
    db = 10 * np.log10(sxx / ref_power + EPS)
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    for col, (w0, w1) in enumerate(wins):
        a0, a1, a2 = axes[0, col], axes[1, col], axes[2, col]
        _plot_channels(a0, data, fs, w0, w1)
        _shade(a0, det, labels=col == 0)
        a0.set_xlim(w0, w1)
        a0.set_title(f"{'Start' if col == 0 else 'End'} of recording ({w0:.2f}-{w1:.2f} s)")
        _plot_rms(a1, det)
        a1.set_xlim(w0, w1)
        for r in det["speech"]:
            a1.axvspan(r["start_s"], r["end_s"], color="green", alpha=0.08)
        for r in det.get("ambiguous", []):
            a1.axvspan(r["start_s"], r["end_s"], color="orange", alpha=0.2)
        a1.set_xlabel("Time (s)")
        m = a2.pcolormesh(t, f, db, shading="auto", cmap="magma", vmin=-cfg.spec_range_db, vmax=0)
        a2.set_xlim(w0, w1)
        a2.set_ylim(0, min(cfg.spec_max_hz, fs / 2))
        a2.set_xlabel("Time (s)")
        a2.set_ylabel("Frequency (Hz)")
        a2.grid(False)
        for r in det["active"]:
            for tt in (r["start_s"], r["end_s"]):
                if w0 <= tt <= w1:
                    a2.axvline(tt, color="cyan", lw=0.8, ls="--")
        if col == 1:
            fig.colorbar(m, ax=axes[2, :], label="Power (dB re. original peak)", pad=0.01)
    axes[0, 0].legend(loc="upper left", fontsize=8)
    axes[1, 0].legend(loc="lower left", fontsize=7)
    fig.suptitle(f"{title} - boundary review (cyan dashed = detected active-region edges)", fontsize=14)
    fig.savefig(path)
    plt.close(fig)


def plot_comparison(orig, proc, fs, title, path: Path) -> None:
    """Original and processed shown with identical time scale and amplitude limits."""
    tmax = orig.shape[0] / fs
    ymax = max(np.max(np.abs(orig)), np.max(np.abs(proc))) * 1.1 if proc.size else 1
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, sharey=True)
    for ax, d, name in zip(axes, (orig, proc), ("Original", "Processed (silence trimmed)")):
        _plot_channels(ax, d, fs)
        ax.set_title(f"{name} - {d.shape[0] / fs:.3f} s")
        ax.legend(loc="upper right")
    axes[0].set_xlabel("")
    axes[0].set_xlim(0, tmax)
    axes[0].set_ylim(-ymax, ymax)
    fig.suptitle(f"{title} - original vs processed (same axes)", fontsize=14)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_spectrogram(panels: list[tuple[str, np.ndarray]], fs: int, title: str, path: Path, cfg: Config,
                     ref_power: float | None = None) -> None:
    """Spectrograms on a shared dB scale. 0 dB = ref_power (default: the strongest panel's peak power)."""
    specs = [compute_spectrogram(to_mono(d), fs, cfg) for _, d in panels]
    ref = ref_power if ref_power is not None else max(float(s[2].max()) for s in specs) + EPS
    fmax = min(cfg.spec_max_hz, fs / 2)
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 4.2 * len(panels)), squeeze=False)
    for ax, (name, d), (f, t, sxx) in zip(axes[:, 0], panels, specs):
        db = 10 * np.log10(sxx / ref + EPS)
        mesh = ax.pcolormesh(t, f, db, shading="auto", cmap="magma", vmin=-cfg.spec_range_db, vmax=0)
        ax.set_ylim(0, fmax)
        ax.set_xlim(0, d.shape[0] / fs)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"{title} - {name} spectrogram")
        ax.grid(False)
        fig.colorbar(mesh, ax=ax, label="Power (dB re. peak power of the original recording)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_spectrogram_resolution(x: np.ndarray, fs: int, title: str, path: Path, cfg: Config, ref_power: float) -> None:
    """Compare FFT sizes (hop = nfft/8) to document the time/frequency resolution trade-off."""
    sizes = [512, 1024, 2048]
    fig, axes = plt.subplots(len(sizes), 1, figsize=(10, 3.6 * len(sizes)), sharex=True)
    mono = to_mono(x)
    for ax, nfft in zip(axes, sizes):
        f, t, sxx = compute_spectrogram(mono, fs, cfg, nfft=nfft, hop=nfft // 8)
        mesh = ax.pcolormesh(t, f, 10 * np.log10(sxx / ref_power + EPS), shading="auto", cmap="magma",
                             vmin=-cfg.spec_range_db, vmax=0)
        ax.set_ylim(0, min(cfg.spec_max_hz, fs / 2))
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"FFT {nfft}: window {1000 * nfft / fs:.1f} ms, bin spacing {fs / nfft:.1f} Hz")
        ax.grid(False)
        fig.colorbar(mesh, ax=ax, label="dB re. original peak")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{title} - STFT resolution comparison (Hann window)", fontsize=14)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- real-recording analysis
def summarise_detection(det: dict) -> dict:
    return {k: v for k, v in det.items() if k not in ("frame_rms_db", "frame_times_s")}


def sensitivity_sweep(data: np.ndarray, fs: int, cfg: Config) -> list[dict]:
    """How much do the trimmed boundaries move when the main detector parameters change?"""
    rows = []
    for frame_ms in (20.0, 25.0, 40.0):
        for low in (6.0, 8.0, 10.0):
            c = replace(cfg, frame_ms=frame_ms, low_margin_db=low)
            det = detect_silence(data, fs, c)
            proc, _ = trim_silence(data, fs, det, c)
            act = det["active"]
            rows.append({"frame_ms": frame_ms, "low_margin_db": low,
                         "first_active_start_s": act[0]["start_s"] if act else None,
                         "last_active_end_s": act[-1]["end_s"] if act else None,
                         "n_speech": len(det["speech"]), "n_ambiguous": len(det["ambiguous"]),
                         "processed_duration_s": proc.shape[0] / fs,
                         "percent_removed": (1 - proc.shape[0] / data.shape[0]) * 100})
    return rows


def process_recording(key: str, entry: dict, out: Path, cfg: Config) -> dict:
    path: Path = entry["path"]
    phrase = entry["declared_utterance"]
    sha_before = sha256_of(path)
    data, fs = load_audio(path)
    meta = get_metadata(path, data, fs)
    stats = signal_stats(data)
    det = detect_silence(data, fs, cfg)
    proc, trim_info = trim_silence(data, fs, det, cfg)

    base_det = detect_silence_baseline(data, fs, cfg)
    base_proc, _ = trim_silence(data, fs, base_det, cfg)

    plots, proc_dir = out / "plots", out / "processed_audio"
    paths = {"waveform": plots / f"{key}_waveform.png", "silence_detection": plots / f"{key}_silence_detection.png",
             "boundary_review": plots / f"{key}_boundary_review.png",
             "trimmed_waveform": plots / f"{key}_trimmed_waveform.png",
             "spectrogram": plots / f"{key}_spectrogram.png",
             "spectrogram_processed": plots / f"{key}_spectrogram_processed.png",
             "spectrogram_resolution": plots / f"{key}_spectrogram_resolution.png"}
    proc_path = proc_dir / f"{path.stem}_processed.wav"
    label = f'"{phrase}" ({path.name})'
    ref = spectrogram_reference(data, fs, cfg)

    plot_waveform(data, fs, label, paths["waveform"])
    plot_silence(data, fs, det, label, paths["silence_detection"])
    plot_boundary_review(data, fs, det, label, paths["boundary_review"], cfg, ref)
    plot_comparison(data, proc, fs, label, paths["trimmed_waveform"])
    plot_spectrogram([("original", data)], fs, label, paths["spectrogram"], cfg, ref)
    plot_spectrogram([("processed", proc)], fs, label, paths["spectrogram_processed"], cfg, ref)
    plot_spectrogram_resolution(data, fs, label, paths["spectrogram_resolution"], cfg, ref)
    sf.write(str(proc_path), proc, fs, subtype="PCM_16")

    validation = validate_wav(proc_path, fs, data.shape[1], proc, stats["peak_amplitude"])
    validation["joins"] = check_joins(proc, trim_info["join_samples"])
    validation["checks"]["joins_without_discontinuity"] = validation["joins"]["ok"]
    sha_after = sha256_of(path)
    validation["checks"]["original_file_unchanged"] = sha_before == sha_after
    validation["passed"] = all(validation["checks"].values())
    validation["original_sha256"] = sha_after

    rel = lambda p: p.relative_to(out.parent).as_posix()  # noqa: E731
    return {"declared_utterance": phrase, "identity_source": entry["identity_source"],
            "content_verification": "Declared utterance only. No speech recognition was run, so the spoken content "
                                    "is NOT independently verified.",
            "metadata": meta, "statistics": stats,
            "silence_detection": summarise_detection(det), "trimming": trim_info,
            "reduction": duration_reduction(data.shape[0], proc.shape[0], fs),
            "processed_stats": signal_stats(proc), "processed_file": rel(proc_path),
            "baseline": {"silence_detection": summarise_detection(base_det),
                         "reduction": duration_reduction(data.shape[0], base_proc.shape[0], fs)},
            "sensitivity": sensitivity_sweep(data, fs, cfg),
            "validation": validation, "plots": {k: rel(v) for k, v in paths.items()}}


def run(audio_dir: Path, out: Path, cfg: Config) -> dict:
    from evaluation import run_synthetic_evaluation  # local import: evaluation depends on this module
    from report import write_report

    _style()
    for sub in ("plots", "processed_audio"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    found = discover_recordings(audio_dir)
    results = {"course": "EC24313 Artificial Intelligence and Machine Learning", "student": "Naman Gupta",
               "config": asdict(cfg), "recordings": {}}
    for key, entry in found.items():
        print(f"[{key}] processing {entry['path'].name}")
        results["recordings"][key] = process_recording(key, entry, out, cfg)
        r = results["recordings"][key]
        print(f"  fs={r['metadata']['sampling_frequency_hz']} Hz, dur={r['metadata']['duration_s']:.3f} s, "
              f"removed={r['reduction']['percent_removed']:.1f}% (baseline {r['baseline']['reduction']['percent_removed']:.1f}%), "
              f"valid={r['validation']['passed']}")
    print("[synthetic] running controlled evaluation")
    results["synthetic_evaluation"] = run_synthetic_evaluation(cfg)
    review = ROOT / "manual_review.json"
    results["manual_review"] = json.loads(review.read_text(encoding="utf-8")) if review.is_file() else None
    frozen = out / "baseline_results.json"
    results["frozen_baseline_file"] = frozen.name if frozen.is_file() else None
    (out / "analysis_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_report(results, cfg, out / "analysis_report.md")
    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--audio-dir", type=Path, default=ROOT / "audio")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = p.parse_args(argv)
    try:
        results = run(args.audio_dir, args.output_dir, Config())
    except Exception as exc:  # critical failure -> non-zero exit
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0 if all(r["validation"]["passed"] for r in results["recordings"].values()) else 2


if __name__ == "__main__":
    sys.exit(main())
