#!/usr/bin/env python3
"""EC24313 Artificial Intelligence and Machine Learning - Assignment 1.

Analyses two voice recordings (time-domain waveform, silence detection,
silence trimming, spectrogram) and writes plots, processed WAV files,
analysis_results.json and analysis_report.md.

Usage:  python assignment1.py [--audio-dir audio] [--output-dir outputs]
All paths are relative to this file's directory unless absolute.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
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

# Recording key -> (filename keywords used for discovery, spoken phrase claimed by the assignment)
RECORDINGS = {
    "bit": (("bit",), "BIT"),
    "name": (("name", "naman_gupta", "gupta"), "Naman Gupta"),
}


# --------------------------------------------------------------------------- config
@dataclass
class Config:
    frame_ms: float = 25.0          # analysis frame length
    hop_ms: float = 10.0            # hop between frames
    noise_percentile: float = 10.0  # percentile of frame RMS (dB) taken as noise floor
    speech_percentile: float = 95.0  # percentile taken as speech level
    floor_margin_db: float = 12.0   # threshold >= noise floor + this
    speech_range_db: float = 35.0   # threshold >= speech level - this
    min_silence_ms: float = 100.0   # shorter low-energy runs are treated as pauses inside speech
    min_speech_ms: float = 30.0     # shorter active runs are treated as noise bursts
    edge_pad_ms: float = 100.0       # padding kept around speech when trimming leading/trailing silence
    max_internal_gap_ms: float = 400.0  # internal gaps longer than this are shortened
    kept_gap_ms: float = 200.0      # length an over-long internal gap is shortened to
    fade_ms: float = 5.0            # fade at splice points to avoid clicks
    spec_nfft: int = 1024
    spec_hop: int = 128
    spec_range_db: float = 80.0
    spec_max_hz: float = 8000.0


# --------------------------------------------------------------------------- loading
def discover_recordings(audio_dir: Path) -> dict[str, Path]:
    """Locate the two recordings by filename (searches subdirectories)."""
    files = sorted(p for p in audio_dir.rglob("*") if p.suffix.lower() in {".wav", ".flac", ".ogg"})
    found: dict[str, Path] = {}
    for key, (keywords, _) in RECORDINGS.items():
        matches = [p for p in files if any(k in p.stem.lower() for k in keywords)]
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected exactly one '{key}' recording in {audio_dir}, found {len(matches)}")
        found[key] = matches[0]
    return found


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Return float64 samples shaped (n_samples, n_channels) and the sampling frequency."""
    data, fs = sf.read(str(path), dtype="float64", always_2d=True)
    return data, int(fs)


def to_mono(data: np.ndarray) -> np.ndarray:
    return data.mean(axis=1)


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


# --------------------------------------------------------------------------- silence detection
def frame_rms_db(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    """Short-time RMS in dB full scale. Signals shorter than a frame give one frame."""
    if x.size == 0:
        return np.array([])
    if x.size < frame:
        x = np.pad(x, (0, frame - x.size))
    n_frames = 1 + (x.size - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n_frames)[:, None]
    return 20 * np.log10(np.sqrt(np.mean(x[idx] ** 2, axis=1)) + EPS)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end_exclusive) of consecutive True runs."""
    if mask.size == 0:
        return []
    d = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def detect_silence(x: np.ndarray, fs: int, cfg: Config) -> dict:
    """Adaptive short-time-energy silence detector.

    threshold_dB = max(noise_floor + floor_margin, speech_level - speech_range)
    where noise_floor / speech_level are low / high percentiles of the frame RMS.
    Returns silent and speech intervals in seconds plus detector internals.
    """
    duration = x.size / fs if fs else 0.0
    frame = max(1, int(round(cfg.frame_ms * fs / 1000)))
    hop = max(1, int(round(cfg.hop_ms * fs / 1000)))
    result = {"method": "short-time RMS energy with adaptive percentile threshold",
              "frame_ms": cfg.frame_ms, "hop_ms": cfg.hop_ms, "frame_samples": frame, "hop_samples": hop,
              "min_silence_ms": cfg.min_silence_ms, "min_speech_ms": cfg.min_speech_ms,
              "silence": [], "speech": [], "threshold_db": None, "noise_floor_db": None, "speech_level_db": None}
    db = frame_rms_db(x, frame, hop)
    if db.size == 0 or np.max(np.abs(x)) < 1e-9:  # empty or digital silence
        if duration > 0:
            result["silence"] = [_region(0.0, duration, "entire recording")]
        return result

    floor = float(np.percentile(db, cfg.noise_percentile))
    level = float(np.percentile(db, cfg.speech_percentile))
    thr = max(floor + cfg.floor_margin_db, level - cfg.speech_range_db)
    active = db > thr

    hop_s = hop / fs
    # frame i covers [i*hop, i*hop+frame); treat it as speaking for its centre hop-sized span
    min_sil = max(1, int(round(cfg.min_silence_ms / cfg.hop_ms)))
    min_sp = max(1, int(round(cfg.min_speech_ms / cfg.hop_ms)))
    for s, e in _runs(~active):            # short pauses inside speech are not silence
        if e - s < min_sil and s > 0 and e < active.size:
            active[s:e] = True
    for s, e in _runs(active):             # very short bursts are noise, not speech
        if e - s < min_sp:
            active[s:e] = False

    def to_time(i0: int, i1: int) -> tuple[float, float]:
        t0 = 0.0 if i0 == 0 else (i0 * hop + frame / 2) / fs
        t1 = duration if i1 >= active.size else (i1 * hop + frame / 2) / fs
        return float(t0), float(min(t1, duration))

    sil = _runs(~active)
    n = active.size
    for s, e in sil:
        t0, t1 = to_time(s, e)
        kind = "leading" if s == 0 else "trailing" if e == n else "internal"
        if t1 - t0 >= cfg.min_silence_ms / 1000 or kind != "internal":
            result["silence"].append(_region(t0, t1, kind))
    result["speech"] = [_region(*to_time(s, e), "speech") for s, e in _runs(active)]
    result.update(threshold_db=float(thr), noise_floor_db=floor, speech_level_db=level,
                  frame_rms_db=db.tolist(), frame_times_s=((np.arange(n) * hop + frame / 2) / fs).tolist())
    return result


def _region(t0: float, t1: float, kind: str) -> dict:
    return {"start_s": float(t0), "end_s": float(t1), "duration_s": float(t1 - t0), "type": kind}


# --------------------------------------------------------------------------- trimming
def trim_silence(data: np.ndarray, fs: int, detection: dict, cfg: Config) -> tuple[np.ndarray, dict]:
    """Remove leading/trailing silence (keeping padding) and shorten over-long internal gaps.

    Never modifies `data`; returns a new array with the original channel layout and rate.
    """
    n = data.shape[0]
    speech = detection["speech"]
    if not speech:  # nothing to trim safely
        return data.copy(), {"kept_segments": [[0.0, n / fs if fs else 0.0]], "shortened_gaps": [],
                             "preserved_gaps": [], "note": "no speech detected; audio left unchanged"}
    pad = int(round(cfg.edge_pad_ms * fs / 1000))
    start = max(0, int(round(speech[0]["start_s"] * fs)) - pad)
    end = min(n, int(round(speech[-1]["end_s"] * fs)) + pad)

    keep: list[list[int]] = [[start, end]]
    shortened, preserved = [], []
    half = int(round(cfg.kept_gap_ms * fs / 2000))
    for a, b in zip(speech[:-1], speech[1:]):
        g0, g1 = int(round(a["end_s"] * fs)), int(round(b["start_s"] * fs))
        gap_ms = (g1 - g0) * 1000 / fs
        rec = {"start_s": g0 / fs, "end_s": g1 / fs, "duration_s": (g1 - g0) / fs}
        if gap_ms > cfg.max_internal_gap_ms and g1 - g0 > 2 * half:
            keep = [k for k in keep]  # split the kept range around the removed middle
            last = keep[-1]
            keep[-1] = [last[0], g0 + half]
            keep.append([g1 - half, last[1]])
            shortened.append(rec)
        else:
            preserved.append(rec)

    fade = int(round(cfg.fade_ms * fs / 1000))
    pieces = []
    for i, (s, e) in enumerate(keep):
        seg = data[s:e].copy()
        if i > 0 and seg.shape[0] > 2 * fade > 0:
            seg[:fade] *= np.linspace(0, 1, fade)[:, None]
        if i < len(keep) - 1 and seg.shape[0] > 2 * fade > 0:
            seg[-fade:] *= np.linspace(1, 0, fade)[:, None]
        pieces.append(seg)
    out = np.concatenate(pieces, axis=0)
    info = {"kept_segments": [[s / fs, e / fs] for s, e in keep], "shortened_gaps": shortened,
            "preserved_gaps": preserved,
            "note": "internal gaps <= max_internal_gap_ms preserved to keep natural rhythm"}
    return out, info


def duration_reduction(orig_samples: int, new_samples: int, fs: int) -> dict:
    o, p = orig_samples / fs, new_samples / fs
    return {"original_duration_s": o, "processed_duration_s": p, "removed_duration_s": o - p,
            "percent_removed": (o - p) / o * 100 if o > 0 else 0.0}


def validate_wav(path: Path, expected_fs: int, expected_channels: int, expected_samples: int) -> dict:
    data, fs = sf.read(str(path), dtype="float64", always_2d=True)
    checks = {"nonempty": data.size > 0, "all_finite": bool(np.all(np.isfinite(data))) if data.size else False,
              "not_all_zero": bool(np.any(data != 0)),
              "sampling_frequency_preserved": fs == expected_fs,
              "channels_preserved": data.shape[1] == expected_channels,
              "num_samples_matches_report": data.shape[0] == expected_samples,
              "no_clipping": bool(np.max(np.abs(data)) < 1.0) if data.size else False}
    return {"file": str(path), "checks": checks, "passed": all(checks.values()),
            "num_samples": int(data.shape[0]), "sampling_frequency_hz": int(fs),
            "peak_amplitude": float(np.max(np.abs(data))) if data.size else 0.0}


# --------------------------------------------------------------------------- plotting
def _style() -> None:
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12, "axes.grid": True,
                         "grid.alpha": 0.3, "figure.dpi": 100, "savefig.dpi": DPI})


def _plot_channels(ax, data: np.ndarray, fs: int, t_offset: float = 0.0) -> None:
    t = np.arange(data.shape[0]) / fs + t_offset
    colors = ["#1f77b4", "#d95f02"]
    for c in range(data.shape[1]):
        label = ("Left", "Right")[c] if data.shape[1] == 2 else (f"Channel {c + 1}" if data.shape[1] > 1 else "Mono")
        ax.plot(t, data[:, c], lw=0.6, color=colors[c % 2], alpha=0.85 if c == 0 else 0.6, label=label)
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


def plot_silence(data, fs, det, title, path: Path) -> None:
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.6]})
    _plot_channels(ax, data, fs)
    for i, r in enumerate(det["silence"]):
        ax.axvspan(r["start_s"], r["end_s"], color="red", alpha=0.2, label="Silence" if i == 0 else None)
    for i, r in enumerate(det["speech"]):
        ax.axvspan(r["start_s"], r["end_s"], color="green", alpha=0.1, label="Speech (active)" if i == 0 else None)
    ax.set_title(f"{title} - silence detection")
    ax.legend(loc="upper right", ncol=2)
    if det.get("frame_times_s"):
        ax2.plot(det["frame_times_s"], det["frame_rms_db"], color="k", lw=1, label="Frame RMS")
        ax2.axhline(det["threshold_db"], color="red", ls="--", label=f"Threshold ({det['threshold_db']:.1f} dB)")
        ax2.axhline(det["noise_floor_db"], color="gray", ls=":", label=f"Noise floor ({det['noise_floor_db']:.1f} dB)")
        ax2.legend(loc="lower right", ncol=3, fontsize=9)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frame RMS (dBFS)")
    ax2.set_ylim(-80, 0)
    ax2.set_xlim(0, data.shape[0] / fs)
    fig.tight_layout()
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


def compute_spectrogram(x: np.ndarray, fs: int, cfg: Config):
    """STFT power spectrogram (Hann window). nfft is reduced for very short signals."""
    if x.size < 8:  # too short for an STFT: zero-pad so the call cannot fail
        x = np.pad(x, (0, 8 - x.size))
    nperseg = int(min(cfg.spec_nfft, x.size))
    hop = max(1, min(cfg.spec_hop, nperseg // 2))
    f, t, sxx = signal.spectrogram(x, fs=fs, window="hann", nperseg=nperseg, noverlap=nperseg - hop,
                                   scaling="spectrum", mode="psd")
    return f, t, sxx


def plot_spectrogram(panels: list[tuple[str, np.ndarray]], fs: int, title: str, path: Path, cfg: Config) -> None:
    """One or more spectrograms sharing the same dB colour scale (referenced to the first panel's maximum)."""
    specs = [compute_spectrogram(to_mono(d), fs, cfg) for _, d in panels]
    ref = max(float(s[2].max()) for s in specs) + EPS
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
        fig.colorbar(mesh, ax=ax, label="Power (dB relative to peak)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- pipeline
def process_recording(key: str, path: Path, out: Path, cfg: Config) -> dict:
    phrase = RECORDINGS[key][1]
    data, fs = load_audio(path)
    meta = get_metadata(path, data, fs)
    stats = signal_stats(data)
    mono = to_mono(data)
    det = detect_silence(mono, fs, cfg)
    proc, trim_info = trim_silence(data, fs, det, cfg)

    plots, proc_dir = out / "plots", out / "processed_audio"
    paths = {"waveform": plots / f"{key}_waveform.png", "silence_detection": plots / f"{key}_silence_detection.png",
             "trimmed_waveform": plots / f"{key}_trimmed_waveform.png",
             "spectrogram": plots / f"{key}_spectrogram.png",
             "spectrogram_processed": plots / f"{key}_spectrogram_processed.png"}
    proc_path = proc_dir / f"{path.stem}_processed.wav"
    label = f'"{phrase}" ({path.name})'

    plot_waveform(data, fs, label, paths["waveform"])
    plot_silence(data, fs, det, label, paths["silence_detection"])
    plot_comparison(data, proc, fs, label, paths["trimmed_waveform"])
    plot_spectrogram([("original", data)], fs, label, paths["spectrogram"], cfg)
    plot_spectrogram([("processed", proc)], fs, label, paths["spectrogram_processed"], cfg)
    sf.write(str(proc_path), proc, fs, subtype="PCM_16")

    validation = validate_wav(proc_path, fs, data.shape[1], proc.shape[0])
    det_out = {k: v for k, v in det.items() if k not in ("frame_rms_db", "frame_times_s")}
    rel = lambda p: p.relative_to(out.parent).as_posix()  # noqa: E731
    return {"spoken_phrase_per_assignment": phrase,
            "content_verification": "Phrase taken from filename only; spoken content cannot be verified automatically.",
            "metadata": meta, "statistics": stats, "silence_detection": det_out, "trimming": trim_info,
            "reduction": duration_reduction(data.shape[0], proc.shape[0], fs),
            "processed_stats": signal_stats(proc), "processed_file": rel(proc_path),
            "validation": validation, "plots": {k: rel(v) for k, v in paths.items()}}


def write_report(results: dict, cfg: Config, path: Path) -> None:
    R = results["recordings"]
    keys = list(R)
    f = lambda v, d=3: f"{v:.{d}f}"  # noqa: E731
    L: list[str] = []
    a = L.append
    a("# EC24313 Artificial Intelligence and Machine Learning - Assignment 1\n")
    a("**Course:** EC24313 - Artificial Intelligence and Machine Learning  ")
    a("**Student:** Naman Gupta  ")
    a("**Topic:** Voice recording, time-domain analysis, silence detection/removal and spectrogram\n")
    a("## 1. Objective\n")
    a("To record two short utterances (\"BIT\" and \"Naman Gupta\"), study them in the time domain, locate silent regions, "
      "produce silence-minimised versions and examine their time-frequency content with a spectrogram.\n")
    a("## 2. Tools and Libraries\n")
    a("Python with NumPy (numerics), SoundFile (WAV I/O and header metadata), SciPy (STFT/spectrogram) and Matplotlib (plots). "
      "Everything runs on a CPU; run `python assignment1.py`.\n")
    a("## 3. Methodology\n")
    a("1. Task 1: the original recordings are in `audio/` and are never modified (read-only access). Spoken content is "
      "inferred from the file names only and was not verified automatically.")
    a("2. Task 2: each file is read, sampling frequency/samples/channels taken from the header, then amplitude vs. time is plotted.")
    a("3. Task 3: frame-wise RMS energy (dB) with an adaptive threshold (below).")
    a("4. Task 4: leading/trailing silence trimmed with padding; over-long internal gaps shortened.")
    a("5. Task 5: Hann-window STFT power spectrogram.\n")
    a("## 4. Sampling Frequency and Time-Domain Analysis\n")
    a("The sampling frequency is the number of amplitude samples stored per second. It is read directly from the WAV header "
      "(`soundfile.info`), not estimated from the signal. It is unrelated to the fundamental frequency (pitch) of the voice, "
      "which is typically ~85-255 Hz for adult speech; the Nyquist limit fs/2 only bounds the highest representable frequency. "
      "Duration = number of samples / fs. For stereo files, the plots show both channels and analysis uses their average.\n")
    a("| Recording | File | fs (Hz) | Channels | Samples | Duration (s) | Peak | RMS |")
    a("|---|---|---|---|---|---|---|---|")
    for k in keys:
        r = R[k]; m, s = r["metadata"], r["statistics"]
        a(f"| {r['spoken_phrase_per_assignment']} | {m['source_file']} | {m['sampling_frequency_hz']} | {m['channels']} | "
          f"{m['num_samples']} | {f(m['duration_s'])} | {f(s['peak_amplitude'], 4)} | {f(s['rms_amplitude'], 4)} |")
    a("")
    for k in keys:
        a(f"![{k} waveform](../{R[k]['plots']['waveform']})\n")
    a("## 5. Silent Region Identification\n")
    a(f"Method: short-time RMS energy with frame = {cfg.frame_ms} ms and hop = {cfg.hop_ms} ms. "
      f"Threshold (dB) = max(noise floor + {cfg.floor_margin_db} dB, speech level - {cfg.speech_range_db} dB), where the noise floor is the "
      f"{cfg.noise_percentile:.0f}th and the speech level the {cfg.speech_percentile:.0f}th percentile of the frame RMS values of that recording, "
      f"so the threshold adapts to each recording's noise floor. Low-energy runs shorter than {cfg.min_silence_ms} ms are treated as "
      f"pauses within speech, and active runs shorter than {cfg.min_speech_ms} ms are discarded as noise bursts. "
      "This is an amplitude-based decision only: it does not identify words or phonemes, and weak sounds (e.g. unvoiced fricatives, "
      "final consonants) may be classed as silence.\n")
    for k in keys:
        d = R[k]["silence_detection"]
        a(f"**{R[k]['spoken_phrase_per_assignment']}** - noise floor {f(d['noise_floor_db'], 1) if d['noise_floor_db'] is not None else 'n/a'} dB, "
          f"threshold {f(d['threshold_db'], 1) if d['threshold_db'] is not None else 'n/a'} dB.\n")
        a("| Type | Start (s) | End (s) | Duration (s) |")
        a("|---|---|---|---|")
        for r in d["silence"]:
            a(f"| {r['type']} | {f(r['start_s'])} | {f(r['end_s'])} | {f(r['duration_s'])} |")
        if not d["silence"]:
            a("| none detected | - | - | - |")
        a("")
        a(f"![{k} silence detection](../{R[k]['plots']['silence_detection']})\n")
    a("## 6. Audio Preprocessing and Silence Removal\n")
    a(f"Leading and trailing silence is removed while keeping {cfg.edge_pad_ms} ms padding around the first/last detected speech so that soft onsets and decays are not cut. "
      f"Internal gaps are only shortened if longer than {cfg.max_internal_gap_ms} ms (then reduced to {cfg.kept_gap_ms} ms with {cfg.fade_ms} ms fades at the joins); "
      "shorter gaps are preserved because they are natural pauses/stop closures. The sampling frequency, channel count and bit depth are unchanged, and no gain is applied.\n")
    a("| Recording | Original (s) | Processed (s) | Removed (s) | Removed (%) | Internal gaps shortened | Internal gaps kept |")
    a("|---|---|---|---|---|---|---|")
    for k in keys:
        r = R[k]; d = r["reduction"]
        a(f"| {r['spoken_phrase_per_assignment']} | {f(d['original_duration_s'])} | {f(d['processed_duration_s'])} | "
          f"{f(d['removed_duration_s'])} | {f(d['percent_removed'], 1)} | {len(r['trimming']['shortened_gaps'])} | {len(r['trimming']['preserved_gaps'])} |")
    a("")
    for k in keys:
        v = R[k]["validation"]
        a(f"- {R[k]['spoken_phrase_per_assignment']}: processed file `{R[k]['processed_file']}` validation "
          f"{'passed' if v['passed'] else 'FAILED'} (" + ", ".join(f"{c}={'ok' if ok else 'fail'}" for c, ok in v["checks"].items()) + ").")
    a("")
    for k in keys:
        a(f"![{k} original vs processed](../{R[k]['plots']['trimmed_waveform']})\n")
    a("## 7. Spectrogram Analysis\n")
    a(f"A spectrogram is the magnitude of the Short-Time Fourier Transform: the signal is cut into short overlapping windows "
      f"(Hann window, {cfg.spec_nfft} samples, hop {cfg.spec_hop} samples), each is Fourier-transformed, and the power in dB is drawn with time on the x-axis, frequency on the y-axis "
      "and colour for energy. Speech is non-stationary, so this shows when energy appears and at which frequencies: voiced sounds "
      "give horizontal harmonic stripes spaced at the pitch, vowel formants appear as broad energy bands, and fricatives/bursts appear as "
      f"broadband energy at high frequency. Frequency axis is limited to {min(cfg.spec_max_hz, 24000):.0f} Hz (or fs/2 if lower). "
      "The window length trades time resolution against frequency resolution. Colour is dB relative to the peak power of the plotted file.\n")
    for k in keys:
        a(f"![{k} spectrogram](../{R[k]['plots']['spectrogram']})\n")
        a(f"Processed counterpart: ![{k} processed spectrogram](../{R[k]['plots']['spectrogram_processed']})\n")
    a("## 8. Results Summary and Discussion\n")
    for k in keys:
        r = R[k]; d = r["reduction"]; s = r["silence_detection"]["silence"]
        lead = sum(x["duration_s"] for x in s if x["type"] == "leading")
        trail = sum(x["duration_s"] for x in s if x["type"] == "trailing")
        internal = [x for x in s if x["type"] == "internal"]
        a(f"- **{r['spoken_phrase_per_assignment']}**: leading silence {f(lead)} s, trailing silence {f(trail)} s, "
          f"{len(internal)} internal silent region(s) of at least {cfg.min_silence_ms:.0f} ms; "
          f"{f(d['percent_removed'], 1)}% of the duration removed.")
    a("\nThe percentages are results of the chosen thresholds and padding; a different threshold would change them. "
      "Detected regions should be checked against the annotated plots, which is how they were assessed.\n")
    a("## 9. Limitations\n")
    a("- Silence is decided from short-time energy only; quiet speech or background noise near the threshold can be misclassified.")
    a("- Percentile-based thresholds assume each file contains both background and speech frames.")
    a("- Which phrase each file contains is taken from its filename and was not verified by listening or recognition.")
    a("- Low-level background noise (e.g. breath or room sound) near the threshold can be flagged as active, so some non-speech segments (notably in the noisy tails) may remain in the processed audio.")
    a("- No perceptual/listening test of intelligibility was carried out in this automated run.")
    a("- Spectrogram appearance depends on window length and dB range.\n")
    a("## 10. Conclusion\n")
    a("Both recordings were loaded, analysed, trimmed and visualised with a single reproducible script. "
      "Measured values for sampling frequency, duration, silence intervals and reductions are given in the tables above and in `analysis_results.json`.\n")
    path.write_text("\n".join(L), encoding="utf-8")


def run(audio_dir: Path, out: Path, cfg: Config) -> dict:
    _style()
    for sub in ("plots", "processed_audio"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    found = discover_recordings(audio_dir)
    results = {"course": "EC24313 Artificial Intelligence and Machine Learning", "student": "Naman Gupta",
               "config": asdict(cfg), "recordings": {}}
    for key, path in found.items():
        print(f"[{key}] processing {path.name}")
        results["recordings"][key] = process_recording(key, path, out, cfg)
        r = results["recordings"][key]
        print(f"  fs={r['metadata']['sampling_frequency_hz']} Hz, dur={r['metadata']['duration_s']:.3f} s, "
              f"removed={r['reduction']['percent_removed']:.1f}%, valid={r['validation']['passed']}")
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
