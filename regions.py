"""Evidence gathering for ambiguous low-level regions, join plots and listening-test material.

Everything here is automated measurement or plotting. Nothing in this module can decide whether a region is
speech: the output is evidence plus a conservative label, and the region is preserved either way.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import stats

import assignment1 as a

PRE_POST_S = 0.2   # context shown around a region in the focus plots
CLIP_PAD_S = 0.3   # context included in listening clips


def _frame_mask(det_times: np.ndarray, t0: float, t1: float) -> np.ndarray:
    return (det_times >= t0) & (det_times <= t1)


def _summ(feats: dict, mask: np.ndarray, cfg: a.Config) -> dict | None:
    if mask.sum() == 0:
        return None
    per = feats["periodicity"][mask]
    return {"frames": int(mask.sum()),
            "rms_db_median": float(np.median(feats["rms_db"][mask])),
            "periodicity_median": float(np.median(per)),
            "periodic_frame_fraction": float((per >= cfg.min_periodicity).mean()),
            "zcr_median": float(np.median(feats["zcr"][mask])),
            "spectral_flatness_median": float(np.median(feats["spectral_flatness"][mask])),
            "low_freq_fraction_<300Hz_median": float(np.median(feats["low_freq_fraction"][mask]))}


def region_features(data: np.ndarray, fs: int, det: dict, region: dict, cfg: a.Config) -> dict:
    """Features of one region compared with confident-speech frames and confident-silence frames of the same file."""
    feats = a.frame_features(data, fs, cfg)
    frame, hop = a._frame_params(fs, cfg)
    times = (np.arange(len(feats["rms_db"])) * hop + frame / 2) / fs
    reg = _summ(feats, _frame_mask(times, region["start_s"], region["end_s"]), cfg)
    sp = np.zeros_like(times, bool)
    for r in det["speech"]:
        sp |= _frame_mask(times, r["start_s"], r["end_s"])
    si = np.zeros_like(times, bool)
    for r in det["silence"]:
        si |= _frame_mask(times, r["start_s"], r["end_s"])
    si &= feats["rms_db"] > -90  # exclude digital zeros
    i0, i1 = int(region["start_s"] * fs), int(region["end_s"] * fs)
    seg = a.to_mono(data)[i0:i1]
    kurt = float(stats.kurtosis(seg)) if seg.size > 8 else None
    crest = float(np.max(np.abs(seg)) / (np.sqrt(np.mean(seg ** 2)) + a.EPS)) if seg.size else None
    # per-channel level difference: a click or handling noise often hits one channel more than the other
    ch_db = [float(10 * np.log10(np.mean(data[i0:i1, c] ** 2) + a.EPS)) for c in range(data.shape[1])]
    sp_idx = np.concatenate([np.arange(int(r["start_s"] * fs), int(r["end_s"] * fs)) for r in det["speech"]]) if det["speech"] else np.array([], int)
    ch_sp = [float(10 * np.log10(np.mean(data[sp_idx, c] ** 2) + a.EPS)) for c in range(data.shape[1])] if sp_idx.size else None
    out = {"region": region, "reference_speech_channel_level_db": ch_sp, "region_features": reg, "kurtosis_excess": kurt, "crest_factor": crest,
           "channel_level_db": ch_db, "reference_confident_speech": _summ(feats, sp, cfg),
           "reference_confident_silence": _summ(feats, si, cfg)}
    pf = reg["periodic_frame_fraction"] if reg else 0.0
    if reg is None:
        label = "too short to measure"
    elif pf >= 0.8:
        label = "strongly periodic: voiced-like (consistent with speech, but also with any periodic sound)"
    elif kurt is not None and kurt > 6 and pf < 0.3:
        label = "impulsive and non-periodic: click-like"
    elif pf < 0.3:
        label = "non-periodic, noise-like: breath, background noise or unvoiced sound; speech not excluded"
    else:
        label = ("inconclusive: borderline periodicity (low-frequency noise and weak voicing both give this); "
                 "speech, breath and background noise cannot be separated")
    out["automated_label"] = label
    out["decision"] = ("removable: automated evidence is inconclusive, so removal rests on the manual listening finding"
                       if cfg.trim_ambiguous else "preserved (evidence insufficient to separate speech from noise or breath)")
    return out


def plot_region_focus(data, proc, proc_offset_s, fs, det, region, title, path: Path, cfg: a.Config, ref_power: float) -> None:
    """Original waveform, processed waveform (on the original time axis), energy/periodicity and spectrogram."""
    dur = data.shape[0] / fs
    w0, w1 = max(0.0, region["start_s"] - PRE_POST_S), min(dur, region["end_s"] + PRE_POST_S)
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True, constrained_layout=True)
    # original
    a._plot_channels(axes[0], data, fs, w0, w1)
    a._shade(axes[0], det)
    axes[0].set_title(f"{title}: original, region {region['start_s']:.3f}-{region['end_s']:.3f} s")
    axes[0].legend(loc="upper left", fontsize=8, ncol=3)
    # processed on original time axis
    i0 = int(round(proc_offset_s * fs))
    t = (np.arange(proc.shape[0]) + i0) / fs
    m = (t >= w0) & (t <= w1)
    for c in range(proc.shape[1]):
        axes[1].plot(t[m], proc[m, c], lw=0.6, color=["#1f77b4", "#d95f02"][c % 2], alpha=0.8)
    a._shade(axes[1], det, labels=False)
    axes[1].axvline(proc_offset_s, color="k", ls="--", lw=1, label=f"processed file starts here ({proc_offset_s:.4f} s)")
    axes[1].set_ylabel("Amplitude (full scale = 1)")
    axes[1].set_title("Processed output placed on the original time axis (empty where audio was removed)")
    axes[1].legend(loc="upper left", fontsize=8)
    # energy + periodicity
    feats = a.frame_features(data, fs, cfg)
    frame, hop = a._frame_params(fs, cfg)
    times = (np.arange(len(feats["rms_db"])) * hop + frame / 2) / fs
    ax = axes[2]
    ax.plot(times, feats["rms_db"], "k", lw=1, label="Frame RMS (dBFS)")
    ax.axhline(det["high_threshold_db"], color="green", ls="--", label="High threshold")
    ax.axhline(det["threshold_db"], color="red", ls="--", label="Low threshold")
    ax.axhline(det["noise_floor_db"], color="gray", ls=":", label="Noise floor")
    ax.set_ylim(-80, 0)
    ax.set_ylabel("Frame RMS (dBFS)")
    ax2 = ax.twinx()
    ax2.plot(times, feats["periodicity"], color="purple", lw=1, label="Periodicity (0-1)")
    ax2.axhline(cfg.min_periodicity, color="purple", ls=":", lw=0.8)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Periodicity (autocorrelation peak)")
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, ncol=3)
    # spectrogram
    f, tt, sxx = a.compute_spectrogram(a.to_mono(data), fs, cfg)
    mesh = axes[3].pcolormesh(tt, f, 10 * np.log10(sxx / ref_power + a.EPS), shading="auto", cmap="magma",
                              vmin=-cfg.spec_range_db, vmax=0)
    axes[3].set_ylim(0, min(cfg.spec_max_hz, fs / 2))
    axes[3].set_ylabel("Frequency (Hz)")
    axes[3].set_xlabel("Time (s)")
    axes[3].grid(False)
    for r in det["active"]:
        for tx in (r["start_s"], r["end_s"]):
            axes[3].axvline(tx, color="cyan", ls="--", lw=0.8)
    axes[3].axvline(region["start_s"], color="white", lw=1.5)
    axes[3].axvline(region["end_s"], color="white", lw=1.5)
    fig.colorbar(mesh, ax=axes[3], label="Power (dB re. original peak)", pad=0.01)
    axes[0].set_xlim(w0, w1)
    fig.savefig(path)
    plt.close(fig)


def analyse_ambiguous_regions(key: str, data, proc, fs, det, trim_info, out: Path, cfg: a.Config, ref_power: float) -> list[dict]:
    offset = trim_info["kept_segments"][0][0] if trim_info.get("kept_segments") else 0.0
    results = []
    for i, region in enumerate(det["ambiguous"]):
        entry = region_features(data, fs, det, region, cfg)
        path = out / "plots" / f"{key}_ambiguous_region_{i + 1}.png"
        plot_region_focus(data, proc, offset, fs, det, region, f"{key} ambiguous region {i + 1}", path, cfg, ref_power)
        entry["focus_plot"] = path.relative_to(out.parent).as_posix()
        entry["kept_in_processed_output"] = bool(
            any(k0 <= region["start_s"] + 1e-6 and k1 >= region["end_s"] - 1e-6 for k0, k1 in trim_info["kept_segments"]))
        results.append(entry)
    return results


# --------------------------------------------------------------------------- joins
def plot_join_check(join_audio: dict, fs: int, path: Path, window_ms: float = 12.0) -> None:
    labels = {"natural": "Natural case: speech bursts around a 1.2 s noise gap (configured fade)",
              "plain_cut_fade_0ms": "Stress case: loud sine, plain cut (fade 0 ms)",
              "configured_fade": "Stress case: loud sine, configured fade"}
    fig, axes = plt.subplots(len(labels), 1, figsize=(10, 9), constrained_layout=True)
    for ax, (k, title) in zip(axes, labels.items()):
        sig, joins = join_audio[k]
        j = joins[0]
        half = int(window_ms * fs / 2000)
        sl = slice(max(0, j - half), min(sig.shape[0], j + half))
        tt = (np.arange(sl.start, sl.stop) - j) / fs * 1000
        ax.plot(tt, sig[sl, 0], ".-", lw=0.8, ms=3)
        ax.axvline(0, color="red", ls="--", label="join")
        ax.set_title(title)
        ax.set_xlabel("Time relative to join (ms)")
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right")
    fig.suptitle("Synthetic join check (numerical continuity only; not a listening test)", fontsize=13)
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- listening material
def write_listening_materials(found: dict, results: dict, out: Path, join_audio: dict, fs_synth: int) -> dict:
    """Write short clips and a checklist for manual listening. Nothing here records a listening result."""
    d = out / "listening_test"
    d.mkdir(parents=True, exist_ok=True)
    lines = ["# Manual listening checklist", "",
             "Generated by `python assignment1.py`. **No listening has been performed by the pipeline.** "
             "Use headphones in a quiet room. Clips are cut from the original recordings (not modified otherwise).",
             "To record your findings, create `manual_listening.json` in the repository root (see the end of this file); "
             "the next run will include it in the report as manual observations.", ""]
    clips = []
    for key, entry in found.items():
        data, fs = a.load_audio(entry["path"])
        for i, reg in enumerate(results["recordings"][key]["region_analysis"]):
            r = reg["region"]
            s0, s1 = max(0.0, r["start_s"] - CLIP_PAD_S), min(data.shape[0] / fs, r["end_s"] + CLIP_PAD_S)
            name = f"{key}_ambiguous_region_{i + 1}_{s0:.2f}-{s1:.2f}s.wav"
            sf.write(str(d / name), data[int(s0 * fs):int(s1 * fs)], fs, subtype="PCM_16")
            clips.append(name)
            lines += [f"## {key}: ambiguous region {i + 1} ({r['start_s']:.3f}-{r['end_s']:.3f} s)", "",
                      f"Clip: `outputs/listening_test/{name}` (starts at {s0:.2f} s of the original; the region begins "
                      f"{r['start_s'] - s0:.2f} s into the clip).",
                      f"Automated evidence: {reg['automated_label']}.", "",
                      "- [ ] Is there an audible word, syllable or consonant inside the region?",
                      "- [ ] Does the region sound like breath, mouth click, room/background noise, or handling noise?",
                      "- [ ] If the region were removed, would any intended speech be lost? (yes / no / cannot tell)",
                      f"- [ ] Is the recording identifiable as the declared utterance \"{results['recordings'][key]['declared_utterance']}\"?",
                      ""]
    for k, fname in (("plain_cut_fade_0ms", "synthetic_join_plain_cut.wav"), ("configured_fade", "synthetic_join_with_fade.wav")):
        sig, joins = join_audio[k]
        sf.write(str(d / fname), sig, fs_synth, subtype="PCM_16")
        clips.append(fname)
    sig, joins = join_audio["natural"]
    sf.write(str(d / "synthetic_join_natural.wav"), sig, fs_synth, subtype="PCM_16")
    clips.append("synthetic_join_natural.wav")
    lines += ["## Synthetic join clips (generated test signals, not your recordings)", "",
              "Compare `synthetic_join_plain_cut.wav` and `synthetic_join_with_fade.wav` (a loud tone cut at the join, "
              "around 0.5 s into the clip) and listen for a click or thump at the join.",
              "- [ ] Click audible in the plain cut?  - [ ] Click audible with the fade?",
              "Then listen to `synthetic_join_natural.wav` (two harmonic bursts joined across a shortened noise gap).",
              "- [ ] Audible discontinuity or change in background noise at the join?", "",
              "## Template for `manual_listening.json`", "", "```json",
              '{"listener": "name or initials", "date": "YYYY-MM-DD", "setup": "headphones model / room",',
              ' "observations": [{"item": "bit ambiguous region 1", "finding": "free text", "speech_present": "yes|no|cannot tell"}]}',
              "```"]
    (out / "listening_checklist.md").write_text("\n".join(lines), encoding="utf-8")
    return {"clips": clips, "checklist": "outputs/listening_checklist.md", "listening_performed": False}
