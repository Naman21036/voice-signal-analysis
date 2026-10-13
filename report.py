"""Markdown report generation from analysis_results.json-style results."""
from __future__ import annotations

from pathlib import Path

import evaluation as ev


def _f(v, d=3) -> str:
    return "n/a" if v is None else f"{v:.{d}f}"


def _regions_table(regs: list[dict]) -> list[str]:
    rows = ["| Type | Start (s) | End (s) | Duration (s) |", "|---|---|---|---|"]
    rows += [f"| {r['type']} | {_f(r['start_s'])} | {_f(r['end_s'])} | {_f(r['duration_s'])} |" for r in regs]
    if not regs:
        rows.append("| none | - | - | - |")
    return rows


def write_report(results: dict, cfg, path: Path) -> None:
    R = results["recordings"]
    keys = list(R)
    L: list[str] = []
    a = L.append

    a("# EC24313 Artificial Intelligence and Machine Learning - Assignment 1\n")
    a("**Course:** EC24313 - Artificial Intelligence and Machine Learning  ")
    a("**Student:** Naman Gupta  ")
    a("**Topic:** Voice recording, time-domain analysis, silence detection/removal and spectrogram\n")
    a("Every number below was produced by `python assignment1.py` from the actual recordings, or (where labelled "
      "*synthetic*) from deterministic test signals with known ground truth. Manual observations and assumptions "
      "are labelled as such.\n")

    a("## 1. Objective\n")
    a("To study two short utterances (declared as \"BIT\" and \"Naman Gupta\") in the time domain, locate silent regions "
      "with a robust detector, produce conservatively trimmed versions, and examine the time-frequency content with a "
      "spectrogram. Priority order: do not remove genuine speech; preserve soft onsets/decays; remove only confidently "
      "identified silence; preserve ambiguous low-level regions.\n")

    a("## 2. Tools and Libraries\n")
    a("Python, NumPy (frame energy, statistics), SoundFile (WAV I/O and header metadata), SciPy (STFT), Matplotlib (plots), "
      "pytest (tests). CPU only; no external services.\n")

    a("## 3. Methodology\n")
    a("1. **Identity:** each file is tied to its declared utterance through `audio/manifest.json`. This is a declaration, "
      "not verification: no speech recognition was run, so the spoken content is not independently confirmed.")
    a("2. **Time domain:** header metadata (sampling frequency, channels, samples), peak and RMS, waveform plots.")
    a("3. **Silence detection:** hysteresis energy detector (Section 5) compared with the original single-threshold baseline.")
    a("4. **Trimming:** conservative boundary trimming; ambiguous regions preserved (Section 6).")
    a("5. **Validation:** re-open each processed file and check integrity, join discontinuities and original-file hashes.")
    a("6. **Spectrogram:** Hann-window STFT, shared dB reference (Section 7).\n")

    a("## 4. Sampling Frequency and Time-Domain Analysis\n")
    a("The sampling frequency is the number of amplitude samples stored per second. It is read from the WAV header "
      "(`soundfile.info`), not estimated from the signal, and it is unrelated to the speaker's fundamental frequency "
      "(pitch, typically ~85-255 Hz for adults). Duration = samples / fs. The Nyquist limit fs/2 is the highest "
      "representable frequency. Both files are stereo; plots show both channels and the detector pools energy over both channels.\n")
    a("| Recording | Declared utterance | File | fs (Hz) | Channels | Samples | Duration (s) | Peak | RMS |")
    a("|---|---|---|---|---|---|---|---|---|")
    for k in keys:
        r = R[k]; m, s = r["metadata"], r["statistics"]
        a(f"| {k} | {r['declared_utterance']} | {m['source_file']} | {m['sampling_frequency_hz']} | {m['channels']} | "
          f"{m['num_samples']} | {_f(m['duration_s'])} | {_f(s['peak_amplitude'], 4)} | {_f(s['rms_amplitude'], 4)} |")
    a("")
    for k in keys:
        a(f"![{k} waveform](../{R[k]['plots']['waveform']})\n")

    a("## 5. Silent Region Identification\n")
    a("### 5.1 Baseline detector (original implementation)\n")
    a(f"Mono mix, {cfg.frame_ms:g} ms frames / {cfg.hop_ms:g} ms hop, threshold = max(P{cfg.noise_percentile:g} + {cfg.floor_margin_db:g} dB, "
      f"P{cfg.speech_percentile:g} - {cfg.speech_range_db:g} dB), pauses < {cfg.min_silence_ms:g} ms bridged, runs < {cfg.min_speech_ms:g} ms dropped. "
      "Weaknesses examined: the noise floor is a percentile of the same frames being classified (biased when little silence exists), "
      "channel averaging can cancel out-of-phase channels, and a single threshold has no way to express uncertainty.\n")
    a("### 5.2 Improved detector (default)\n")
    a(f"- Frame RMS in dB over **all channels pooled** ({cfg.frame_ms:g} ms frames, {cfg.hop_ms:g} ms hop; each frame owns the hop-sized span around its centre, so onsets are not delayed by half a frame).")
    a(f"- **Noise floor** = median of the quietest {cfg.noise_fraction * 100:.0f} % of frames after excluding digital-zero frames "
      f"(< {cfg.noise_exclude_db:g} dBFS, which carry no noise information). The MAD of those frames is reported as a stationarity check only; "
      "using it to set thresholds pushed the threshold into soft speech on the real recordings and was rejected.")
    a(f"- **Hysteresis:** low (hold) threshold = noise + {cfg.low_margin_db:g} dB; high (onset) threshold = noise + {cfg.high_margin_db:g} dB, "
      f"kept at least {cfg.level_gap_db:g} dB below the speech level (95th percentile).")
    a(f"- Confident **speech** needs >= {cfg.min_core_ms:g} ms above the high threshold and extends at most {cfg.max_extension_ms:g} ms beyond that core. "
      "Other low-level activity above the hold threshold is **ambiguous** (breath, noise or weak speech) and is preserved by default.")
    a(f"- Pauses shorter than {cfg.min_silence_ms:g} ms are bridged; active runs shorter than {cfg.min_speech_ms:g} ms are discarded.")
    a(f"- If speech level - noise floor < {cfg.min_snr_db:g} dB, no region is declared silent.")
    a("- Spectral flux, zero-crossing rate and spectral centroid were inspected on the real recordings around the suspect regions. "
      "They did not separate the low-frequency rumble/click regions from weak voiced sound reliably (centroids and ZCR overlapped), so they were **not adopted**.\n")
    a("| Recording | Noise floor (dB) | MAD (dB) | Low thr. (dB) | High thr. (dB) | Speech level (dB) | Level - noise (dB) |")
    a("|---|---|---|---|---|---|---|")
    for k in keys:
        d = R[k]["silence_detection"]
        a(f"| {k} | {_f(d['noise_floor_db'], 1)} | {_f(d.get('noise_mad_db'), 1)} | {_f(d['low_threshold_db'], 1)} | "
          f"{_f(d['high_threshold_db'], 1)} | {_f(d['speech_level_db'], 1)} | {_f(d.get('level_minus_noise_db'), 1)} |")
    a("")
    for k in keys:
        d, b = R[k]["silence_detection"], R[k]["baseline"]["silence_detection"]
        a(f"**{k} ({R[k]['declared_utterance']}) - improved detector**\n")
        a("\n".join(_regions_table(sorted(d["silence"] + d["speech"] + d["ambiguous"], key=lambda r: r["start_s"]))))
        a("")
        a(f"Baseline silence intervals for comparison: " + "; ".join(
            f"{r['type']} {_f(r['start_s'])}-{_f(r['end_s'])} s" for r in b["silence"]) + ".\n")
        for fl in d["flags"]:
            a(f"- Review flag: {fl}")
        a("")
        a(f"![{k} silence detection](../{R[k]['plots']['silence_detection']})\n")

    a("### 5.3 Boundary review of the suspected regions\n")
    a("The zoomed figures show waveform, frame RMS with both thresholds, and spectrogram at the start and end of each recording. "
      "Cyan dashed lines are the detected active-region edges.\n")
    for k in keys:
        a(f"![{k} boundary review](../{R[k]['plots']['boundary_review']})\n")
    mr = results.get("manual_review")
    if mr:
        a(f"**Manual observations** ({mr['basis']}):\n")
        a("| Recording | Region | Observation | Judgement |")
        a("|---|---|---|---|")
        for e in mr["reviews"]:
            a(f"| {e['recording']} | {e['region']} | {e['observation']} | {e['judgement']} |")
        a("")
    else:
        a("No manual review file was found (`manual_review.json`).\n")

    a("### 5.4 Sensitivity to parameters (real recordings)\n")
    a("Boundaries and reduction when the frame length and the low-threshold margin are varied (all else default):\n")
    a("| Recording | Frame (ms) | Low margin (dB) | First active (s) | Last active (s) | Speech / ambiguous regions | Removed (%) |")
    a("|---|---|---|---|---|---|---|")
    for k in keys:
        for s in R[k]["sensitivity"]:
            a(f"| {k} | {s['frame_ms']:g} | {s['low_margin_db']:g} | {_f(s['first_active_start_s'])} | {_f(s['last_active_end_s'])} | "
              f"{s['n_speech']} / {s['n_ambiguous']} | {_f(s['percent_removed'], 1)} |")
    a("")

    a("## 6. Audio Preprocessing and Silence Removal\n")
    a(f"- Leading/trailing: cut to the first/last **active** region (speech or ambiguous) with {cfg.edge_pad_ms:g} ms context, "
      f"and a {cfg.fade_ms:g} ms fade at each cut that is not the original file edge. The padding exceeds the frame length plus the measured onset ramps, so soft onsets are kept.")
    a(f"- Internal: a gap is shortened (to {cfg.kept_gap_ms:g} ms) only if it is longer than {cfg.max_internal_gap_ms:g} ms **and** both neighbours are confident speech. "
      "Natural pauses, stop closures and gaps next to ambiguous regions are preserved.")
    a(f"- `trim_ambiguous` = {cfg.trim_ambiguous}: ambiguous regions are not removed. The table also shows what removing them would do, for information only.")
    a("- Sampling frequency, channels and 16-bit PCM format are unchanged; no gain is applied.\n")
    a("| Recording | Original (s) | Processed (s) | Removed (s) | Removed (%) | Gaps shortened | Gaps preserved |")
    a("|---|---|---|---|---|---|---|")
    for k in keys:
        r = R[k]; d = r["reduction"]
        a(f"| {k} | {_f(d['original_duration_s'])} | {_f(d['processed_duration_s'], 4)} | {_f(d['removed_duration_s'], 4)} | "
          f"{_f(d['percent_removed'], 1)} | {len(r['trimming']['shortened_gaps'])} | {len(r['trimming']['preserved_gaps'])} |")
    a("")
    a("**Baseline versus improved (measured on the real recordings)**\n")
    a("| Recording | Baseline processed (s) | Baseline removed (%) | Improved processed (s) | Improved removed (%) | Ambiguous kept (s) |")
    a("|---|---|---|---|---|---|")
    for k in keys:
        r = R[k]; b = r["baseline"]["reduction"]; d = r["reduction"]
        amb = sum(x["duration_s"] for x in r["silence_detection"]["ambiguous"])
        a(f"| {k} | {_f(b['processed_duration_s'], 4)} | {_f(b['percent_removed'], 1)} | {_f(d['processed_duration_s'], 4)} | "
          f"{_f(d['percent_removed'], 1)} | {_f(amb)} |")
    a("")
    a("A smaller reduction than the baseline is intentional where the baseline cut into low-level material that cannot be shown to be non-speech. "
      "The reduction is not a quality metric.\n")
    for k in keys:
        a(f"![{k} original vs processed](../{R[k]['plots']['trimmed_waveform']})\n")

    a("### 6.1 Processed-audio integrity (measured)\n")
    checks = list(R[keys[0]]["validation"]["checks"])
    a("| Check | " + " | ".join(keys) + " |")
    a("|---|" + "---|" * len(keys))
    for c in checks:
        a(f"| {c} | " + " | ".join("pass" if R[k]["validation"]["checks"][c] else "FAIL" for k in keys) + " |")
    a("")
    a("Joins: " + "; ".join(f"{k}: {R[k]['validation']['joins']['checked']} edited join(s)" for k in keys) + ". "
      "These checks establish signal integrity only. Intelligibility and audible artefacts were **not** assessed by listening.\n")

    a("## 7. Spectrogram Analysis\n")
    a(f"A spectrogram is the magnitude of the Short-Time Fourier Transform: the signal is cut into short overlapping windows "
      f"(Hann, {cfg.spec_nfft} samples, hop {cfg.spec_hop} samples), each is Fourier-transformed and the power in dB is shown against time and frequency. "
      "Voiced sounds give harmonic stripes spaced at the pitch, vowel formants appear as broad bands, bursts and fricatives as broadband energy.\n")
    fs = R[keys[0]]["metadata"]["sampling_frequency_hz"]
    a(f"At fs = {fs} Hz the window is {1000 * cfg.spec_nfft / fs:.2f} ms long and the nominal bin spacing is {fs / cfg.spec_nfft:.3f} Hz "
      f"(hop {1000 * cfg.spec_hop / fs:.2f} ms). Harmonics of a ~100-200 Hz voice are therefore about 2-4 bins apart (resolved), while 21 ms is short "
      "enough to follow vowel-to-consonant changes. The comparison figures show 512 / 1024 / 2048-point windows: 512 blurs harmonics, 2048 smears timing; "
      "1024 was kept as the compromise, judged on these recordings by inspection rather than by an objective criterion. "
      f"The vertical axis stops at min({cfg.spec_max_hz:g} Hz, Nyquist). **0 dB is the peak spectrogram power of the original recording**, and the processed spectrogram uses the same reference, "
      "so colours are comparable. Individual harmonics or formants of a 1-2 s utterance should not be over-interpreted: pitch varies within a window and there is no repetition to average over.\n")
    for k in keys:
        a(f"![{k} spectrogram](../{R[k]['plots']['spectrogram']})\n")
        a(f"Processed: ![{k} processed spectrogram](../{R[k]['plots']['spectrogram_processed']})\n")
        a(f"Resolution comparison: ![{k} spectrogram resolution](../{R[k]['plots']['spectrogram_resolution']})\n")

    a("## 8. Controlled Evaluation on Synthetic Signals\n")
    E = results["synthetic_evaluation"]
    a(f"Deterministic synthetic signals (harmonic 'speech-like' bursts plus filtered noise, fs = {E['sampling_frequency_hz']} Hz) with known "
      "speech intervals. They test the algorithms only and are not substitutes for the recordings. Activity metrics use a "
      f"{E['grid_ms']:g} ms grid; for the improved detector confident speech and ambiguous regions both count as active (both are preserved). "
      "Boundary error = detected first start / last end minus truth (negative start error = starts early, i.e. conservative). "
      "Preserved = fraction of true speech samples that remain after trimming.\n")
    a("| Scenario | Baseline F1 | Improved F1 | Baseline preserved | Improved preserved | Baseline start/end err (ms) | Improved start/end err (ms) | Baseline dur. err (s) | Improved dur. err (s) |")
    a("|---|---|---|---|---|---|---|---|---|")
    for name, s in E["scenarios"].items():
        b, i = s["baseline"], s["improved"]
        be = "-" if b["start_error_ms"] is None else f"{b['start_error_ms']:.0f} / {b['end_error_ms']:.0f}"
        ie = "-" if i["start_error_ms"] is None else f"{i['start_error_ms']:.0f} / {i['end_error_ms']:.0f}"
        a(f"| {name} | {b['f1']:.2f} | {i['f1']:.2f} | {b['speech_preserved_fraction']:.3f} | {i['speech_preserved_fraction']:.3f} | "
          f"{be} | {ie} | {b['duration_error_s']:+.2f} | {i['duration_error_s']:+.2f} |")
    a("")
    worst = min(s["improved"]["speech_preserved_fraction"] for s in E["scenarios"].values())
    a(f"Lowest speech-preserved fraction of the improved pipeline across scenarios: {worst:.3f}. "
      "Observations: the improved detector recovers the speech when the file contains little silence and when the channels are out of phase "
      "(both break the baseline), and it declines to trim when speech and noise are too close to separate (SNR 10 dB: nothing is trimmed, everything is ambiguous). "
      "A 15 ms click and a 300 ms noise burst inside silence are still classed as speech/active by both detectors - energy alone cannot tell loud noise from speech.\n")
    a("No precision/recall/F1 is reported for the real recordings because no independently labelled ground truth exists for them.\n")

    a("## 9. Results Summary and Discussion\n")
    for k in keys:
        r = R[k]; d = r["reduction"]; s = r["silence_detection"]
        lead = sum(x["duration_s"] for x in s["silence"] if x["type"] == "leading")
        trail = sum(x["duration_s"] for x in s["silence"] if x["type"] == "trailing")
        internal = [x for x in s["silence"] if x["type"] == "internal"]
        amb = s["ambiguous"]
        a(f"- **{k} ({r['declared_utterance']})**: confident leading silence {_f(lead)} s (includes leading digital zeros), trailing silence {_f(trail)} s, "
          f"{len(internal)} internal silent region(s), {len(amb)} ambiguous region(s) "
          f"({', '.join(f'{x['start_s']:.2f}-{x['end_s']:.2f} s' for x in amb) or 'none'}); {_f(d['percent_removed'], 1)} % of the duration removed.")
    a("")
    a("Which of the ambiguous regions are breath, room noise or weak speech cannot be settled from amplitude and the spectrograms alone, so they are kept. "
      "Removing them (`trim_ambiguous=True`) would shorten the files further but risks clipping speech; that is a decision for listening review.\n")

    a("## 10. Limitations\n")
    a("- Silence detection is energy-based. Loud non-speech sounds (clicks, noise bursts) look like speech; very quiet speech near the noise floor may be labelled ambiguous or, in the worst case, silence.")
    a("- Thresholds assume the quietest frames are representative of the background noise; this fails if the recording has no noise-only frames. The minimum-separation guard only partly mitigates this.")
    a("- The utterance in each file is declared in `audio/manifest.json`; it was not verified by recognition or by listening.")
    a("- Manual boundary observations were made from plots by the author of the analysis, not by listening, and are not ground truth.")
    a("- Join/click checks measure sample discontinuities, not audibility. No intelligibility test was performed, and no recognition-accuracy claim is made.")
    a("- Synthetic results show behaviour on simplified signals (stationary filtered noise, harmonic bursts) and need not transfer to every real recording.")
    a("- Spectrogram appearance depends on window length, hop and dB range.\n")

    a("## 11. Conclusion\n")
    a("Both recordings were analysed with a hysteresis detector that is at least as good as the original on every synthetic scenario tested for speech "
      "preservation, handles two failure cases of the baseline, and flags uncertain low-level regions instead of deleting them. "
      "On the real recordings it removes only confidently identified silence; remaining uncertainty is stated above. "
      "All measured values are in `analysis_results.json`; the frozen pre-change results are in `baseline_results.json`.\n")
    path.write_text("\n".join(L), encoding="utf-8")
