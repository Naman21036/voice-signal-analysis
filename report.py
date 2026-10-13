"""Markdown report generation from analysis_results.json-style results."""
from __future__ import annotations

from pathlib import Path

import evaluation as ev


def _f(v, d=3) -> str:
    return "n/a" if v is None else f"{v:.{d}f}"


def _lr_text(ch) -> str:
    return "n/a (not stereo)" if not ch or len(ch) < 2 else f"{ch[0]:.1f} / {ch[1]:.1f}"


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
      "*synthetic*) from deterministic test signals with known ground truth. Each kind of evidence is labelled:\n")
    a("| Evidence class | Where it appears | What it can and cannot show |")
    a("|---|---|---|")
    a("| **Measured** (automatic, real recordings) | Sections 4, 5.1-5.2, 5.4, 6, 7 | Durations, levels, detected regions; no ground truth for speech/non-speech |")
    a("| **Recognition-based** (offline ASR, real recordings) | Section 3.1 | Supporting evidence for the declared utterance; not ground truth |")
    a("| **Automated region evidence** (features, real recordings) | Section 5.3 | Describes ambiguous regions; cannot identify speech |")
    a("| **Manual observation** (plots inspected by the analyst) | Section 5.3 | Judgement from images; not a listening test |")
    a("| **Manual listening** | Section 5.3 (only if `manual_listening.json` exists) | Human listening results, if provided |")
    a("| **Synthetic** (generated signals, known truth) | Sections 5.5, 6.2, 8 | Algorithm behaviour on simplified signals only |")
    a("| **Assumptions / unverified** | Section 10 | Listed explicitly |")
    a("")

    a("## 1. Objective\n")
    a("To study two short utterances (declared as \"BIT\" and \"Naman Gupta\") in the time domain, locate silent regions "
      "with a robust detector, produce conservatively trimmed versions, and examine the time-frequency content with a "
      "spectrogram. Priority order: do not remove genuine speech; preserve soft onsets/decays; remove only confidently "
      "identified silence; preserve ambiguous low-level regions.\n")

    a("## 2. Tools and Libraries\n")
    a("Python, NumPy (frame energy, statistics), SoundFile (WAV I/O and header metadata), SciPy (STFT), Matplotlib (plots), "
      "pytest (tests). Optional, only for `verify_content.py`: faster-whisper with a locally cached model and the Windows built-in System.Speech recogniser. CPU only; audio is never uploaded.\n")

    a("## 3. Methodology\n")
    a("1. **Identity:** each file is tied to its declared utterance through `audio/manifest.json` (a declaration), then "
      "cross-checked with offline speech recognition where available (Section 3.1).")
    a("2. **Time domain:** header metadata (sampling frequency, channels, samples), peak and RMS, waveform plots.")
    a("3. **Silence detection:** hysteresis energy detector (Section 5) compared with the original single-threshold baseline.")
    a("4. **Trimming:** conservative boundary trimming; ambiguous regions preserved (Section 6).")
    a("5. **Validation:** re-open each processed file and check integrity, join discontinuities and original-file hashes.")
    a("6. **Spectrogram:** Hann-window STFT, shared dB reference (Section 7).\n")

    a("### 3.1 Verification of spoken content (recognition-based evidence)\n")
    cv = results.get("content_verification")
    if cv:
        a(cv["basis"] + " Command: `python verify_content.py`. Audio was converted to 16 kHz mono in a temporary directory "
          "(originals untouched). Comparison ignores case and punctuation (`B-I-T` = `bit`); a similarity of at least 0.8 counts as consistent.\n")
        a("| Recording | Declared | Engine | Transcript | Similarity | Consistent? | Engine confidence |")
        a("|---|---|---|---|---|---|---|")
        for k, v in cv["recordings"].items():
            for e in v["engines"]:
                if not e.get("available"):
                    a(f"| {k} | {v['declared_utterance']} | {e['engine']} | unavailable: {e.get('reason', '')} | - | - | - |")
                    continue
                conf = (f"avg log-prob {e['avg_logprob']}" if "avg_logprob" in e else f"confidence {e.get('confidence')}")
                a(f"| {k} | {v['declared_utterance']} | {e['engine']} | \"{e['transcript']}\" | {e['comparison']['similarity']:.2f} | "
                  f"{'yes' if e['comparison']['consistent_with_expected'] else 'no'} | {conf} |")
        a("")
        for k, v in cv["recordings"].items():
            a(f"- **{k}**: status `{v['status']}` - {v['note']} Confidence: {v['confidence']}.")
        a("\nThe Whisper model is a local copy; the Windows dictation engine is a weak recogniser for single short words, so its "
          "disagreement is weak evidence against the declared utterance, and Whisper's agreement is not proof. "
          "**Manual verification procedure:** " + " ".join(f"({i + 1}) {s}" for i, s in enumerate(cv["manual_procedure"])) + "\n")
    else:
        a("No recognition run was recorded (`outputs/content_verification.json` missing): the utterances are **unverified** and only declared. "
          "Run `python verify_content.py`, or follow the manual procedure in `outputs/listening_checklist.md`.\n")

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
    a(f"- **Periodicity gate** (`use_periodicity_gate` = {cfg.use_periodicity_gate}): a high-energy cluster is accepted as confident speech only if it contains at least "
      f"{cfg.min_periodic_ms:g} ms of frames whose normalised autocorrelation peak ({cfg.pitch_min_hz:g}-{cfg.pitch_max_hz:g} Hz lags, {cfg.periodicity_frame_ms:g} ms window) is >= {cfg.min_periodicity:g}; "
      "otherwise it is labelled ambiguous (still preserved). Evaluated in Section 5.5. It detects voicing, not speech.")
    a("- Zero-crossing rate, spectral flux, spectral flatness and low-frequency energy fraction were also measured on controlled signals (Section 5.5) and were not adopted.\n")
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

    a("**Automated evidence for each ambiguous region (real recordings).** Features are medians over the region's frames; "
      "the references are the same file's confident-speech frames and confident-silence frames (digital zeros excluded). "
      "Periodicity is the autocorrelation peak (0-1); 'periodic fraction' is the share of frames at or above the gate threshold. "
      "Channel levels are pooled RMS (dBFS) of left/right over the region, against the same measure over the confident speech.\n")
    a("| Recording | Region (s) | Level (dB) | Periodic fraction | Region / speech / silence periodicity | <300 Hz energy share (region / speech / silence) | Kurtosis | Crest | L / R level (dB) | Speech L / R (dB) | Automated label |")
    a("|---|---|---|---|---|---|---|---|---|---|---|")
    for k in keys:
        for e in R[k]["region_analysis"]:
            f, sp, si = e["region_features"], e["reference_confident_speech"], e["reference_confident_silence"]
            lf = "low_freq_fraction_<300Hz_median"
            a(f"| {k} | {_f(e['region']['start_s'])}-{_f(e['region']['end_s'])} | {_f(f['rms_db_median'], 1)} | {_f(f['periodic_frame_fraction'], 2)} | "
              f"{_f(f['periodicity_median'], 2)} / {_f(sp['periodicity_median'], 2)} / {_f(si['periodicity_median'], 2) if si else 'n/a'} | "
              f"{_f(f[lf], 2)} / {_f(sp[lf], 2)} / {_f(si[lf], 2) if si else 'n/a'} | {_f(e['kurtosis_excess'], 1)} | {_f(e['crest_factor'], 1)} | "
              f"{_lr_text(e['channel_level_db'])} | "
              f"{_lr_text(e['reference_speech_channel_level_db'])} | {e['automated_label']} |")
    a("")
    regs = [(k, e) for k in keys for e in R[k]["region_analysis"]]
    if not regs:
        a("No ambiguous regions were detected in these recordings, so no region evidence is reported.\n")
    gaps =[e["reference_confident_speech"]["rms_db_median"] - e["region_features"]["rms_db_median"] for k, e in regs]
    pfs = [e["region_features"]["periodic_frame_fraction"] for k, e in regs]
    def _d(ch):
        return None if not ch or len(ch) < 2 else ch[0] - ch[1]
    lr_reg = [_d(e["channel_level_db"]) for k, e in regs]
    lr_sp = [_d(e["reference_speech_channel_level_db"]) for k, e in regs]
    more_left = [f"{k} {e['region']['start_s']:.2f}-{e['region']['end_s']:.2f} s" for (k, e), r, s_ in zip(regs, lr_reg, lr_sp) if r is not None and s_ is not None and r > s_ + 3]
    if regs: a(f"Reading (measured): the ambiguous regions are {min(gaps, default=0):.0f}-{max(gaps, default=0):.0f} dB below the median level of the confident speech, and "
      f"{min(pfs, default=0) * 100:.0f}-{max(pfs, default=0) * 100:.0f} % of their frames pass the periodicity threshold (confident speech: see table). "
      "Periodicity sits between the silence reference and voiced speech and most energy is below 300 Hz, so these features do not separate weak voicing "
      "from low-frequency noise or breath: the automated label is inconclusive. "
      + (f"Regions with a left-minus-right level difference more than 3 dB larger than the speech's ({'; '.join(more_left)}) "
         "are more left-channel dominant than the speech, which would fit a source other than the speaker's voice (handling noise, a breath near one microphone), "
         "but this is only a hint. " if more_left else "")
      + "All regions are **preserved** ("
      + ("all present in the processed output" if all(e["kept_in_processed_output"] for k, e in regs) else "NOT all present - check")
      + ").\n")
    for k in keys:
        for i, e in enumerate(R[k]["region_analysis"]):
            a(f"![{k} ambiguous region {i + 1} focus](../{e['focus_plot']})\n")
    a("Each focus figure shows, from top to bottom: the original waveform with classified regions; the processed output placed on the original time axis "
      "(confirming the region is present); frame RMS with both thresholds and the periodicity curve; and the spectrogram with detected edges (cyan) and the region edges (white).\n")
    ml = results.get("manual_listening")
    if ml:
        a(f"**Manual listening observations** (listener: {ml.get('listener', 'n/a')}, {ml.get('date', 'n/a')}; setup: {ml.get('setup', 'n/a')}):\n")
        for o in ml.get("observations", []):
            a(f"- {o.get('item')}: {o.get('finding')} (speech present: {o.get('speech_present', 'n/a')})")
        a("")
    else:
        a("**Manual listening: not performed.** No `manual_listening.json` was supplied. `outputs/listening_checklist.md` lists clip files "
          "(`outputs/listening_test/`) with timestamps and questions for each ambiguous region. Until someone listens, the identity of these regions remains unresolved.\n")

    a("### 5.4 Sensitivity to parameters (real recordings)\n")
    a("Boundaries and reduction when the frame length and the low-threshold margin are varied (all else default):\n")
    a("| Recording | Frame (ms) | Low margin (dB) | First active (s) | Last active (s) | Speech / ambiguous regions | Removed (%) |")
    a("|---|---|---|---|---|---|---|")
    for k in keys:
        for s in R[k]["sensitivity"]:
            a(f"| {k} | {s['frame_ms']:g} | {s['low_margin_db']:g} | {_f(s['first_active_start_s'])} | {_f(s['last_active_end_s'])} | "
              f"{s['n_speech']} / {s['n_ambiguous']} | {_f(s['percent_removed'], 1)} |")
    a("")

    a("### 5.5 Loud non-speech sounds: do extra features help? (synthetic evaluation)\n")
    FE = results["feature_evaluation"]
    a(FE["note"] + "\n")
    a("Frame-level separability of voiced speech-like frames from loud non-speech frames (AUC = probability that a speech frame scores higher than a "
      "non-speech frame; 0.5 = no separation; a value far below 0.5 means the feature separates them in the opposite direction):\n")
    a("| Feature | AUC |")
    a("|---|---|")
    for k, v in FE["feature_auc"].items():
        a(f"| {k} | {_f(v, 2)} |")
    a("")
    a("Detector comparison on the speech class (rates on a 10 ms grid; FP = non-speech frames accepted as confident speech; "
      "FN = true-speech frames not accepted as confident speech; Active FN = true-speech frames that would be trimmed away):\n")
    a("| Scenario | Energy-only FP | FN | Active FN | With periodicity gate FP | FN | Active FN |")
    a("|---|---|---|---|---|---|---|")
    for name, s_ in FE["scenarios"].items():
        e, g = s_["energy_only"], s_["with_periodicity_gate"]
        a(f"| {name} | {e['speech_false_positive_rate']:.2f} | {e['speech_false_negative_rate']:.2f} | {e['active_false_negative_rate']:.2f} | "
          f"{g['speech_false_positive_rate']:.2f} | {g['speech_false_negative_rate']:.2f} | {g['active_false_negative_rate']:.2f} |")
    a("")
    changed = [k for k in keys if R[k]["periodicity_gate_changed_regions"]]
    a("Findings: the periodicity gate cuts the speech false-positive rate for clicks, white-noise bursts and breath-like noise, and leaves quiet voiced "
      "speech and an unvoiced sound next to voiced speech unaffected. It does **not** help against a periodic low-frequency thump, and a standalone "
      "unvoiced sound (a fricative with no voiced neighbour) is no longer accepted as confident speech: it becomes ambiguous and is still preserved "
      "(Active FN stays 0). Its practical effect is classification and flagging, because ambiguous regions are preserved either way. The other features were "
      "not adopted: their AUCs come from stationary synthetic noise, which differs from real speech transitions, and no labelled real data exists to validate them. "
      "The pipeline therefore stays **energy-based with an optional voicing check**; it is not a validated speech/non-speech classifier. "
      "On the real recordings, running with the gate off and on "
      + ("gave identical regions." if not changed else "gave different regions for: " + ", ".join(changed) + ".") + "\n")

    a("## 6. Audio Preprocessing and Silence Removal\n")
    a(f"- Leading/trailing: cut to the first/last **active** region (speech or ambiguous) with {cfg.edge_pad_ms:g} ms context, "
      f"and a {cfg.fade_ms:g} ms fade at each cut that is not the original file edge. The padding exceeds the frame length plus the measured onset ramps, so soft onsets are kept.")
    a(f"- Internal: a gap is shortened (to {cfg.kept_gap_ms:g} ms) only if it is longer than {cfg.max_internal_gap_ms:g} ms **and** both neighbours are confident speech. "
      "Natural pauses, stop closures and gaps next to ambiguous regions are preserved.")
    a(f"- `trim_ambiguous` = {cfg.trim_ambiguous}: ambiguous regions are not removed.")
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
    checks = list(dict.fromkeys(c for k in keys for c in R[k]["validation"]["checks"]))
    a("| Check | " + " | ".join(keys) + " |")
    a("|---|" + "---|" * len(keys))
    for c in checks:
        a(f"| {c} | " + " | ".join(("pass" if R[k]["validation"]["checks"][c] else "FAIL") if c in R[k]["validation"]["checks"] else "not applicable"
                                   for k in keys) + " |")
    a("")
    a("These checks establish signal integrity only. Intelligibility and audible artefacts were **not** assessed by listening.\n")

    a("### 6.2 Join validation\n")
    for k in keys:
        j = R[k]["validation"]["joins"]
        a(f"- **Real recording {k}**: {j['joins_created']} join(s) created. {j['applicability']}.")
    a("")
    sj = results["synthetic_join_validation"]
    nat, st = sj["natural_case"], sj["stress_case"]
    a("Because no join exists in the real outputs, the join code was exercised on **synthetic** signals (not the recordings):\n")
    a("| Case | Joins | Largest step at join | 99.9th-percentile step of file | Within reference? |")
    a("|---|---|---|---|---|")
    a(f"| Natural: two speech bursts around a 1.2 s noise gap (configured {cfg.fade_ms:g} ms fade) | {nat['joins_created']} | "
      f"{_f(nat['check']['join_max_step'], 4)} | {_f(nat['check']['reference_p99_9_step'], 4)} | {'yes' if nat['check']['ok'] else 'NO'} |")
    for lab, name in (("plain_cut_fade_0ms", "Stress: loud 233.3 Hz sine, plain cut (no fade)"), ("configured_fade", "Stress: same, configured fade")):
        c = st[lab]
        a(f"| {name} | {c['checked']} | {_f(c['join_max_step'], 4)} | {_f(c['reference_p99_9_step'], 4)} | {'yes' if c['ok'] else 'NO'} |")
    a("")
    a("The stress case is built so the cut lands at high amplitude with a phase jump: the plain cut produces a step as large as "
      "the signal itself and fails the check, while the fade removes it, so the check can detect a bad join and the fade code works. "
      "In real use, joins only occur inside confident silence between two speech regions, where the signal is near the noise floor. "
      f"![synthetic join check](../{sj['plot']})\n")
    a("This is **numerical continuity** (no sample jump). It is not evidence of perceptual smoothness: an audible change in background-noise level "
      "or room tone at a join is not measured. Perceptual smoothness has not been evaluated; `outputs/listening_test/synthetic_join_*.wav` and "
      "`outputs/listening_checklist.md` are provided for a listening check, and none has been performed.\n")

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
    a("Status of each earlier limitation after this round of work:\n")
    a("| Limitation | Status | Evidence |")
    a("|---|---|---|")
    cvs = {k: (cv or {"recordings": {}})["recordings"].get(k, {}).get("status", "unverified") for k in keys}
    a("| Spoken content only declared | **Partly addressed, still unverified** | Offline ASR run (Section 3.1): statuses "
      + ", ".join(f"{k}: {v}" for k, v in cvs.items()) + ". ASR is not ground truth; no listening check. |")
    a("| Ambiguous low-level regions unresolved | **Investigated, still unresolved** | Features and plots (Section 5.3) are inconclusive; the regions are preserved. A listening check is needed. |")
    a("| Join quality not tested on real audio | **Clarified** | No joins exist in the real outputs (Section 6.2). Numerical continuity was validated on synthetic signals only. |")
    a("| Perceptual smoothness of joins | **Unchanged (not evaluated)** | Listening material is provided; no listening has been performed unless `manual_listening.json` exists. |")
    a("| Loud non-speech sounds look like speech | **Partly mitigated** | The periodicity gate reduces false positives on synthetic clicks, noise bursts and breath-like noise (Section 5.5); it fails on a periodic thump and is not validated on real labelled data. |")
    a("")
    a("Further limitations:\n")
    a("- Detection is energy-based with an optional voicing check. It is not a validated speech/non-speech classifier; very quiet speech near the noise floor may be labelled ambiguous or, in the worst case, silence.")
    a("- Thresholds assume the quietest frames represent the background noise; this fails if a recording has no noise-only frames. The minimum-separation guard only partly mitigates this.")
    a("- A standalone unvoiced sound is not accepted as confident speech by the periodicity gate (it is kept as ambiguous).")
    a("- Manual boundary observations come from plots, not listening, and are not ground truth.")
    a("- Synthetic results show behaviour on simplified signals and need not transfer to the real recordings.")
    a("- Spectrogram appearance depends on window length, hop and dB range.\n")

    a("## 11. Conclusion\n")
    a("Both recordings were analysed with a conservative hysteresis detector. Offline recognition gives partial, non-definitive support for the declared utterances; "
      "the ambiguous low-level regions could not be classified from automated evidence and are preserved; no joins were created in the real outputs, so join behaviour "
      "is validated on synthetic signals only; and a periodicity gate measurably reduces false speech labels on synthetic non-speech sounds without changing the real-recording regions. "
      "Listening checks remain necessary for the ambiguous regions, the spoken content and the perceptual quality of joins. "
      "All measured values are in `analysis_results.json`; the frozen pre-change results are in `baseline_results.json`.\n")
    path.write_text("\n".join(L), encoding="utf-8")
