# EC24313 Artificial Intelligence and Machine Learning - Assignment 1

**Course:** EC24313 - Artificial Intelligence and Machine Learning  
**Student:** Naman Gupta  
**Topic:** Voice recording, time-domain analysis, silence detection/removal and spectrogram

Every number below was produced by `python assignment1.py` from the actual recordings, or (where labelled *synthetic*) from deterministic test signals with known ground truth. Manual observations and assumptions are labelled as such.

## 1. Objective

To study two short utterances (declared as "BIT" and "Naman Gupta") in the time domain, locate silent regions with a robust detector, produce conservatively trimmed versions, and examine the time-frequency content with a spectrogram. Priority order: do not remove genuine speech; preserve soft onsets/decays; remove only confidently identified silence; preserve ambiguous low-level regions.

## 2. Tools and Libraries

Python, NumPy (frame energy, statistics), SoundFile (WAV I/O and header metadata), SciPy (STFT), Matplotlib (plots), pytest (tests). CPU only; no external services.

## 3. Methodology

1. **Identity:** each file is tied to its declared utterance through `audio/manifest.json`. This is a declaration, not verification: no speech recognition was run, so the spoken content is not independently confirmed.
2. **Time domain:** header metadata (sampling frequency, channels, samples), peak and RMS, waveform plots.
3. **Silence detection:** hysteresis energy detector (Section 5) compared with the original single-threshold baseline.
4. **Trimming:** conservative boundary trimming; ambiguous regions preserved (Section 6).
5. **Validation:** re-open each processed file and check integrity, join discontinuities and original-file hashes.
6. **Spectrogram:** Hann-window STFT, shared dB reference (Section 7).

## 4. Sampling Frequency and Time-Domain Analysis

The sampling frequency is the number of amplitude samples stored per second. It is read from the WAV header (`soundfile.info`), not estimated from the signal, and it is unrelated to the speaker's fundamental frequency (pitch, typically ~85-255 Hz for adults). Duration = samples / fs. The Nyquist limit fs/2 is the highest representable frequency. Both files are stereo; plots show both channels and the detector pools energy over both channels.

| Recording | Declared utterance | File | fs (Hz) | Channels | Samples | Duration (s) | Peak | RMS |
|---|---|---|---|---|---|---|---|---|
| bit | BIT | Naman_BIT.wav | 48000 | 2 | 118656 | 2.472 | 0.9523 | 0.1615 |
| name | Naman Gupta | Naman_Name.wav | 48000 | 2 | 87552 | 1.824 | 0.9477 | 0.1666 |

![bit waveform](../outputs/plots/bit_waveform.png)

![name waveform](../outputs/plots/name_waveform.png)

## 5. Silent Region Identification

### 5.1 Baseline detector (original implementation)

Mono mix, 25 ms frames / 10 ms hop, threshold = max(P10 + 12 dB, P95 - 35 dB), pauses < 100 ms bridged, runs < 30 ms dropped. Weaknesses examined: the noise floor is a percentile of the same frames being classified (biased when little silence exists), channel averaging can cancel out-of-phase channels, and a single threshold has no way to express uncertainty.

### 5.2 Improved detector (default)

- Frame RMS in dB over **all channels pooled** (25 ms frames, 10 ms hop; each frame owns the hop-sized span around its centre, so onsets are not delayed by half a frame).
- **Noise floor** = median of the quietest 10 % of frames after excluding digital-zero frames (< -90 dBFS, which carry no noise information). The MAD of those frames is reported as a stationarity check only; using it to set thresholds pushed the threshold into soft speech on the real recordings and was rejected.
- **Hysteresis:** low (hold) threshold = noise + 8 dB; high (onset) threshold = noise + 20 dB, kept at least 10 dB below the speech level (95th percentile).
- Confident **speech** needs >= 30 ms above the high threshold and extends at most 150 ms beyond that core. Other low-level activity above the hold threshold is **ambiguous** (breath, noise or weak speech) and is preserved by default.
- Pauses shorter than 100 ms are bridged; active runs shorter than 30 ms are discarded.
- If speech level - noise floor < 15 dB, no region is declared silent.
- Spectral flux, zero-crossing rate and spectral centroid were inspected on the real recordings around the suspect regions. They did not separate the low-frequency rumble/click regions from weak voiced sound reliably (centroids and ZCR overlapped), so they were **not adopted**.

| Recording | Noise floor (dB) | MAD (dB) | Low thr. (dB) | High thr. (dB) | Speech level (dB) | Level - noise (dB) |
|---|---|---|---|---|---|---|
| bit | -42.6 | 1.1 | -34.6 | -22.6 | -9.6 | 33.0 |
| name | -42.4 | 1.4 | -34.4 | -22.4 | -9.7 | 32.7 |

**bit (BIT) - improved detector**

| Type | Start (s) | End (s) | Duration (s) |
|---|---|---|---|
| leading | 0.000 | 0.158 | 0.158 |
| ambiguous | 0.158 | 0.608 | 0.450 |
| speech | 0.608 | 1.718 | 1.110 |
| internal | 1.718 | 1.817 | 0.100 |
| speech | 1.817 | 2.098 | 0.280 |
| trailing | 2.098 | 2.472 | 0.374 |

Baseline silence intervals for comparison: leading 0.000-0.182 s; internal 0.472-0.723 s; internal 1.712-1.823 s; trailing 2.092-2.472 s.

- Review flag: 1 ambiguous low-level region(s) kept: level is above the noise floor but never reaches the speech-onset threshold

![bit silence detection](../outputs/plots/bit_silence_detection.png)

**name (Naman Gupta) - improved detector**

| Type | Start (s) | End (s) | Duration (s) |
|---|---|---|---|
| leading | 0.000 | 0.158 | 0.158 |
| ambiguous | 0.158 | 0.287 | 0.130 |
| speech | 0.287 | 1.347 | 1.060 |
| internal | 1.347 | 1.528 | 0.180 |
| ambiguous | 1.528 | 1.824 | 0.296 |

Baseline silence intervals for comparison: leading 0.000-0.343 s; internal 1.052-1.153 s; internal 1.333-1.692 s.

- Review flag: 2 ambiguous low-level region(s) kept: level is above the noise floor but never reaches the speech-onset threshold

![name silence detection](../outputs/plots/name_silence_detection.png)

### 5.3 Boundary review of the suspected regions

The zoomed figures show waveform, frame RMS with both thresholds, and spectrogram at the start and end of each recording. Cyan dashed lines are the detected active-region edges.

![bit boundary review](../outputs/plots/bit_boundary_review.png)

![name boundary review](../outputs/plots/name_boundary_review.png)

**Manual observations** (visual inspection of the boundary-review plots (waveform, frame RMS, spectrogram) by the analyst; no listening test, so these are observations, not ground truth):

| Recording | Region | Observation | Judgement |
|---|---|---|---|
| bit | 0.16-0.61 s (ambiguous) | Low-level (-35 to -22 dB) impulsive activity: isolated spikes mostly in the left channel and a vertical broadband line near 0.26 s in the spectrogram; no harmonic stripes. Voiced low-frequency energy only starts near 0.72 s. | Probably mouth/handling clicks or breath rather than speech, but not provable from the plots. Preserved (conservative). |
| bit | 0.61-0.72 s (confident speech by extension) | Frame RMS dips to the noise floor around 0.56-0.70 s and the vowel's harmonic band appears at about 0.72 s. This stretch is labelled speech only because it lies within the 150 ms onset extension. | Boundary is conservative (starts about 0.1 s before the visible onset). Acceptable. |
| bit | end of file, trailing silence from about 2.10 s | Vowel energy decays smoothly into a flat noise floor (about -42 dB) with no harmonic structure in the spectrogram. | Confident silence; trimmed with 100 ms padding. |
| name | 0.16-0.29 s (ambiguous) and 0.29-0.44 s (speech by extension) | Broadband low-level noise about 8-12 dB above the floor with no harmonic structure; clear voiced onset (harmonics, rising RMS) at about 0.44 s. | Likely background noise or breath; ambiguous, preserved. The earlier baseline had cut this to start at 0.34 s. |
| name | 1.35-1.53 s (internal silence) | After the vowel of the last syllable decays (about 1.3 s) the frame RMS sits at the noise floor for about 0.18 s. | Confident internal silence but shorter than 400 ms and next to an ambiguous region, so it is preserved, not shortened. |
| name | 1.53-1.82 s (ambiguous, kept to end of file) | Low-level activity (-35 to -25 dB), mostly energy below about 300 Hz, no word-like harmonic structure after the burst and vowel that end near 1.3 s. | Probably breath, handling or room noise, but unresolved. Preserved; it is the main remaining uncertainty. |

### 5.4 Sensitivity to parameters (real recordings)

Boundaries and reduction when the frame length and the low-threshold margin are varied (all else default):

| Recording | Frame (ms) | Low margin (dB) | First active (s) | Last active (s) | Speech / ambiguous regions | Removed (%) |
|---|---|---|---|---|---|---|
| bit | 20 | 6 | 0.135 | 2.105 | 1 / 1 | 12.2 |
| bit | 20 | 8 | 0.155 | 2.095 | 2 / 1 | 13.4 |
| bit | 20 | 10 | 0.165 | 2.095 | 2 / 1 | 13.8 |
| bit | 25 | 6 | 0.147 | 2.107 | 1 / 1 | 12.6 |
| bit | 25 | 8 | 0.158 | 2.098 | 2 / 1 | 13.4 |
| bit | 25 | 10 | 0.168 | 2.098 | 2 / 1 | 13.8 |
| bit | 40 | 6 | 0.145 | 2.115 | 1 / 1 | 12.2 |
| bit | 40 | 8 | 0.155 | 2.105 | 1 / 1 | 13.0 |
| bit | 40 | 10 | 0.165 | 2.095 | 1 / 1 | 13.8 |
| name | 20 | 6 | 0.155 | 1.824 | 1 / 2 | 3.0 |
| name | 20 | 8 | 0.155 | 1.824 | 1 / 2 | 3.0 |
| name | 20 | 10 | 0.165 | 1.824 | 1 / 2 | 3.6 |
| name | 25 | 6 | 0.158 | 1.824 | 1 / 2 | 3.2 |
| name | 25 | 8 | 0.158 | 1.824 | 1 / 2 | 3.2 |
| name | 25 | 10 | 0.158 | 1.824 | 1 / 2 | 3.2 |
| name | 40 | 6 | 0.145 | 1.824 | 1 / 2 | 2.5 |
| name | 40 | 8 | 0.155 | 1.824 | 1 / 2 | 3.0 |
| name | 40 | 10 | 0.155 | 1.824 | 1 / 2 | 3.0 |

## 6. Audio Preprocessing and Silence Removal

- Leading/trailing: cut to the first/last **active** region (speech or ambiguous) with 100 ms context, and a 5 ms fade at each cut that is not the original file edge. The padding exceeds the frame length plus the measured onset ramps, so soft onsets are kept.
- Internal: a gap is shortened (to 200 ms) only if it is longer than 400 ms **and** both neighbours are confident speech. Natural pauses, stop closures and gaps next to ambiguous regions are preserved.
- `trim_ambiguous` = False: ambiguous regions are not removed. The table also shows what removing them would do, for information only.
- Sampling frequency, channels and 16-bit PCM format are unchanged; no gain is applied.

| Recording | Original (s) | Processed (s) | Removed (s) | Removed (%) | Gaps shortened | Gaps preserved |
|---|---|---|---|---|---|---|
| bit | 2.472 | 2.1400 | 0.3320 | 13.4 | 0 | 2 |
| name | 1.824 | 1.7665 | 0.0575 | 3.2 | 0 | 2 |

**Baseline versus improved (measured on the real recordings)**

| Recording | Baseline processed (s) | Baseline removed (%) | Improved processed (s) | Improved removed (%) | Ambiguous kept (s) |
|---|---|---|---|---|---|
| bit | 2.1100 | 14.6 | 2.1400 | 13.4 | 0.450 |
| name | 1.5815 | 13.3 | 1.7665 | 3.2 | 0.426 |

A smaller reduction than the baseline is intentional where the baseline cut into low-level material that cannot be shown to be non-speech. The reduction is not a quality metric.

![bit original vs processed](../outputs/plots/bit_trimmed_waveform.png)

![name original vs processed](../outputs/plots/name_trimmed_waveform.png)

### 6.1 Processed-audio integrity (measured)

| Check | bit | name |
|---|---|---|
| reopened | pass | pass |
| nonempty | pass | pass |
| all_finite | pass | pass |
| not_all_zero | pass | pass |
| sampling_frequency_preserved | pass | pass |
| channels_preserved | pass | pass |
| num_samples_matches_report | pass | pass |
| matches_processed_array_within_1lsb | pass | pass |
| no_clipped_samples | pass | pass |
| peak_not_above_original | pass | pass |
| joins_without_discontinuity | pass | pass |
| original_file_unchanged | pass | pass |

Joins: bit: 0 edited join(s); name: 0 edited join(s). These checks establish signal integrity only. Intelligibility and audible artefacts were **not** assessed by listening.

## 7. Spectrogram Analysis

A spectrogram is the magnitude of the Short-Time Fourier Transform: the signal is cut into short overlapping windows (Hann, 1024 samples, hop 128 samples), each is Fourier-transformed and the power in dB is shown against time and frequency. Voiced sounds give harmonic stripes spaced at the pitch, vowel formants appear as broad bands, bursts and fricatives as broadband energy.

At fs = 48000 Hz the window is 21.33 ms long and the nominal bin spacing is 46.875 Hz (hop 2.67 ms). Harmonics of a ~100-200 Hz voice are therefore about 2-4 bins apart (resolved), while 21 ms is short enough to follow vowel-to-consonant changes. The comparison figures show 512 / 1024 / 2048-point windows: 512 blurs harmonics, 2048 smears timing; 1024 was kept as the compromise, judged on these recordings by inspection rather than by an objective criterion. The vertical axis stops at min(8000 Hz, Nyquist). **0 dB is the peak spectrogram power of the original recording**, and the processed spectrogram uses the same reference, so colours are comparable. Individual harmonics or formants of a 1-2 s utterance should not be over-interpreted: pitch varies within a window and there is no repetition to average over.

![bit spectrogram](../outputs/plots/bit_spectrogram.png)

Processed: ![bit processed spectrogram](../outputs/plots/bit_spectrogram_processed.png)

Resolution comparison: ![bit spectrogram resolution](../outputs/plots/bit_spectrogram_resolution.png)

![name spectrogram](../outputs/plots/name_spectrogram.png)

Processed: ![name processed spectrogram](../outputs/plots/name_spectrogram_processed.png)

Resolution comparison: ![name spectrogram resolution](../outputs/plots/name_spectrogram_resolution.png)

## 8. Controlled Evaluation on Synthetic Signals

Deterministic synthetic signals (harmonic 'speech-like' bursts plus filtered noise, fs = 16000 Hz) with known speech intervals. They test the algorithms only and are not substitutes for the recordings. Activity metrics use a 10 ms grid; for the improved detector confident speech and ambiguous regions both count as active (both are preserved). Boundary error = detected first start / last end minus truth (negative start error = starts early, i.e. conservative). Preserved = fraction of true speech samples that remain after trimming.

| Scenario | Baseline F1 | Improved F1 | Baseline preserved | Improved preserved | Baseline start/end err (ms) | Improved start/end err (ms) | Baseline dur. err (s) | Improved dur. err (s) |
|---|---|---|---|---|---|---|---|---|
| clean | 0.99 | 0.99 | 1.000 | 1.000 | 2 / 13 | -13 / 8 | +0.03 | +0.04 |
| snr_30dB | 0.99 | 0.99 | 1.000 | 1.000 | 2 / 13 | -3 / 8 | +0.02 | +0.02 |
| snr_20dB | 1.00 | 1.00 | 1.000 | 1.000 | 2 / 2 | -3 / -2 | +0.00 | +0.00 |
| snr_10dB | 0.00 | 0.64 | 1.000 | 1.000 | - | -500 / 600 | +1.20 | +1.20 |
| quiet_second_word_(-20dB) | 0.99 | 0.99 | 1.000 | 1.000 | -8 / 2 | -13 / -2 | +0.02 | +0.02 |
| pause_150ms | 0.99 | 0.99 | 1.000 | 1.000 | -8 / 13 | -13 / 7 | +0.02 | +0.02 |
| pause_600ms | 0.99 | 0.99 | 1.000 | 1.000 | 2 / 13 | -3 / 8 | +0.03 | +0.03 |
| little_silence_(speech_fills_file) | 0.48 | 1.00 | 0.981 | 1.000 | 152 / -67 | -3 / 8 | -0.22 | +0.00 |
| near_guard_(SNR_16dB) | 0.99 | 1.00 | 1.000 | 1.000 | 12 / -7 | -3 / -2 | -0.04 | +0.00 |
| abrupt_onsets_(no_ramp) | 0.99 | 0.99 | 1.000 | 1.000 | -8 / 13 | -13 / 8 | +0.04 | +0.04 |
| stereo_opposite_polarity | 0.00 | 0.99 | 1.000 | 1.000 | - | -13 / 8 | +1.20 | +0.04 |
| stereo_unequal_levels_(R=-20dB) | 0.99 | 0.99 | 1.000 | 1.000 | 2 / 13 | -13 / 8 | +0.03 | +0.04 |
| click_in_leading_silence | 0.98 | 0.97 | 1.000 | 1.000 | -308 / 13 | -312 / 8 | +0.33 | +0.34 |
| noise_burst_300ms_(-34dBFS)_in_silence | 0.86 | 0.86 | 1.000 | 1.000 | -408 / 13 | -412 / 8 | +0.43 | +0.43 |

Lowest speech-preserved fraction of the improved pipeline across scenarios: 1.000. Observations: the improved detector recovers the speech when the file contains little silence and when the channels are out of phase (both break the baseline), and it declines to trim when speech and noise are too close to separate (SNR 10 dB: nothing is trimmed, everything is ambiguous). A 15 ms click and a 300 ms noise burst inside silence are still classed as speech/active by both detectors - energy alone cannot tell loud noise from speech.

No precision/recall/F1 is reported for the real recordings because no independently labelled ground truth exists for them.

## 9. Results Summary and Discussion

- **bit (BIT)**: confident leading silence 0.158 s (includes leading digital zeros), trailing silence 0.374 s, 1 internal silent region(s), 1 ambiguous region(s) (0.16-0.61 s); 13.4 % of the duration removed.
- **name (Naman Gupta)**: confident leading silence 0.158 s (includes leading digital zeros), trailing silence 0.000 s, 1 internal silent region(s), 2 ambiguous region(s) (0.16-0.29 s, 1.53-1.82 s); 3.2 % of the duration removed.

Which of the ambiguous regions are breath, room noise or weak speech cannot be settled from amplitude and the spectrograms alone, so they are kept. Removing them (`trim_ambiguous=True`) would shorten the files further but risks clipping speech; that is a decision for listening review.

## 10. Limitations

- Silence detection is energy-based. Loud non-speech sounds (clicks, noise bursts) look like speech; very quiet speech near the noise floor may be labelled ambiguous or, in the worst case, silence.
- Thresholds assume the quietest frames are representative of the background noise; this fails if the recording has no noise-only frames. The minimum-separation guard only partly mitigates this.
- The utterance in each file is declared in `audio/manifest.json`; it was not verified by recognition or by listening.
- Manual boundary observations were made from plots by the author of the analysis, not by listening, and are not ground truth.
- Join/click checks measure sample discontinuities, not audibility. No intelligibility test was performed, and no recognition-accuracy claim is made.
- Synthetic results show behaviour on simplified signals (stationary filtered noise, harmonic bursts) and need not transfer to every real recording.
- Spectrogram appearance depends on window length, hop and dB range.

## 11. Conclusion

Both recordings were analysed with a hysteresis detector that is at least as good as the original on every synthetic scenario tested for speech preservation, handles two failure cases of the baseline, and flags uncertain low-level regions instead of deleting them. On the real recordings it removes only confidently identified silence; remaining uncertainty is stated above. All measured values are in `analysis_results.json`; the frozen pre-change results are in `baseline_results.json`.
