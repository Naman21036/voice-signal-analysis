# EC24313 Artificial Intelligence and Machine Learning - Assignment 1

**Course:** EC24313 - Artificial Intelligence and Machine Learning  
**Student:** Naman Gupta  
**Topic:** Voice recording, time-domain analysis, silence detection/removal and spectrogram

## 1. Objective

To record two short utterances ("BIT" and "Naman Gupta"), study them in the time domain, locate silent regions, produce silence-minimised versions and examine their time-frequency content with a spectrogram.

## 2. Tools and Libraries

Python with NumPy (numerics), SoundFile (WAV I/O and header metadata), SciPy (STFT/spectrogram) and Matplotlib (plots). Everything runs on a CPU; run `python assignment1.py`.

## 3. Methodology

1. Task 1: the original recordings are in `audio/` and are never modified (read-only access). Spoken content is inferred from the file names only and was not verified automatically.
2. Task 2: each file is read, sampling frequency/samples/channels taken from the header, then amplitude vs. time is plotted.
3. Task 3: frame-wise RMS energy (dB) with an adaptive threshold (below).
4. Task 4: leading/trailing silence trimmed with padding; over-long internal gaps shortened.
5. Task 5: Hann-window STFT power spectrogram.

## 4. Sampling Frequency and Time-Domain Analysis

The sampling frequency is the number of amplitude samples stored per second. It is read directly from the WAV header (`soundfile.info`), not estimated from the signal. It is unrelated to the fundamental frequency (pitch) of the voice, which is typically ~85-255 Hz for adult speech; the Nyquist limit fs/2 only bounds the highest representable frequency. Duration = number of samples / fs. For stereo files, the plots show both channels and analysis uses their average.

| Recording | File | fs (Hz) | Channels | Samples | Duration (s) | Peak | RMS |
|---|---|---|---|---|---|---|---|
| BIT | Naman_BIT.wav | 48000 | 2 | 118656 | 2.472 | 0.9523 | 0.1615 |
| Naman Gupta | Naman_Name.wav | 48000 | 2 | 87552 | 1.824 | 0.9477 | 0.1666 |

![bit waveform](../outputs/plots/bit_waveform.png)

![name waveform](../outputs/plots/name_waveform.png)

## 5. Silent Region Identification

Method: short-time RMS energy with frame = 25.0 ms and hop = 10.0 ms. Threshold (dB) = max(noise floor + 12.0 dB, speech level - 35.0 dB), where the noise floor is the 10th and the speech level the 95th percentile of the frame RMS values of that recording, so the threshold adapts to each recording's noise floor. Low-energy runs shorter than 100.0 ms are treated as pauses within speech, and active runs shorter than 30.0 ms are discarded as noise bursts. This is an amplitude-based decision only: it does not identify words or phonemes, and weak sounds (e.g. unvoiced fricatives, final consonants) may be classed as silence.

**BIT** - noise floor -44.6 dB, threshold -32.6 dB.

| Type | Start (s) | End (s) | Duration (s) |
|---|---|---|---|
| leading | 0.000 | 0.182 | 0.182 |
| internal | 0.472 | 0.723 | 0.250 |
| internal | 1.712 | 1.823 | 0.110 |
| trailing | 2.092 | 2.472 | 0.380 |

![bit silence detection](../outputs/plots/bit_silence_detection.png)

**Naman Gupta** - noise floor -44.4 dB, threshold -32.4 dB.

| Type | Start (s) | End (s) | Duration (s) |
|---|---|---|---|
| leading | 0.000 | 0.343 | 0.343 |
| internal | 1.052 | 1.153 | 0.100 |
| internal | 1.333 | 1.692 | 0.360 |

![name silence detection](../outputs/plots/name_silence_detection.png)

## 6. Audio Preprocessing and Silence Removal

Leading and trailing silence is removed while keeping 100.0 ms padding around the first/last detected speech so that soft onsets and decays are not cut. Internal gaps are only shortened if longer than 400.0 ms (then reduced to 200.0 ms with 5.0 ms fades at the joins); shorter gaps are preserved because they are natural pauses/stop closures. The sampling frequency, channel count and bit depth are unchanged, and no gain is applied.

| Recording | Original (s) | Processed (s) | Removed (s) | Removed (%) | Internal gaps shortened | Internal gaps kept |
|---|---|---|---|---|---|---|
| BIT | 2.472 | 2.110 | 0.362 | 14.6 | 0 | 2 |
| Naman Gupta | 1.824 | 1.581 | 0.243 | 13.3 | 0 | 2 |

- BIT: processed file `outputs/processed_audio/Naman_BIT_processed.wav` validation passed (nonempty=ok, all_finite=ok, not_all_zero=ok, sampling_frequency_preserved=ok, channels_preserved=ok, num_samples_matches_report=ok, no_clipping=ok).
- Naman Gupta: processed file `outputs/processed_audio/Naman_Name_processed.wav` validation passed (nonempty=ok, all_finite=ok, not_all_zero=ok, sampling_frequency_preserved=ok, channels_preserved=ok, num_samples_matches_report=ok, no_clipping=ok).

![bit original vs processed](../outputs/plots/bit_trimmed_waveform.png)

![name original vs processed](../outputs/plots/name_trimmed_waveform.png)

## 7. Spectrogram Analysis

A spectrogram is the magnitude of the Short-Time Fourier Transform: the signal is cut into short overlapping windows (Hann window, 1024 samples, hop 128 samples), each is Fourier-transformed, and the power in dB is drawn with time on the x-axis, frequency on the y-axis and colour for energy. Speech is non-stationary, so this shows when energy appears and at which frequencies: voiced sounds give horizontal harmonic stripes spaced at the pitch, vowel formants appear as broad energy bands, and fricatives/bursts appear as broadband energy at high frequency. Frequency axis is limited to 8000 Hz (or fs/2 if lower). The window length trades time resolution against frequency resolution. Colour is dB relative to the peak power of the plotted file.

![bit spectrogram](../outputs/plots/bit_spectrogram.png)

Processed counterpart: ![bit processed spectrogram](../outputs/plots/bit_spectrogram_processed.png)

![name spectrogram](../outputs/plots/name_spectrogram.png)

Processed counterpart: ![name processed spectrogram](../outputs/plots/name_spectrogram_processed.png)

## 8. Results Summary and Discussion

- **BIT**: leading silence 0.182 s, trailing silence 0.380 s, 2 internal silent region(s) of at least 100 ms; 14.6% of the duration removed.
- **Naman Gupta**: leading silence 0.343 s, trailing silence 0.000 s, 2 internal silent region(s) of at least 100 ms; 13.3% of the duration removed.

The percentages are results of the chosen thresholds and padding; a different threshold would change them. Detected regions should be checked against the annotated plots, which is how they were assessed.

## 9. Limitations

- Silence is decided from short-time energy only; quiet speech or background noise near the threshold can be misclassified.
- Percentile-based thresholds assume each file contains both background and speech frames.
- Which phrase each file contains is taken from its filename and was not verified by listening or recognition.
- Low-level background noise (e.g. breath or room sound) near the threshold can be flagged as active, so some non-speech segments (notably in the noisy tails) may remain in the processed audio.
- No perceptual/listening test of intelligibility was carried out in this automated run.
- Spectrogram appearance depends on window length and dB range.

## 10. Conclusion

Both recordings were loaded, analysed, trimmed and visualised with a single reproducible script. Measured values for sampling frequency, duration, silence intervals and reductions are given in the tables above and in `analysis_results.json`.
