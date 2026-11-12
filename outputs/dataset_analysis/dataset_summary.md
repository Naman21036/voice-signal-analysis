# Dataset Analysis Summary

Course: EC24313 Artificial Intelligence and Machine Learning  
Project: Voice-Based Gender Classification (Assignment 2)  
Author: Naman Gupta (BTECH/10887/24)

## 1. Class Distribution

| Class | Label | Folder Name | Valid Recordings | Percentage | Unique Speakers |
|---|---|---|---|---|---|
| **Boys** (Class 1) | 0 | `VOICE DATASET BOYS` | 103 | 89.6% | 56 |
| **Girls** (Class 2) | 1 | `VOICE DATASET GIRLS` | 12 | 10.4% | 7 |
| **Total** | - | - | **115** | **100.0%** | **63** |

## 2. Duration Statistics

- **Total Duration of Dataset:** 286.19 seconds (4.77 minutes)
- **Mean Duration per Recording:** 2.49 ± 1.73 seconds
- **Minimum Duration:** 0.46 seconds
- **Maximum Duration:** 10.20 seconds

## 3. Sampling Rates and Channels

- **Sampling Rate Distribution:**
  - 16000 Hz: 10 recordings (8.7%)
  - 22050 Hz: 2 recordings (1.7%)
  - 44100 Hz: 31 recordings (27.0%)
  - 48000 Hz: 72 recordings (62.6%)
- **Channel Distribution:**
  - Mono (1 channel): 86 recordings (74.8%)
  - Stereo (2 channels): 29 recordings (25.2%)

## 4. Utterance Content Distribution

- **BIT:** 47 recordings
- **Both:** 19 recordings
- **Name:** 46 recordings
- **Unspecified:** 3 recordings

## 5. File Integrity and Exclusions

- **Exact Duplicate Files (SHA-256):** 0
- **Invalid / Excluded Files:** 33 files (recorded in `invalid_files.csv`)
  - All exclusions were due to non-WAV media formats (.mp3, .mp4, .m4a, missing extensions) as per assignment specifications.
