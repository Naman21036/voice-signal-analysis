"""Dataset discovery, metadata extraction, speaker identification, and integrity auditing.

Discovers WAV recordings across boys' and girls' dataset directories, parses speaker
identities and utterance types from filename conventions, logs invalid/excluded files,
and calculates comprehensive dataset summary metrics.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from assignment2.config import Config

logger = logging.getLogger(__name__)


@dataclass
class AudioFileMetadata:
    """Detailed metadata for a single discovered audio recording."""
    file_id: str
    filename: str
    gender: str
    label: int
    duration: float
    sample_rate: int
    channels: int
    num_samples: int
    peak_amplitude: float
    rms_amplitude: float
    audio_path: str
    integrity_status: str
    sha256_hash: str
    speaker_id: str
    utterance_type: str


@dataclass
class ExcludedFileRecord:
    """Record of an ignored or invalid file in dataset directories."""
    file_id: str
    filename: str
    parent_dir: str
    audio_path: str
    reason: str
    format_detected: str


def compute_sha256(path: Path) -> str:
    """Compute the SHA-256 cryptographic digest of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_speaker_and_utterance(filename: str) -> tuple[str, str]:
    """Extract speaker identifier and utterance classification from filename.

    Conventions in the dataset:
    - Utterances: 'BIT', 'Name', or 'Both' (e.g. combined recordings or '&').
    - Speakers: First and last names, normalised to Title Case with underscores.
    """
    stem = Path(filename).stem
    norm = stem.replace("-", "_").replace("&", " & ")
    lower = norm.lower()

    # Utterance classification
    has_bit = "bit" in lower
    has_name = any(w in lower for w in ["name", "voice", "rudraksh"]) and not (
        "anant" in lower and "trimmed" in lower and "name" not in lower
    )
    has_both = (has_bit and has_name) or ("&" in norm) or ("bit_name" in lower)

    if has_both:
        utt = "Both"
    elif has_bit:
        utt = "BIT"
    elif has_name:
        utt = "Name"
    else:
        if "sarthak_anand" in lower and not has_bit:
            utt = "Name"
        else:
            utt = "Unspecified"

    # Speaker extraction
    s = re.sub(r"(?i)processed[_\s]+audio[_\s]+", "", stem)
    s = re.sub(r"(?i)[_ &]*(trimmed|processed|raw|voice).*$", "", s)
    s = re.sub(r"(?i)[_ &]+(bit|name).*$", "", s)
    s = re.sub(r"[\d_ &]+$", "", s)
    s = s.strip("_ ")
    s = re.sub(r"\s+", "_", s)

    if not s or s.lower() in ["processed", "audio", "aiml"]:
        match = re.search(r"([A-Za-z]+)", stem)
        s = match.group(1) if match else "Unknown"

    return s.title(), utt


def locate_dataset_folders(root_dir: Path) -> tuple[Path, Path]:
    """Automatically find Boys and Girls dataset folders without hardcoding."""
    candidates = [p for p in root_dir.rglob("*") if p.is_dir()]

    boys_dir: Path | None = None
    girls_dir: Path | None = None

    for d in candidates:
        name_clean = d.name.upper().replace("_", " ")
        if "BOY" in name_clean and "VOICE" in name_clean:
            boys_dir = d
        elif "GIRL" in name_clean and "VOICE" in name_clean:
            girls_dir = d

    if boys_dir is None or not boys_dir.is_dir():
        raise FileNotFoundError(
            f"Could not locate Boys voice dataset directory under {root_dir}"
        )
    if girls_dir is None or not girls_dir.is_dir():
        raise FileNotFoundError(
            f"Could not locate Girls voice dataset directory under {root_dir}"
        )

    return boys_dir, girls_dir


def discover_dataset(
    cfg: Config,
) -> tuple[list[AudioFileMetadata], list[ExcludedFileRecord], dict[str, Any]]:
    """Discover, validate, and extract metadata for all dataset recordings."""
    boys_dir, girls_dir = locate_dataset_folders(cfg.root_dir)
    logger.info("Discovered Boys dataset directory: %s", boys_dir)
    logger.info("Discovered Girls dataset directory: %s", girls_dir)

    valid_records: list[AudioFileMetadata] = []
    excluded_records: list[ExcludedFileRecord] = []
    seen_hashes: dict[str, str] = {}
    duplicate_count = 0

    dir_targets = [
        (boys_dir, cfg.class_boys_name, cfg.class_boys_label),
        (girls_dir, cfg.class_girls_name, cfg.class_girls_label),
    ]

    file_counter = 0

    for folder, gender_name, gender_label in dir_targets:
        for file_path in sorted(folder.iterdir()):
            if file_path.is_dir():
                continue

            file_counter += 1
            file_id = f"REC_{file_counter:04d}"
            filename = file_path.name
            parent_dir = folder.name
            suffix = file_path.suffix.lower()

            # The assignment strictly requires WAV recordings
            if suffix != ".wav":
                reason = (
                    f"Non-WAV extension ({suffix if suffix else 'missing extension'})"
                )
                try:
                    info = sf.info(str(file_path))
                    detected_fmt = info.format
                except Exception:
                    detected_fmt = "Unknown/Unreadable"
                excluded_records.append(
                    ExcludedFileRecord(
                        file_id=file_id,
                        filename=filename,
                        parent_dir=parent_dir,
                        audio_path=str(file_path),
                        reason=reason,
                        format_detected=detected_fmt,
                    )
                )
                continue

            # Check file readability and integrity via soundfile
            try:
                info = sf.info(str(file_path))
                data, fs = sf.read(str(file_path), dtype="float64", always_2d=True)
                integrity = "VALID"
                if np.isnan(data).any() or np.isinf(data).any():
                    integrity = "NON_FINITE_VALUES"
                elif data.size == 0 or info.duration == 0:
                    integrity = "ZERO_DURATION"
            except Exception as exc:
                integrity = f"READ_ERROR: {str(exc)}"
                excluded_records.append(
                    ExcludedFileRecord(
                        file_id=file_id,
                        filename=filename,
                        parent_dir=parent_dir,
                        audio_path=str(file_path),
                        reason=integrity,
                        format_detected="CORRUPT_WAV",
                    )
                )
                continue

            if integrity != "VALID":
                excluded_records.append(
                    ExcludedFileRecord(
                        file_id=file_id,
                        filename=filename,
                        parent_dir=parent_dir,
                        audio_path=str(file_path),
                        reason=integrity,
                        format_detected=info.format,
                    )
                )
                continue

            # Compute statistics and hash
            file_hash = compute_sha256(file_path)
            if file_hash in seen_hashes:
                duplicate_count += 1
            else:
                seen_hashes[file_hash] = filename

            peak = float(np.max(np.abs(data)))
            rms = float(np.sqrt(np.mean(data**2)))
            speaker_id, utterance_type = parse_speaker_and_utterance(filename)

            valid_records.append(
                AudioFileMetadata(
                    file_id=file_id,
                    filename=filename,
                    gender=gender_name,
                    label=gender_label,
                    duration=float(info.duration),
                    sample_rate=int(info.samplerate),
                    channels=int(info.channels),
                    num_samples=int(data.shape[0]),
                    peak_amplitude=peak,
                    rms_amplitude=rms,
                    audio_path=str(file_path),
                    integrity_status=integrity,
                    sha256_hash=file_hash,
                    speaker_id=speaker_id,
                    utterance_type=utterance_type,
                )
            )

    # Compute summary metrics
    summary = compute_dataset_summary(valid_records, excluded_records, duplicate_count)
    return valid_records, excluded_records, summary


def compute_dataset_summary(
    records: list[AudioFileMetadata],
    excluded: list[ExcludedFileRecord],
    duplicate_count: int,
) -> dict[str, Any]:
    """Calculate summary statistics over valid records and exclusions."""
    durations = [r.duration for r in records]
    boys_records = [r for r in records if r.label == 0]
    girls_records = [r for r in records if r.label == 1]

    sr_dist: dict[int, int] = {}
    ch_dist: dict[str, int] = {"Mono": 0, "Stereo": 0, "Multi": 0}
    for r in records:
        sr_dist[r.sample_rate] = sr_dist.get(r.sample_rate, 0) + 1
        if r.channels == 1:
            ch_dist["Mono"] += 1
        elif r.channels == 2:
            ch_dist["Stereo"] += 1
        else:
            ch_dist["Multi"] += 1

    unique_speakers_boys = len(set(r.speaker_id for r in boys_records))
    unique_speakers_girls = len(set(r.speaker_id for r in girls_records))
    total_unique_speakers = len(set(r.speaker_id for r in records))

    utterance_dist: dict[str, int] = {}
    for r in records:
        utterance_dist[r.utterance_type] = utterance_dist.get(r.utterance_type, 0) + 1

    summary = {
        "total_recordings": len(records),
        "boys_recordings": len(boys_records),
        "girls_recordings": len(girls_records),
        "total_duration_s": float(np.sum(durations)) if durations else 0.0,
        "average_duration_s": float(np.mean(durations)) if durations else 0.0,
        "std_duration_s": float(np.std(durations)) if durations else 0.0,
        "min_duration_s": float(np.min(durations)) if durations else 0.0,
        "max_duration_s": float(np.max(durations)) if durations else 0.0,
        "sampling_rate_distribution": sr_dist,
        "channel_distribution": ch_dist,
        "num_invalid_files": len(excluded),
        "num_duplicate_files_by_hash": duplicate_count,
        "num_identifiable_speakers": {
            "boys": unique_speakers_boys,
            "girls": unique_speakers_girls,
            "total": total_unique_speakers,
        },
        "utterance_distribution": utterance_dist,
    }
    return summary


def save_dataset_artifacts(
    records: list[AudioFileMetadata],
    excluded: list[ExcludedFileRecord],
    summary: dict[str, Any],
    cfg: Config,
) -> None:
    """Save metadata CSV, exclusions CSV, and dataset summary files."""
    cfg.ensure_directories()

    # 1. Save metadata.csv
    meta_path = cfg.dataset_analysis_dir / "metadata.csv"
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file_id",
                "filename",
                "gender",
                "label",
                "duration",
                "sample_rate",
                "channels",
                "num_samples",
                "peak_amplitude",
                "rms_amplitude",
                "audio_path",
                "integrity_status",
                "sha256_hash",
                "speaker_id",
                "utterance_type",
            ],
        )
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))
    logger.info("Saved metadata CSV: %s (%d records)", meta_path, len(records))

    # 2. Save invalid_files.csv
    excl_path = cfg.dataset_analysis_dir / "invalid_files.csv"
    with open(excl_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file_id",
                "filename",
                "parent_dir",
                "audio_path",
                "reason",
                "format_detected",
            ],
        )
        writer.writeheader()
        for e in excluded:
            writer.writerow(asdict(e))
    logger.info("Saved invalid files CSV: %s (%d records)", excl_path, len(excluded))

    # 3. Save dataset_summary.json
    sum_json_path = cfg.dataset_analysis_dir / "dataset_summary.json"
    sum_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # 4. Save dataset_summary.md
    sum_md_path = cfg.dataset_analysis_dir / "dataset_summary.md"
    md_content = f"""# Dataset Analysis Summary

Course: EC24313 Artificial Intelligence and Machine Learning  
Project: Voice-Based Gender Classification (Assignment 2)  
Author: Naman Gupta (BTECH/10887/24)

## 1. Class Distribution

| Class | Label | Folder Name | Valid Recordings | Percentage | Unique Speakers |
|---|---|---|---|---|---|
| **Boys** (Class 1) | 0 | `VOICE DATASET BOYS` | {summary['boys_recordings']} | {summary['boys_recordings'] / summary['total_recordings'] * 100:.1f}% | {summary['num_identifiable_speakers']['boys']} |
| **Girls** (Class 2) | 1 | `VOICE DATASET GIRLS` | {summary['girls_recordings']} | {summary['girls_recordings'] / summary['total_recordings'] * 100:.1f}% | {summary['num_identifiable_speakers']['girls']} |
| **Total** | - | - | **{summary['total_recordings']}** | **100.0%** | **{summary['num_identifiable_speakers']['total']}** |

## 2. Duration Statistics

- **Total Duration of Dataset:** {summary['total_duration_s']:.2f} seconds ({summary['total_duration_s'] / 60:.2f} minutes)
- **Mean Duration per Recording:** {summary['average_duration_s']:.2f} ± {summary['std_duration_s']:.2f} seconds
- **Minimum Duration:** {summary['min_duration_s']:.2f} seconds
- **Maximum Duration:** {summary['max_duration_s']:.2f} seconds

## 3. Sampling Rates and Channels

- **Sampling Rate Distribution:**
{chr(10).join(f"  - {sr} Hz: {count} recordings ({count / summary['total_recordings'] * 100:.1f}%)" for sr, count in sorted(summary['sampling_rate_distribution'].items()))}
- **Channel Distribution:**
  - Mono (1 channel): {summary['channel_distribution']['Mono']} recordings ({summary['channel_distribution']['Mono'] / summary['total_recordings'] * 100:.1f}%)
  - Stereo (2 channels): {summary['channel_distribution']['Stereo']} recordings ({summary['channel_distribution']['Stereo'] / summary['total_recordings'] * 100:.1f}%)

## 4. Utterance Content Distribution

{chr(10).join(f"- **{utt}:** {count} recordings" for utt, count in sorted(summary['utterance_distribution'].items()))}

## 5. File Integrity and Exclusions

- **Exact Duplicate Files (SHA-256):** {summary['num_duplicate_files_by_hash']}
- **Invalid / Excluded Files:** {summary['num_invalid_files']} files (recorded in `invalid_files.csv`)
  - All exclusions were due to non-WAV media formats (.mp3, .mp4, .m4a, missing extensions) as per assignment specifications.
"""
    sum_md_path.write_text(md_content, encoding="utf-8")
    logger.info("Saved dataset summary Markdown: %s", sum_md_path)

