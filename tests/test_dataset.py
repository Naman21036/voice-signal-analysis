"""Automated tests for Assignment 2 dataset discovery and metadata extraction."""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from assignment2.config import Config
from assignment2.dataset import (
    compute_dataset_summary,
    compute_sha256,
    discover_dataset,
    locate_dataset_folders,
    parse_speaker_and_utterance,
)


def test_locate_dataset_folders():
    cfg = Config()
    boys_dir, girls_dir = locate_dataset_folders(cfg.root_dir)
    assert boys_dir.is_dir()
    assert girls_dir.is_dir()
    assert "BOY" in boys_dir.name.upper()
    assert "GIRL" in girls_dir.name.upper()


def test_parse_speaker_and_utterance():
    # Test BIT utterance
    spk1, utt1 = parse_speaker_and_utterance("ABHAY_PATHAK_BIT.wav")
    assert spk1 == "Abhay_Pathak"
    assert utt1 == "BIT"

    # Test Name utterance
    spk2, utt2 = parse_speaker_and_utterance("Avantika_NAME.wav")
    assert spk2 == "Avantika"
    assert utt2 == "Name"

    # Test Combined utterance
    spk3, utt3 = parse_speaker_and_utterance("Jalaj_BIT & Jalaj_Name.wav")
    assert spk3 == "Jalaj"
    assert utt3 == "Both"

    # Test trimmed variant
    spk4, utt4 = parse_speaker_and_utterance("Mehul_BIT_trimmed.wav")
    assert spk4 == "Mehul"
    assert utt4 == "BIT"


def test_discover_dataset_integrity():
    cfg = Config()
    records, excluded, summary = discover_dataset(cfg)

    # 1. Verify file counts
    assert len(records) == 115
    assert summary["boys_recordings"] == 103
    assert summary["girls_recordings"] == 12

    # 2. Verify numeric labels
    for r in records:
        if r.gender == "Boys":
            assert r.label == 0
        elif r.gender == "Girls":
            assert r.label == 1
        assert r.duration > 0.0
        assert r.sample_rate in [16000, 22050, 44100, 48000]
        assert r.channels in [1, 2]
        assert r.integrity_status == "VALID"

    # 3. Verify exclusions
    assert len(excluded) == 33
    for e in excluded:
        assert "Non-WAV" in e.reason or "READ_ERROR" in e.reason


def test_compute_dataset_summary():
    cfg = Config()
    records, excluded, summary = discover_dataset(cfg)
    assert summary["total_recordings"] == 115
    assert summary["total_duration_s"] > 200.0
    assert summary["num_identifiable_speakers"]["total"] == 63
    assert summary["num_identifiable_speakers"]["boys"] == 56
    assert summary["num_identifiable_speakers"]["girls"] == 7

