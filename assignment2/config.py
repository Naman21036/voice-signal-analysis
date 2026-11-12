"""Centralized configuration for Assignment 2 Gender Classification Pipeline.

Defines all signal processing parameters, directory paths, feature settings,
model hyperparameters, and random seeds for reproducibility.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Master configuration dataclass for Assignment 2."""

    # Project directories
    root_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])

    # Audio search patterns / directories
    dataset_boys_dir_name: str = "VOICE DATASET BOYS"
    dataset_girls_dir_name: str = "VOICE DATASET GIRLS"

    # Preprocessing parameters
    target_sample_rate: int = 16000     # 16 kHz standard
    pre_emphasis: float = 0.97          # high-frequency boost coefficient
    min_speech_duration_s: float = 0.2  # minimum length of speech to consider valid

    # MFCC extraction parameters (standard academic speech configuration)
    n_mfcc: int = 13                    # exactly 13 coefficients per frame
    n_mels: int = 26                    # 26 Mel filterbank channels
    n_fft: int = 512                    # 512-point FFT (32 ms window @ 16 kHz)
    win_length: int = 400               # 400 samples (25 ms analysis window @ 16 kHz)
    hop_length: int = 160               # 160 samples (10 ms frame shift @ 16 kHz)
    window: str = "hann"                # Hann window function
    f_min: float = 0.0                  # minimum frequency for Mel bank (Hz)
    f_max: float = 8000.0               # Nyquist frequency @ 16 kHz

    # Silence trimming parameters (reused from Assignment 1)
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    edge_pad_ms: float = 100.0          # context padding before/after speech

    # Dataset & Labeling
    class_boys_name: str = "Boys"
    class_girls_name: str = "Girls"
    class_boys_label: int = 0           # Primary positive class: Class 1 (Boys) -> label 0
    class_girls_label: int = 1          # Negative class: Class 2 (Girls) -> label 1

    # Reproducibility
    random_seed: int = 42

    # Model splitting and cross-validation
    test_size: float = 0.20
    n_splits_cv: int = 4                # K-fold splits respecting group boundaries

    # Plotting configuration
    dpi: int = 300
    color_boys: str = "#2563EB"         # Blue
    color_girls: str = "#E11D48"        # Rose / Crimson

    # Output paths
    @property
    def outputs_dir(self) -> Path:
        return self.root_dir / "outputs"

    @property
    def dataset_analysis_dir(self) -> Path:
        return self.outputs_dir / "dataset_analysis"

    @property
    def processed_audio_dir(self) -> Path:
        return self.outputs_dir / "processed_audio" / "assignment2"

    @property
    def features_dir(self) -> Path:
        return self.outputs_dir / "features"

    @property
    def plots_dir(self) -> Path:
        return self.outputs_dir / "plots" / "assignment2"

    @property
    def models_dir(self) -> Path:
        return self.outputs_dir / "models"

    @property
    def metrics_dir(self) -> Path:
        return self.outputs_dir / "metrics"

    @property
    def reports_dir(self) -> Path:
        return self.outputs_dir / "reports"

    def ensure_directories(self) -> None:
        """Create all required output subdirectories."""
        for d in [
            self.dataset_analysis_dir,
            self.processed_audio_dir,
            self.features_dir,
            self.plots_dir,
            self.models_dir,
            self.metrics_dir,
            self.reports_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

