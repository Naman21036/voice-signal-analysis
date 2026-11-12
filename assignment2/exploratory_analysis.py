"""Exploratory Data Analysis (EDA) and visualization execution for Assignment 2.

Computes acoustic and demographic distributions, evaluates feature discriminative power,
and generates all required figures:
- Raw vs 16 kHz preprocessed waveform comparisons
- MFCC heatmaps (Boys vs Girls)
- Mandatory 2D PCA scatter plots (by class and by speaker)
- Pairwise MFCC correlation matrix
- MFCC coefficient boxplot distributions
- Class balance, duration, and sampling frequency distributions
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy import stats

from assignment2.config import Config
from assignment2.dataset import AudioFileMetadata
from assignment2.feature_extraction import extract_mfcc_frames
from assignment2.visualization import (
    plot_dataset_exploratory_charts,
    plot_mfcc_correlation_matrix,
    plot_mfcc_distributions,
    plot_mfcc_heatmaps,
    plot_pca_scatter,
    plot_pca_speakers_scatter,
    plot_waveform_comparison,
)

logger = logging.getLogger(__name__)


def compute_feature_statistics(
    X: np.ndarray, y: np.ndarray, feature_names: list[str]
) -> list[dict[str, Any]]:
    """Compute mean, std, and two-sample Welch t-test for each MFCC between Boys and Girls."""
    stats_list: list[dict[str, Any]] = []
    idx_boys = np.where(y == 0)[0]
    idx_girls = np.where(y == 1)[0]

    for i, name in enumerate(feature_names):
        boys_vals = X[idx_boys, i]
        girls_vals = X[idx_girls, i]

        t_stat, p_val = stats.ttest_ind(boys_vals, girls_vals, equal_var=False)

        stats_list.append({
            "feature": name,
            "boys_mean": float(np.mean(boys_vals)),
            "boys_std": float(np.std(boys_vals)),
            "girls_mean": float(np.mean(girls_vals)),
            "girls_std": float(np.std(girls_vals)),
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "statistically_significant": bool(p_val < 0.05),
        })
    return stats_list


def run_exploratory_analysis(
    records: list[AudioFileMetadata],
    preprocessed_signals: dict[str, np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    summary: dict[str, Any],
    cfg: Config,
) -> dict[str, Any]:
    """Execute complete EDA and output all visualization artifacts."""
    cfg.ensure_directories()
    plots_dir = cfg.plots_dir

    # 1. Dataset demographic and audio distribution charts
    plot_dataset_exploratory_charts(records, summary, plots_dir, cfg, dpi=cfg.dpi)

    # 2. Select representative recordings for waveform comparisons and MFCC heatmaps
    rep_boy_rec = next(r for r in records if r.label == 0 and r.utterance_type == "BIT")
    rep_girl_rec = next(r for r in records if r.label == 1 and r.utterance_type == "BIT")

    raw_boy, fs_boy = sf.read(rep_boy_rec.audio_path, dtype="float64", always_2d=True)
    raw_girl, fs_girl = sf.read(rep_girl_rec.audio_path, dtype="float64", always_2d=True)

    proc_boy = preprocessed_signals[rep_boy_rec.file_id]
    proc_girl = preprocessed_signals[rep_girl_rec.file_id]

    plot_waveform_comparison(
        raw_boy,
        fs_boy,
        proc_boy,
        cfg.target_sample_rate,
        f"{rep_boy_rec.speaker_id} · Boys (BIT)",
        plots_dir / "waveform_comparison_boy.png",
        dpi=cfg.dpi,
    )

    plot_waveform_comparison(
        raw_girl,
        fs_girl,
        proc_girl,
        cfg.target_sample_rate,
        f"{rep_girl_rec.speaker_id} · Girls (BIT)",
        plots_dir / "waveform_comparison_girl.png",
        dpi=cfg.dpi,
    )

    # 3. MFCC Heatmaps
    boy_frames = extract_mfcc_frames(proc_boy, cfg)
    girl_frames = extract_mfcc_frames(proc_girl, cfg)

    plot_mfcc_heatmaps(
        boy_frames,
        girl_frames,
        rep_boy_rec.speaker_id,
        rep_girl_rec.speaker_id,
        plots_dir / "mfcc_heatmaps_comparison.png",
        hop_ms=cfg.hop_ms,
        dpi=cfg.dpi,
    )

    # 4. Mandatory PCA Scatter Plot (by Gender)
    pca_model, X_pca = plot_pca_scatter(
        X, y, records, plots_dir / "pca_scatter_plot.png", cfg, dpi=cfg.dpi
    )

    # 5. PCA Scatter Plot (by Speaker Identity)
    plot_pca_speakers_scatter(
        X_pca, y, records, plots_dir / "pca_speakers_scatter_plot.png", dpi=cfg.dpi
    )

    # 6. MFCC Correlation Matrix
    plot_mfcc_correlation_matrix(
        X, feature_names, plots_dir / "mfcc_correlation_heatmap.png", dpi=cfg.dpi
    )

    # 7. MFCC Distributions Boxplot
    plot_mfcc_distributions(
        X, y, feature_names, plots_dir / "mfcc_distributions_boxplot.png", cfg, dpi=cfg.dpi
    )

    # 8. Compute statistical feature ranking
    feature_stats = compute_feature_statistics(X, y, feature_names)

    logger.info("Exploratory analysis complete. All figures generated in %s", plots_dir)
    return {
        "pca_explained_variance_ratio": pca_model.explained_variance_ratio_.tolist(),
        "feature_statistics": feature_stats,
    }

