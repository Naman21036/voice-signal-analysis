"""Publication-quality academic visualization for Assignment 2.

Generates high-resolution figures (300 DPI) adhering to academic styling:
1. Waveform comparison (raw vs preprocessed mono @ 16 kHz).
2. MFCC heatmaps for representative Boys and Girls recordings.
3. Mandatory 2D PCA scatter plot with explained variance percentages.
4. PCA scatter plot coloured by speaker identity.
5. MFCC coefficient correlation heatmap.
6. MFCC coefficient distributions comparison (Boys vs Girls).
7. Dataset exploratory charts (class balance, duration distribution).
8. Confusion matrix heatmaps (Class 1: Boys, Class 2: Girls).
9. Receiver Operating Characteristic (ROC) curves with AUC.
10. Multi-model performance comparison chart.
11. Neural Network training and validation loss convergence curves.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from assignment2.config import Config
from assignment2.dataset import AudioFileMetadata

logger = logging.getLogger(__name__)

# Consistent styling configuration
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "STIXGeneral", "Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.grid": True,
    "grid.color": "#E2E8F0",
    "grid.linewidth": 0.6,
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#0F172A",
    "text.color": "#0F172A",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def plot_waveform_comparison(
    raw_audio: np.ndarray,
    orig_fs: int,
    proc_audio: np.ndarray,
    target_fs: int,
    recording_title: str,
    out_path: Path,
    dpi: int = 300,
) -> None:
    """Plot time-domain comparison between raw audio and 16 kHz preprocessed audio."""
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 3.8), sharex=False, constrained_layout=True)

    t_raw = np.arange(raw_audio.shape[0]) / orig_fs
    if raw_audio.ndim == 2 and raw_audio.shape[1] > 1:
        axes[0].plot(t_raw, raw_audio[:, 0], color="#64748B", lw=0.6, alpha=0.7, label="Left Channel")
        axes[0].plot(t_raw, raw_audio[:, 1], color="#94A3B8", lw=0.6, alpha=0.7, label="Right Channel")
        axes[0].legend(loc="upper right", fontsize=7.5, framealpha=0.8)
    else:
        axes[0].plot(t_raw, raw_audio if raw_audio.ndim == 1 else raw_audio[:, 0], color="#64748B", lw=0.6)

    axes[0].set_title(f"Original Waveform ({recording_title} · {orig_fs} Hz)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylim(-1.05, 1.05)

    t_proc = np.arange(proc_audio.shape[0]) / target_fs
    axes[1].plot(t_proc, proc_audio, color="#2563EB", lw=0.7)
    axes[1].set_title(f"Preprocessed Mono Waveform (Resampled to {target_fs} Hz · Trimmed & Normalized)")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylim(-1.05, 1.05)

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    logger.info("Saved waveform comparison plot: %s", out_path)


def plot_mfcc_heatmaps(
    boys_frames: np.ndarray,
    girls_frames: np.ndarray,
    boys_name: str,
    girls_name: str,
    out_path: Path,
    hop_ms: float = 10.0,
    dpi: int = 300,
) -> None:
    """Plot 13-coefficient MFCC heatmaps for representative Boys and Girls recordings."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.6), sharey=True, constrained_layout=True)

    # frames shape: (n_frames, 13) -> transpose to (13, n_frames)
    t1 = np.arange(boys_frames.shape[0]) * (hop_ms / 1000.0)
    t2 = np.arange(girls_frames.shape[0]) * (hop_ms / 1000.0)

    vmin = min(float(np.min(boys_frames)), float(np.min(girls_frames)))
    vmax = max(float(np.max(boys_frames)), float(np.max(girls_frames)))

    im1 = ax1.imshow(
        boys_frames.T,
        origin="lower",
        aspect="auto",
        extent=[0, t1[-1] if len(t1) > 0 else 1.0, 1, 13],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    ax1.set_title(f"Class 1: Boys ({boys_name})", fontweight="bold", color="#2563EB")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("MFCC Index (1–13)")
    ax1.set_yticks(range(1, 14))

    im2 = ax2.imshow(
        girls_frames.T,
        origin="lower",
        aspect="auto",
        extent=[0, t2[-1] if len(t2) > 0 else 1.0, 1, 13],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    ax2.set_title(f"Class 2: Girls ({girls_name})", fontweight="bold", color="#E11D48")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_yticks(range(1, 14))

    cbar = fig.colorbar(im2, ax=[ax1, ax2], orientation="vertical", shrink=0.85, pad=0.03)
    cbar.set_label("Cepstral Coefficient Magnitude", fontsize=8.5)

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    logger.info("Saved MFCC heatmaps: %s", out_path)


def plot_pca_scatter(
    X: np.ndarray,
    y: np.ndarray,
    records: list[AudioFileMetadata],
    out_path: Path,
    cfg: Config,
    dpi: int = 300,
) -> tuple[PCA, np.ndarray]:
    """Generate mandatory 2D PCA scatter plot of 13-dimensional MFCC vectors.

    Standardizes features, projects to first 2 principal components,
    and plots samples coloured by gender with explained variance percentages.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=cfg.random_seed)
    X_pca = pca.fit_transform(X_scaled)
    evr = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(6.8, 5.0), constrained_layout=True)

    idx_boys = np.where(y == 0)[0]
    idx_girls = np.where(y == 1)[0]

    ax.scatter(
        X_pca[idx_boys, 0],
        X_pca[idx_boys, 1],
        c=cfg.color_boys,
        label=f"Class 1: Boys (n = {len(idx_boys)})",
        alpha=0.80,
        s=48,
        edgecolors="white",
        linewidths=0.7,
    )
    ax.scatter(
        X_pca[idx_girls, 0],
        X_pca[idx_girls, 1],
        c=cfg.color_girls,
        label=f"Class 2: Girls (n = {len(idx_girls)})",
        alpha=0.90,
        s=64,
        marker="s",
        edgecolors="white",
        linewidths=0.7,
    )

    ax.set_title(
        f"PCA Projection of 13-Dimensional Acoustic MFCC Vectors\n"
        f"Total Explained Variance: {(evr[0] + evr[1]) * 100:.1f}%",
        fontweight="bold",
    )
    ax.set_xlabel(f"Principal Component 1 ({evr[0] * 100:.1f}% Variance)")
    ax.set_ylabel(f"Principal Component 2 ({evr[1] * 100:.1f}% Variance)")
    ax.axhline(0, color="#94A3B8", lw=0.6, ls="--", alpha=0.5)
    ax.axvline(0, color="#94A3B8", lw=0.6, ls="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9)

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    logger.info("Saved mandatory PCA scatter plot: %s", out_path)
    return pca, X_pca


def plot_pca_speakers_scatter(
    X_pca: np.ndarray,
    y: np.ndarray,
    records: list[AudioFileMetadata],
    out_path: Path,
    dpi: int = 300,
) -> None:
    """Generate PCA scatter plot colored by speaker clusters to visualize intra-speaker consistency."""
    speakers = [r.speaker_id for r in records]
    unique_speakers = list(dict.fromkeys(speakers))

    fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)

    # Use a varied colormap for speaker IDs
    cmap = plt.cm.get_cmap("tab20", len(unique_speakers))

    for i, spk in enumerate(unique_speakers):
        spk_indices = [idx for idx, s in enumerate(speakers) if s == spk]
        is_girl = y[spk_indices[0]] == 1
        marker = "s" if is_girl else "o"
        ax.scatter(
            X_pca[spk_indices, 0],
            X_pca[spk_indices, 1],
            c=[cmap(i)],
            marker=marker,
            s=45 if not is_girl else 60,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.4,
            label=f"{spk} ({'G' if is_girl else 'B'})" if len(spk_indices) > 1 else None,
        )
        if len(spk_indices) > 1:
            pts = X_pca[spk_indices]
            ax.plot(pts[:, 0], pts[:, 1], color=cmap(i), lw=0.7, alpha=0.4, ls=":")

    ax.set_title(
        "Acoustic MFCC Space Grouped by Speaker Identity\n"
        "(Dashed lines connect multiple recordings from the same speaker)",
        fontweight="bold",
    )
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    # Show legend only for speakers with multiple recordings to prevent overcrowding
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=6.8, ncol=2)

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    logger.info("Saved speaker PCA scatter plot: %s", out_path)


def plot_mfcc_correlation_matrix(
    X: np.ndarray,
    feature_names: list[str],
    out_path: Path,
    dpi: int = 300,
) -> None:
    """Plot Pearson correlation matrix heatmap for all 13 MFCC coefficients."""
    corr = np.corrcoef(X, rowvar=False)

    fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 6.8},
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation (r)"},
        xticklabels=feature_names,
        yticklabels=feature_names,
        ax=ax,
    )
    ax.set_title("Pairwise Correlation Matrix of 13 MFCC Acoustic Features", fontweight="bold")

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    logger.info("Saved MFCC correlation matrix: %s", out_path)


def plot_mfcc_distributions(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    out_path: Path,
    cfg: Config,
    dpi: int = 300,
) -> None:
    """Plot boxplot distributions of all 13 MFCC coefficients partitioned by gender."""
    fig, axes = plt.subplots(3, 5, figsize=(9.5, 5.5), constrained_layout=True)
    flat_axes = axes.flatten()

    for i in range(13):
        ax = flat_axes[i]
        vals_boys = X[y == 0, i]
        vals_girls = X[y == 1, i]

        bp = ax.boxplot(
            [vals_boys, vals_girls],
            tick_labels=["Boys", "Girls"],
            patch_artist=True,
            widths=0.6,
            medianprops={"color": "black", "linewidth": 1.0},
        )
        bp["boxes"][0].set_facecolor(cfg.color_boys)
        bp["boxes"][0].set_alpha(0.65)
        bp["boxes"][1].set_facecolor(cfg.color_girls)
        bp["boxes"][1].set_alpha(0.65)

        ax.set_title(feature_names[i], fontsize=8.5, fontweight="bold")
        ax.tick_params(axis="both", labelsize=7.5)

    # Hide unused subplots
    for j in range(13, len(flat_axes)):
        flat_axes[j].axis("off")

    fig.suptitle("Acoustic MFCC Coefficient Distributions by Gender Class", fontsize=11, fontweight="bold")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    logger.info("Saved MFCC distributions boxplot: %s", out_path)


def plot_dataset_exploratory_charts(
    records: list[AudioFileMetadata],
    summary: dict[str, Any],
    out_dir: Path,
    cfg: Config,
    dpi: int = 300,
) -> None:
    """Generate exploratory dataset charts: class distribution, durations, and sample rates."""
    # 1. Class Distribution Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.2), constrained_layout=True)

    classes = ["Boys\n(Class 1)", "Girls\n(Class 2)"]
    counts = [summary["boys_recordings"], summary["girls_recordings"]]
    colors = [cfg.color_boys, cfg.color_girls]

    bars = ax1.bar(classes, counts, color=colors, width=0.55, edgecolor="#1E293B", linewidth=0.8)
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, h + 1.5, f"{int(h)} ({h / len(records) * 100:.1f}%)", ha="center", va="bottom", fontsize=8.5)
    ax1.set_ylim(0, max(counts) * 1.18)
    ax1.set_title("Recordings per Gender Class", fontweight="bold")
    ax1.set_ylabel("Number of Recordings")

    # Speaker Distribution
    spk_counts = [summary["num_identifiable_speakers"]["boys"], summary["num_identifiable_speakers"]["girls"]]
    bars2 = ax2.bar(classes, spk_counts, color=colors, width=0.55, edgecolor="#1E293B", linewidth=0.8, alpha=0.8)
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, h + 0.8, f"{int(h)}", ha="center", va="bottom", fontsize=8.5)
    ax2.set_ylim(0, max(spk_counts) * 1.18)
    ax2.set_title("Identifiable Unique Speakers", fontweight="bold")
    ax2.set_ylabel("Unique Speakers Count")

    fig.savefig(out_dir / "class_distribution.png", dpi=dpi)
    plt.close(fig)

    # 2. Duration Distribution Chart
    fig, ax = plt.subplots(figsize=(6.2, 3.2), constrained_layout=True)
    durations_boys = [r.duration for r in records if r.label == 0]
    durations_girls = [r.duration for r in records if r.label == 1]

    ax.hist(
        [durations_boys, durations_girls],
        bins=15,
        color=[cfg.color_boys, cfg.color_girls],
        label=[f"Boys (Mean: {np.mean(durations_boys):.2f}s)", f"Girls (Mean: {np.mean(durations_girls):.2f}s)"],
        edgecolor="#1E293B",
        lw=0.6,
        stacked=True,
    )
    ax.set_title("Recording Duration Distribution by Class", fontweight="bold")
    ax.set_xlabel("Duration (seconds)")
    ax.set_ylabel("Number of Audio Recordings")
    ax.legend(loc="upper right")

    fig.savefig(out_dir / "duration_distribution.png", dpi=dpi)
    plt.close(fig)

    # 3. Sampling Rate Distribution
    fig, ax = plt.subplots(figsize=(5.5, 3.0), constrained_layout=True)
    srs = [str(k) + " Hz" for k in sorted(summary["sampling_rate_distribution"].keys())]
    sr_vals = [summary["sampling_rate_distribution"][int(k.split()[0])] for k in srs]
    bars_sr = ax.bar(srs, sr_vals, color="#0D9488", width=0.5, edgecolor="#1E293B", linewidth=0.7)
    for b in bars_sr:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2.0, h + 1.0, f"{int(h)} ({h / len(records) * 100:.1f}%)", ha="center", va="bottom", fontsize=8.5)
    ax.set_ylim(0, max(sr_vals) * 1.2)
    ax.set_title("Original Audio Sampling Frequency Distribution", fontweight="bold")
    ax.set_ylabel("Count")

    fig.savefig(out_dir / "sampling_rate_distribution.png", dpi=dpi)
    plt.close(fig)
    logger.info("Saved dataset exploratory charts to %s", out_dir)


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    out_path: Path,
    dpi: int = 300,
) -> None:
    """Generate annotated Confusion Matrix heatmap with Class 1: Boys and Class 2: Girls."""
    fig, ax = plt.subplots(figsize=(4.8, 4.0), constrained_layout=True)

    # Class 1: Boys (label 0), Class 2: Girls (label 1)
    labels = ["Class 1: Boys", "Class 2: Girls"]

    # Annotations with counts and percentages
    total = np.sum(cm)
    annot = [
        [f"TP = {cm[0, 0]}\n({cm[0, 0] / total * 100:.1f}%)", f"FN = {cm[0, 1]}\n({cm[0, 1] / total * 100:.1f}%)"],
        [f"FP = {cm[1, 0]}\n({cm[1, 0] / total * 100:.1f}%)", f"TN = {cm[1, 1]}\n({cm[1, 1] / total * 100:.1f}%)"],
    ]

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=1.2,
        linecolor="#0F172A",
        ax=ax,
        annot_kws={"size": 10.0, "fontweight": "bold"},
    )
    ax.set_title(f"Confusion Matrix: {model_name}\n(Speaker-Independent Test Set)", fontweight="bold")
    ax.set_xlabel("Predicted Class", fontweight="bold")
    ax.set_ylabel("Actual Ground Truth Class", fontweight="bold")

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    logger.info("Saved confusion matrix heatmap: %s", out_path)


def plot_roc_curve(
    roc_dict: dict[str, tuple[np.ndarray, np.ndarray, float]],
    out_path: Path,
    dpi: int = 300,
) -> None:
    """Generate ROC curve comparing models with AUC scores and chance diagonal."""
    fig, ax = plt.subplots(figsize=(5.6, 4.5), constrained_layout=True)

    colors = {"Support Vector Machine": "#2563EB", "Random Forest": "#16A34A", "Neural Network (MLP)": "#9333EA"}

    for name, (fpr, tpr, auc_val) in roc_dict.items():
        color = colors.get(name, "#475569")
        ax.plot(fpr, tpr, color=color, lw=2.0, label=f"{name} (AUC = {auc_val:.3f})")

    ax.plot([0, 1], [0, 1], color="#94A3B8", lw=1.2, ls="--", label="Random Classifier (AUC = 0.500)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontweight="bold")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    logger.info("Saved ROC curves: %s", out_path)


def plot_model_comparison(
    metrics_table: list[dict[str, Any]],
    out_path: Path,
    dpi: int = 300,
) -> None:
    """Generate grouped bar chart comparing performance metrics across all models."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)

    models = [m["Model"] for m in metrics_table]
    metric_keys = ["Accuracy", "Sensitivity", "Precision", "Specificity", "F1 Score", "ROC AUC"]
    n_models = len(models)
    n_metrics = len(metric_keys)

    x = np.arange(n_metrics)
    width = 0.24

    palette = ["#2563EB", "#16A34A", "#9333EA"]

    for i, m in enumerate(metrics_table):
        vals = [m[k] for k in metric_keys]
        offset = (i - (n_models - 1) / 2) * width
        rects = ax.bar(x + offset, vals, width, label=m["Model"], color=palette[i % len(palette)], edgecolor="#1E293B", lw=0.6)
        for r in rects:
            h = r.get_height()
            ax.text(r.get_x() + r.get_width() / 2.0, h + 0.015, f"{h:.2f}", ha="center", va="bottom", fontsize=6.8, rotation=45)

    ax.set_ylabel("Score (0.00 – 1.00)")
    ax.set_title("Evaluation Metric Comparison Across Machine Learning Classifiers", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_keys, fontweight="bold")
    ax.set_ylim(0, 1.18)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    logger.info("Saved model comparison bar chart: %s", out_path)


def plot_neural_net_loss(
    train_losses: list[float],
    val_losses: list[float],
    out_path: Path,
    dpi: int = 300,
) -> None:
    """Plot training and validation loss curves of the Neural Network (MLP)."""
    fig, ax = plt.subplots(figsize=(6.0, 3.6), constrained_layout=True)

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, color="#2563EB", lw=1.8, label="Training Loss (Cross-Entropy)")
    if val_losses:
        ax.plot(epochs, val_losses, color="#E11D48", lw=1.8, ls="--", label="Validation Loss")

    ax.set_title("Multi-Layer Perceptron (MLP) Training & Validation Loss", fontweight="bold")
    ax.set_xlabel("Epoch / Iteration")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right", fontsize=8.5)

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    logger.info("Saved Neural Network loss curves: %s", out_path)

