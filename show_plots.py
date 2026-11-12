#!/usr/bin/env python3
"""Interactive Plot Viewer for Assessment 2: Voice-Based Gender Classification.

Allows viewing all 16 generated plots interactively via Matplotlib.

Usage:
    python show_plots.py              # Opens interactive viewer with Next/Previous controls
    python show_plots.py --list       # Lists all available plots
    python show_plots.py --show scatter # Displays specific plot (e.g. pca_scatter_plot)
    python show_plots.py --category dataset # Displays plots from a specific category
"""
import argparse
import sys
from pathlib import Path

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from matplotlib.widgets import Button
except ImportError:
    print("Error: matplotlib is required. Install via pip install matplotlib")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent
PLOTS_DIR = ROOT / "outputs" / "plots" / "assignment2"

PLOT_CATALOG = [
    # Category 1: Feature Extraction & Scatter Plots (Requirement 2)
    {
        "id": "pca_scatter_plot",
        "file": "pca_scatter_plot.png",
        "category": "scatter",
        "title": "PCA 2D Scatter Plot: Mel-Frequency Cepstral Coefficients (13 MFCCs)",
        "desc": "Primary scatter plot of 115 voice recordings projected onto PC1 (20.0%) and PC2 (12.9%). Shows separation between Class 1 (Boys) and Class 2 (Girls)."
    },
    {
        "id": "pca_speakers",
        "file": "pca_speakers_scatter_plot.png",
        "category": "scatter",
        "title": "PCA Scatter Plot: Grouped by 63 Unique Speakers",
        "desc": "Demonstrates intra-speaker clustering across repeated utterances ('BIT' vs Name), proving why Stratified Group K-Fold is essential to prevent data leakage."
    },
    {
        "id": "mfcc_heatmaps",
        "file": "mfcc_heatmaps_comparison.png",
        "category": "mfcc",
        "title": "MFCC Heatmaps: Representative Boy vs Girl Utterance",
        "desc": "Time-frequency evolution across 13 MFCC coefficients over frame steps. Demonstrates energy concentration differences across vocal tract resonances."
    },
    {
        "id": "mfcc_boxplot",
        "file": "mfcc_distributions_boxplot.png",
        "category": "mfcc",
        "title": "MFCC Feature Distributions: Boys vs Girls (Coefficients 1 to 13)",
        "desc": "Boxplot distributions of mean MFCC coefficients. Highlights strongest discriminating coefficients (MFCC 1, 2, 4, and 7)."
    },
    {
        "id": "mfcc_correlation",
        "file": "mfcc_correlation_heatmap.png",
        "category": "mfcc",
        "title": "MFCC Feature Correlation Matrix (Pearson r)",
        "desc": "Empirical verification of DCT decorrelation properties; coefficients exhibit near-orthogonality with localized formant correlations."
    },
    {
        "id": "waveform_boy",
        "file": "waveform_comparison_boy.png",
        "category": "waveforms",
        "title": "Time-Domain Waveform & Spectrogram: Class 1 (Boy)",
        "desc": "Pre- and post-silence trimming waveforms and Short-Time Fourier Transform (STFT) spectrogram."
    },
    {
        "id": "waveform_girl",
        "file": "waveform_comparison_girl.png",
        "category": "waveforms",
        "title": "Time-Domain Waveform & Spectrogram: Class 2 (Girl)",
        "desc": "Pre- and post-silence trimming waveforms and higher fundamental frequency formant representation."
    },

    # Category 2: Machine Learning Evaluation & Metrics (Requirement 3 & 4)
    {
        "id": "cm_svm",
        "file": "confusion_matrix_support_vector_machine.png",
        "category": "models",
        "title": "Confusion Matrix: Support Vector Machine (RBF Kernel)",
        "desc": "Test evaluation on unseen speakers: 16 True Boys (100% Sensitivity), 2 True Girls, 3 False Boys (due to class imbalance)."
    },
    {
        "id": "cm_rf",
        "file": "confusion_matrix_random_forest.png",
        "category": "models",
        "title": "Confusion Matrix: Random Forest (Balanced Weighting)",
        "desc": "Ensemble evaluation across 50 decision trees with depth constraint: 100% Boys Sensitivity, 40% Girls Specificity."
    },
    {
        "id": "cm_mlp",
        "file": "confusion_matrix_neural_network_mlp.png",
        "category": "models",
        "title": "Confusion Matrix: Neural Network (MLP 13 -> 64 -> 32 -> 2)",
        "desc": "Deep architecture with Adam optimizer, ReLU activation, and Softmax classification output."
    },
    {
        "id": "roc_curves",
        "file": "roc_curves_comparison.png",
        "category": "models",
        "title": "Receiver Operating Characteristic (ROC) Curves Comparison",
        "desc": "True Positive Rate vs False Positive Rate: SVM (AUC = 1.000), Random Forest (AUC = 1.000), Neural Net MLP (AUC = 0.975)."
    },
    {
        "id": "model_barchart",
        "file": "model_comparison_barchart.png",
        "category": "models",
        "title": "Comparative Evaluation Metrics: Accuracy, Sensitivity, Precision, Specificity, F1",
        "desc": "Side-by-side metric benchmarking across SVM, Random Forest, and Neural Network (MLP)."
    },
    {
        "id": "neural_net_loss",
        "file": "neural_net_loss_curves.png",
        "category": "models",
        "title": "Neural Network Training & Validation Loss Curves",
        "desc": "Cross-entropy loss convergence over 200 epochs with early stopping monitoring."
    },

    # Category 3: Dataset Analysis & Demographics (Requirement 1)
    {
        "id": "class_distribution",
        "file": "class_distribution.png",
        "category": "dataset",
        "title": "Gender Class Distribution: Class 1 (Boys) vs Class 2 (Girls)",
        "desc": "Empirical distribution across 115 voice recordings: 103 Boys (89.6%) vs 12 Girls (10.4%)."
    },
    {
        "id": "duration_distribution",
        "file": "duration_distribution.png",
        "category": "dataset",
        "title": "Audio Duration Distribution Across Classes",
        "desc": "Histogram showing recording lengths (Mean = 1.63s, Range = 0.38s - 4.96s) across utterance types."
    },
    {
        "id": "sampling_rate",
        "file": "sampling_rate_distribution.png",
        "category": "dataset",
        "title": "Original Sampling Rate Distribution",
        "desc": "Heterogeneity in raw audio hardware (44.1 kHz, 48 kHz, 16 kHz) normalized to uniform 16 kHz via polyphase resampling."
    },
]


class PlotViewer:
    def __init__(self, catalog):
        self.catalog = [item for item in catalog if (PLOTS_DIR / item["file"]).is_file()]
        if not self.catalog:
            print(f"Error: No plot images found in {PLOTS_DIR}")
            sys.exit(1)
        self.index = 0

        self.fig, self.ax = plt.subplots(figsize=(11, 7.5))
        plt.subplots_adjust(bottom=0.15, top=0.90, left=0.05, right=0.95)
        self.fig.canvas.manager.set_window_title("Assessment 2 - Voice Gender Classification Visualizer")

        # Buttons
        ax_prev = plt.axes([0.15, 0.03, 0.15, 0.06])
        ax_next = plt.axes([0.70, 0.03, 0.15, 0.06])
        self.btn_prev = Button(ax_prev, "◀ Previous")
        self.btn_next = Button(ax_next, "Next ▶")
        self.btn_prev.on_clicked(self.prev_plot)
        self.btn_next.on_clicked(self.next_plot)

        # Keyboard shortcuts
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.show_current()

    def show_current(self):
        self.ax.clear()
        item = self.catalog[self.index]
        img_path = PLOTS_DIR / item["file"]
        img = mpimg.imread(str(img_path))
        self.ax.imshow(img)
        self.ax.axis("off")

        total = len(self.catalog)
        self.fig.suptitle(
            f"[{self.index + 1}/{total}] {item['title']}\n{item['desc']}",
            fontsize=11,
            fontweight="bold",
            color="#1a237e",
            wrap=True,
        )
        self.fig.canvas.draw_idle()

    def next_plot(self, event=None):
        self.index = (self.index + 1) % len(self.catalog)
        self.show_current()

    def prev_plot(self, event=None):
        self.index = (self.index - 1) % len(self.catalog)
        self.show_current()

    def on_key(self, event):
        if event.key in ("right", " ", "n"):
            self.next_plot()
        elif event.key in ("left", "p"):
            self.prev_plot()
        elif event.key in ("q", "escape"):
            plt.close(self.fig)


def list_plots():
    print(f"\nAvailable Assessment 2 Generated Plots ({PLOTS_DIR}):\n")
    print(f"{'#':<3} | {'Plot ID':<20} | {'Category':<10} | {'Filename':<35} | {'Title'}")
    print("-" * 110)
    for idx, item in enumerate(PLOT_CATALOG, 1):
        exists = "✓" if (PLOTS_DIR / item["file"]).is_file() else "✗ MISSING"
        print(f"{idx:<3} | {item['id']:<20} | {item['category']:<10} | {item['file']:<35} | {exists} {item['title'][:40]}")
    print("\nRun: python show_plots.py              # to launch interactive browser")
    print("Run: python show_plots.py --show <id>   # to display a specific plot\n")


def show_single_plot(plot_id: str):
    matches = [p for p in PLOT_CATALOG if p["id"].lower() == plot_id.lower() or p["file"].lower() == plot_id.lower()]
    if not matches:
        print(f"Error: Unknown plot ID '{plot_id}'. Use --list to see available plots.")
        return
    item = matches[0]
    path = PLOTS_DIR / item["file"]
    if not path.is_file():
        print(f"Error: File not found: {path}")
        return

    img = mpimg.imread(str(path))
    plt.figure(figsize=(10, 7))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"{item['title']}\n{item['desc']}", fontsize=11, fontweight="bold", color="#1a237e")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="View Assessment 2 plots interactively.")
    parser.add_argument("--list", action="store_true", help="List all generated plots and exit.")
    parser.add_argument("--show", type=str, help="Show a specific plot by ID or filename.")
    parser.add_argument("--category", type=str, choices=["scatter", "mfcc", "waveforms", "models", "dataset"],
                        help="Filter viewer by category.")
    args = parser.parse_args()

    if args.list:
        list_plots()
        return

    if args.show:
        show_single_plot(args.show)
        return

    catalog = PLOT_CATALOG
    if args.category:
        catalog = [p for p in PLOT_CATALOG if p["category"] == args.category]

    viewer = PlotViewer(catalog)
    print("\nLaunching Interactive Plot Viewer...")
    print("Controls: Next / Prev buttons, Left / Right arrow keys, 'q' or Esc to close window.\n")
    plt.show()


if __name__ == "__main__":
    main()

