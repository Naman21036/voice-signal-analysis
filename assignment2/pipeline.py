"""Single-command end-to-end execution pipeline for Assignment 2.

Executes all 21 systematic steps:
1. Dataset discovery
2. Dataset validation
3. Audio metadata extraction
4. Speaker identification validation
5. Audio segmentation check
6. Audio preprocessing (resampling to 16 kHz, phase-safe mono, silence boundary detection)
7. MFCC extraction (13 coefficients per frame)
8. Feature CSV generation (13-dim primary and 26-dim optional)
9. Exploratory analysis
10. Mandatory PCA scatter plot generation
11. Dataset splitting (speaker-independent Stratified Group K-Fold)
12. Model training (SVM, Random Forest, MLP Neural Network)
13. Hyperparameter tuning within training folds
14. Model comparison
15. Final model selection
16. Confusion matrix generation
17. Metric calculation (Accuracy, Sensitivity, Precision, Specificity, F1)
18. ROC curve and AUC generation
19. Model saving (.joblib and metadata)
20. Results and comparative experiments export
21. Comprehensive academic report generation (Markdown & publication PDF)
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from assignment2.config import Config
from assignment2.dataset import discover_dataset, save_dataset_artifacts
from assignment2.experiments import run_comparative_experiments
from assignment2.exploratory_analysis import run_exploratory_analysis
from assignment2.feature_extraction import extract_dataset_features
from assignment2.preprocessing import preprocess_dataset
from assignment2.report_generator import generate_markdown_report, generate_pdf_report
from assignment2.train import train_and_evaluate_all_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def run_full_pipeline(cfg: Config | None = None) -> dict[str, Any]:
    """Execute complete end-to-end Assignment 2 workflow."""
    cfg = cfg or Config()
    start_time = time.time()
    logger.info("=" * 65)
    logger.info(" STARTING ASSIGNMENT 2: VOICE GENDER CLASSIFICATION PIPELINE")
    logger.info("=" * 65)

    # STEP 1–4: Dataset Discovery, Validation, Metadata, and Speaker Identification
    logger.info("[STEP 1-4] Discovering dataset and extracting metadata...")
    records, excluded, summary = discover_dataset(cfg)
    save_dataset_artifacts(records, excluded, summary, cfg)
    logger.info(
        "Discovered %d valid recordings (%d Boys, %d Girls across %d unique speakers).",
        summary["total_recordings"],
        summary["boys_recordings"],
        summary["girls_recordings"],
        summary["num_identifiable_speakers"]["total"],
    )

    # STEP 5–6: Audio Preprocessing (Phase-safe mono, 16 kHz resampling, silence boundaries)
    logger.info("[STEP 5-6] Running standardized 16 kHz audio preprocessing...")
    preprocessed_signals = preprocess_dataset(records, cfg, save_processed_wav=True)

    # STEP 7–8: MFCC Feature Extraction (13-dim primary and 26-dim optional CSVs)
    logger.info("[STEP 7-8] Extracting 13-dimensional MFCC acoustic features...")
    X, y, feature_names = extract_dataset_features(records, preprocessed_signals, cfg)

    # STEP 9–10: Exploratory Data Analysis & Mandatory PCA Scatter Plot
    logger.info("[STEP 9-10] Executing exploratory data analysis and generating plots...")
    eda_results = run_exploratory_analysis(
        records, preprocessed_signals, X, y, feature_names, summary, cfg
    )

    # STEP 11–19: Training, Tuning, Comparison, Selection, Confusion Matrix, ROC, Model Saving
    logger.info("[STEP 11-19] Training, tuning, and evaluating ML classifiers (SVM, RF, MLP)...")
    train_results = train_and_evaluate_all_models(X, y, records, feature_names, cfg)
    best_model_name = train_results["best_model_name"]
    metrics_list = train_results["metrics_list"]

    # STEP 20: Comparative Experiments (Utterance types and feature dimensions)
    logger.info("[STEP 20] Executing comparative experiments (BIT vs Name vs Combined)...")
    exp_results = run_comparative_experiments(records, preprocessed_signals, cfg)

    # STEP 21: Academic Report Generation (Markdown & Publication PDF)
    logger.info("[STEP 21] Compiling academic Markdown and PDF reports...")
    md_report_path = generate_markdown_report(
        summary,
        eda_results["feature_statistics"],
        metrics_list,
        best_model_name,
        exp_results,
        cfg,
    )
    pdf_report_path = generate_pdf_report(
        summary,
        eda_results["feature_statistics"],
        metrics_list,
        best_model_name,
        exp_results,
        cfg,
    )

    elapsed = time.time() - start_time
    logger.info("=" * 65)
    logger.info(" ASSIGNMENT 2 PIPELINE COMPLETED SUCCESSFULLY IN %.2f SECONDS", elapsed)
    logger.info("=" * 65)
    logger.info("Summary of Deliverables:")
    logger.info("  - Metadata CSV:        %s", cfg.dataset_analysis_dir / "metadata.csv")
    logger.info("  - 13-dim Features CSV: %s", cfg.features_dir / "mfcc_features_13.csv")
    logger.info("  - Best Model:          %s", train_results["model_save_path"])
    logger.info("  - Mandatory PCA Plot:  %s", cfg.plots_dir / "pca_scatter_plot.png")
    logger.info("  - Markdown Report:     %s", md_report_path)
    logger.info("  - Academic PDF Report: %s", pdf_report_path)
    logger.info("=" * 65)

    return {
        "summary": summary,
        "train_results": train_results,
        "exp_results": exp_results,
        "md_report_path": str(md_report_path),
        "pdf_report_path": str(pdf_report_path),
    }


def main() -> None:
    """CLI entry point for running Assignment 2."""
    parser = argparse.ArgumentParser(
        description="Run Assignment 2 Voice-Based Gender Classification Pipeline."
    )
    args = parser.parse_args()
    try:
        run_full_pipeline()
    except Exception as exc:
        logger.exception("Pipeline execution failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

