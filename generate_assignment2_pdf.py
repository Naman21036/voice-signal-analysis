#!/usr/bin/env python3
"""Generate AIML_Assignment2_Report_Naman_Gupta.pdf from measured results.

Usage:
    python generate_assignment2_pdf.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import shutil
from assignment2.config import Config
from assignment2.dataset import discover_dataset
from assignment2.evaluate import EvaluationMetrics
from assignment2.report_generator import generate_pdf_report


def main():
    cfg = Config()
    records, excluded, summary = discover_dataset(cfg)

    # Load evaluated metrics
    metrics_path = cfg.metrics_dir / "model_comparison.json"
    if not metrics_path.is_file():
        print("Metrics not found. Running pipeline first...")
        from assignment2.pipeline import run_full_pipeline
        run_full_pipeline(cfg)
        return

    metrics_json = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics_list = [EvaluationMetrics(**m) for m in metrics_json]

    # Load comparative experiments
    exp_path = cfg.metrics_dir / "comparative_experiments.json"
    exp_results = json.loads(exp_path.read_text(encoding="utf-8")) if exp_path.is_file() else {}

    # Load metadata
    meta_path = cfg.models_dir / "model_metadata.json"
    best_model_name = "Support Vector Machine"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        best_model_name = meta.get("best_model_name", "Support Vector Machine")

    pdf_out = generate_pdf_report(summary, [], metrics_list, best_model_name, exp_results, cfg)
    # Copy to root directory for easy access
    root_pdf = ROOT / "AIML_Assignment2_Report_Naman_Gupta.pdf"
    shutil.copy2(pdf_out, root_pdf)
    print(f"Academic PDF report generated successfully at:")
    print(f"  - {pdf_out}")
    print(f"  - {root_pdf}")


if __name__ == "__main__":
    main()

