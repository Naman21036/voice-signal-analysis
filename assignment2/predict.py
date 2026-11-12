"""Inference and prediction interface for new WAV audio recordings.

Loads the saved best machine learning pipeline, extracts the 13-dimensional
MFCC acoustic vector using standard 16 kHz preprocessing, and outputs:
- Predicted numeric class (0 = Class 1: Boys, 1 = Class 2: Girls)
- Predicted gender label
- Class posterior probabilities
- Full 13-dimensional MFCC feature vector
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import soundfile as sf

from assignment2.config import Config
from assignment2.feature_extraction import extract_features_13
from assignment2.preprocessing import preprocess_signal

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def predict_audio_file(
    audio_path: str | Path,
    model_path: str | Path | None = None,
    cfg: Config | None = None,
) -> dict[str, Any]:
    """Run end-to-end inference on a single WAV audio file."""
    cfg = cfg or Config()
    audio_path = Path(audio_path).resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load trained model pipeline
    if model_path is None:
        model_path = cfg.models_dir / "best_gender_classifier.joblib"
    else:
        model_path = Path(model_path).resolve()

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Please train the model first."
        )

    pipeline = joblib.load(model_path)

    # 1. Read and validate audio
    data, fs = sf.read(str(audio_path), dtype="float64", always_2d=True)
    if data.size == 0:
        raise ValueError(f"Audio file contains zero samples: {audio_path}")

    # 2. Preprocess to 16 kHz mono
    processed, trim_info = preprocess_signal(data, fs, cfg, apply_trimming=True)

    # 3. Extract 13 MFCC features
    features_13 = extract_features_13(processed, cfg)
    feat_matrix = features_13.reshape(1, -1)

    # 4. Predict
    pred_label = int(pipeline.predict(feat_matrix)[0])
    gender_name = cfg.class_boys_name if pred_label == 0 else cfg.class_girls_name
    class_name = f"Class {pred_label + 1}: {gender_name}"

    # Probabilities
    probs = {}
    if hasattr(pipeline, "predict_proba"):
        prob_array = pipeline.predict_proba(feat_matrix)[0]
        # Label 0 is Boys, Label 1 is Girls
        probs = {
            "Boys": float(prob_array[0]),
            "Girls": float(prob_array[1]),
        }

    feature_names = [f"MFCC_{i}" for i in range(1, cfg.n_mfcc + 1)]
    feature_dict = {name: float(val) for name, val in zip(feature_names, features_13)}

    result = {
        "file": audio_path.name,
        "audio_path": str(audio_path),
        "predicted_class_id": pred_label + 1,       # Class 1: Boys, Class 2: Girls
        "predicted_label": pred_label,              # 0: Boys, 1: Girls
        "predicted_gender": gender_name,
        "class_name": class_name,
        "probabilities": probs,
        "preprocessing_info": trim_info,
        "mfcc_features": feature_dict,
    }
    return result


def main() -> None:
    """CLI entrypoint for gender prediction on a WAV file."""
    parser = argparse.ArgumentParser(
        description="Predict gender (Boys vs Girls) from a voice WAV recording."
    )
    parser.add_argument("audio_path", type=str, help="Path to input WAV recording.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model pipeline (.joblib). Default: outputs/models/best_gender_classifier.joblib",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print output as structured JSON string.",
    )

    args = parser.parse_args()

    try:
        res = predict_audio_file(args.audio_path, model_path=args.model)
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print("\n" + "=" * 55)
            print(" VOICE GENDER CLASSIFICATION RESULT")
            print("=" * 55)
            print(f"File:               {res['file']}")
            print(f"Predicted Class:    Class {res['predicted_class_id']} ({res['predicted_gender']})")
            print(f"Predicted Gender:   {res['predicted_gender']}")
            if res["probabilities"]:
                print(f"Boys Probability:   {res['probabilities']['Boys'] * 100:.2f}%")
                print(f"Girls Probability:  {res['probabilities']['Girls'] * 100:.2f}%")
            print("-" * 55)
            print("Extracted 13-Dimensional MFCC Features:")
            for k, v in res["mfcc_features"].items():
                print(f"  {k:<8}: {v:>9.4f}")
            print("=" * 55 + "\n")
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

