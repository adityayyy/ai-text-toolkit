"""
Evaluation script - "how accurate is the detector?"

Loads the saved classifier (models/detector_logreg.joblib) and evaluates it on
the HC3 held-out test sample. Prints accuracy / precision / recall / F1 /
ROC-AUC / confusion matrix.

Usage (from the project root, after training):
    python -m detector.evaluate
    python -m detector.evaluate --test 50   # smaller/faster
"""

import argparse

from detector.train import evaluate_saved


def main():
    ap = argparse.ArgumentParser(description="Evaluate the trained detector")
    ap.add_argument("--test", type=int, default=150, help="test samples per class")
    args = ap.parse_args()
    evaluate_saved(args.test)


if __name__ == "__main__":
    main()
