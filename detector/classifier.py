"""
Logistic Regression classifier over the detector feature vector.

Pipeline: StandardScaler -> LogisticRegression.
The scaler is REQUIRED: perplexity (~20-100) and variance features (0-1000+)
live on wildly different scales, and raw LogisticRegression would be dominated
by the large-magnitude features.

SPEED/QUALITY TRADEOFF: LogisticRegression trains in milliseconds on ~1000
samples and infers in microseconds. Upgrade path: swap in MLPClassifier or a
small gradient-boosted model inside the same Pipeline - nothing else changes.
"""

import os

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Saved classifier lives in <project root>/models/
MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
DETECTOR_PKL = os.path.join(MODELS_DIR, "detector_logreg.joblib")


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])


class AIDetectorClassifier:
    """Thin wrapper: fit / predict_proba(AI class) / save / load."""

    def __init__(self):
        self.pipeline = build_pipeline()

    def fit(self, X, y):
        self.pipeline.fit(X, y)

    def predict_proba(self, X):
        """Probability of the AI class (class 1), shape (n,)."""
        return self.pipeline.predict_proba(X)[:, 1]

    def save(self, path: str = DETECTOR_PKL):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.pipeline, path)
        print(f"[classifier] saved -> {path}")

    def load(self, path: str = DETECTOR_PKL):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No trained classifier at {path}. Run: python -m detector.train")
        self.pipeline = joblib.load(path)
        print(f"[classifier] loaded <- {path}")
