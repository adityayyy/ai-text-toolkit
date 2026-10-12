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
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.settings import get_settings
from core.exceptions import DetectorLoadError, DetectorNotReadyError


def build_pipeline() -> Pipeline:
    """Build the StandardScaler + LogisticRegression pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])


class AIDetectorClassifier:
    """Thin wrapper: fit / predict_proba(AI class) / save / load."""

    def __init__(self):
        self.pipeline = build_pipeline()
        self._loaded = False

    def fit(self, X, y):
        """Train the classifier."""
        self.pipeline.fit(X, y)
        self._loaded = True

    def predict_proba(self, X) -> np.ndarray:
        """Probability of the AI class (class 1), shape (n,)."""
        if not self._loaded:
            raise DetectorNotReadyError("Classifier not fitted or loaded")
        return self.pipeline.predict_proba(X)[:, 1]

    def predict(self, X) -> np.ndarray:
        """Predict class labels (0=human, 1=AI)."""
        if not self._loaded:
            raise DetectorNotReadyError("Classifier not fitted or loaded")
        return self.pipeline.predict(X)

    def save(self, path: str | Path = None):
        """Save the trained classifier to disk."""
        if path is None:
            path = get_settings().detector_model_full_path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)
        self._loaded = True
        print(f"[classifier] saved -> {path}")

    def load(self, path: str | Path = None):
        """Load a trained classifier from disk."""
        if path is None:
            path = get_settings().detector_model_full_path
        path = Path(path)
        if not path.exists():
            raise DetectorLoadError(
                f"No trained classifier at {path}. Run: python -m core.detector.training"
            )
        self.pipeline = joblib.load(path)
        self._loaded = True
        print(f"[classifier] loaded <- {path}")

    @property
    def is_loaded(self) -> bool:
        return self._loaded