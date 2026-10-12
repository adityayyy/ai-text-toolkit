"""
Feature extraction - combines perplexity + burstiness + stylometric features
into the complete feature vector for classification.
"""

from typing import List

import numpy as np

from config.constants import FEATURE_NAMES
from core.detector.perplexity import PerplexityComputer, compute_burstiness
from core.detector.stylometric import compute_stylometric_features
from core.detector.models import FeatureVector, StylometricFeatures, PerplexityResult


class FeatureExtractor:
    """Extracts the complete feature vector from text."""

    def __init__(self, perplexity_computer: PerplexityComputer):
        self.perplexity_computer = perplexity_computer

    def extract(self, text: str) -> FeatureVector:
        """
        Build the complete feature vector in FEATURE_NAMES order.
        
        Returns:
            FeatureVector with perplexity, burstiness, and stylometric features
        """
        # Perplexity analysis (includes burstiness computation)
        ppl_result = self.perplexity_computer.analyze(text)
        burstiness = compute_burstiness(ppl_result.sentence_ppls)
        
        # Stylometric features
        styl_dict = compute_stylometric_features(text)
        stylometric = StylometricFeatures(**styl_dict)
        
        return FeatureVector(
            perplexity=ppl_result.perplexity,
            burstiness=burstiness,
            stylometric=stylometric,
        )

    def extract_batch(self, texts: list[str]) -> list[FeatureVector]:
        """Extract features for multiple texts."""
        return [self.extract(text) for text in texts]


def extract_features(text: str, perplexity_computer: PerplexityComputer) -> np.ndarray:
    """
    Convenience function to extract features as numpy array.
    Used by training script and pipeline.
    """
    extractor = FeatureExtractor(perplexity_computer)
    fv = extractor.extract(text)
    return fv.to_array()