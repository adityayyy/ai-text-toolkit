# core.detector — AI-text detection engine

from .models import (
    DetectionResult,
    PerplexityResult,
    StylometricFeatures,
    FeatureVector,
    SentenceScore,
)
from .perplexity import PerplexityComputer, compute_burstiness, split_sentences, split_sentences_with_spans
from .stylometric import compute_stylometric_features, cliche_hits, ngram_repetition_top, split_sentences
from .classifier import AIDetectorClassifier
from .features import FeatureExtractor, extract_features
from .pipeline import AIDetector, demo_inference, DEMO_TEXTS
from .training import main as train_main, evaluate_saved, load_hc3, load_synthetic

__all__ = [
    # Models
    "DetectionResult",
    "PerplexityResult",
    "StylometricFeatures",
    "FeatureVector",
    "SentenceScore",
    # Perplexity
    "PerplexityComputer",
    "compute_burstiness",
    "split_sentences",
    "split_sentences_with_spans",
    # Stylometric
    "compute_stylometric_features",
    "cliche_hits",
    "ngram_repetition_top",
    "split_sentences",
    # Classifier
    "AIDetectorClassifier",
    # Features
    "FeatureExtractor",
    "extract_features",
    # Pipeline
    "AIDetector",
    "demo_inference",
    "DEMO_TEXTS",
    # Training
    "train_main",
    "evaluate_saved",
    "load_hc3",
    "load_synthetic",
]