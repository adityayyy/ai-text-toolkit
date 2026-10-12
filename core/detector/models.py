"""Data models for the detector module."""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class DetectionResult:
    """Complete detection result."""
    ai_probability: float
    is_ai: bool
    label: str
    feature_breakdown: dict
    sentence_scores: list[dict]
    top_repeated_ngrams: list[tuple[str, int]]
    n_tokens: int
    elapsed_seconds: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API serialization."""
        return {
            "ai_probability": self.ai_probability,
            "is_ai": self.is_ai,
            "label": self.label,
            "feature_breakdown": self.feature_breakdown,
            "sentence_scores": self.sentence_scores,
            "top_repeated_ngrams": self.top_repeated_ngrams,
            "n_tokens": self.n_tokens,
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass
class PerplexityResult:
    """Result from perplexity analysis."""
    perplexity: float
    sentence_ppls: list[tuple[str, float]]
    n_tokens: int


@dataclass
class StylometricFeatures:
    """Computed stylometric features."""
    avg_sentence_length: float
    sentence_length_var: float
    type_token_ratio: float
    punctuation_ratio: float
    bigram_uniqueness: float
    avg_word_length: float
    pos_noun: float
    pos_verb: float
    pos_adj: float
    pos_adv: float
    pos_pron: float
    pos_other: float
    
    def to_dict(self) -> dict:
        return {
            "avg_sentence_length": self.avg_sentence_length,
            "sentence_length_var": self.sentence_length_var,
            "type_token_ratio": self.type_token_ratio,
            "punctuation_ratio": self.punctuation_ratio,
            "bigram_uniqueness": self.bigram_uniqueness,
            "avg_word_length": self.avg_word_length,
            "pos_noun": self.pos_noun,
            "pos_verb": self.pos_verb,
            "pos_adj": self.pos_adj,
            "pos_adv": self.pos_adv,
            "pos_pron": self.pos_pron,
            "pos_other": self.pos_other,
        }
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array in FEATURE_NAMES[2:] order."""
        from config.constants import FEATURE_NAMES
        feature_names = FEATURE_NAMES[2:]  # Skip perplexity and burstiness
        return np.array([getattr(self, name) for name in feature_names], dtype=np.float32)


@dataclass
class FeatureVector:
    """Complete feature vector for classification."""
    perplexity: float
    burstiness: float
    stylometric: StylometricFeatures
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array in FEATURE_NAMES order."""
        from config.constants import FEATURE_NAMES
        values = {
            "perplexity": self.perplexity,
            "burstiness": self.burstiness,
            **self.stylometric.to_dict(),
        }
        return np.array([values[name] for name in FEATURE_NAMES], dtype=np.float32)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API serialization."""
        return {
            "perplexity": self.perplexity,
            "burstiness": self.burstiness,
            **self.stylometric.to_dict(),
        }


@dataclass
class SentenceScore:
    """Per-sentence AI-likeness score."""
    sentence: str
    ai_score: float