"""
Tests for the ai-text-toolkit core modules.
"""

import pytest
import numpy as np

# Test detector module imports
def test_detector_imports():
    from core.detector import (
        AIDetector,
        PerplexityComputer,
        compute_burstiness,
        split_sentences,
        compute_stylometric_features,
        cliche_hits,
        ngram_repetition_top,
        AIDetectorClassifier,
        FeatureExtractor,
        extract_features,
    )
    assert AIDetector is not None
    assert PerplexityComputer is not None
    assert compute_burstiness is not None
    assert split_sentences is not None
    assert compute_stylometric_features is not None
    assert cliche_hits is not None
    assert ngram_repetition_top is not None
    assert AIDetectorClassifier is not None
    assert FeatureExtractor is not None
    assert extract_features is not None

def test_humanizer_imports():
    from core.humanizer import (
        Humanizer,
        GenerationBackend,
        LlamaCppBackend,
        OpenAICompatibleBackend,
        HumanizeResult,
        create_humanizer,
        HUMANIZE_SYSTEM_PROMPT,
    )
    assert Humanizer is not None
    assert GenerationBackend is not None
    assert LlamaCppBackend is not None
    assert OpenAICompatibleBackend is not None
    assert HumanizeResult is not None
    assert create_humanizer is not None
    assert HUMANIZE_SYSTEM_PROMPT is not None

def test_config_imports():
    from config import (
        Settings,
        get_settings,
        FEATURE_NAMES,
        MODEL_NAME,
        MIN_WORDS,
        HUMANIZE_SYSTEM_PROMPT,
        CLICHES,
    )
    assert Settings is not None
    assert get_settings is not None
    assert FEATURE_NAMES is not None
    assert len(FEATURE_NAMES) == 14
    assert MODEL_NAME == "gpt2"
    assert MIN_WORDS == 40
    assert HUMANIZE_SYSTEM_PROMPT is not None
    assert CLICHES is not None

def test_split_sentences():
    from core.detector import split_sentences
    
    text = "Hello world. How are you? I'm fine!"
    sentences = split_sentences(text)
    assert len(sentences) == 3
    assert "Hello world." in sentences
    assert "How are you?" in sentences
    assert "I'm fine!" in sentences

def test_compute_burstiness():
    from core.detector import compute_burstiness
    
    # Low variance = AI-like
    low_var = compute_burstiness([("a", 10.0), ("b", 10.5), ("c", 9.8)])
    # High variance = human-like
    high_var = compute_burstiness([("a", 5.0), ("b", 50.0), ("c", 100.0)])
    
    assert low_var < high_var
    assert low_var < 10
    assert high_var > 100

def test_cliche_hits():
    from core.detector import cliche_hits
    
    text = "In conclusion, it is important to note that AI is great."
    hits = cliche_hits(text)
    assert hits == 2  # "in conclusion" + "it is important to note"
    
    text2 = "Just a normal sentence without cliches."
    assert cliche_hits(text2) == 0

def test_ngram_repetition_top():
    from core.detector import ngram_repetition_top
    
    text = "the cat sat on the mat and the cat slept"
    ngrams = ngram_repetition_top(text, n=2, k=2)
    assert len(ngrams) >= 1
    assert ("the cat", 2) in ngrams

def test_stylometric_features():
    from core.detector import compute_stylometric_features
    
    text = "Hello world. This is a test sentence. Another one here."
    features = compute_stylometric_features(text)
    
    required_keys = [
        "avg_sentence_length", "sentence_length_var", "type_token_ratio",
        "punctuation_ratio", "bigram_uniqueness", "avg_word_length",
        "pos_noun", "pos_verb", "pos_adj", "pos_adv", "pos_pron", "pos_other"
    ]
    
    for key in required_keys:
        assert key in features
        assert isinstance(features[key], float)
    
    # Values should be reasonable
    assert 0 <= features["type_token_ratio"] <= 1
    assert 0 <= features["bigram_uniqueness"] <= 1
    assert features["avg_word_length"] > 0
    assert features["avg_sentence_length"] > 0

def test_feature_names():
    from config.constants import FEATURE_NAMES
    
    assert len(FEATURE_NAMES) == 14
    assert FEATURE_NAMES[0] == "perplexity"
    assert FEATURE_NAMES[1] == "burstiness"
    assert "pos_noun" in FEATURE_NAMES
    assert "pos_other" in FEATURE_NAMES

def test_settings():
    from config.settings import get_settings, Settings
    
    settings = get_settings()
    assert isinstance(settings, Settings)
    assert settings.detector_model_path == "gpt2"
    assert settings.detector_device == "auto"
    assert settings.api_port == 8000

def test_exceptions():
    from core.exceptions import (
        AITextToolkitError,
        DetectorError,
        DetectorNotReadyError,
        HumanizerError,
        BackendNotAvailableError,
        ModelNotFoundError,
    )
    
    # Test exception hierarchy
    assert issubclass(DetectorNotReadyError, DetectorError)
    assert issubclass(DetectorError, AITextToolkitError)
    assert issubclass(BackendNotAvailableError, HumanizerError)
    assert issubclass(ModelNotFoundError, HumanizerError)
    assert issubclass(HumanizerError, AITextToolkitError)

def test_humanize_system_prompt():
    from config.constants import HUMANIZE_SYSTEM_PROMPT
    
    assert "humanizer" in HUMANIZE_SYSTEM_PROMPT.lower()
    assert "contraction" in HUMANIZE_SYSTEM_PROMPT.lower()
    assert "meaning" in HUMANIZE_SYSTEM_PROMPT.lower()

# Performance test (marked as slow)
@pytest.mark.slow
def test_perplexity_computer_load():
    """Test that PerplexityComputer can be instantiated (slow - loads GPT-2)."""
    from core.detector import PerplexityComputer
    pc = PerplexityComputer(device="cpu")
    assert pc.model is not None
    assert pc.tokenizer is not None