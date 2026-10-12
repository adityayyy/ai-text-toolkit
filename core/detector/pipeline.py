"""
Main AI-text detector: combines perplexity + burstiness + stylometric features
into one feature vector, classified by a trained LogisticRegression.

Usage (from the project root, after training):
    python -m core.detector.pipeline

Output for a text:
- ai_probability   0-100 (probability the text is AI-generated)
- is_ai            threshold at 50
- feature_breakdown  all underlying signals (perplexity, burstiness, stylometrics)
- sentence_scores    per-sentence AI-likeness, highest first (for highlighting)

PERFORMANCE: one GPT-2 forward pass (~0.3-0.5s for 500 words on CPU) + cheap
feature extraction  ->  well under the 2-3s target.

SENTENCE SCORING - HEURISTIC (ponytail: relative-perplexity heuristic, upgrade
path = a second per-sentence classifier trained on per-sentence features):
AI sentences tend to sit BELOW the document's mean perplexity (very smooth and
predictable) and to contain AI-cliche phrases. So per sentence:
    z    = (sentence_ppl - mean_ppl) / std_ppl      (negative z = suspicious)
    base = sigmoid(-1.5 * z)                        (0-1)
    ai_score = clamp(base + 0.25 if cliche else base) * 100
This ranks sentences honestly ("most AI-like relative to this document") without
a second trained model.
"""

import time
from typing import List

import numpy as np

from config.constants import FEATURE_NAMES, CLICHES
from config.settings import get_settings

from core.detector.perplexity import PerplexityComputer, compute_burstiness, split_sentences_with_spans
from core.detector.stylometric import compute_stylometric_features, cliche_hits, ngram_repetition_top
from core.detector.classifier import AIDetectorClassifier
from core.detector.features import FeatureExtractor
from core.detector.models import DetectionResult, SentenceScore, PerplexityResult
from core.exceptions import DetectorNotReadyError, FeatureExtractionError


class AIDetector:
    """Full detection pipeline: GPT-2 signals + stylometrics + classifier."""

    def __init__(self, model_name: str = None, device: str = None):
        settings = get_settings()
        self.lm = PerplexityComputer(
            model_name or settings.detector_model_path,
            device or settings.detector_device
        )
        self.feature_extractor = FeatureExtractor(self.lm)
        self.classifier = AIDetectorClassifier()
        self._ready = False

    def load(self, path: str = None):
        """Load the trained classifier (fast; the LM was loaded in __init__)."""
        self.classifier.load(path)
        self._ready = True

    @property
    def ready(self) -> bool:
        return self._ready

    # ── Feature extraction ───────────────────────────────────────────
    def extract_features(self, text: str) -> np.ndarray:
        """Build the 14-dim feature vector in FEATURE_NAMES order."""
        fv = self.feature_extractor.extract(text)
        return fv.to_array()

    # ── Prediction ───────────────────────────────────────────────────
    def predict(self, text: str) -> DetectionResult:
        if not self._ready:
            raise DetectorNotReadyError(
                "Classifier not loaded. Run: python -m core.detector.training"
            )

        t0 = time.time()
        
        # Single GPT-2 forward pass gives us perplexity + per-sentence ppls
        ppl_result = self.lm.analyze(text)
        burstiness = compute_burstiness(ppl_result.sentence_ppls)
        stylometric = compute_stylometric_features(text)
        
        values = {
            "perplexity": ppl_result.perplexity,
            "burstiness": burstiness,
            **stylometric,
        }

        X = np.array([[values[n] for n in FEATURE_NAMES]], dtype=np.float32)
        ai_prob = float(self.classifier.predict_proba(X)[0]) * 100.0

        # Per-sentence scores
        sentence_scores = self._sentence_scores(ppl_result.sentence_ppls)
        
        # Top repeated n-grams
        top_ngrams = ngram_repetition_top(text)

        return DetectionResult(
            ai_probability=round(ai_prob, 1),
            is_ai=ai_prob >= 50.0,
            label="AI-generated" if ai_prob >= 50.0 else "Likely human",
            feature_breakdown={k: round(v, 4) for k, v in values.items()},
            sentence_scores=[{"sentence": s.sentence, "ai_score": s.ai_score} for s in sentence_scores],
            top_repeated_ngrams=top_ngrams,
            n_tokens=ppl_result.n_tokens,
            elapsed_seconds=round(time.time() - t0, 2),
        )

    # ── Per-sentence highlighting ────────────────────────────────────
    def _sentence_scores(self, sentence_ppls: list[tuple[str, float]]) -> list[SentenceScore]:
        """Heuristic per-sentence AI score - see module docstring."""
        ppls = np.array([p for _, p in sentence_ppls], dtype=float)
        mean = float(ppls.mean()) if len(ppls) else 0.0
        std = float(ppls.std()) if len(ppls) else 0.0

        out = []
        for sent, ppl in sentence_ppls:
            z = (ppl - mean) / std if std > 0 else 0.0
            base = 1.0 / (1.0 + np.exp(1.5 * z))        # sigmoid(-1.5 * z)
            if cliche_hits(sent) > 0:
                base += 0.25
            out.append(SentenceScore(sentence=sent, ai_score=round(min(base, 1.0) * 100, 1)))
        out.sort(key=lambda d: d.ai_score, reverse=True)
        return out


# ── Demo / smoke test ────────────────────────────────────────────────
DEMO_TEXTS = {
    "human-style": ("Been thinking about switching my note-taking setup again. "
                    "I tried Notion last year, lasted about three weeks before I "
                    "went back to pen and paper. There's something about crossing "
                    "out a finished task that an app just can't replicate. "
                    "Anyway, the exam is on Friday and my notes are a mess."),
    "ai-style": ("In conclusion, selecting an appropriate note-taking system is "
                 "essential for academic success. Furthermore, digital tools "
                 "offer robust organizational capabilities that enhance "
                 "productivity. It is important to note that consistency plays "
                 "a crucial role in maintaining effective study habits. "
                 "Additionally, students should carefully evaluate their "
                 "individual needs before making a decision."),
}


def demo_inference():
    detector = AIDetector()
    detector.load()
    for name, text in DEMO_TEXTS.items():
        print(f"\n=== {name} ===")
        import json
        print(json.dumps(detector.predict(text).to_dict(), indent=2))


if __name__ == "__main__":
    import time  # for demo
    demo_inference()