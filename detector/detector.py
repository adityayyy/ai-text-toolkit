"""
Main AI-text detector: combines perplexity + burstiness + stylometric features
into one feature vector, classified by a trained LogisticRegression.

Usage (from the project root, after training):
    python -m detector.detector

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

import json
import os
import sys
import time

import numpy as np

# Absolute imports so this works both as `python -m detector.detector` and as
# a direct script (python detector/detector.py).
from detector.perplexity import PerplexityComputer, compute_burstiness
from detector.stylometric import (compute_stylometric_features, cliche_hits,
                                  ngram_repetition_top, split_sentences)
from detector.classifier import AIDetectorClassifier, DETECTOR_PKL

# Feature vector layout - SINGLE SOURCE OF TRUTH (train.py builds vectors from
# this list, so never reorder without retraining).
FEATURE_NAMES = [
    "perplexity",           # overall GPT-2 perplexity (lower = more AI-like)
    "burstiness",           # variance of per-sentence ppl (lower = more AI-like)
    "avg_sentence_length",
    "sentence_length_var",
    "type_token_ratio",
    "punctuation_ratio",
    "bigram_uniqueness",    # 1.0 = no repeated bigrams (lower = more AI-like)
    "avg_word_length",
    "pos_noun", "pos_verb", "pos_adj", "pos_adv", "pos_pron", "pos_other",
]


class AIDetector:
    """Full detection pipeline: GPT-2 signals + stylometrics + classifier."""

    def __init__(self, model_name: str = "gpt2", device: str = "cpu"):
        self.lm = PerplexityComputer(model_name, device)  # swap LM here
        self.classifier = AIDetectorClassifier()
        self.ready = False

    def load(self, path: str = DETECTOR_PKL):
        """Load the trained classifier (fast; the LM was loaded in __init__)."""
        self.classifier.load(path)
        self.ready = True

    # ── Feature extraction ───────────────────────────────────────────
    def extract_features(self, text: str) -> np.ndarray:
        """Build the 14-dim feature vector in FEATURE_NAMES order."""
        res = self.lm.analyze(text)
        burst = compute_burstiness(res["sentence_ppls"])
        styl = compute_stylometric_features(text)
        values = {"perplexity": res["perplexity"], "burstiness": burst, **styl}
        return np.array([values[name] for name in FEATURE_NAMES], dtype=np.float32)

    # ── Prediction ───────────────────────────────────────────────────
    def predict(self, text: str) -> dict:
        if not self.ready:
            raise RuntimeError("Classifier not loaded. Run: python -m detector.train")

        t0 = time.time()
        res = self.lm.analyze(text)
        burst = compute_burstiness(res["sentence_ppls"])
        styl = compute_stylometric_features(text)
        values = {"perplexity": res["perplexity"], "burstiness": burst, **styl}

        X = np.array([[values[n] for n in FEATURE_NAMES]], dtype=np.float32)
        ai_prob = float(self.classifier.predict_proba(X)[0]) * 100.0

        return {
            "ai_probability": round(ai_prob, 1),
            "is_ai": ai_prob >= 50.0,
            "label": "AI-generated" if ai_prob >= 50.0 else "Likely human",
            "feature_breakdown": {k: round(v, 4) for k, v in values.items()},
            "sentence_scores": self._sentence_scores(res["sentence_ppls"]),
            "top_repeated_ngrams": ngram_repetition_top(text),
            "n_tokens": res["n_tokens"],
            "elapsed_seconds": round(time.time() - t0, 2),
        }

    # ── Per-sentence highlighting ────────────────────────────────────
    def _sentence_scores(self, sentence_ppls: list[tuple[str, float]]) -> list[dict]:
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
            out.append({"sentence": sent, "ai_score": round(min(base, 1.0) * 100, 1)})
        out.sort(key=lambda d: d["ai_score"], reverse=True)
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
    detector.load()   # requires a trained classifier from train.py
    for name, text in DEMO_TEXTS.items():
        print(f"\n=== {name} ===")
        print(json.dumps(detector.predict(text), indent=2))


if __name__ == "__main__":
    demo_inference()
