"""
Stylometric features computed directly from text - no ML model needed.

These are cheap (<1ms) surface statistics that differ between human and
AI-generated writing:
- Average sentence length + variance (AI: uniform, medium-length sentences)
- Type-token ratio (AI: often *lower* unique-word share in long passages)
- Punctuation ratio (AI: predictable comma/period pattern)
- Bigram uniqueness (AI: repeated connective phrases -> more repeated bigrams)
- Average word length
- POS distribution via nltk (AI: heavier on nouns/adjectives, fewer pronouns)

Also hosts the AI-CLICHE phrase list, used by the per-sentence scorer in
detector.py ("in conclusion", "it is important to note", "delve", ...).
"""

import re
from collections import Counter

import numpy as np

# AI-cliche phrases: small, measurable, directly greppable. Extend freely.
CLICHES = [
    "in conclusion", "it is important to note", "it is worth noting",
    "it can be argued", "in today's", "in summary", "firstly", "secondly",
    "furthermore", "moreover", "additionally", "nevertheless", "therefore",
    "overall", "delve", "tapestry", "navigate the", "unlock the", "foster",
    "leverage", "robust framework", "plays a crucial role", "in essence",
]

_WORD_RE = re.compile(r"[A-Za-z]")
_STRIP_RE = ".,!?;:'\"()[]{}-"


def split_sentences(text: str) -> list[str]:
    """Split into sentences, keeping ending punctuation (mirror of perplexity.py)."""
    return [m.group().strip()
            for m in re.finditer(r"\s*[^.!?]+[.!?]*", text)
            if m.group().strip()]


def _pos_distribution(words: list[str]) -> dict:
    """
    Proportion of nouns / verbs / adjectives / adverbs / pronouns / other.
    Self-healing: downloads the small nltk tagger data (~5MB) on first use.
    """
    try:
        import nltk
        try:
            nltk.pos_tag(["probe"])          # raises LookupError if data missing
        except LookupError:
            # Modern nltk (>=3.8.2) uses the _eng resource names; grab both
            # old + new names so any version works.
            for pkg in ("averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"):
                nltk.download(pkg, quiet=True)
        tags = [tag for _, tag in nltk.pos_tag(words)]
    except Exception:
        # nltk unavailable (fully offline): zeroed POS features.
        # First run needs internet anyway (GPT-2 download), so this is rare.
        return {"pos_noun": 0.0, "pos_verb": 0.0, "pos_adj": 0.0,
                "pos_adv": 0.0, "pos_pron": 0.0, "pos_other": 0.0}

    n = len(tags) or 1

    def frac(prefixes: tuple) -> float:
        return sum(1 for t in tags if t.startswith(prefixes)) / n

    covered = (frac(("NN",)) + frac(("VB",)) + frac(("JJ",))
               + frac(("RB",)) + frac(("PRP", "WP")))
    return {
        "pos_noun": frac(("NN",)),
        "pos_verb": frac(("VB",)),
        "pos_adj": frac(("JJ",)),
        "pos_adv": frac(("RB",)),
        "pos_pron": frac(("PRP", "WP")),
        "pos_other": 1.0 - covered,
    }


def compute_stylometric_features(text: str) -> dict:
    """Compute all stylometric features. Returns a fixed-key dict matching
    FEATURE_NAMES[2:] in detector.py."""
    sentences = split_sentences(text)
    tokens = text.split()

    zeros = {
        "avg_sentence_length": 0.0, "sentence_length_var": 0.0,
        "type_token_ratio": 0.0, "punctuation_ratio": 0.0,
        "bigram_uniqueness": 1.0, "avg_word_length": 0.0,
        "pos_noun": 0.0, "pos_verb": 0.0, "pos_adj": 0.0,
        "pos_adv": 0.0, "pos_pron": 0.0, "pos_other": 0.0,
    }
    if not tokens:
        return zeros

    # Words = tokens that actually contain letters, stripped of punctuation.
    words = [w.lower().strip(_STRIP_RE) for w in tokens if _WORD_RE.search(w)]
    words = [w for w in words if w]
    if not words:
        return zeros

    # Sentence length stats
    sent_lens = [len(s.split()) for s in sentences] or [len(tokens)]
    avg_sent_len = float(np.mean(sent_lens))
    sent_len_var = float(np.var(sent_lens))

    # Type-token ratio (vocabulary diversity)
    ttr = len(set(words)) / len(words)

    # Punctuation ratio (punctuation-bearing tokens / all tokens)
    punct_ratio = sum(1 for t in tokens if re.search(r"[^\w\s]", t)) / len(tokens)

    # Bigram uniqueness: 1.0 = no repeated bigrams, lower = more repetition
    bigrams = list(zip(words, words[1:]))
    bigram_unique = (len(set(bigrams)) / len(bigrams)) if bigrams else 1.0

    # Average word length
    avg_word_len = float(np.mean([len(w) for w in words]))

    return {
        "avg_sentence_length": avg_sent_len,
        "sentence_length_var": sent_len_var,
        "type_token_ratio": ttr,
        "punctuation_ratio": punct_ratio,
        "bigram_uniqueness": bigram_unique,
        "avg_word_length": avg_word_len,
        **_pos_distribution(words),
    }


def cliche_hits(text: str) -> int:
    """Count AI-cliche phrases in the text (case-insensitive)."""
    low = text.lower()
    return sum(1 for c in CLICHES if c in low)


def ngram_repetition_top(text: str, n: int = 3, k: int = 3) -> list[tuple[str, int]]:
    """Most repeated n-grams (diagnostic helper for the results panel)."""
    words = [w.lower().strip(_STRIP_RE) for w in text.split() if _WORD_RE.search(w)]
    grams = Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    return [((" ".join(g)), c) for g, c in grams.most_common(k) if c > 1]
