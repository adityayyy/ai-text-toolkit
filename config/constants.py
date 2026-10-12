"""Shared constants - single source of truth for feature names, prompts, etc."""

# Feature vector layout - SINGLE SOURCE OF TRUTH
# train.py builds vectors from this list, so never reorder without retraining
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

# GPT-2 model for perplexity computation
MODEL_NAME = "gpt2"  # ~500MB download on first run, then cached locally

# Training constants
MIN_WORDS = 40  # Ignore answers shorter than this (noisy for stylometrics)

# AI-cliche phrases for per-sentence scoring
CLICHES = [
    "in conclusion", "it is important to note", "it is worth noting",
    "it can be argued", "in today's", "in summary", "firstly", "secondly",
    "furthermore", "moreover", "additionally", "nevertheless", "therefore",
    "overall", "delve", "tapestry", "navigate the", "unlock the", "foster",
    "leverage", "robust framework", "plays a crucial role", "in essence",
]

# Shared system prompt for humanization (single source of truth)
HUMANIZE_SYSTEM_PROMPT = """You are a text humanizer. Rewrite the input text to sound more natural and human-written, while preserving the exact meaning and information content.

Guidelines:
- Vary sentence length and structure significantly (short, medium, long sentences mixed)
- Use contractions naturally (don't, can't, it's, I've, you're, etc.)
- Introduce minor imperfections: occasional redundancy, conversational fillers, varied transitions
- Avoid AI-typical phrasing: "In conclusion", "It is important to note", "Furthermore", "Moreover", "Additionally", "delve", "tapestry", "navigate", "unlock", "foster", "leverage", "robust framework", "plays a crucial role", "in essence", "overall", "therefore", "nevertheless", "firstly", "secondly"
- Don't use overly balanced parallel structure in every sentence
- Keep the original meaning 100% intact — no hallucination, no added facts
- Write like a real person explaining something to a friend: natural flow, personal voice
- Preserve formatting (paragraphs, lists, code blocks) if present

Return ONLY the rewritten text, no explanations or meta-commentary."""

import re

# Regex constants for text processing
_WORD_RE = re.compile(r"[A-Za-z]")
_STRIP_RE = re.compile(r"[.,!?;:'\"()\[\]{}-]")

# String version of strip characters for str.strip()
_STRIP_CHARS = ".,!?;:'\"()[]{}-"

# External provider configurations
EXTERNAL_PROVIDERS = {
    "gemini": {
        "name": "Gemini 1.5 Flash",
        "url_template": "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}",
        "model": "gemini-1.5-flash",
    },
    "openai": {
        "name": "GPT-3.5 Turbo",
        "url": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-3.5-turbo",
    },
    "nvidia": {
        "name": "Nemotron 3 Ultra",
        "url": "https://integrate.api.nvidia.com/v1/chat/completions",
        "model": "nvidia/nemotron-3-ultra",
    },
}