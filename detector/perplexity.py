"""
Perplexity + burstiness computation using GPT-2 small (fully local, CPU/GPU).

Why GPT-2 small (~124M params)?
- Small enough to run fast on a CPU-only laptop (~0.3-0.5s per forward pass
  for a 500-word input).
- Perplexity under a generic LM is a strong signal: AI-generated text tends to
  be LOW-perplexity (the model finds it very predictable) while human text is
  higher and more uneven.

How it works (one forward pass per document):
1. Tokenize the text and ask GPT-2 for logits at every position.
2. Convert logits to log-probabilities, then pick out the log-probability that
   GPT-2 assigned to the token that ACTUALLY appears next  ->  per-token
   log-likelihood.
3. exp(-mean(log-likelihood)) = overall perplexity.
4. Using character offsets from the tokenizer, group tokens into sentences and
   average per sentence  ->  per-sentence perplexity.

Burstiness = variance of per-sentence perplexity across the document.
Human text: HIGH variance (some sentences predictable, some complex).
AI text:    LOW variance (uniformly smooth predictability).

SPEED/QUALITY TRADEOFF (swap points):
- model_name="gpt2"  ->  fastest CPU option. Swap to "gpt2-medium"/"distilgpt2"
  etc. by changing one string; larger models give better perplexity estimates
  but run slower.
- Single 1024-token window (GPT-2's context limit): documents longer than
  ~750 words get truncated for the perplexity signal (ponytail: fine for the
  500-word target; chunk long docs into windows if you need more).

GPU SUPPORT (auto-detects):
- CUDA (NVIDIA) — standard
- MPS (Apple Silicon) — standard
- DirectML (AMD/Intel GPU on Windows) — install `torch-directml`, use device="dml"
- Intel XPU (Intel Arc/iGPU) — install `intel-extension-for-pytorch`, use device="xpu"
- ROCm (AMD GPU on Linux) — standard CUDA API
- Auto-detects CUDA/MPS; falls back to CPU
- Pass device="cpu"/"cuda"/"mps"/"dml"/"xpu" to force
"""

import math
import re

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

MODEL_NAME = "gpt2"  # ~500MB download on first run, then cached locally


def _resolve_device(device: str) -> str:
    """Resolve device string - auto-detect CUDA/MPS/DirectML/XPU if 'auto'."""
    if device and device != "auto":
        return device
    
    # Standard CUDA (NVIDIA)
    if torch.cuda.is_available():
        return "cuda"
    
    # Apple Silicon MPS
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    
    # DirectML (AMD/Intel GPU on Windows) - requires torch-directml
    try:
        import torch_directml
        if torch_directml.is_available():
            return "dml"
    except ImportError:
        pass
    
    # Intel XPU (Intel Arc/iGPU) - requires intel-extension-for-pytorch
    try:
        import intel_extension_for_pytorch as ipex
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
    except ImportError:
        pass
    
    return "cpu"


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping the ending punctuation."""
    return [m.group().strip()
            for m in re.finditer(r"\s*[^.!?]+[.!?]*", text)
            if m.group().strip()]


def split_sentences_with_spans(text: str) -> list[tuple[int, int, str]]:
    """Same as split_sentences but also returns each sentence's (start, end)
    character offsets, so tokens can be assigned to sentences."""
    spans = []
    for m in re.finditer(r"\s*[^.!?]+[.!?]*", text):
        s = m.group().strip()
        if not s:
            continue
        spans.append((m.start(), m.start() + m.end(), s))
    return spans


class PerplexityComputer:
    """Loads GPT-2 small once; computes overall + per-sentence perplexity."""

    def __init__(self, model_name: str = MODEL_NAME, device: str = "auto"):
        self.device = _resolve_device(device)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print(f"[perplexity] Using device: {self.device}")

    @torch.no_grad()
    def analyze(self, text: str) -> dict:
        """
        One forward pass returns everything we need from the LM:
            {"perplexity": float,
             "sentence_ppls": [(sentence, perplexity), ...],
             "n_tokens": int}
        """
        spans = split_sentences_with_spans(text)
        if not text or not text.strip():
            return {"perplexity": 0.0, "sentence_ppls": [], "n_tokens": 0}

        enc = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,   # lets us map tokens -> sentences
            add_special_tokens=False,
            truncation=True,
            max_length=1024,               # GPT-2 context window (see header)
        )
        input_ids = enc["input_ids"].to(self.device)
        offsets = enc["offset_mapping"][0].tolist()

        logits = self.model(input_ids).logits[0]            # (seq_len, vocab)
        log_probs = torch.log_softmax(logits, dim=-1)

        # Token i is PREDICTED by position i-1, so:
        # loglik[i] = log_probs[i-1, input_ids[i]]
        target_ids = input_ids[0]
        logliks = log_probs[:-1].gather(1, target_ids[1:].unsqueeze(1)).squeeze(1)
        token_offsets = offsets[1:]                          # aligns with logliks

        n_tokens = int(logliks.numel())
        if n_tokens == 0:
            # Text too short to score (e.g. 1 token) - neutral output
            return {"perplexity": 0.0,
                    "sentence_ppls": [(s, 0.0) for _, _, s in spans],
                    "n_tokens": n_tokens}

        overall_ppl = math.exp(-float(logliks.mean().item()))

        # Walk tokens in order and bucket them into sentences via char offsets.
        # (Whitespace tokens belong to the following sentence's span, which is
        # how the spans above are cut, so every token lands in exactly one
        # sentence in practice; rare strays are simply skipped.)
        sentence_ppls = []
        cursor = 0
        for s_start, s_end, s_text in spans:
            vals = []
            while cursor < n_tokens and token_offsets[cursor][0] < s_end:
                if token_offsets[cursor][0] >= s_start:
                    vals.append(float(logliks[cursor].item()))
                cursor += 1
            if vals:
                sentence_ppls.append((s_text, math.exp(-float(np.mean(vals)))))
            else:
                # Sentence produced no tokens (only possible for the truncated
                # tail of a very long doc): reuse the document average.
                sentence_ppls.append((s_text, overall_ppl))

        return {"perplexity": overall_ppl,
                "sentence_ppls": sentence_ppls,
                "n_tokens": n_tokens}


def compute_burstiness(sentence_ppls: list[tuple[str, float]]) -> float:
    """
    Burstiness = variance of per-sentence perplexity (higher -> more human).
    Needs at least 2 sentences; shorter texts return 0.0 (neutral-ish).
    """
    vals = [p for _, p in sentence_ppls]
    if len(vals) < 2:
        return 0.0
    return float(np.var(vals))
