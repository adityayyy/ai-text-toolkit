# humanizer — text rewriting engine (Part 2)
# Backends: llama-cpp-python (local GGUF) + OpenAI-compatible API (swappable)

from .humanizer import (
    Humanizer,
    GenerationBackend,
    LlamaCppBackend,
    OpenAICompatibleBackend,
    HumanizeResult,
    create_humanizer,
    HUMANIZE_SYSTEM_PROMPT,
)

__all__ = [
    "Humanizer",
    "GenerationBackend",
    "LlamaCppBackend",
    "OpenAICompatibleBackend",
    "HumanizeResult",
    "create_humanizer",
    "HUMANIZE_SYSTEM_PROMPT",
]