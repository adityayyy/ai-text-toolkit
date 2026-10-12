# core.humanizer.backends — Generation backends

from .base import GenerationBackend
from .llama_cpp import LlamaCppBackend
from .openai_compat import OpenAICompatibleBackend

__all__ = [
    "GenerationBackend",
    "LlamaCppBackend",
    "OpenAICompatibleBackend",
]