"""Abstract base class for generation backends."""

from abc import ABC, abstractmethod


class GenerationBackend(ABC):
    """Abstract interface for text generation backends."""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """Generate text given system + user prompts. Return the generated text."""
        pass

    @abstractmethod
    def name(self) -> str:
        """Short name for logging (e.g., 'llama-cpp', 'openai-api')."""
        pass

    def close(self):
        """Optional cleanup for backends with persistent connections."""
        pass