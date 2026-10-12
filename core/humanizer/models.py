"""Data models for the humanizer module."""

from dataclasses import dataclass


@dataclass
class HumanizeResult:
    """Result of humanization with before/after detection scores."""
    original_text: str
    humanized_text: str
    original_ai_prob: float
    humanized_ai_prob: float
    backend_used: str
    elapsed_seconds: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API serialization."""
        return {
            "original_text": self.original_text,
            "humanized_text": self.humanized_text,
            "original_ai_probability": self.original_ai_prob,
            "humanized_ai_probability": self.humanized_ai_prob,
            "backend_used": self.backend_used,
            "elapsed_seconds": self.elapsed_seconds,
        }