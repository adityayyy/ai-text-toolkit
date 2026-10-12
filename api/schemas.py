"""Pydantic request/response schemas for the API."""

from pydantic import BaseModel, Field
from typing import Literal, Optional
from typing_extensions import Annotated


# ── Detection ─────────────────────────────────────────────────────────
class DetectRequest(BaseModel):
    text: Annotated[str, Field(..., min_length=1, max_length=10000, description="Text to analyze")]


class SentenceScore(BaseModel):
    sentence: str
    ai_score: float


class DetectResponse(BaseModel):
    ai_probability: float
    is_ai: bool
    label: str
    feature_breakdown: dict
    sentence_scores: list[SentenceScore]
    top_repeated_ngrams: list[tuple[str, int]]
    n_tokens: int
    elapsed_seconds: float


# ── Humanization ──────────────────────────────────────────────────────
class HumanizeRequest(BaseModel):
    text: Annotated[str, Field(..., min_length=1, max_length=10000, description="Text to humanize")]
    re_detect: bool = Field(default=True, description="Re-run detector on humanized text")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=512, ge=1, le=2048)


class HumanizeResponse(BaseModel):
    original_text: str
    humanized_text: str
    original_ai_probability: float
    humanized_ai_probability: Optional[float] = None
    backend_used: str
    elapsed_seconds: float


# ── External provider proxy ───────────────────────────────────────────
class ExternalHumanizeRequest(BaseModel):
    text: Annotated[str, Field(..., min_length=1, max_length=10000, description="Text to humanize")]
    provider: Literal["gemini", "openai", "nvidia"]
    api_key: Annotated[str, Field(..., description="API key for the external provider (never logged/stored)")]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=512, ge=1, le=2048)


class ExternalHumanizeResponse(BaseModel):
    humanized_text: str
    provider: str
    elapsed_seconds: float


# ── Health ────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    detector_loaded: bool
    humanizer_ready: bool