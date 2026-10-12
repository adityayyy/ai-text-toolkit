"""Environment configuration using Pydantic Settings."""

import os
from pathlib import Path
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Project paths
    project_root: Path = Path(__file__).parent.parent
    models_dir: Path = Path(__file__).parent.parent / "models"
    
    # Detector settings
    detector_model_path: str = "gpt2"
    detector_device: str = "auto"  # auto, cpu, cuda, mps, dml, xpu
    detector_model_file: str = "detector_logreg.joblib"
    
    # Humanizer settings
    humanizer_backend: str = "local"  # local, api
    humanizer_model_path: Optional[str] = None
    humanizer_api_url: str = "http://localhost:11434/v1"
    humanizer_api_model: str = "qwen2.5:1.5b-instruct"
    humanizer_api_key: str = "not-needed"
    
    # Llama.cpp settings (local backend)
    llama_n_ctx: int = 2048
    llama_n_threads: int = 0  # 0 = auto
    llama_n_gpu_layers: int = 0  # CPU-only by default
    llama_verbose: bool = False
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_cors_origins: list[str] = ["*"]
    api_reload: bool = False
    
    # External provider settings (for /humanize-external proxy)
    # These are NOT loaded from env for security - passed per-request
    # but we define defaults here for documentation
    gemini_model: str = "gemini-1.5-flash"
    openai_model: str = "gpt-3.5-turbo"
    nvidia_model: str = "nvidia/nemotron-3-ultra"
    
    # Training defaults
    train_samples_per_class: int = 500
    test_samples_per_class: int = 150
    
    @property
    def detector_model_full_path(self) -> Path:
        """Full path to the trained classifier."""
        return self.models_dir / self.detector_model_file
    
    @property
    def default_gguf_path(self) -> Path:
        """Default path for GGUF model."""
        return self.models_dir / "qwen2.5-1.5b-instruct-q4_k_m.gguf"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()