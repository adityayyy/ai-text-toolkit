"""FastAPI dependencies for detector and humanizer instances."""

from fastapi import Depends, HTTPException, status
from typing import Optional

from core.detector import AIDetector
from core.humanizer import Humanizer, create_humanizer
from config.settings import get_settings

from core.exceptions import (
    DetectorNotReadyError,
    HumanizerNotConfiguredError,
    BackendNotAvailableError,
    ModelNotFoundError,
)


# Global instances (initialized on startup)
_detector: Optional[AIDetector] = None
_humanizer = None


def get_detector(required: bool = True) -> Optional[AIDetector]:
    """Get the detector instance."""
    if _detector is None:
        if required:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Detector not initialized. Server may be starting up.",
            )
        return None
    return _detector


def get_ready_detector() -> AIDetector:
    """Get the detector instance, ensuring it's ready (model loaded)."""
    detector = get_detector(required=True)
    if not detector.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Detector not trained. Run: python -m core.detector.training",
        )
    return detector


def get_humanizer() -> Optional[object]:
    """Get the humanizer instance (may be None if not configured)."""
    return _humanizer


def get_ready_humanizer() -> object:
    """Get the humanizer instance, ensuring it's configured."""
    if _humanizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Humanizer not configured. Set HUMANIZER_BACKEND and provide model/API.",
        )
    return _humanizer


def init_dependencies():
    """Initialize global dependencies on startup."""
    global _detector, _humanizer
    settings = get_settings()
    
    # Initialize detector
    _detector = AIDetector(
        model_name=settings.detector_model_path,
        device=settings.detector_device
    )
    try:
        _detector.load(str(settings.detector_model_full_path))
        print("[api] Detector loaded successfully")
    except FileNotFoundError:
        print("[api] WARNING: No trained detector found. Run: python -m core.detector.training")
        print("[api] Detector will not work until trained.")
    
    # Initialize humanizer
    _humanizer = None
    backend = settings.humanizer_backend
    model_path = settings.humanizer_model_path or str(settings.default_gguf_path)
    
    if backend == "local":
        try:
            _humanizer = create_humanizer(
                _detector, 
                "local", 
                model_path=model_path,
                n_ctx=settings.llama_n_ctx,
                n_threads=settings.llama_n_threads,
                n_gpu_layers=settings.llama_n_gpu_layers,
                verbose=settings.llama_verbose,
            )
            print(f"[api] Humanizer ready (local: {model_path})")
        except Exception as e:
            print(f"[api] Humanizer local backend unavailable: {e}")
            _print_humanizer_help()
            _humanizer = None
    elif backend == "api":
        try:
            _humanizer = create_humanizer(
                _detector,
                "api",
                base_url=settings.humanizer_api_url,
                model=settings.humanizer_api_model,
                api_key=settings.humanizer_api_key,
            )
            print(f"[api] Humanizer ready (API: {settings.humanizer_api_model}@{settings.humanizer_api_url})")
        except Exception as e:
            print(f"[api] Humanizer API backend unavailable: {e}")
            _humanizer = None


def cleanup_dependencies():
    """Cleanup on shutdown."""
    global _humanizer
    if _humanizer and hasattr(_humanizer, 'backend') and hasattr(_humanizer.backend, 'close'):
        _humanizer.backend.close()


def _print_humanizer_help():
    """Print humanizer setup help."""
    settings = get_settings()
    if isinstance(settings.humanizer_model_path, str) and settings.humanizer_model_path:
        print(f"  Model expected at: {settings.humanizer_model_path}")
    print("  To fix on Windows, install llama-cpp-python via conda:")
    print("    conda install -c conda-forge llama-cpp-python")
    print("  Or download a GGUF model to models/ (e.g., Qwen2.5-1.5B-Instruct Q4_K_M)")
    print("  Or use the API backend instead (set HUMANIZER_BACKEND=api):")
    print("    $env:HUMANIZER_API_URL='http://localhost:11434/v1'")
    print("    $env:HUMANIZER_API_MODEL='qwen2.5:1.5b-instruct'")