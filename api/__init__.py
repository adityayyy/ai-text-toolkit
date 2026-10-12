# api — FastAPI backend package

from .main import app, create_app
from .schemas import (
    DetectRequest,
    DetectResponse,
    HumanizeRequest,
    HumanizeResponse,
    ExternalHumanizeRequest,
    ExternalHumanizeResponse,
    HealthResponse,
)
from .dependencies import init_dependencies, cleanup_dependencies

__all__ = [
    "app",
    "create_app",
    "DetectRequest",
    "DetectResponse",
    "HumanizeRequest",
    "HumanizeResponse",
    "ExternalHumanizeRequest",
    "ExternalHumanizeResponse",
    "HealthResponse",
    "init_dependencies",
    "cleanup_dependencies",
]