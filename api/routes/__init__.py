# api.routes — All API route modules

from .detect import router as detect_router
from .humanize import router as humanize_router
from .external import router as external_router
from .health import router as health_router

__all__ = [
    "detect_router",
    "humanize_router",
    "external_router",
    "health_router",
]