"""
FastAPI application factory for ai-text-toolkit.

Run:
    python -m api.main
    # or: uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is on path for absolute imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.routes import detect_router, humanize_router, external_router, health_router
from api.dependencies import init_dependencies, cleanup_dependencies
from config.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown."""
    # Startup
    init_dependencies()
    yield
    # Shutdown
    cleanup_dependencies()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="ai-text-toolkit API",
        description="Local AI text detector + humanizer",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS for local frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health_router)
    app.include_router(detect_router)
    app.include_router(humanize_router)
    app.include_router(external_router)

    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )