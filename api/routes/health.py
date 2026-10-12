"""Health check API routes."""

from fastapi import APIRouter, Depends, Response

from api.schemas import HealthResponse
from api.dependencies import get_detector, get_humanizer
from core.detector import AIDetector

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def health_check(
    detector = Depends(get_detector),  # Use get_detector to allow None
    humanizer = Depends(get_humanizer),
    response: Response = None,
):
    """Health check endpoint for monitoring."""
    detector_ready = detector is not None and detector.ready
    humanizer_ready = humanizer is not None
    
    status = "ok" if detector_ready and humanizer_ready else "degraded"
    if response:
        response.status_code = 200 if status == "ok" else 503
    
    return HealthResponse(
        status=status,
        detector_loaded=detector is not None and detector.ready,
        humanizer_ready=humanizer is not None,
    )