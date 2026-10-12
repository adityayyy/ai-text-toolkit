"""Humanization API routes."""

import asyncio
from fastapi import APIRouter, Depends, HTTPException, status

from api.schemas import HumanizeRequest, HumanizeResponse
from api.dependencies import get_ready_detector, get_ready_humanizer
from core.detector import AIDetector
from core.humanizer import Humanizer

router = APIRouter(prefix="/humanize", tags=["Humanization"])


@router.post("", response_model=HumanizeResponse)
async def humanize_text(
    request: HumanizeRequest,
    detector: AIDetector = Depends(get_ready_detector),
    humanizer: Humanizer = Depends(get_ready_humanizer),
):
    """
    Humanize text using the configured backend.
    
    Returns:
    - original_text: the input text
    - humanized_text: the rewritten text
    - original_ai_probability: AI% before humanization
    - humanized_ai_probability: AI% after (if re_detect=true)
    - backend_used: which generation backend was used
    - elapsed_seconds: total processing time
    """
    try:
        # Run synchronous humanize in thread pool to avoid blocking event loop
        result = await asyncio.to_thread(
            humanizer.humanize,
            request.text,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )

        return HumanizeResponse(
            original_text=result.original_text,
            humanized_text=result.humanized_text,
            original_ai_probability=result.original_ai_prob,
            humanized_ai_probability=result.humanized_ai_prob if request.re_detect else None,
            backend_used=result.backend_used,
            elapsed_seconds=result.elapsed_seconds,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Humanization failed: {str(e)}",
        )