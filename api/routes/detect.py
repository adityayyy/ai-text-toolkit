"""Detection API routes."""

from fastapi import APIRouter, Depends, HTTPException, status

from api.schemas import DetectRequest, DetectResponse
from api.dependencies import get_ready_detector
from core.detector import AIDetector

router = APIRouter(prefix="/detect", tags=["Detection"])


@router.post("", response_model=DetectResponse)
def detect_text(
    request: DetectRequest,
    detector: AIDetector = Depends(get_ready_detector),
):
    """
    Analyze text for AI-generated content.
    
    Returns:
    - ai_probability: 0-100% probability the text is AI-generated
    - is_ai: boolean (threshold at 50%)
    - label: "AI-generated" or "Likely human"
    - feature_breakdown: all 14 underlying signals
    - sentence_scores: per-sentence AI-likeness scores
    - top_repeated_ngrams: most repeated 3-grams
    - n_tokens: number of GPT-2 tokens processed
    - elapsed_seconds: processing time
    """
    try:
        result = detector.predict(request.text)
        return DetectResponse(**result.to_dict())
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detection failed: {str(e)}",
        )