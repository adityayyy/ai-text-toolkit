"""External provider proxy routes - avoids CORS by proxying through our backend."""

import time
import httpx
from fastapi import APIRouter, HTTPException, status, Depends
from typing import Literal

from api.schemas import ExternalHumanizeRequest, ExternalHumanizeResponse
from api.dependencies import get_ready_detector
from config.constants import HUMANIZE_SYSTEM_PROMPT, EXTERNAL_PROVIDERS
from core.detector import AIDetector

router = APIRouter(prefix="/humanize-external", tags=["External Providers"])


# CRITICAL: api_key is used ONLY in-memory for this single outbound request.
# It is NEVER logged, stored, or persisted beyond the scope of this handler.
async def call_external_provider(
    provider: Literal["gemini", "openai", "nvidia"],
    prompt: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Call external provider API."""
    provider_config = EXTERNAL_PROVIDERS[provider]
    
    if provider == "gemini":
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{provider_config['model']}:generateContent?key={api_key}"
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "topP": 0.9, "maxOutputTokens": max_tokens}
        }
        headers = {"Content-Type": "application/json"}
    elif provider == "openai":
        url = provider_config["url"]
        body = {
            "model": provider_config["model"],
            "messages": [
                {"role": "system", "content": HUMANIZE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "top_p": 0.9,
            "max_tokens": max_tokens
        }
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    elif provider == "nvidia":
        url = provider_config["url"]
        body = {
            "model": provider_config["model"],
            "messages": [
                {"role": "system", "content": HUMANIZE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "top_p": 0.9,
            "max_tokens": max_tokens
        }
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    else:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Unknown provider: {provider}")

    async with httpx.AsyncClient(timeout=60.0) as client:
        res = await client.post(url, json=body, headers=headers)
        if not res.is_success:
            err = res.json() if res.headers.get("content-type", "").startswith("application/json") else {}
            msg = err.get("error", {}).get("message") or err.get("message") or f"{provider} API error: {res.status_code}"
            raise HTTPException(res.status_code, msg)
        data = res.json()
    
    if provider == "gemini":
        return data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
    else:
        return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()


@router.post("", response_model=ExternalHumanizeResponse)
async def humanize_external(
    request: ExternalHumanizeRequest,
    detector: AIDetector = Depends(get_ready_detector),
):
    """
    Proxy humanize requests to external providers (Gemini/OpenAI/NVIDIA)
    to avoid CORS issues. The API key is passed in the request body and is
    used ONLY in-memory for the single outbound request — NEVER logged,
    stored, or persisted beyond this handler's scope.
    """
    # Build the prompt with system instructions + user text
    prompt = f"{HUMANIZE_SYSTEM_PROMPT}\n\nRewrite this text to sound more natural and human:\n\n{request.text}"

    t0 = time.time()
    humanized = await call_external_provider(
        provider=request.provider,
        prompt=prompt,
        api_key=request.api_key,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )
    elapsed = time.time() - t0

    if not humanized:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Empty response from {request.provider}")

    return ExternalHumanizeResponse(
        humanized_text=humanized,
        provider=request.provider,
        elapsed_seconds=round(elapsed, 2),
    )