"""
FastAPI backend for ai-text-toolkit.

Endpoints:
- POST /detect       → AI detection result (ai_probability, features, sentence scores)
- POST /humanize     → Humanize text + optional re-detect (before/after AI%)

Run:
    python -m api.main
    # or: uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

import asyncio
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
import httpx

# Ensure project root is on path for absolute imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from detector.detector import AIDetector
from humanizer.humanizer import (
    Humanizer,
    GenerationBackend,
    LlamaCppBackend,
    OpenAICompatibleBackend,
    create_humanizer,
)


# ── Global singletons (initialized on startup) ─────────────────────
detector: AIDetector = None
humanizer: Humanizer = None


# ── Pydantic schemas ───────────────────────────────────────────────
class DetectRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze")


class SentenceScore(BaseModel):
    sentence: str
    ai_score: float


class DetectResponse(BaseModel):
    ai_probability: float
    is_ai: bool
    label: str
    feature_breakdown: dict
    sentence_scores: list[SentenceScore]
    # ngram_repetition_top() returns list[tuple[str, int]] = [(phrase, count)]
    top_repeated_ngrams: list[tuple[str, int]]
    n_tokens: int
    elapsed_seconds: float


class HumanizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to humanize")
    re_detect: bool = Field(default=True, description="Re-run detector on humanized text")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=512, ge=1, le=2048)


class HumanizeResponse(BaseModel):
    original_text: str
    humanized_text: str
    original_ai_probability: float
    humanized_ai_probability: float = None
    backend_used: str
    elapsed_seconds: float


# ── External provider proxy schemas ────────────────────────────────────
class ExternalHumanizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to humanize")
    provider: Literal["gemini", "openai", "nvidia"]
    api_key: str = Field(..., description="API key for the external provider (never logged/stored)")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=512, ge=1, le=2048)


class ExternalHumanizeResponse(BaseModel):
    humanized_text: str
    provider: str
    elapsed_seconds: float


# Helper: call external provider via our proxy (server-to-provider, avoids CORS)
# CRITICAL: api_key is used ONLY in-memory for this single outbound request.
# It is NEVER logged, stored, or persisted beyond the scope of this handler.
async def call_external_provider(provider: str, prompt: str, api_key: str, temperature: float, max_tokens: int) -> str:
    if provider == "gemini":
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "topP": 0.9, "maxOutputTokens": max_tokens}
        }
        headers = {"Content-Type": "application/json"}
    elif provider == "openai":
        url = "https://api.openai.com/v1/chat/completions"
        body = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": HUMANIZE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": max_tokens
        }
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    elif provider == "nvidia":
        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        body = {
            "model": "nvidia/nemotron-3-ultra",
            "messages": [
                {"role": "system", "content": HUMANIZE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": max_tokens
        }
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    else:
        raise HTTPException(400, f"Unknown provider: {provider}")

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


# ── Shared system prompt (must be at module level for proxy) ───────────
HUMANIZE_SYSTEM_PROMPT = """You are a text humanizer. Rewrite the input text to sound more natural and human-written, while preserving the exact meaning and information content.

Guidelines:
- Vary sentence length and structure significantly (short, medium, long sentences mixed)
- Use contractions naturally (don't, can't, it's, I've, you're, etc.)
- Introduce minor imperfections: occasional redundancy, conversational fillers, varied transitions
- Avoid AI-typical phrasing: "In conclusion", "It is important to note", "Furthermore", "Moreover", "Additionally", "delve", "tapestry", "navigate", "unlock", "foster", "leverage", "robust framework", "plays a crucial role", "in essence", "overall", "therefore", "nevertheless", "firstly", "secondly"
- Don't use overly balanced parallel structure in every sentence
- Keep the original meaning 100% intact — no hallucination, no added facts
- Write like a real person explaining something to a friend: natural flow, personal voice
- Preserve formatting (paragraphs, lists, code blocks) if present

Return ONLY the rewritten text, no explanations or meta-commentary."""


# ── Startup / shutdown ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector, humanizer

    # Load detector
    detector = AIDetector()
    try:
        detector.load()
        print("[api] Detector loaded")
    except FileNotFoundError:
        print("[api] WARNING: No trained detector found. Run: python -m detector.train")
        print("[api] Detector will not work until trained.")

    # Initialize humanizer (local backend if model exists, else None)
    # Configure via env vars or edit here:
    # HUMANIZER_BACKEND=local|api
    # HUMANIZER_MODEL_PATH=/path/to/model.gguf
    # HUMANIZER_API_URL=http://localhost:11434/v1
    # HUMANIZER_API_MODEL=qwen2.5:1.5b-instruct
    backend = os.getenv("HUMANIZER_BACKEND", "local")
    model_path = os.getenv(
        "HUMANIZER_MODEL_PATH",
        os.path.join(PROJECT_ROOT, "models", "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
    )

    if backend == "local":
        try:
            humanizer = create_humanizer(detector, "local", model_path=model_path)
            print(f"[api] Humanizer ready (local: {model_path})")
        except (RuntimeError, FileNotFoundError) as e:
            # Windows-friendly error with conda guidance
            if "conda" in str(e).lower() or "Windows" in str(e) or isinstance(e, FileNotFoundError):
                if isinstance(e, FileNotFoundError):
                    print(f"[api] Humanizer local backend: model not found at {model_path}")
                else:
                    print(f"[api] Humanizer local backend unavailable:")
                    print(f"  {e}")
                print("  To fix on Windows, install llama-cpp-python via conda:")
                print("    conda install -c conda-forge llama-cpp-python")
                print("  Or download a GGUF model to models/ (e.g., Qwen2.5-1.5B-Instruct Q4_K_M)")
                print("  Or use the API backend instead (set HUMANIZER_BACKEND=api):")
                print("    $env:HUMANIZER_API_URL='http://localhost:11434/v1'")
                print("    $env:HUMANIZER_API_MODEL='qwen2.5:1.5b-instruct'")
            else:
                print(f"[api] Humanizer local backend unavailable: {e}")
            humanizer = None
    elif backend == "api":
        api_url = os.getenv("HUMANIZER_API_URL", "http://localhost:11434/v1")
        api_model = os.getenv("HUMANIZER_API_MODEL", "qwen2.5:1.5b-instruct")
        try:
            humanizer = create_humanizer(detector, "api", base_url=api_url, model=api_model)
            print(f"[api] Humanizer ready (API: {api_model}@{api_url})")
        except Exception as e:
            print(f"[api] Humanizer API backend unavailable: {e}")
            humanizer = None

    yield

    # Cleanup
    if humanizer and hasattr(humanizer.backend, "close"):
        humanizer.backend.close()


# ── FastAPI app ────────────────────────────────────────────────────
app = FastAPI(
    title="ai-text-toolkit API",
    description="Local AI text detector + humanizer",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Local only; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ──────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "detector_loaded": detector is not None and detector.ready,
        "humanizer_ready": humanizer is not None,
    }


@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest):
    if not detector or not detector.ready:
        raise HTTPException(503, "Detector not trained. Run: python -m detector.train")
    result = detector.predict(req.text)
    return DetectResponse(**result)


@app.post("/humanize", response_model=HumanizeResponse)
async def humanize(req: HumanizeRequest):
    if not humanizer:
        raise HTTPException(
            503,
            "Humanizer not configured. Set HUMANIZER_BACKEND and provide model/API.",
        )
    if not detector or not detector.ready:
        raise HTTPException(503, "Detector not trained (needed for re-detect).")

    # Run synchronous humanize in thread pool to avoid blocking event loop
    result = await asyncio.to_thread(
        humanizer.humanize,
        req.text,
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_tokens,
    )

    resp = HumanizeResponse(
        original_text=result.original_text,
        humanized_text=result.humanized_text,
        original_ai_probability=result.original_ai_prob,
        humanized_ai_probability=result.humanized_ai_prob if req.re_detect else None,
        backend_used=result.backend_used,
        elapsed_seconds=result.elapsed_seconds,
    )
    return resp


# ── External provider proxy endpoint ───────────────────────────────────
# Proxies humanize requests to external providers (Gemini/OpenAI/NVIDIA)
# to avoid CORS issues. The API key is passed in the request body and is
# used ONLY in-memory for the single outbound request — NEVER logged,
# stored, or persisted beyond this handler's scope.
@app.post("/humanize-external", response_model=ExternalHumanizeResponse)
async def humanize_external(req: ExternalHumanizeRequest):
    # Build the prompt with system instructions + user text
    prompt = f"{HUMANIZE_SYSTEM_PROMPT}\n\nRewrite this text to sound more natural and human:\n\n{req.text}"

    import time
    t0 = time.time()
    humanized = await call_external_provider(
        provider=req.provider,
        prompt=prompt,
        api_key=req.api_key,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )
    elapsed = time.time() - t0

    if not humanized:
        raise HTTPException(502, f"Empty response from {req.provider}")

    return ExternalHumanizeResponse(
        humanized_text=humanized,
        provider=req.provider,
        elapsed_seconds=round(elapsed, 2),
    )


# ── Entrypoint ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)