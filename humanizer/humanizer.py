"""
Humanizer — text rewriting engine with swappable generation backends.

Backends:
- Local: llama-cpp-python (GGUF models like Qwen2.5-1.5B-Instruct, Phi-3-mini)
  Windows: install via conda (recommended) or use pre-built wheels
  conda install -c conda-forge llama-cpp-python
  OR: pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
- Remote: OpenAI-compatible API (Ollama, LM Studio, OpenAI, etc.)

GPU ACCELERATION for llama.cpp (build-time flags):
- NVIDIA CUDA:        CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
- AMD ROCm (Linux):   CMAKE_ARGS="-DLLAMA_HIPBLAS=on" pip install llama-cpp-python
- AMD/Intel DirectML (Windows): CMAKE_ARGS="-DLLAMA_DIRECTML=on" pip install llama-cpp-python
- OpenCL (any GPU/iGPU): CMAKE_ARGS="-DLLAMA_OPENCL=on" pip install llama-cpp-python
- Vulkan (any GPU):   CMAKE_ARGS="-DLLAMA_VULKAN=on" pip install llama-cpp-python
- Metal (Apple):      CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python (default on macOS)

For Windows AMD/Intel integrated GPU: use DirectML build
  pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/directml

The generate() call is abstracted behind a common interface so the pipeline
doesn't care which backend is used. Configure via Humanizer(backend="local"/"api").

After humanizing, the text is automatically re-run through the detector
to show before/after AI-probability scores.
"""

import os
import sys
from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass

# Lazy import for llama-cpp-python — only imported when LlamaCppBackend is instantiated
# This allows the module to be imported even when llama-cpp-python is not installed
_LLAMA_CPP = {"Llama": None, "available": None, "error": None}

def _get_llama_cpp():
    """Lazily import llama_cpp with Windows-friendly error message."""
    if _LLAMA_CPP["Llama"] is not None or _LLAMA_CPP["error"] is not None:
        return _LLAMA_CPP["Llama"], _LLAMA_CPP["error"] is None

    try:
        from llama_cpp import Llama
        _LLAMA_CPP["Llama"] = Llama
        _LLAMA_CPP["available"] = True
        _LLAMA_CPP["error"] = None
        return Llama, True
    except ImportError as e:
        if sys.platform == "win32":
            msg = (
                "llama-cpp-python not installed or failed to build on Windows.\n"
                "  EASIEST FIX (no compiler needed):\n"
                "    conda install -c conda-forge llama-cpp-python\n"
                "  OR use pre-built wheels:\n"
                "    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu\n"
                "  OR install Visual Studio Build Tools with 'Desktop development with C++' workload."
            )
        else:
            msg = "llama-cpp-python not installed. Run: pip install llama-cpp-python"
        _LLAMA_CPP["error"] = RuntimeError(msg)
        _LLAMA_CPP["available"] = False
        raise _LLAMA_CPP["error"] from e


# Optional remote backend
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


@dataclass
class HumanizeResult:
    original_text: str
    humanized_text: str
    original_ai_prob: float
    humanized_ai_prob: float
    backend_used: str
    elapsed_seconds: float


class GenerationBackend(ABC):
    """Abstract interface for text generation backends."""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """Generate text given system + user prompts. Return the generated text."""
        pass

    @abstractmethod
    def name(self) -> str:
        """Short name for logging (e.g., 'llama-cpp', 'openai-api')."""
        pass


class LlamaCppBackend(GenerationBackend):
    """Local GGUF model via llama-cpp-python."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: int = 0,  # 0 = auto
        n_gpu_layers: int = 0,  # CPU-only by default
        verbose: bool = False,
    ):
        Llama, _ = _get_llama_cpp()  # raises RuntimeError with Windows guidance if not available

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"GGUF model not found: {model_path}")

        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )
        self.model_path = model_path

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        # llama-cpp-python uses chat format for instruct models
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        # Use chat_completion for instruct models (Qwen, Phi, etc.)
        resp = self.model.create_chat_completion(
            messages=messages,
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            max_tokens=kwargs.get("max_tokens", 512),
            stop=kwargs.get("stop", None),
        )
        return resp["choices"][0]["message"]["content"].strip()

    def name(self) -> str:
        return f"llama-cpp({os.path.basename(self.model_path)})"


class OpenAICompatibleBackend(GenerationBackend):
    """OpenAI-compatible API (local or remote)."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",  # Ollama default
        api_key: str = "not-needed",
        model: str = "qwen2.5:1.5b-instruct",
        timeout: float = 60.0,
    ):
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx not installed. Run: pip install httpx")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.client = httpx.Client(timeout=timeout)

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        resp = self.client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "max_tokens": kwargs.get("max_tokens", 512),
                "stream": False,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def name(self) -> str:
        return f"openai-compat({self.model}@{self.base_url})"

    def close(self):
        self.client.close()


# ── System prompt ──────────────────────────────────────────────────
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


# ── Humanizer pipeline ─────────────────────────────────────────────
class Humanizer:
    """
    Main humanization pipeline:
    1. Generate humanized text via configured backend
    2. Re-run through detector to get before/after AI probability
    3. Return structured result
    """

    def __init__(
        self,
        detector,
        backend: GenerationBackend,
    ):
        self.detector = detector
        self.backend = backend

    def humanize(self, text: str, **gen_kwargs) -> HumanizeResult:
        import time
        t0 = time.time()

        # Original AI probability
        orig_result = self.detector.predict(text)
        orig_ai = orig_result["ai_probability"]

        # Build user prompt
        user_prompt = f"Rewrite this text to sound more natural and human:\n\n{text}"

        # Generate
        humanized = self.backend.generate(
            HUMANIZE_SYSTEM_PROMPT,
            user_prompt,
            **gen_kwargs,
        )

        # Re-detect
        new_result = self.detector.predict(humanized)
        new_ai = new_result["ai_probability"]

        return HumanizeResult(
            original_text=text,
            humanized_text=humanized,
            original_ai_prob=orig_ai,
            humanized_ai_prob=new_ai,
            backend_used=self.backend.name(),
            elapsed_seconds=round(time.time() - t0, 2),
        )


# ── Factory / convenience ──────────────────────────────────────────
def create_humanizer(
    detector,
    backend: str = "local",
    model_path: Optional[str] = None,
    **backend_kwargs,
) -> Humanizer:
    """
    Factory to create a Humanizer with the desired backend.

    Args:
        detector: Trained AIDetector instance (from detector.detector)
        backend: "local" (llama-cpp) or "api" (OpenAI-compatible)
        model_path: Path to GGUF model (required for local backend)
        **backend_kwargs: Passed to backend constructor

    Returns:
        Humanizer instance ready to use.
    """
    if backend == "local":
        if not model_path:
            raise ValueError("model_path required for local backend")
        try:
            be = LlamaCppBackend(model_path, **backend_kwargs)
        except RuntimeError as e:
            if "Windows" in str(e) or "conda" in str(e).lower():
                raise RuntimeError(
                    f"Failed to initialize local backend: {e}\n"
                    "  On Windows, install llama-cpp-python via conda (recommended):\n"
                    "    conda install -c conda-forge llama-cpp-python\n"
                    "  Or use the API backend instead (Ollama/LM Studio):\n"
                    "    humanizer = create_humanizer(det, 'api', base_url='http://localhost:11434/v1', model='qwen2.5:1.5b-instruct')"
                ) from e
            raise
    elif backend == "api":
        be = OpenAICompatibleBackend(**backend_kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'local' or 'api'.")
    return Humanizer(detector, be)


# ── Demo / CLI ─────────────────────────────────────────────────────
def demo_humanize():
    """Quick demo - requires trained detector and (for local) a GGUF model."""
    from detector.detector import AIDetector

    det = AIDetector()
    det.load()

    # Try local backend if model exists, else show how to configure
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models",
        "qwen2.5-1.5b-instruct-q4_k_m.gguf",
    )

    if os.path.exists(model_path):
        try:
            print(f"[demo] Using local model: {model_path}")
            humanizer = create_humanizer(det, "local", model_path=model_path)
        except RuntimeError as e:
            print(f"[demo] Local backend failed: {e}")
            print("[demo] Falling back to API backend instructions...")
            _print_api_instructions()
            return
    else:
        print("[demo] No local GGUF model found.")
        print(f"       Expected at: {model_path}")
        _print_api_instructions()
        return

    test_text = (
        "In conclusion, artificial intelligence has transformed the way we interact "
        "with technology. Furthermore, it is important to note that these systems "
        "learn from vast amounts of data. Additionally, they identify patterns that "
        "humans might miss. Therefore, the potential benefits are enormous."
    )

    print("\n=== Humanizing ===")
    result = humanizer.humanize(test_text)
    print(f"Original AI%:   {result.original_ai_prob:.1f}%")
    print(f"Humanized AI%:  {result.humanized_ai_prob:.1f}%")
    print(f"Delta:          {result.humanized_ai_prob - result.original_ai_prob:+.1f}%")
    print(f"Backend:        {result.backend_used}")
    print(f"Time:           {result.elapsed_seconds}s")
    print(f"\n--- Humanized ---\n{result.humanized_text}")


def _print_api_instructions():
    """Print instructions for configuring API backend."""
    print("\n  To use the humanizer, configure an API backend:")
    print("    # Option 1: Ollama (local, runs GGUF via Ollama)")
    print("    humanizer = create_humanizer(det, 'api',")
    print("        base_url='http://localhost:11434/v1',")
    print("        model='qwen2.5:1.5b-instruct')")
    print("    # Option 2: LM Studio (local GUI)")
    print("    humanizer = create_humanizer(det, 'api',")
    print("        base_url='http://localhost:1234/v1',")
    print("        model='qwen2.5-1.5b-instruct')")
    print("    # Option 3: OpenAI-compatible remote API")
    print("    humanizer = create_humanizer(det, 'api',")
    print("        base_url='https://api.openai.com/v1',")
    print("        model='gpt-3.5-turbo', api_key='your-key')")


if __name__ == "__main__":
    demo_humanize()