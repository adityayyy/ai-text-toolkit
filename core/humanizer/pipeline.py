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

import time
from typing import Optional

from config.constants import HUMANIZE_SYSTEM_PROMPT
from config.settings import get_settings

from core.humanizer.backends import GenerationBackend, LlamaCppBackend, OpenAICompatibleBackend
from core.humanizer.models import HumanizeResult
from core.exceptions import HumanizerNotConfiguredError, BackendNotAvailableError, ModelNotFoundError


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
        t0 = time.time()

        # Original AI probability
        orig_result = self.detector.predict(text)
        orig_ai = orig_result.ai_probability

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
        new_ai = new_result.ai_probability

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
        detector: Trained AIDetector instance (from core.detector)
        backend: "local" (llama-cpp) or "api" (OpenAI-compatible)
        model_path: Path to GGUF model (required for local backend)
        **backend_kwargs: Passed to backend constructor

    Returns:
        Humanizer instance ready to use.
    """
    settings = get_settings()
    
    if backend == "local":
        if not model_path:
            model_path = str(settings.default_gguf_path)
        try:
            be = LlamaCppBackend(
                model_path,
                n_ctx=backend_kwargs.pop("n_ctx", settings.llama_n_ctx),
                n_threads=backend_kwargs.pop("n_threads", settings.llama_n_threads),
                n_gpu_layers=backend_kwargs.pop("n_gpu_layers", settings.llama_n_gpu_layers),
                verbose=backend_kwargs.pop("verbose", settings.llama_verbose),
            )
        except BackendNotAvailableError as e:
            if "Windows" in str(e) or "conda" in str(e).lower():
                raise BackendNotAvailableError(
                    f"Failed to initialize local backend: {e}\n"
                    "  On Windows, install llama-cpp-python via conda (recommended):\n"
                    "    conda install -c conda-forge llama-cpp-python\n"
                    "  Or use the API backend instead (Ollama/LM Studio):\n"
                    "    humanizer = create_humanizer(det, 'api', base_url='http://localhost:11434/v1', model='qwen2.5:1.5b-instruct')"
                ) from e
            raise
        except ModelNotFoundError:
            raise ModelNotFoundError(
                f"GGUF model not found at: {model_path}\n"
                f"  Download a model (e.g., Qwen2.5-1.5B-Instruct Q4_K_M ~1GB) to:\n"
                f"    {settings.models_dir}/\n"
                f"  Or use the API backend instead (set HUMANIZER_BACKEND=api):\n"
                f"    HUMANIZER_API_URL='http://localhost:11434/v1'\n"
                f"    HUMANIZER_API_MODEL='qwen2.5:1.5b-instruct'"
            )
    elif backend == "api":
        be = OpenAICompatibleBackend(
            base_url=backend_kwargs.pop("base_url", settings.humanizer_api_url),
            api_key=backend_kwargs.pop("api_key", settings.humanizer_api_key),
            model=backend_kwargs.pop("model", settings.humanizer_api_model),
            **backend_kwargs,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'local' or 'api'.")
    
    return Humanizer(detector, be)


# ── Demo / CLI ─────────────────────────────────────────────────────
def demo_humanize():
    """Quick demo - requires trained detector and (for local) a GGUF model."""
    from core.detector import AIDetector

    det = AIDetector()
    det.load()

    # Try local backend if model exists, else show how to configure
    settings = get_settings()
    model_path = settings.default_gguf_path

    if model_path.exists():
        try:
            print(f"[demo] Using local model: {model_path}")
            humanizer = create_humanizer(det, "local", model_path=str(model_path))
        except BackendNotAvailableError as e:
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