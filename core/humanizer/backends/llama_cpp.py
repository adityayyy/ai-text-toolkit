"""Local GGUF model via llama-cpp-python."""

import os
import sys
from typing import Optional

from core.humanizer.backends.base import GenerationBackend
from core.exceptions import BackendNotAvailableError, ModelNotFoundError


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
        _LLAMA_CPP["error"] = BackendNotAvailableError(msg)
        _LLAMA_CPP["available"] = False
        raise _LLAMA_CPP["error"] from e


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
        Llama, _ = _get_llama_cpp()  # raises BackendNotAvailableError with Windows guidance if not available

        if not os.path.exists(model_path):
            raise ModelNotFoundError(f"GGUF model not found: {model_path}")

        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )
        self.model_path = model_path
        self._n_ctx = n_ctx
        self._n_threads = n_threads
        self._n_gpu_layers = n_gpu_layers

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

    def close(self):
        """Release model resources."""
        if hasattr(self, 'model'):
            del self.model