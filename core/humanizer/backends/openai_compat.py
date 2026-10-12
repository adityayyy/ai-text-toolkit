"""OpenAI-compatible API (local or remote)."""

import httpx
from typing import Optional

from core.humanizer.backends.base import GenerationBackend
from core.exceptions import BackendNotAvailableError


class OpenAICompatibleBackend(GenerationBackend):
    """OpenAI-compatible API (local or remote)."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",  # Ollama default
        api_key: str = "not-needed",
        model: str = "qwen2.5:1.5b-instruct",
        timeout: float = 60.0,
    ):
        try:
            import httpx
        except ImportError:
            raise BackendNotAvailableError("httpx not installed. Run: pip install httpx")
        
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
        """Close the HTTP client."""
        self.client.close()