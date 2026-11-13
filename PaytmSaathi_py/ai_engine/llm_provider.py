"""Configurable LLM provider abstraction.

Supported providers (selected with the LLM_PROVIDER environment variable):

* ``mock``    - Deterministic stub for development and tests when no API key
                is available. Never invents policy facts.
* ``openai``  - Any OpenAI-compatible chat API (OpenAI, Azure OpenAI, Together,
                Groq, local llama.cpp server, etc.) via stdlib urllib.
* ``ollama``  - Local Ollama instance (http://localhost:11434).

API keys are read only from environment variables and are never logged.
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .exceptions import LLMProviderError, ValidationError


class LLMProvider(ABC):
    """Interface: generate text, or generate a JSON object validated later by Pydantic."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        ...

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        schema: dict,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
    ) -> dict:
        ...


class MockLLMProvider(LLMProvider):
    """Deterministic mock for development and offline testing.

    Honest about its limits: it cannot know the uploaded policy, so question
    answering returns the insufficient-evidence message and structured
    extraction returns every field as unknown. No network or API key needed.
    """

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        return (
            "I could not find enough information in the uploaded policy "
            "to answer this reliably."
        )

    def generate_structured(
        self,
        prompt: str,
        schema: dict,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
    ) -> dict:
        return {}


class OpenAICompatibleProvider(LLMProvider):
    """OpenAI-compatible chat API (https://platform.openai.com or similar)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = (
            base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")
        self.model = model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        if not self.api_key:
            raise LLMProviderError(
                "OPENAI_API_KEY is not set. Set it in .env or use LLM_PROVIDER=mock."
            )

    def _post_chat(
        self, messages: list[dict], temperature: float, max_tokens: int
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=60) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")[:500]
            raise LLMProviderError(f"LLM API error {exc.code}: {detail}")
        except URLError as exc:
            raise LLMProviderError(f"LLM connection failed: {exc}")
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError(f"Unexpected LLM API response: {exc}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self._post_chat(messages, temperature, max_tokens)

    def generate_structured(
        self,
        prompt: str,
        schema: dict,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
    ) -> dict:
        system = (system_prompt or "") + (
            "\n\nReturn ONLY a valid JSON object matching this schema: "
            + json.dumps(schema)
        )


class OllamaProvider(LLMProvider):
    """Local Ollama instance (default http://localhost:11434)."""

    def __init__(
        self, base_url: Optional[str] = None, model: Optional[str] = None
    ):
        self.base_url = (
            base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
        ).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL") or "llama3.2"

    def _post_chat(
        self, messages: list[dict], temperature: float, max_tokens: int
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")[:500]
            raise LLMProviderError(f"Ollama API error {exc.code}: {detail}")
        except URLError as exc:
            raise LLMProviderError(f"Ollama connection failed: {exc}")
        try:
            return body["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMProviderError(f"Unexpected Ollama response: {exc}")


def mask_secret(value: Optional[str]) -> str:
    """Masked representation of a secret for safe logging."""
    if not value:
        return ""
    if len(value) <= 7:
        return "***"
    return value[:3] + "***" + value[-4:]


def _parse_json_text(text: str) -> dict:
    """Extract a JSON object from LLM output, tolerating markdown fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n", "", cleaned)
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"LLM returned invalid JSON: {exc}")


def get_llm_provider(provider_name: Optional[str] = None) -> LLMProvider:
    """Factory: build the provider configured by the environment."""
    name = (provider_name or os.getenv("LLM_PROVIDER", "mock")).lower().strip()
    if name == "mock":
        return MockLLMProvider()
    if name == "openai":
        return OpenAICompatibleProvider()
    if name == "ollama":
        return OllamaProvider()
    raise LLMProviderError(
        f"Unsupported LLM provider: {name}. Use 'mock', 'openai', or 'ollama'."
    )
