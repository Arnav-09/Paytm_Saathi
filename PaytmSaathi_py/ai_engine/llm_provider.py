"""Configurable LLM provider abstraction.

Supported providers (selected with the LLM_PROVIDER environment variable):

* ``mock``    - Deterministic stub for development and tests when no API key
                is available. Never invents policy facts.
* ``groq``    - Groq's free-tier, OpenAI-compatible API (GROQ_API_KEY).
* ``gemini``  - Google Gemini's OpenAI-compatible endpoint (GEMINI_API_KEY).
* ``openai``  - Any OpenAI-compatible chat API (OpenAI, Together, llama.cpp...).
* ``ollama``  - Local Ollama instance (http://localhost:11434).

API keys are read only from environment variables and are never logged.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .exceptions import LLMProviderError, ValidationError

USER_AGENT = "PaytmSaathi/1.0 (+python-urllib)"

# name -> (key env, base url, model env, default model, label)
PRESETS: dict[str, tuple[str, str, str, str, str]] = {
    "groq": ("GROQ_API_KEY", "https://api.groq.com/openai/v1", "GROQ_MODEL",
             "openai/gpt-oss-120b", "Groq"),
    "gemini": ("GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1beta/openai",
               "GEMINI_MODEL", "gemini-2.0-flash", "Gemini"),
}


class LLMProvider(ABC):
    """Interface: generate text, or generate a JSON object validated later by Pydantic."""

    label: str = "LLM"
    model: str = ""

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

    label = "Demo mode"
    model = "offline"

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


def _messages(prompt: str, system_prompt: Optional[str]) -> list[dict]:
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def _structured_system(system_prompt: Optional[str], schema: dict) -> str:
    return (system_prompt or "") + (
        "\n\nReturn ONLY a valid JSON object (no markdown, no prose) matching this schema: "
        + json.dumps(schema)
    )


class OpenAICompatibleProvider(LLMProvider):
    """OpenAI-compatible chat API (OpenAI, Groq, Gemini, Together, llama.cpp...)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        label: str = "OpenAI-compatible",
        key_env: str = "OPENAI_API_KEY",
    ):
        self.api_key = api_key or os.getenv(key_env)
        self.base_url = (
            base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")
        self.model = model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        self.label = label
        if not self.api_key:
            raise LLMProviderError(
                f"{key_env} is not set. Add it to .env or use LLM_PROVIDER=mock."
            )

    def _post_chat(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        json_mode: bool = False,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": USER_AGENT,
        }
        last_error = ""
        for attempt in range(3):  # retry on free-tier rate limits / transient errors
            request = Request(f"{self.base_url}/chat/completions", data=data,
                              headers=headers, method="POST")
            try:
                with urlopen(request, timeout=60) as response:
                    body = json.loads(response.read().decode("utf-8"))
                break
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="ignore")[:300]
                if exc.code in (429, 500, 502, 503) and attempt < 2:
                    retry_after = exc.headers.get("retry-after") if exc.headers else None
                    try:
                        wait = min(float(retry_after), 8.0) if retry_after else 2.0 * (attempt + 1)
                    except ValueError:
                        wait = 2.0 * (attempt + 1)
                    time.sleep(wait)
                    last_error = detail
                    continue
                if exc.code == 429:
                    raise LLMProviderError(
                        f"The AI service ({self.label}) is busy (free-tier rate limit). "
                        "Please try again in a minute.")
                if exc.code in (401, 403):
                    raise LLMProviderError(
                        f"The AI service ({self.label}) rejected the API key. Check the key in .env.")
                raise LLMProviderError(f"LLM API error {exc.code}: {detail}")
            except URLError as exc:
                raise LLMProviderError(f"Could not reach the AI service ({self.label}): {exc.reason}")
        else:
            raise LLMProviderError(f"The AI service ({self.label}) is busy. {last_error}")
        try:
            return body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError(f"Unexpected LLM API response: {exc}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        return self._post_chat(_messages(prompt, system_prompt), temperature, max_tokens)

    def generate_structured(
        self,
        prompt: str,
        schema: dict,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
    ) -> dict:
        messages = _messages(prompt, _structured_system(system_prompt, schema))
        try:
            text = self._post_chat(messages, temperature, 4096, json_mode=True)
        except LLMProviderError as exc:
            if "response_format" not in str(exc) and "json" not in str(exc).lower():
                raise
            text = self._post_chat(messages, temperature, 4096)  # provider without JSON mode
        return _parse_json_text(text)


class OllamaProvider(LLMProvider):
    """Local Ollama instance (default http://localhost:11434)."""

    label = "Ollama"

    def __init__(
        self, base_url: Optional[str] = None, model: Optional[str] = None
    ):
        self.base_url = (
            base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
        ).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL") or "llama3.2"

    def _post_chat(
        self, messages: list[dict], temperature: float, max_tokens: int, json_mode: bool = False
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if json_mode:
            payload["format"] = "json"
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

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        return self._post_chat(_messages(prompt, system_prompt), temperature, max_tokens)

    def generate_structured(
        self,
        prompt: str,
        schema: dict,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
    ) -> dict:
        text = self._post_chat(_messages(prompt, _structured_system(system_prompt, schema)),
                               temperature, 4096, json_mode=True)
        return _parse_json_text(text)


def mask_secret(value: Optional[str]) -> str:
    """Masked representation of a secret for safe logging."""
    if not value:
        return ""
    if len(value) <= 7:
        return "***"
    return value[:3] + "***" + value[-4:]


def _parse_json_text(text: str) -> dict:
    """Extract a JSON object from LLM output, tolerating markdown fences and prose."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n", "", cleaned)
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError as exc:
                raise ValidationError(f"LLM returned invalid JSON: {exc}")
        raise ValidationError("LLM returned invalid JSON.")


def get_llm_provider(provider_name: Optional[str] = None) -> LLMProvider:
    """Factory: build the provider configured by the environment."""
    name = (provider_name or os.getenv("LLM_PROVIDER", "mock")).lower().strip()
    if name == "mock":
        return MockLLMProvider()
    if name in PRESETS:
        key_env, base_url, model_env, default_model, label = PRESETS[name]
        if not os.getenv(key_env):
            logging.getLogger("paytm_saathi").warning(
                "%s is not set; running in demo mode (no AI model).", key_env)
            return MockLLMProvider()
        return OpenAICompatibleProvider(
            api_key=os.getenv(key_env), base_url=base_url,
            model=os.getenv(model_env) or default_model, label=label, key_env=key_env)
    if name == "openai":
        return OpenAICompatibleProvider(label="OpenAI")
    if name == "ollama":
        return OllamaProvider()
    raise LLMProviderError(
        f"Unsupported LLM provider: {name}. Use 'mock', 'groq', 'gemini', 'openai', or 'ollama'."
    )


def provider_status() -> dict:
    """What the UI should show: label, model and whether a real model is connected."""
    name = os.getenv("LLM_PROVIDER", "mock").lower().strip()
    if name in PRESETS and not os.getenv(PRESETS[name][0]):
        return {"connected": False, "label": "Demo mode", "model": "",
                "problem": f"{PRESETS[name][0]} is missing in .env"}
    try:
        llm = get_llm_provider()
    except LLMProviderError as exc:
        return {"connected": False, "label": "Demo mode", "model": "", "problem": exc.message}
    connected = not isinstance(llm, MockLLMProvider)
    return {"connected": connected, "label": llm.label, "model": llm.model if connected else "", "problem": None}

