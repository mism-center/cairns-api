from __future__ import annotations

from typing import List, Sequence
import os
from urllib.parse import urlparse

import requests

from models.base import ChatMessage, ChatModel, EmbeddingModel


class OllamaChatBackend(ChatModel):
    def __init__(
        self,
        base_url: str,
        model_name: str,
        timeout_seconds: float = 120.0,
        temperature: float = 0.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature

    def generate(self, messages: Sequence[ChatMessage]) -> str:
        payload = {
            "model": self.model_name,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()

        if isinstance(body.get("message"), dict):
            return body["message"].get("content", "")
        if "response" in body:
            return body.get("response", "")
        raise RuntimeError(f"Unexpected Ollama chat response format: {body}")


class OllamaEmbeddingBackend(EmbeddingModel):
    def __init__(
        self,
        base_url: str,
        model_name: str,
        timeout_seconds: float = 120.0,
        dimensions: int | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.dimensions = dimensions

    def _ensure_ollama_host_env(self):
        if os.getenv("OLLAMA_HOST"):
            return
        parsed = urlparse(self.base_url)
        host = parsed.netloc if parsed.netloc else self.base_url.replace("http://", "").replace("https://", "")
        if host:
            os.environ["OLLAMA_HOST"] = host

    def _get_response_field(self, response, field: str):
        if isinstance(response, dict):
            return response.get(field)
        if hasattr(response, field):
            return getattr(response, field)
        if hasattr(response, "model_dump"):
            try:
                dumped = response.model_dump()  # type: ignore[attr-defined]
                if isinstance(dumped, dict):
                    return dumped.get(field)
            except Exception:
                return None
        return None

    def _short_error(self, exc: Exception) -> str:
        message = str(exc).replace("\n", " ").strip()
        if len(message) > 320:
            message = message[:320] + "..."
        return f"{type(exc).__name__}: {message}"

    def _embed_via_ollama_sdk(self, texts: Sequence[str]) -> List[List[float]] | None:
        self._ensure_ollama_host_env()
        sdk_payload = {"model": self.model_name, "input": list(texts)}
        if self.dimensions is not None:
            sdk_payload["dimensions"] = self.dimensions

        # Primary SDK path: ollama.embed(model=..., input=[...]).
        try:
            from ollama import embed as sdk_embed  # type: ignore

            response = sdk_embed(**sdk_payload)
            vectors = self._get_response_field(response, "embeddings")
            if isinstance(vectors, list) and len(vectors) == len(texts):
                return vectors
        except Exception:
            pass

        # Alternate SDK path for other installed ollama SDK variants.
        try:
            from ollama import Client  # type: ignore
        except Exception:
            return None

        client = Client(host=self.base_url)
        response = client.embed(**sdk_payload)
        vectors = self._get_response_field(response, "embeddings")
        if isinstance(vectors, list) and len(vectors) == len(texts):
            return vectors
        raise RuntimeError(f"Unexpected ollama SDK embed response: {response}")

    def _embed_via_http(self, texts: Sequence[str]) -> List[List[float]]:
        # Try current Ollama embed API first (batch).
        payload = {"model": self.model_name, "input": list(texts)}
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        response = requests.post(
            f"{self.base_url}/api/embed",
            json=payload,
            timeout=self.timeout_seconds,
        )
        if response.ok:
            body = response.json()
            if isinstance(body.get("embeddings"), list):
                return body["embeddings"]

        # Try OpenAI-compatible endpoint exposed by some Ollama setups (batch).
        openai_response = requests.post(
            f"{self.base_url}/v1/embeddings",
            json=payload,
            timeout=self.timeout_seconds,
        )
        if openai_response.ok:
            openai_body = openai_response.json()
            data = openai_body.get("data")
            if not isinstance(data, list):
                raise RuntimeError(f"Unexpected /v1/embeddings response format: {openai_body}")
            vectors: List[List[float]] = []
            for item in data:
                vector = item.get("embedding") if isinstance(item, dict) else None
                if not isinstance(vector, list):
                    raise RuntimeError(f"Unexpected embedding item in /v1/embeddings response: {item}")
                vectors.append(vector)
            if len(vectors) == len(texts):
                return vectors

        # Final fallback: legacy per-text endpoint.
        vectors: List[List[float]] = []
        for text in texts:
            legacy_response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model_name, "prompt": text},
                timeout=self.timeout_seconds,
            )
            legacy_response.raise_for_status()
            legacy_body = legacy_response.json()
            vector = legacy_body.get("embedding")
            if not isinstance(vector, list):
                raise RuntimeError(f"Unexpected legacy Ollama embedding response format: {legacy_body}")
            vectors.append(vector)
        return vectors

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []

        sdk_error: Exception | None = None
        try:
            sdk_vectors = self._embed_via_ollama_sdk(texts)
            if sdk_vectors is not None:
                return sdk_vectors
            sdk_error = RuntimeError("SDK embedding path returned no vectors.")
        except Exception as exc:
            sdk_error = exc

        http_error: Exception | None = None
        try:
            return self._embed_via_http(texts)
        except Exception as exc:
            http_error = exc

        sdk_msg = self._short_error(sdk_error) if isinstance(sdk_error, Exception) else "None"
        http_msg = self._short_error(http_error) if isinstance(http_error, Exception) else "None"
        raise RuntimeError(
            "Failed to create embeddings via both SDK and HTTP Ollama paths. "
            f"SDK error: {sdk_msg}; HTTP error: {http_msg}"
        ) from (http_error or sdk_error)
