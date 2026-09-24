from __future__ import annotations

import os
from typing import Any

from models.base import ChatModel, EmbeddingModel
from models.ollama_backend import OllamaChatBackend, OllamaEmbeddingBackend
from models.openai_backend import OpenAIChatBackend, OpenAIEmbeddingBackend
from models.stub_backend import StubChatBackend, StubEmbeddingBackend


def _cfg(config: Any, key: str, default: Any = None) -> Any:
    if config is not None and hasattr(config, key):
        return getattr(config, key)
    return os.getenv(key, default)


def _resolve_backend_name(config: Any = None) -> str:
    backend = _cfg(config, "MODEL_BACKEND", "")
    if backend:
        return str(backend).lower()

    legacy = str(_cfg(config, "LLM_SERVER_TYPE", "openai")).lower()
    if legacy in {"vllm", "openai"}:
        return "openai"
    if legacy in {"ollama"}:
        return "ollama"
    return legacy


def get_chat_model(config: Any = None) -> ChatModel:
    backend = _resolve_backend_name(config)
    timeout_seconds = float(_cfg(config, "MODEL_TIMEOUT_SECONDS", 120))
    temperature = float(_cfg(config, "GEN_TEMPERATURE", 0))

    if backend == "ollama":
        return OllamaChatBackend(
            base_url=str(_cfg(config, "OLLAMA_BASE_URL", "http://localhost:11434")),
            model_name=str(_cfg(config, "OLLAMA_CHAT_MODEL", "llama3.1:latest")),
            timeout_seconds=timeout_seconds,
            temperature=temperature,
        )
    if backend == "openai":
        return OpenAIChatBackend(
            model_name=str(_cfg(config, "OPENAI_CHAT_MODEL", _cfg(config, "GEN_MODEL_NAME", "gpt-4o-mini"))),
            api_key=str(_cfg(config, "OPENAI_API_KEY", _cfg(config, "GEN_API_KEY", "EMPTY"))),
            base_url=_cfg(config, "OPENAI_BASE_URL", _cfg(config, "LLM_URL", None)),
            timeout_seconds=timeout_seconds,
            temperature=temperature,
        )
    return StubChatBackend(backend_name=backend)


def get_embedding_model(config: Any = None) -> EmbeddingModel:
    backend = _resolve_backend_name(config)
    timeout_seconds = float(_cfg(config, "MODEL_TIMEOUT_SECONDS", 120))

    if backend == "ollama":
        dimensions = _cfg(config, "OLLAMA_EMBED_DIMENSIONS", None)
        if dimensions in ("", None):
            parsed_dimensions = None
        else:
            parsed_dimensions = int(dimensions)
        return OllamaEmbeddingBackend(
            base_url=str(_cfg(config, "OLLAMA_BASE_URL", "http://localhost:11434")),
            model_name=str(_cfg(config, "OLLAMA_EMBED_MODEL", "bge-m3")),
            timeout_seconds=timeout_seconds,
            dimensions=parsed_dimensions,
        )
    if backend == "openai":
        return OpenAIEmbeddingBackend(
            model_name=str(
                _cfg(
                    config,
                    "OPENAI_EMBED_MODEL",
                    _cfg(config, "EMB_MODEL_NAME", "text-embedding-3-small"),
                )
            ),
            api_key=str(_cfg(config, "OPENAI_API_KEY", _cfg(config, "GEN_API_KEY", "EMPTY"))),
            base_url=_cfg(config, "OPENAI_BASE_URL", _cfg(config, "EMBEDDING_URL", _cfg(config, "LLM_URL", None))),
            timeout_seconds=timeout_seconds,
        )
    return StubEmbeddingBackend(backend_name=backend)
