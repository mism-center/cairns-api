from __future__ import annotations

from typing import List, Sequence

from models.base import ChatMessage, ChatModel, EmbeddingModel


class StubChatBackend(ChatModel):
    def __init__(self, backend_name: str):
        self.backend_name = backend_name

    def generate(self, messages: Sequence[ChatMessage]) -> str:
        raise RuntimeError(
            f"Chat backend '{self.backend_name}' is not configured in this repository. "
            "Set MODEL_BACKEND to 'ollama' or 'openai' and configure matching environment variables."
        )


class StubEmbeddingBackend(EmbeddingModel):
    def __init__(self, backend_name: str):
        self.backend_name = backend_name

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        raise RuntimeError(
            f"Embedding backend '{self.backend_name}' is not configured in this repository. "
            "Set MODEL_BACKEND to 'ollama' or 'openai' and configure matching environment variables."
        )
