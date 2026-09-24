from __future__ import annotations

from typing import List, Sequence

from openai import OpenAI

from models.base import ChatMessage, ChatModel, EmbeddingModel


class OpenAIChatBackend(ChatModel):
    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str | None = None,
        timeout_seconds: float = 120.0,
        temperature: float = 0.0,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)
        self.model_name = model_name
        self.temperature = temperature

    def generate(self, messages: Sequence[ChatMessage]) -> str:
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=self.temperature,
        )
        if not completion.choices:
            return ""
        return completion.choices[0].message.content or ""


class OpenAIEmbeddingBackend(EmbeddingModel):
    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str | None = None,
        timeout_seconds: float = 120.0,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)
        self.model_name = model_name

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(model=self.model_name, input=list(texts))
        return [item.embedding for item in response.data]
