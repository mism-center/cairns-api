from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


class ChatModel(ABC):
    @abstractmethod
    def generate(self, messages: Sequence[ChatMessage]) -> str:
        """Generate a single text response from chat messages."""


class EmbeddingModel(ABC):
    @abstractmethod
    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed a batch of texts into vectors."""
