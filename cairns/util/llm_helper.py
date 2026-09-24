import asyncio
from typing import Any, Callable, Sequence

from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from models.base import ChatMessage
from models.factory import get_chat_model, get_embedding_model


class BackendEmbeddingAdapter(Embeddings):
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embedding_model.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        vectors = self.embedding_model.embed([text])
        return vectors[0] if vectors else []


class LLMFactory:

    @staticmethod
    def get_raw_llm(config):
        backend = config.MODEL_BACKEND.lower()
        if backend == "openai":
            return ChatOpenAI(
                api_key=config.OPENAI_API_KEY,
                base_url=config.OPENAI_BASE_URL,
                model=config.OPENAI_CHAT_MODEL,
                temperature=config.GEN_TEMPERATURE,
            )
        if backend == "ollama":
            return ChatOllama(
                base_url=config.OLLAMA_BASE_URL,
                model=config.OLLAMA_CHAT_MODEL,
                temperature=config.GEN_TEMPERATURE,
            )
        raise ValueError(
            f"Unsupported MODEL_BACKEND '{config.MODEL_BACKEND}'. "
            "Supported values in this repository are: openai, ollama."
        )

    @classmethod
    def get_llm(cls, config):
        backend = get_chat_model(config=config)

        def _invoke(payload: Any):
            messages = cls._normalize_messages(payload)
            text = backend.generate(messages)
            return AIMessage(content=text, response_metadata={})

        async def _ainvoke(payload: Any):
            return await asyncio.to_thread(_invoke, payload)

        return RunnableLambda(_invoke, afunc=_ainvoke) | RunnableLambda(cls.strip_thought)

    @staticmethod
    def get_embeddings(config):
        return BackendEmbeddingAdapter(get_embedding_model(config=config))

    @staticmethod
    def _normalize_messages(payload: Any) -> list[ChatMessage]:
        if payload is None:
            return []

        if isinstance(payload, str):
            return [ChatMessage(role="user", content=payload)]

        if hasattr(payload, "to_messages"):
            payload = payload.to_messages()

        if isinstance(payload, BaseMessage):
            payload = [payload]

        if isinstance(payload, Sequence):
            normalized = []
            for item in payload:
                if isinstance(item, BaseMessage):
                    role = {
                        "human": "user",
                        "ai": "assistant",
                        "system": "system",
                        "tool": "tool",
                    }.get(getattr(item, "type", "human"), "user")
                    normalized.append(ChatMessage(role=role, content=str(item.content)))
                elif isinstance(item, tuple) and len(item) >= 2:
                    normalized.append(ChatMessage(role=str(item[0]), content=str(item[1])))
                elif isinstance(item, dict) and "content" in item:
                    normalized.append(
                        ChatMessage(
                            role=str(item.get("role", "user")),
                            content=str(item.get("content", "")),
                        )
                    )
                else:
                    normalized.append(ChatMessage(role="user", content=str(item)))
            return normalized

        return [ChatMessage(role="user", content=str(payload))]

    @staticmethod
    def strip_thought(message: AIMessage):
        if not isinstance(message.content, str):
            return message
        if "</think>" in message.content:
            messages = message.content.split("</think>")
            thought = messages[0].replace("<think>", "").replace("</think>", "")
            message.content = messages[-1].strip("\n\n")
            message.response_metadata["thought"] = thought
        return message


class DeferredLLM:
    """
    Lazily creates a concrete LLM instance when any attribute/method is first accessed.
    The creation happens in the thread (and event loop) that triggers the first access.
    `factory` should be a callable that takes no args and returns a concrete LLM instance.
    """
    def __init__(self, factory: Callable[[], object]):
        self._factory = factory
        self._llm = None

    def _ensure(self):
        if self._llm is None:
            self._llm = self._factory()

    def __getattr__(self, item):
        # called from the thread/method that uses the llm
        self._ensure()
        return getattr(self._llm, item)
