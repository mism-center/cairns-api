import html
from operator import itemgetter
from typing import Any, Dict, List

import config
import config as app_config
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langfuse import Langfuse
from models.user_question import Question
from qdrant_client import QdrantClient
from util.chat_history_util import format_chat_history, format_chat_history_text
from util.llm_helper import LLMFactory
from util.prompt_loader import get_prompt_text


class QuestionLookupChain:
    """Performs ToolDB vector lookup and generates an answer grounded in retrieved tools."""

    _SNIPPET_MAX_CHARS: int = 500   # max characters kept from tool description/canonical_text
    _URL_DISPLAY_LIMIT: int = 5     # max URLs included in the XML context block

    def __init__(self, config):
        self.config = config
        self.q_client_sync = QdrantClient(url=config.QDRANT_URL)
        self.collection_name = getattr(config, "TOOLDB_QDRANT_COLLECTION", config.QDRANT_COLLECTION_NAME)
        self.embeddings = LLMFactory.get_embeddings(config)

        # initialize llm
        self.llm = LLMFactory.get_llm(config=config)
        self.langfuse_client = None
        if config.LANGFUSE_ENABLED:
            self.langfuse_client = Langfuse(
                secret_key=config.LANGFUSE_SECRET_KEY,
                public_key=config.LANGFUSE_PUBLIC_KEY,
                host=config.LANGFUSE_HOST,
            )

        # PROMPTS
        self.REPHRASE_PROMPT = self._get_prompt_from_langfuse("REPHRASE_PROMPT")
        self.ANSWER_GENERATION_PROMPT = self._create_answer_generation_prompt("ANSWER_GENERATION_PROMPT")

    def as_retrieval_chain(self, lookup_parameters=None) -> Runnable:
        if lookup_parameters is None:
            lookup_parameters = {"k": getattr(self.config, "TOOLDB_QV_TOP_K", 20)}
        top_k = int(lookup_parameters.get("k", getattr(self.config, "TOOLDB_QV_TOP_K", 20)))

        rephrase_branch = RunnableBranch(
            (
                RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                    run_name="HasChatHistoryCheck"
                ),
                RunnablePassthrough.assign(
                    # REPHRASE_PROMPT is a plain-text PromptTemplate: its
                    # {chat_history} slot needs text, not Message objects.
                    # Passing Message objects here causes MESSAGE_COERCION_FAILURE.
                    chat_history=lambda x: format_chat_history_text(x["chat_history"])
                ).with_config(run_name="format_chat_history")
                | self.REPHRASE_PROMPT
                | self.llm.with_config(name="rephrase_user_query")
                | StrOutputParser(),
            ),
            RunnableLambda(itemgetter("input")),
        ).with_config(run_name="rephrase_based_on_chat")

        retrieval_chain = rephrase_branch | RunnableLambda(
            lambda user_query: self._retrieve_tool_cards(user_query=user_query, top_k=top_k),
            afunc=lambda user_query: self._retrieve_tool_cards_async(user_query=user_query, top_k=top_k),
            name="qdrant_tool_lookup",
        )
        return retrieval_chain.with_config(run_name="retrieve_documents")

    def as_generative_chain(self, lookup_parameters=None) -> Runnable:
        if lookup_parameters is None:
            lookup_parameters = {"k": getattr(self.config, "TOOLDB_QV_TOP_K", 20)}

        retrieval = self.as_retrieval_chain(lookup_parameters=lookup_parameters).with_config(
            run_name="retrieval"
        )
        _inputs = RunnableParallel(
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: format_chat_history(x["chat_history"]),
                "retrieval": retrieval,
            }
        ).with_types(input_type=Question).with_config(run_name="question_lookup_chain")

        answer_chain = (
            self.ANSWER_GENERATION_PROMPT | self.llm | StrOutputParser()
        ).with_config(name="answer_generation")

        generative_chain = _inputs | RunnableParallel(
            {
                "output": RunnableLambda(
                    lambda x: {
                        "input": x["input"],
                        "context": x.get("retrieval", {}).get("context", "<tools></tools>"),
                        "chat_history": x["chat_history"],
                    }
                )
                | answer_chain,
                "extra": RunnableLambda(
                    lambda x: {
                        "evidence_cards": x.get("retrieval", {}).get("evidence_cards", []),
                        "retrieval_query": x.get("retrieval", {}).get("retrieval_query", x["input"]),
                    }
                ),
            }
        )
        return config.configure_langfuse(generative_chain)

    async def _retrieve_tool_cards_async(self, user_query: str, top_k: int) -> dict:
        import asyncio

        return await asyncio.to_thread(self._retrieve_tool_cards, user_query, top_k)

    def _retrieve_tool_cards(self, user_query: str, top_k: int) -> dict:
        query_vector = self.embeddings.embed_query(user_query)
        try:
            search_hits = self.q_client_sync.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
        except Exception:
            return {
                "retrieval_query": user_query,
                "context": "<tools></tools>",
                "evidence_cards": [],
            }

        cards_by_tool: Dict[str, Dict[str, Any]] = {}
        for hit in search_hits:
            payload = dict(hit.payload or {})
            tool_id = str(
                payload.get("tool_id")
                or payload.get("identifier")
                or payload.get("doc_id")
                or hit.id
            )
            tool_name = payload.get("tool_name") or payload.get("name") or tool_id
            snippet = (payload.get("canonical_text") or payload.get("description") or "")[: self._SNIPPET_MAX_CHARS]
            matched_fields = payload.get("fields_used") or []
            if not isinstance(matched_fields, list):
                matched_fields = [str(matched_fields)]

            if tool_id not in cards_by_tool:
                cards_by_tool[tool_id] = {
                    "tool_id": tool_id,
                    "name": tool_name,
                    "snippet": snippet,
                    "matched_fields": matched_fields,
                    "score": float(hit.score),
                    "payload": payload,
                }
            else:
                existing = cards_by_tool[tool_id]
                existing["score"] = max(existing["score"], float(hit.score))
                existing["matched_fields"] = list(
                    dict.fromkeys(existing["matched_fields"] + matched_fields)
                )
                if len(snippet) > len(existing["snippet"]):
                    existing["snippet"] = snippet

        cards = sorted(cards_by_tool.values(), key=lambda x: x["score"], reverse=True)[:top_k]
        return {
            "retrieval_query": user_query,
            "context": self._format_tool_context(cards),
            "evidence_cards": cards,
        }

    @staticmethod
    def _format_tool_context(cards: List[Dict[str, Any]]) -> str:
        if not cards:
            return "<tools></tools>"

        tool_blocks = []
        for card in cards:
            payload = card.get("payload", {})
            why = ", ".join(card.get("matched_fields", []))
            urls = payload.get("urls") or []
            urls_string = ", ".join(urls[: QuestionLookupChain._URL_DISPLAY_LIMIT]) if isinstance(urls, list) else str(urls)
            tool_blocks.append(
                f'<tool id="{html.escape(str(card.get("tool_id", "")))}">'
                f"<name>{html.escape(str(card.get('name', '')))}</name>"
                f"<score>{card.get('score', 0):.6f}</score>"
                f"<why_matched>{html.escape(why)}</why_matched>"
                f"<snippet>{html.escape(str(card.get('snippet', '')))}</snippet>"
                f"<urls>{html.escape(urls_string)}</urls>"
                f"</tool>"
            )
        return "<tools>" + "\n".join(tool_blocks) + "</tools>"

    def _get_raw_from_langfuse(self, prompt_name: str) -> str:
        return get_prompt_text(prompt_name=prompt_name, config=self.config, langfuse_client=self.langfuse_client)

    def _get_prompt_from_langfuse(self, prompt_name: str) -> PromptTemplate:
        return PromptTemplate.from_template(template=self._get_raw_from_langfuse(prompt_name))

    def _create_answer_generation_prompt(self, prompt_name: str) -> ChatPromptTemplate:
        template = self._get_raw_from_langfuse(prompt_name)
        return ChatPromptTemplate.from_messages(
            [
                ("system", template),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
            ]
        )


if __name__ == "__main__":
    cls = QuestionLookupChain(config=app_config)
    user_q = Question(chat_history=[], input="Find tools for RNA-seq differential expression")
    qa_chain = cls.as_generative_chain()
    response = qa_chain.invoke(user_q.dict())
    print(response)
