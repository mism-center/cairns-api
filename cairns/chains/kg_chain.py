import asyncio
import html
import json
import re
from typing import Any, Dict, List

import config
import config as app_config
from databases.sqlite_kg import SQLiteKGDB
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel
from langfuse import Langfuse
from models.user_question import Question
from util.chat_history_util import format_chat_history
from util.llm_helper import LLMFactory
from util.prompt_loader import get_prompt_text


class KGChain:
    TOOLDB_TERM_NODE_MAP = {
        "EDAM_TOPIC": {"key": "node_id", "name_key": "name"},
        "EDAM_OPERATION": {"key": "node_id", "name_key": "name"},
        "FORMAT": {"key": "node_id", "name_key": "name"},
        "LANGUAGE": {"key": "name", "name_key": "name"},
        "APP_CATEGORY": {"key": "name", "name_key": "name"},
        "OS": {"key": "name", "name_key": "name"},
    }

    # ── term-matching thresholds ──────────────────────────────────────────────
    # Labels this short (e.g. "R", "C") are matched only on exact token boundary
    # to avoid spurious hits inside longer words.
    _SHORT_LABEL_MAX_LEN: int = 2
    # Score bonus per character for substring matches, capped at 1.0.
    # Longer label names get a slightly higher score to prefer specific matches.
    _NAME_LENGTH_NORM: float = 80.0
    # Minimum token-overlap ratio required to treat a partial match as valid.
    _TOKEN_OVERLAP_THRESHOLD: float = 0.6
    # Multiplier applied to KG_TERM_MATCH_LIMIT when pre-loading the term index.
    # We load more terms than the per-query limit because the index is built once
    # and reused across queries that differ in how many terms they surface.
    _TERM_INDEX_LOAD_FACTOR: int = 100
    # Truncation for tool description snippets stored in evidence cards.
    _SNIPPET_MAX_CHARS: int = 500

    def __init__(self, config):
        self.config = config
        self.llm = LLMFactory.get_llm(config=config)
        self.langfuse_client = None
        if config.LANGFUSE_ENABLED:
            self.langfuse_client = Langfuse(
                secret_key=config.LANGFUSE_SECRET_KEY,
                public_key=config.LANGFUSE_PUBLIC_KEY,
                host=config.LANGFUSE_HOST,
            )
        self.ANSWER_GENERATION_PROMPT = self._create_answer_generation_prompt("ANSWER_GENERATION_PROMPT_KG_APP")
        self.kg_backend = str(getattr(config, "KG_BACKEND", "redis") or "redis").lower()
        if self.kg_backend == "sqlite":
            self.graph = SQLiteKGDB(config)
        else:
            # Imported lazily: falkordb is only needed for the redis/falkor
            # KG backend, so sqlite deployments don't need it installed.
            from databases.redis_graph import RedisGraphDB
            self.graph = RedisGraphDB(config)
        self._term_index = None

    def as_retrival_chain(self):
        return RunnableParallel(
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: format_chat_history(x["chat_history"]),
                "context": RunnableLambda(
                    lambda x: self._retrieve_kg_context_sync(x["input"]),
                    afunc=lambda x: self._retrieve_kg_context_async(x["input"]),
                    name="retrieve_tooldb_kg",
                ).with_config(run_name="retrieval"),
            }
        )

    def as_generative_chain(self):
        retrival_chain = self.as_retrival_chain().with_config(run_name="kg_lookup_chain")
        answer_chain = RunnableBranch(
            (
                RunnableLambda(lambda x: bool(x.get("context"))).with_config(run_name="has_context"),
                (self.ANSWER_GENERATION_PROMPT | self.llm | StrOutputParser()).with_config(
                    run_name="answer_generation"
                ),
            ),
            RunnableLambda(lambda _: "No matching tools were found in the ToolDB knowledge graph.").with_config(
                run_name="no_data"
            ),
        )
        generative_chain = retrival_chain | RunnableParallel(
            {
                "output": RunnableLambda(
                    lambda x: {
                        "input": x["input"],
                        "context": x.get("context", {}).get("context", ""),
                        "chat_history": x["chat_history"],
                    }
                )
                | answer_chain,
                "extra": RunnableLambda(
                    lambda x: {
                        "knowledge_graph": x.get("context", {}).get("extra_data", {}).get(
                            "knowledge_graph", {}
                        ),
                        "matched_terms": x.get("context", {}).get("extra_data", {}).get(
                            "matched_terms", []
                        ),
                        "evidence_cards": x.get("context", {}).get("evidence_cards", []),
                    }
                ),
            }
        )
        return config.configure_langfuse(generative_chain)

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\-\s/]", " ", text.lower())).strip()

    @staticmethod
    def _escape_cypher(value: str) -> str:
        return (value or "").replace("\\", "\\\\").replace("'", "\\'")

    def _load_term_index(self) -> list[dict]:
        if self._term_index is not None:
            return self._term_index

        index = []
        for label, spec in self.TOOLDB_TERM_NODE_MAP.items():
            key_name = spec["key"]
            name_key = spec["name_key"]
            rows = self.graph.load_term_rows(
                label=label,
                key_name=key_name,
                name_key=name_key,
                limit=self.config.KG_TERM_MATCH_LIMIT * self._TERM_INDEX_LOAD_FACTOR,
            )
            for row in rows:
                node_key = str(row[0] or "").strip()
                node_name = str(row[1] or "").strip()
                if not node_key or not node_name:
                    continue
                normalized = self._normalize_text(node_name)
                if not normalized:
                    continue
                index.append(
                    {
                        "label": label,
                        "node_key": node_key,
                        "node_name": node_name,
                        "normalized_name": normalized,
                    }
                )
        self._term_index = index
        return index

    def _deterministic_term_match(self, query: str) -> list[dict]:
        query_norm = self._normalize_text(query)
        if not query_norm:
            return []

        query_tokens = set(query_norm.split())
        matches = []
        for item in self._load_term_index():
            name_norm = item["normalized_name"]
            if not name_norm:
                continue
            score = 0.0
            if len(name_norm) <= self._SHORT_LABEL_MAX_LEN:
                # Very short labels (e.g. "R", "C") require an exact token match
                # to avoid spurious substring hits inside longer words.
                if name_norm in query_tokens:
                    score = 1.0
            elif name_norm in query_norm:
                score = 1.0 + min(1.0, len(name_norm) / self._NAME_LENGTH_NORM)
            else:
                name_tokens = set(name_norm.split())
                overlap = len(query_tokens.intersection(name_tokens))
                if overlap:
                    overlap_ratio = overlap / max(1, len(name_tokens))
                    if overlap_ratio >= self._TOKEN_OVERLAP_THRESHOLD:
                        score = overlap_ratio
            if score > 0:
                matches.append({**item, "score": score})

        matches.sort(key=lambda x: x["score"], reverse=True)
        limit = max(self.config.KG_TERM_MATCH_LIMIT, 1)
        return matches[:limit]

    def _llm_assisted_terms(self, query: str) -> list[str]:
        if not self.config.KG_ENABLE_LLM_PARSING:
            return []

        prompt = (
            "Extract up to 8 concise technical terms from the query that could map to "
            "topics, operations, file formats, programming languages, application categories, or operating systems. "
            "Return a comma-separated list only.\n\n"
            f"Query: {query}"
        )
        try:
            response = self.llm.invoke(prompt)
            content = getattr(response, "content", str(response))
            values = [value.strip() for value in re.split(r"[,\n;]", content) if value.strip()]
            return values[:8]
        except Exception:
            return []

    def _resolve_terms(self, query: str) -> list[dict]:
        deterministic_matches = self._deterministic_term_match(query)
        if deterministic_matches:
            return deterministic_matches

        llm_terms = self._llm_assisted_terms(query)
        if not llm_terms:
            return deterministic_matches

        expanded_query = f"{query} {' '.join(llm_terms)}"
        return self._deterministic_term_match(expanded_query)

    def _query_tools_for_term(self, term: dict) -> list[dict]:
        label = term["label"]
        node_key = term["node_key"]
        key_name = self.TOOLDB_TERM_NODE_MAP[label]["key"]
        rows_data = self.graph.query_tools_for_term_rows(
            label=label,
            key_name=key_name,
            node_key=node_key,
            limit=self.config.KG_TOOL_RESULT_LIMIT,
        )
        rows = []
        for row in rows_data:
            rows.append(
                {
                    "tool_id": str(row[0] or ""),
                    "tool_name": str(row[1] or row[0] or ""),
                    "tool_description": str(row[2] or ""),
                    "tool_identifier": str(row[3] or ""),
                    "tool_url": str(row[4] or ""),
                    "relation": str(row[5] or ""),
                    "node_key": str(row[6] or ""),
                    "node_name": str(row[7] or term["node_name"]),
                    "node_label": label,
                    # 9th/10th columns added by sqlite_kg; tolerate older rows
                    # and the Redis/FalkorDB backend, which doesn't provide them.
                    "source": str(row[8]) if len(row) > 8 and row[8] else "tooldb",
                    "metadata_json": str(row[9]) if len(row) > 9 and row[9] else "{}",
                }
            )
        return rows

    def _aggregate_results(self, matched_terms: list[dict], tool_rows: list[dict]) -> dict:
        cards_by_tool: Dict[str, Dict[str, Any]] = {}
        kg_nodes = []
        kg_edges = []
        seen_nodes = set()
        seen_edges = set()

        for row in tool_rows:
            tool_id = row["tool_id"]
            if not tool_id:
                continue
            try:
                full_metadata = json.loads(row.get("metadata_json") or "{}")
                if not isinstance(full_metadata, dict):
                    full_metadata = {}
            except (TypeError, ValueError):
                full_metadata = {}

            card = cards_by_tool.setdefault(
                tool_id,
                {
                    "tool_id": tool_id,
                    "name": row["tool_name"],
                    "snippet": row["tool_description"][: self._SNIPPET_MAX_CHARS],
                    "why_matched": [],
                    "matched_terms": [],
                    "score": 0.0,
                    "payload": {
                        "tool_id": tool_id,
                        "tool_name": row["tool_name"],
                        "identifier": row["tool_identifier"],
                        "url": row["tool_url"],
                        # Data source of the tool (tooldb | biomodels); retrieval
                        # method is always KG here.
                        "source": row.get("source", "tooldb"),
                        "retrieval": "kg",
                        # Full source record (includes raw_metadata for sources
                        # that provide it), so KG-only matches carry everything too.
                        "metadata": full_metadata,
                    },
                },
            )
            reason = f"{row['relation']} -> {row['node_name']} ({row['node_label']})"
            if reason not in card["why_matched"]:
                card["why_matched"].append(reason)
            if row["node_name"] not in card["matched_terms"]:
                card["matched_terms"].append(row["node_name"])
            card["score"] += 1.0

            tool_node_id = f"Tool:{tool_id}"
            term_node_id = f"{row['node_label']}:{row['node_key']}"
            if tool_node_id not in seen_nodes:
                seen_nodes.add(tool_node_id)
                kg_nodes.append({"id": tool_node_id, "label": "Tool", "name": row["tool_name"]})
            if term_node_id not in seen_nodes:
                seen_nodes.add(term_node_id)
                kg_nodes.append(
                    {
                        "id": term_node_id,
                        "label": row["node_label"],
                        "name": row["node_name"],
                    }
                )
            edge_id = f"{tool_node_id}:{row['relation']}:{term_node_id}"
            if edge_id not in seen_edges:
                seen_edges.add(edge_id)
                kg_edges.append(
                    {
                        "id": edge_id,
                        "source": tool_node_id,
                        "target": term_node_id,
                        "label": row["relation"],
                    }
                )

        cards = sorted(cards_by_tool.values(), key=lambda x: x["score"], reverse=True)[
            : self.config.KG_TOOL_RESULT_LIMIT
        ]
        context = self._format_cards_as_context(cards)
        return {
            "context": context,
            "evidence_cards": cards,
            "extra_data": {
                "matched_terms": [
                    {
                        "label": term["label"],
                        "node_key": term["node_key"],
                        "node_name": term["node_name"],
                        "score": term["score"],
                    }
                    for term in matched_terms
                ],
                "knowledge_graph": {"nodes": kg_nodes, "edges": kg_edges},
            },
        }

    @staticmethod
    def _format_cards_as_context(cards: list[dict]) -> str:
        if not cards:
            return "<tools></tools>"
        blocks = []
        for card in cards:
            why = "; ".join(card.get("why_matched", []))
            blocks.append(
                f'<tool id="{html.escape(card.get("tool_id", ""))}">'
                f"<name>{html.escape(card.get('name', ''))}</name>"
                f"<score>{card.get('score', 0):.2f}</score>"
                f"<why_matched>{html.escape(why)}</why_matched>"
                f"<snippet>{html.escape(card.get('snippet', ''))}</snippet>"
                f"</tool>"
            )
        return "<tools>" + "\n".join(blocks) + "</tools>"

    def _retrieve_kg_context_sync(self, query: str) -> dict:
        matched_terms = self._resolve_terms(query)
        if not matched_terms:
            return {"context": "<tools></tools>", "evidence_cards": [], "extra_data": {}}

        rows = []
        for term in matched_terms:
            rows.extend(self._query_tools_for_term(term))

        if not rows:
            return {
                "context": "<tools></tools>",
                "evidence_cards": [],
                "extra_data": {"matched_terms": matched_terms, "knowledge_graph": {"nodes": [], "edges": []}},
            }
        return self._aggregate_results(matched_terms=matched_terms, tool_rows=rows)

    async def _retrieve_kg_context_async(self, query: str) -> dict:
        # Keep SQLite access on the same thread to avoid cross-thread connection/cursor issues.
        return self._retrieve_kg_context_sync(query)

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
    kg_agent = KGChain(config=app_config)
    user_q = Question(chat_history=[], input="Find Python tools for FASTQ format processing on Linux")
    qa_chain = kg_agent.as_generative_chain()
    response = qa_chain.invoke(user_q.dict())
    print(response)
