import json
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

import config
import config as app_config
from chains.kg_chain import KGChain
from chains.question_lookup_chain import QuestionLookupChain
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel
from langfuse import Langfuse
from models.user_question import Question
from util.chat_history_util import format_chat_history
from util.llm_helper import LLMFactory
from util.prompt_loader import get_prompt_text


class QVKGChain:
    # ── display / truncation limits ──────────────────────────────────────────
    _SNIPPET_MAX_CHARS: int = 500
    _SUMMARY_MIN_LEN: int = 80    # append second sentence when first is shorter
    _SUMMARY_MAX_LEN: int = 260   # hard truncation for card display
    _URL_DISPLAY_LIMIT: int = 5

    # ── token-budget for context window trimming ──────────────────────────────
    _CONTEXT_MAX_TOKENS: int = 16_000
    # Rough chars-per-token estimate used for cheap budget checks only.
    # Actual token counts vary by model; replace with a real tokenizer if precision matters.
    _CHARS_PER_TOKEN: int = 4

    # ── result-count inference when the user doesn't state a number ──────────
    _REQUESTED_COUNT_DEFAULT: int = 5
    _REQUESTED_COUNT_MAX: int = 10

    def __init__(self, config, default_k=10):
        self.config = config
        self.kg_chain = KGChain(config=config)
        self.qv_chain = QuestionLookupChain(config=config)
        self.llm = LLMFactory.get_llm(config=config)
        self.default_lookup_k = default_k
        self.langfuse_client = None
        if config.LANGFUSE_ENABLED:
            self.langfuse_client = Langfuse(
                secret_key=config.LANGFUSE_SECRET_KEY,
                public_key=config.LANGFUSE_PUBLIC_KEY,
                host=config.LANGFUSE_HOST,
            )

        self.COMBINED_ANSWER_PROMPT = self._create_answer_generation_prompt("QV_KG_PROMPT")
        self.SELECTION_PROMPT = self._create_answer_generation_prompt("QV_KG_SELECTION_PROMPT")

    def as_retrieval_chain(self, lookup_parameters=None):
        """Chain to retrieve data from both sources in parallel"""
        if lookup_parameters is None:
            lookup_parameters = {"k": self.default_lookup_k}

        return RunnableParallel(
            {
                "kg_retrieval": self.kg_chain.as_retrival_chain().with_config(run_name="kg_retrieval"),
                "qv_retrieval": self.qv_chain.as_retrieval_chain(lookup_parameters=lookup_parameters).with_config(
                    run_name="qv_retrieval"
                ),
                "input": lambda x: x["input"],
                "chat_history": lambda x: format_chat_history(x.get("chat_history", [])),
                "user_intent": lambda x: x.get("user_intent", {}),
            }
        )

    @staticmethod
    def _safe_parse_xml(xml_string: str) -> Optional[ET.Element]:
        if not xml_string or not xml_string.strip():
            return None

        xml_string = re.sub(
            u"[^\u0009\u000A\u000D\u0020-\uD7FF\uE000-\uFFFD\u10000-\u10FFFF]+",
            "",
            xml_string,
        )
        xml_string = re.sub(
            r"&(?!(?:amp|lt|gt|apos|quot|#\d+|#x[0-9a-fA-F]+);)",
            "&amp;",
            xml_string,
        )

        try:
            return ET.fromstring(xml_string)
        except ET.ParseError:
            try:
                return ET.fromstring(f"<root>{xml_string}</root>")
            except ET.ParseError:
                print("Warning: Failed to parse XML content. Skipping block.")
                return None

    @staticmethod
    def combine_xml_outputs(kg_context: str, qv_context: str, separator: str = "\n\n") -> str:
        tool_data: Dict[str, str] = {}

        if kg_context.strip():
            root = QVKGChain._safe_parse_xml(kg_context)
            if root is not None:
                for tool_element in root.findall("tool"):
                    tool_id = tool_element.get("id")
                    if tool_id:
                        tool_data[tool_id] = ET.tostring(tool_element, encoding="unicode").strip()

        if qv_context.strip():
            root = QVKGChain._safe_parse_xml(qv_context)
            if root is not None:
                for tool_element in root.findall("tool"):
                    tool_id = tool_element.get("id")
                    if tool_id and tool_id not in tool_data:
                        tool_data[tool_id] = ET.tostring(tool_element, encoding="unicode").strip()

        return "<tools>" + separator.join(tool_data.values()) + "</tools>"

    # Reciprocal Rank Fusion constant. Larger k softens the gap between top ranks;
    # 60 is the value from the original RRF paper and a common default.
    _RRF_K: int = 60

    @staticmethod
    def _rank_within_source(cards: list[dict]) -> dict[str, int]:
        """Map tool_id -> 1-based rank within a single source, by its own score.

        Ranking (not raw score) is what makes KG and QV comparable: KG scores are
        integer constraint-hit counts (e.g. 3.0) while QV scores are 0-1 cosine
        similarities (e.g. 0.78). Their raw magnitudes are not on the same scale,
        so we fuse by rank position instead.
        """
        ordered = sorted(
            (c for c in cards if str(c.get("tool_id", "")).strip()),
            key=lambda c: float(c.get("score", 0.0)),
            reverse=True,
        )
        return {str(c["tool_id"]).strip(): i + 1 for i, c in enumerate(ordered)}

    @staticmethod
    def _dedupe_evidence_cards(kg_cards: list[dict], qv_cards: list[dict]) -> list[dict]:
        """Merge KG + QV cards by tool_id and attach a fused RRF score.

        Each merged card keeps its raw per-source score for display, plus:
          _kg_rank / _qv_rank : 1-based rank within each source (or None)
          _rrf                : Reciprocal Rank Fusion score used for ordering
        A tool found by BOTH sources gets contributions from both -> ranks higher.
        """
        kg_rank = QVKGChain._rank_within_source(kg_cards)
        qv_rank = QVKGChain._rank_within_source(qv_cards)

        by_tool: dict[str, dict] = {}
        for card in kg_cards + qv_cards:
            tool_id = str(card.get("tool_id", "")).strip()
            if not tool_id:
                continue
            if tool_id not in by_tool:
                by_tool[tool_id] = dict(card)
            else:
                existing = by_tool[tool_id]
                existing["score"] = max(float(existing.get("score", 0.0)), float(card.get("score", 0.0)))
                if "why_matched" in existing or "why_matched" in card:
                    existing["why_matched"] = list(
                        dict.fromkeys(existing.get("why_matched", []) + card.get("why_matched", []))
                    )
                if "matched_fields" in existing or "matched_fields" in card:
                    existing["matched_fields"] = list(
                        dict.fromkeys(existing.get("matched_fields", []) + card.get("matched_fields", []))
                    )
                if len(str(card.get("snippet", ""))) > len(str(existing.get("snippet", ""))):
                    existing["snippet"] = card.get("snippet", "")
                if isinstance(card.get("payload"), dict):
                    payload = dict(existing.get("payload", {}))
                    payload.update(card["payload"])
                    existing["payload"] = payload

        k = QVKGChain._RRF_K
        for tool_id, card in by_tool.items():
            kr = kg_rank.get(tool_id)
            qr = qv_rank.get(tool_id)
            rrf = 0.0
            if kr is not None:
                rrf += 1.0 / (k + kr)
            if qr is not None:
                rrf += 1.0 / (k + qr)
            card["_kg_rank"] = kr
            card["_qv_rank"] = qr
            card["_rrf"] = rrf

        # Order by fused RRF score; ties fall back to raw score then tool_id.
        return sorted(
            by_tool.values(),
            key=lambda c: (c.get("_rrf", 0.0), float(c.get("score", 0.0))),
            reverse=True,
        )

    @staticmethod
    def _safe_parse_json_object(raw: str) -> dict:
        if isinstance(raw, dict):
            return raw
        if not isinstance(raw, str):
            return {}

        text = raw.strip()
        if not text:
            return {}

        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            pass

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}

        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    def _infer_requested_count(
        self,
        user_query: str,
        default: Optional[int] = None,
        maximum: Optional[int] = None,
    ) -> int:
        _default = default if default is not None else self._REQUESTED_COUNT_DEFAULT
        _maximum = maximum if maximum is not None else self._REQUESTED_COUNT_MAX

        if not isinstance(user_query, str):
            return _default

        digit_match = re.search(
            r"\b(\d{1,2})\b(?:\s+(?:options?|tools?|recommendations?|choices?|workflows?|pipelines?))?",
            user_query,
            flags=re.IGNORECASE,
        )
        if digit_match:
            requested = int(digit_match.group(1))
            return max(1, min(requested, _maximum))

        word_to_num = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        }
        for word, value in word_to_num.items():
            if re.search(rf"\b{word}\b", user_query, flags=re.IGNORECASE):
                return value

        return _default

    @staticmethod
    def _parse_selection_result(raw_selection: str) -> dict:
        parsed = QVKGChain._safe_parse_json_object(raw_selection)

        selected_ids = parsed.get("selected_tool_ids", [])
        if not isinstance(selected_ids, list):
            selected_ids = []
        selected_ids = [str(tool_id).strip() for tool_id in selected_ids if str(tool_id).strip()]

        missing_constraints = parsed.get("missing_constraints", [])
        if not isinstance(missing_constraints, list):
            missing_constraints = []
        missing_constraints = [str(item).strip() for item in missing_constraints if str(item).strip()]

        return {
            "selected_tool_ids": selected_ids,
            "missing_constraints": missing_constraints,
        }

    @staticmethod
    def _extract_card_description(card: dict) -> str:
        payload = card.get("payload", {}) if isinstance(card.get("payload"), dict) else {}
        candidates = [payload.get("description"), card.get("snippet"), payload.get("canonical_text")]

        for candidate in candidates:
            if not candidate:
                continue
            text = str(candidate).strip()
            if not text:
                continue

            description_match = re.search(
                r"description:\s*(.*?)(?:\s+(?:applicationCategory|programmingLanguage|operatingSystem|keywords|topicCategory|featureList|urls|citation_abstracts|license|source):|$)",
                text,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if description_match:
                text = description_match.group(1)

            text = re.sub(r"\s*\|\s*", ". ", text)
            text = re.sub(r">\s*", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                continue

            sentences = re.split(r"(?<=[.!?])\s+", text)
            summary = sentences[0].strip()
            if len(summary) < QVKGChain._SUMMARY_MIN_LEN and len(sentences) > 1:
                summary = f"{summary} {sentences[1].strip()}".strip()

            if len(summary) > QVKGChain._SUMMARY_MAX_LEN:
                summary = summary[: QVKGChain._SUMMARY_MAX_LEN - 3].rstrip() + "..."
            return summary

        return ""

    def _rerank_evidence_cards(self, evidence_cards: list[dict], user_query: str) -> list[dict]:
        """Sort evidence cards (descending) by the best available relevance signal.

        Prefers the fused RRF score (_rrf) set by _dedupe_evidence_cards, which makes
        KG and QV comparable by rank position. Falls back to the raw per-source score
        for callers that didn't go through dedupe (e.g. single-source paths).

        The tiny position penalty (1e-6 * index) is a stable tiebreaker only: it
        preserves the input order when two cards have identical ranking keys.
        """
        def _key(card: dict) -> float:
            if card.get("_rrf") is not None:
                return float(card["_rrf"])
            return float(card.get("score", 0.0))

        scored = [
            (_key(card) - index * 1e-9, card)
            for index, card in enumerate(evidence_cards)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [card for _, card in scored]

    @staticmethod
    def _select_cards(selection: dict, evidence_cards: list[dict], requested_count: int) -> list[dict]:
        cards_by_id = {
            str(card.get("tool_id", "")).strip(): card
            for card in evidence_cards
            if str(card.get("tool_id", "")).strip()
        }

        selected_cards: list[dict] = []
        used_ids: set[str] = set()

        for tool_id in selection.get("selected_tool_ids", []):
            if tool_id in cards_by_id and tool_id not in used_ids:
                selected_cards.append(cards_by_id[tool_id])
                used_ids.add(tool_id)
            if len(selected_cards) >= requested_count:
                break

        for card in evidence_cards:
            tool_id = str(card.get("tool_id", "")).strip()
            if not tool_id or tool_id in used_ids:
                continue
            selected_cards.append(card)
            used_ids.add(tool_id)
            if len(selected_cards) >= requested_count:
                break

        return selected_cards

    @staticmethod
    def _build_reason_text(card: dict) -> str:
        reasons = card.get("why_matched", [])
        if isinstance(reasons, list) and reasons:
            return str(reasons[0])
        fields = card.get("matched_fields", [])
        if isinstance(fields, list) and fields:
            return "matched fields: " + ", ".join(map(str, fields[:3]))
        return "retrieved from ToolDB evidence"

    def _build_fallback_answer(
        self,
        evidence_cards: list[dict],
        user_query: str,
        reason: str,
        requested_count: Optional[int] = None,
    ) -> str:
        if not evidence_cards:
            return (
                "I don't have enough grounded evidence in ToolDB to answer this query safely. "
                "Please refine the query (domain, data type, platform, or task) and try again."
            )

        limit = requested_count or self._infer_requested_count(user_query=user_query)
        lines = [
            f"I can only use retrieved evidence. {reason}",
            "",
            "Top evidence-backed tools:",
        ]
        for card in evidence_cards[:limit]:
            tool_id = str(card.get("tool_id", "")).strip()
            name = str(card.get("name", tool_id)).strip()
            why = self._extract_card_description(card) or self._build_reason_text(card)
            if tool_id:
                lines.append(f"- {name} [{tool_id}] ({why})")
        lines.append("")
        lines.append("If you want a narrower recommendation, add constraints like organism, input format, and task.")
        return "\n".join(lines)

    def _build_grounded_answer(self, payload: dict) -> AIMessage:
        evidence_cards = payload.get("evidence_cards", []) or []
        user_query = str(payload.get("input", "") or "")
        requested_count = self._infer_requested_count(user_query=user_query)

        if not evidence_cards:
            return AIMessage(
                content=self._build_fallback_answer(
                    evidence_cards=[],
                    user_query=user_query,
                    reason="No evidence cards were retrieved.",
                    requested_count=requested_count,
                ),
                name="qvkg_chain",
            )

        ranked_cards = self._rerank_evidence_cards(evidence_cards, user_query=user_query)
        selection = self._parse_selection_result(str(payload.get("selection", "") or ""))
        selected_cards = self._select_cards(selection, evidence_cards=ranked_cards, requested_count=requested_count)

        if not selected_cards:
            return AIMessage(
                content=self._build_fallback_answer(
                    evidence_cards=ranked_cards,
                    user_query=user_query,
                    reason="The selector could not identify grounded tools.",
                    requested_count=requested_count,
                ),
                name="qvkg_chain",
            )

        lines = [f"Here are {len(selected_cards)} evidence-backed options for this request:", ""]

        for index, card in enumerate(selected_cards, start=1):
            tool_id = str(card.get("tool_id", "")).strip()
            name = str(card.get("name", tool_id)).strip() or tool_id
            summary = self._extract_card_description(card)
            if not summary:
                summary = self._build_reason_text(card).capitalize() + "."
            lines.append(f"{index}. **{name} [{tool_id}]**: {summary}")

        missing_constraints = selection.get("missing_constraints", [])
        if missing_constraints:
            lines.append("")
            lines.append("Evidence is still weak or ambiguous for: " + ", ".join(missing_constraints[:4]) + ".")

        if len(selected_cards) < requested_count:
            lines.append("")
            lines.append(
                "Grounded evidence for additional options is limited; add constraints like organism, input format, or workflow step if you want a narrower list."
            )

        return AIMessage(content="\n".join(lines), name="qvkg_chain")

    def as_generative_chain(self, lookup_parameters=None):
        """Chain to combine results and generate an answer"""
        if lookup_parameters is None:
            lookup_parameters = {"k": self.default_lookup_k}

        retrieval_chain = self.as_retrieval_chain(lookup_parameters)

        combined_chain_data = RunnableLambda(
            lambda x: {
                "input": x["input"],
                "context": self.combine_xml_outputs(
                    kg_context=(
                        x.get("kg_retrieval", {}).get("context", {}).get("context", "<tools></tools>")
                        or "<tools></tools>"
                    ),
                    qv_context=(x.get("qv_retrieval", {}).get("context", "<tools></tools>") or "<tools></tools>"),
                ),
                "chat_history": x["chat_history"],
                "has_context": bool(
                    x.get("kg_retrieval", {}).get("context", {}).get("evidence_cards", [])
                    or x.get("qv_retrieval", {}).get("evidence_cards", [])
                ),
                "kg_extra": x.get("kg_retrieval", {}).get("context", {}).get("extra_data", {}),
                "evidence_cards": self._dedupe_evidence_cards(
                    kg_cards=x.get("kg_retrieval", {}).get("context", {}).get("evidence_cards", []),
                    qv_cards=x.get("qv_retrieval", {}).get("evidence_cards", []),
                ),
                "user_persona": x.get("user_intent", {}).get("as_prompt", ""),
            }
        ).with_config(run_name="combined_chain_data")

        response_branch = RunnableBranch(
            (
                lambda x: x["has_context"],
                RunnableParallel(
                    {
                        "output": RunnableParallel(
                            {
                                "selection": RunnableLambda(
                                    lambda x: {
                                        "input": x["input"],
                                        "context": x["context"],
                                        "chat_history": x["chat_history"],
                                        "user_persona": x["user_persona"],
                                        "requested_count": self._infer_requested_count(x.get("input", "")),
                                    }
                                )
                                | RunnableLambda(self._limit_context_tokens)
                                | self.SELECTION_PROMPT
                                | self.llm.with_config(name="grounded_tool_selection")
                                | StrOutputParser(),
                                "evidence_cards": RunnableLambda(lambda x: x.get("evidence_cards", [])),
                                "input": RunnableLambda(lambda x: x.get("input", "")),
                            }
                        )
                        | RunnableLambda(self._build_grounded_answer),
                        "extra": RunnableLambda(
                            lambda x: {
                                "knowledge_graph": x["kg_extra"].get("knowledge_graph", {}),
                                "matched_terms": x["kg_extra"].get("matched_terms", []),
                                "evidence_cards": x.get("evidence_cards", []),
                            }
                        ),
                        "prompt": RunnableLambda(
                            lambda x: {
                                "input": x["input"],
                                "context": x["context"],
                                "chat_history": x["chat_history"],
                                "user_persona": x["user_persona"],
                                "requested_count": self._infer_requested_count(x.get("input", "")),
                            }
                        )
                        | self.SELECTION_PROMPT,
                    }
                ),
            ),
            RunnableLambda(
                lambda x: {
                    "output": AIMessage(
                        content="No tools or relevant information were found to answer the query.",
                        name="qvkg_chain",
                    ),
                    "extra": {"evidence_cards": []},
                    "prompt": RunnableLambda(
                        lambda x: {
                            "input": x["input"],
                            "context": x["context"],
                            "chat_history": x["chat_history"],
                            "user_persona": x["user_persona"],
                            "requested_count": self._infer_requested_count(x.get("input", "")),
                        }
                    )
                    | self.SELECTION_PROMPT,
                }
            ),
        ).with_config(run_name="response_branch")
        qvkg_chain = retrieval_chain | combined_chain_data | response_branch

        return config.configure_langfuse(qvkg_chain.with_config(run_name="qvkg_lookup_generation"))

    @staticmethod
    def _extract_cited_tool_ids(text: str) -> set[str]:
        if not isinstance(text, str) or not text.strip():
            return set()

        cited_ids: set[str] = set()
        for tool_id in re.findall(r"\[tool_id:([^\]]+)\]", text, flags=re.IGNORECASE):
            cleaned = str(tool_id).strip()
            if cleaned:
                cited_ids.add(cleaned)

        for bracket_value in re.findall(r"\[([^\]]+)\]", text):
            cleaned = str(bracket_value).strip()
            if not cleaned:
                continue
            if ":" in cleaned:
                continue
            cited_ids.add(cleaned)
        return cited_ids

    @staticmethod
    def _normalize_citations(text: str) -> str:
        if not isinstance(text, str):
            return str(text)
        return re.sub(
            r"\[tool_id:([^\]]+)\]",
            lambda m: f"[{str(m.group(1)).strip()}]",
            text,
            flags=re.IGNORECASE,
        )

    def _create_answer_generation_prompt(self, prompt_name):
        user_prompt = self._get_raw_from_langfuse(prompt_name)
        return ChatPromptTemplate(
            [
                ("user", user_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
            ]
        )

    def _get_raw_from_langfuse(self, prompt_name: str) -> str:
        return get_prompt_text(prompt_name=prompt_name, config=self.config, langfuse_client=self.langfuse_client)

    def _limit_context_tokens(self, inputs: Dict) -> Dict:
        def get_token_count(current_inputs):
            messages = self.SELECTION_PROMPT.format_messages(**current_inputs)
            # Character count divided by _CHARS_PER_TOKEN is a cheap approximation.
            # For precise counts swap in a real tokenizer (e.g. tiktoken).
            return sum(len(getattr(m, "content", "")) for m in messages) / self._CHARS_PER_TOKEN

        if get_token_count(inputs) <= self._CONTEXT_MAX_TOKENS:
            return inputs

        try:
            root = ET.fromstring(inputs["context"])
            while get_token_count(inputs) > self._CONTEXT_MAX_TOKENS:
                tools = root.findall("tool")
                if not tools:
                    break
                root.remove(tools[-1])
                inputs["context"] = ET.tostring(root, encoding="unicode")
        except ET.ParseError:
            pass

        return inputs


if __name__ == "__main__":
    import asyncio
    from chains.qvkg_chain import QVKGChain

    qvkg_agent = QVKGChain(config=app_config)
    user_q = Question(
        chat_history=[
            (
                "Saliva",
                "This study focuses on whole-genome sequencing and related phenotypes in asthma. It includes variables related to demographic details, health status, and genetic data. For example, it collects data on age, gender, and other phenotypic characteristics that may be linked to secretory status.",
            )
        ],
        input="variables related to Saliva secretor studies",
    )
    qa_chain = qvkg_agent.as_generative_chain()
    response = asyncio.run(qa_chain.ainvoke(user_q.dict()))

    print(json.dumps(response, indent=2))
