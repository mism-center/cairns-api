"""
LangGraph node helpers.

Conventions:
  - Every node returns a *partial* state dict (only the keys that changed).
  - extra is always merged with the existing state.extra — never replaced wholesale.
  - KG results live in extra["kg_result"] / extra["kg_done"].
  - QV results live in extra["qv_result"] / extra["qv_done"].
"""
import functools
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langfuse.decorators import observe, langfuse_context
from guardrails.input_guard import InputGuard
import config as app_config

import logging_util
from models.agent_state import AgentState
from util.chat_history_util import format_chat_history

logger = logging_util.logger


# ── helpers ───────────────────────────────────────────────────────────────────

def _normalize_ai_message(value: Any, name: str) -> AIMessage:
    """Return an AIMessage without double-wrapping existing AIMessage objects."""
    if isinstance(value, AIMessage):
        if value.name:
            return value
        return AIMessage(
            content=str(value.content or ""),
            name=name,
            response_metadata=value.response_metadata or {},
        )
    if hasattr(value, "content"):
        return AIMessage(content=str(getattr(value, "content", "")), name=name)
    return AIMessage(content=str(value or ""), name=name)


def _merge_extra(current: dict, updates: dict) -> dict:
    """Return a new dict that merges updates into current (non-destructive)."""
    merged = dict(current or {})
    merged.update(updates or {})
    return merged


# ── generic node wrapper (kept for backward-compat with combined_context_graph) ──

@observe()
async def agent_node_dict(state: AgentState, agent, name):
    trace_id = langfuse_context.get_current_trace_id()

    input_data = {
        "input": state.input,
        "chat_history": state.chat_history,
        "extra": state.extra,
        "user_intent": state.user_intent,
        "return_prompt": state.return_prompt,
    }
    raw_result = await agent.ainvoke(input_data)
    result = raw_result if isinstance(raw_result, dict) else {"output": raw_result}

    agent_extra = result.get("extra", {})
    if not isinstance(agent_extra, dict):
        agent_extra = {"raw_extra": str(agent_extra)}

    # Merge — do not overwrite the full extra dict.
    merged_extra = _merge_extra(state.extra, agent_extra)
    merged_extra["trace_id"] = trace_id

    if state.return_prompt:
        merged_extra["context"] = result.get("prompt", "")

    return {
        "output": _normalize_ai_message(result.get("output", ""), name=name),
        "next": result.get("next", ""),
        "input": state.input,
        "extra": merged_extra,
        "user_intent": result.get("user_intent", {}),
    }


# ── KG retrieval node ─────────────────────────────────────────────────────────

async def kg_lookup_node(state: AgentState, kg_retrieval_chain) -> dict:
    """
    Call the KG retrieval chain and store results in extra["kg_result"].
    Does NOT generate a text answer; synthesis_node does that.
    """
    input_data = {
        "input": state.input,
        "chat_history": state.chat_history or [],
    }
    raw = await kg_retrieval_chain.ainvoke(input_data)

    # KGChain.as_retrival_chain() returns:
    # {"input": str, "chat_history": [...], "context": {"context": str, "evidence_cards": [...], "extra_data": {...}}}
    context_data = raw.get("context", {}) if isinstance(raw, dict) else {}
    extra_data = context_data.get("extra_data", {}) if isinstance(context_data, dict) else {}

    kg_result = {
        "evidence_cards": context_data.get("evidence_cards", []),
        "context": context_data.get("context", "<tools></tools>"),
        "matched_terms": extra_data.get("matched_terms", []),
        "knowledge_graph": extra_data.get("knowledge_graph", {}),
    }

    merged_extra = _merge_extra(state.extra, {"kg_result": kg_result, "kg_done": True})
    return {"extra": merged_extra}


# ── QV retrieval node ─────────────────────────────────────────────────────────

async def qv_lookup_node(state: AgentState, qv_retrieval_chain) -> dict:
    """
    Call the QV retrieval chain and store results in extra["qv_result"].
    Does NOT generate a text answer; synthesis_node does that.
    """
    input_data = {
        "input": state.input,
        "chat_history": state.chat_history or [],
    }
    raw = await qv_retrieval_chain.ainvoke(input_data)

    # QuestionLookupChain.as_retrieval_chain() returns:
    # {"retrieval_query": str, "context": str, "evidence_cards": [...]}
    qv_result = {
        "evidence_cards": raw.get("evidence_cards", []) if isinstance(raw, dict) else [],
        "context": raw.get("context", "<tools></tools>") if isinstance(raw, dict) else "<tools></tools>",
        "retrieval_query": raw.get("retrieval_query", state.input) if isinstance(raw, dict) else state.input,
    }

    merged_extra = _merge_extra(state.extra, {"qv_result": qv_result, "qv_done": True})
    return {"extra": merged_extra}


# ── synthesis node ────────────────────────────────────────────────────────────

async def synthesize_node(state: AgentState, qvkg_instance) -> dict:
    """
    Combine KG and QV evidence and generate a final grounded answer.

    Reads extra["kg_result"] and extra["qv_result"]; calls
    QVKGChain._dedupe_evidence_cards + _build_grounded_answer.
    """
    from chains.qvkg_chain import QVKGChain

    extra = state.extra or {}
    kg_result = extra.get("kg_result", {})
    qv_result = extra.get("qv_result", {})

    kg_cards = kg_result.get("evidence_cards", [])
    qv_cards = qv_result.get("evidence_cards", [])

    combined_cards = QVKGChain._dedupe_evidence_cards(kg_cards, qv_cards)
    combined_context = QVKGChain.combine_xml_outputs(
        kg_result.get("context", "<tools></tools>"),
        qv_result.get("context", "<tools></tools>"),
    )

    if not combined_cards:
        answer = AIMessage(
            content=(
                "No relevant ToolDB evidence was retrieved for this query. "
                "Try rephrasing with more specific constraints (OS, format, topic, language)."
            ),
            name="synthesize",
        )
        return {"output": answer}

    # Rank cards and use QVKGChain's grounded answer builder.
    ranked_cards = qvkg_instance._rerank_evidence_cards(combined_cards, state.input)

    # Run LLM selection to choose the best-fitting subset.
    user_intent = state.user_intent or {}
    selection_inputs = {
        "input": state.input,
        "context": combined_context,
        # SELECTION_PROMPT has a MessagesPlaceholder; it needs Message objects,
        # not raw (user, ai) pairs. Without formatting, langchain treats the
        # first element of each pair as a role name -> MESSAGE_COERCION_FAILURE.
        "chat_history": format_chat_history(state.chat_history or []),
        "user_persona": user_intent.get("as_prompt", ""),
        "requested_count": qvkg_instance._infer_requested_count(state.input),
    }
    selection_inputs = qvkg_instance._limit_context_tokens(selection_inputs)

    raw_selection = await (
        qvkg_instance.SELECTION_PROMPT
        | qvkg_instance.llm.with_config(name="synthesize_selection")
    ).ainvoke(selection_inputs)

    from langchain_core.output_parsers import StrOutputParser
    selection_text = StrOutputParser().invoke(raw_selection)

    payload = {
        "evidence_cards": ranked_cards,
        "input": state.input,
        "selection": selection_text,
    }
    answer = qvkg_instance._build_grounded_answer(payload)

    merged_extra = _merge_extra(extra, {"evidence_cards": combined_cards})
    return {"output": answer, "extra": merged_extra}


# ── guardrails node ───────────────────────────────────────────────────────────

async def guardrails_node(state: AgentState):
    guardrails_instance = InputGuard(app_config)
    result = await guardrails_instance.instance.ainvoke({"input": state.input})
    if "I'm sorry, I can't respond to that." in result.get("output", ""):
        return {
            "next": "FINISH",
            "output": AIMessage(
                content=(
                    "I'm sorry, but I can't answer that question. "
                    "I'm designed to support ToolDB-related bioinformatics tool discovery."
                ),
            ),
        }
    return {"next": "continue"}
