"""
Intent analysis utilities used by the supervisor graph.

Only pure node functions live here.
The graph itself is assembled in route_agentic_graph.py.
"""
import json
from langchain_core.messages import HumanMessage

from util.llm_helper import LLMFactory
import config as app_config
from models.agent_state import AgentState


def analyze_intent_and_scope(query: str) -> dict:
    """Classify query intent category and entity scope via two LLM calls."""
    intent_prompt = (
        f"Classify the intent of this query into one or more category numbers: '{query}'.\n"
        "Categories: 1.Factual 2.Explanatory 3.Troubleshooting 4.Decision-support "
        "5.Learning 6.Personal-advice 7.Data-processing 8.Research 9.Not-research-related\n"
        "Return only comma-separated integers, e.g. '1,4'."
    )
    scope_prompt = (
        f"Does this query refer to a single entity or multiple entities: '{query}'?\n"
        "Reply with exactly one word: single OR multiple."
    )

    llm = LLMFactory.get_llm(config=app_config)

    intent_msg = llm.invoke(HumanMessage(content=intent_prompt))
    scope_msg = llm.invoke(HumanMessage(content=scope_prompt))

    intent_text = intent_msg.content if hasattr(intent_msg, "content") else str(intent_msg)
    scope_text = scope_msg.content if hasattr(scope_msg, "content") else str(scope_msg)

    try:
        intents = [int(x.strip()) for x in intent_text.split(",") if x.strip().isdigit()]
    except Exception:
        intents = []

    scope = scope_text.strip().lower()
    if scope not in {"single", "multiple"}:
        scope = "multiple"

    return {"intents": intents, "scope": scope}


def intent_node(state: AgentState) -> dict:
    """Extract intent and scope from state.input, write into state.extra."""
    query = state.input
    analysis = analyze_intent_and_scope(query)

    current_extra = dict(state.extra or {})
    current_extra["intents"] = analysis["intents"]
    current_extra["scope"] = analysis["scope"]

    return {"extra": current_extra}


def extract_user_preferences_node(state: AgentState) -> dict:
    """Infer response-style preferences from chat history, write into state.extra."""
    chat_history = state.chat_history or []
    preference_prompt = (
        "Analyze the user's past messages and infer preferences for response formatting, "
        "content restrictions, and preferred response structure.\n"
        "Respond in JSON with keys: 'blocked_terms' (list), 'response_format' (string).\n"
        "Chat history:\n" + str(chat_history)
    )

    llm = LLMFactory.get_llm(config=app_config)
    response = llm.invoke(HumanMessage(content=preference_prompt))
    response_text = response.content if hasattr(response, "content") else str(response)

    try:
        preferences = json.loads(response_text)
        if not isinstance(preferences, dict):
            raise ValueError
    except Exception:
        preferences = {"blocked_terms": [], "response_format": "list"}

    current_extra = dict(state.extra or {})
    current_extra["user_preferences"] = preferences

    return {"extra": current_extra}
