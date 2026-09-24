from bs4 import BeautifulSoup
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from typing import List, Tuple


def _strip_html(value) -> str:
    """Return plain text from a possibly-HTML string (None-safe)."""
    if isinstance(value, BaseMessage):
        value = value.content
    return BeautifulSoup(str(value or ""), features="html.parser").get_text()


def format_chat_history(chat_history: List[Tuple[str, str]]) -> List[BaseMessage]:
    """Format chat history as a flat list of Human/AI message objects.

    Intended for prompts that use a MessagesPlaceholder. Tolerates malformed
    entries instead of crashing:
      - a (human, ai) pair  -> HumanMessage + AIMessage
      - an already-built BaseMessage -> passed through
      - anything else        -> coerced to a HumanMessage of its text
    """
    buffer: List[BaseMessage] = []
    for item in chat_history or []:
        if isinstance(item, BaseMessage):
            buffer.append(item)
            continue
        if isinstance(item, (list, tuple)) and len(item) == 2:
            human, ai = item
            buffer.append(HumanMessage(content=_strip_html(human)))
            buffer.append(AIMessage(content=_strip_html(ai)))
            continue
        # Fallback: unknown shape -> treat as a single human turn.
        buffer.append(HumanMessage(content=_strip_html(item)))
    return buffer


def format_chat_history_text(chat_history: List[Tuple[str, str]]) -> str:
    """Format chat history as a single plain-text block.

    Intended for plain string PromptTemplates (e.g. REPHRASE_PROMPT) whose
    `{chat_history}` slot expects text, NOT message objects. Feeding message
    objects into a string template triggers MESSAGE_COERCION_FAILURE, so this
    keeps the two paths cleanly separated.
    """
    lines: List[str] = []
    for item in chat_history or []:
        if isinstance(item, BaseMessage):
            role = "Assistant" if isinstance(item, AIMessage) else "User"
            lines.append(f"{role}: {_strip_html(item)}")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            human, ai = item
            lines.append(f"User: {_strip_html(human)}")
            lines.append(f"Assistant: {_strip_html(ai)}")
        else:
            lines.append(f"User: {_strip_html(item)}")
    return "\n".join(lines)
