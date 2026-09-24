from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage

from util.llm_helper import LLMFactory


class SupervisorAgent:
    def __init__(self, config):
        self.llm = LLMFactory.get_llm(config=config)

        # Worker agents managed by this supervisor.
        self.members = {
            "KG_lookup": (
                "Structured knowledge-graph lookup. "
                "Best for single-concept queries: a specific tool, format, OS, language, or topic."
            ),
            "QV_lookup": (
                "Semantic vector search over ToolDB questions. "
                "Best for open-ended, multi-concept, or comparative queries."
            ),
        }
        self.options = ["FINISH"] + list(self.members.keys())

    # ── prompt ────────────────────────────────────────────────────────────────

    def _build_prompt(self) -> ChatPromptTemplate:
        system_prompt = (
            "You are a supervisor coordinating two specialist agents:\n"
            "{member_description}\n\n"
            "The user's intent categories are: {intents}. "
            "The query scope is: {scope}.\n\n"
            "Rules:\n"
            "- If an agent has already run (shown in context), do not route to it again.\n"
            "- Prefer KG_lookup for single-entity or structured-constraint queries.\n"
            "- Prefer QV_lookup for semantic, comparative, or open-ended queries.\n"
            "- Reply FINISH when enough evidence has been gathered.\n\n"
            "Return JSON only: {{\"next\": \"<one of {options}>\"}}"
        )
        return ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="history"),
                ("user", "Query: {input}\n\nWhich agent should act next? Choose one of: {options}"),
            ]
        ).partial(
            options=", ".join(self.options),
            member_description="\n".join(
                f"  {name}: {desc}" for name, desc in self.members.items()
            ),
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _normalize_next(self, raw_next, fallback: str) -> str:
        """Ensure next is a valid option string."""
        if isinstance(raw_next, list):
            raw_next = raw_next[0] if raw_next else fallback
        raw_next = str(raw_next).strip()
        return raw_next if raw_next in self.options else fallback

    # ── chain ─────────────────────────────────────────────────────────────────

    def as_generative_chain(self):
        prompt = self._build_prompt()

        def supervisor_logic(state):
            extra = state.extra if hasattr(state, "extra") else (state.get("extra") or {})
            kg_done = bool(extra.get("kg_done"))
            qv_done = bool(extra.get("qv_done"))

            # Both agents have already run → go to synthesis.
            if kg_done and qv_done:
                return {"next": "FINISH"}

            scope = extra.get("scope") or "multiple"
            intents = extra.get("intents") or []
            query = state.input if hasattr(state, "input") else state.get("input", "")
            chat_history = state.chat_history if hasattr(state, "chat_history") else []

            # Build a one-line context summary so the LLM knows what's been done.
            done_summary = []
            if kg_done:
                done_summary.append("KG_lookup has already run")
            if qv_done:
                done_summary.append("QV_lookup has already run")
            context_note = (
                "Already completed: " + "; ".join(done_summary) + "."
                if done_summary
                else "No agents have run yet."
            )

            filled = prompt.partial(intents=intents, scope=scope)
            messages = filled.invoke(
                {
                    "input": f"{query}\n\n[Context: {context_note}]",
                    "history": [HumanMessage(content=str(m)) for m in (chat_history or [])],
                }
            )

            llm_output = self.llm.invoke(messages)
            try:
                parsed = JsonOutputParser().invoke(llm_output)
            except Exception:
                parsed = {}

            raw_next = parsed.get("next", "") if isinstance(parsed, dict) else ""

            # Fallback: if the chosen agent already ran, pick the other or FINISH.
            next_agent = self._normalize_next(raw_next, fallback="FINISH")
            if next_agent == "KG_lookup" and kg_done:
                next_agent = "QV_lookup" if not qv_done else "FINISH"
            elif next_agent == "QV_lookup" and qv_done:
                next_agent = "KG_lookup" if not kg_done else "FINISH"

            return {"next": next_agent}

        return RunnableLambda(supervisor_logic)
