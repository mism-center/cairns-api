"""
Supervisor + KG/QV two-agent RAG graph.

Graph layout
------------

  START
    │
    ▼
  intent_node          ← writes scope/intents into extra
    │
    ▼
  supervisor ──────────────────────────────────────────┐
    │                                                   │
    │ "KG_lookup"          "QV_lookup"                  │ "FINISH"
    ▼                          ▼                        ▼
  KG_lookup             QV_lookup              synthesize_node
    │                          │                        │
    └──────────────────────────┘                        ▼
                 │                                     END
                 ▼
           supervisor  (loop back — decides next agent or FINISH)


KG_lookup  : runs KGChain.as_retrival_chain(), stores results in extra["kg_result"]
QV_lookup  : runs QuestionLookupChain.as_retrieval_chain(), stores in extra["qv_result"]
supervisor : LLM decides KG_lookup | QV_lookup | FINISH (returns {"next": str})
synthesize : combines KG+QV evidence → final grounded answer
"""
import functools

from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver

import config
import config as app_config
from agents.supervisor import SupervisorAgent
from agents.intent_agent_graph import intent_node
from agents.utils import kg_lookup_node, qv_lookup_node, synthesize_node
from chains.kg_chain import KGChain
from chains.question_lookup_chain import QuestionLookupChain
from chains.qvkg_chain import QVKGChain
from models.agent_state import AgentState


# ── agent / chain instances ────────────────────────────────────────────────────

_kg_chain = KGChain(app_config)
_qv_chain = QuestionLookupChain(app_config)
_qvkg_instance = QVKGChain(app_config)
_supervisor = SupervisorAgent(app_config)

# Partial node functions bound to the concrete chain / instance.
_kg_node = functools.partial(kg_lookup_node, kg_retrieval_chain=_kg_chain.as_retrival_chain())
_qv_node = functools.partial(qv_lookup_node, qv_retrieval_chain=_qv_chain.as_retrieval_chain())
_synth_node = functools.partial(synthesize_node, qvkg_instance=_qvkg_instance)


# ── routing helper ─────────────────────────────────────────────────────────────

def _route_supervisor(state: AgentState) -> str:
    """Read state.next and map to a graph node name or END sentinel."""
    nxt = state.next if hasattr(state, "next") else (state.get("next") or "FINISH")
    members = set(_supervisor.members.keys())
    if nxt in members:
        return nxt
    if nxt == "FINISH":
        return "synthesize"
    # Unexpected value → synthesize to avoid hanging.
    return "synthesize"


# ── graph assembly ─────────────────────────────────────────────────────────────

workflow = StateGraph(AgentState)

workflow.add_node("intent", intent_node)
workflow.add_node("supervisor", _supervisor.as_generative_chain())
workflow.add_node("KG_lookup", _kg_node)
workflow.add_node("QV_lookup", _qv_node)
workflow.add_node("synthesize", _synth_node)

# Linear entry path.
workflow.add_edge(START, "intent")
workflow.add_edge("intent", "supervisor")

# Supervisor routes to an agent OR triggers synthesis.
workflow.add_conditional_edges(
    "supervisor",
    _route_supervisor,
    {
        "KG_lookup": "KG_lookup",
        "QV_lookup": "QV_lookup",
        "synthesize": "synthesize",
    },
)

# Each agent loops back to supervisor for a routing decision.
workflow.add_edge("KG_lookup", "supervisor")
workflow.add_edge("QV_lookup", "supervisor")

# Synthesis is terminal.
workflow.add_edge("synthesize", END)

# ── compile ────────────────────────────────────────────────────────────────────

_extra_kwargs = {}
if config.LANGFUSE_ENABLED:
    from langfuse.callback import CallbackHandler
    from langchain.callbacks.manager import CallbackManager
    _cb = CallbackHandler(
        host=config.LANGFUSE_HOST,
        secret_key=config.LANGFUSE_SECRET_KEY,
        public_key=config.LANGFUSE_PUBLIC_KEY,
    )
    _extra_kwargs["callbacks"] = CallbackManager([_cb])

graph = workflow.compile(checkpointer=MemorySaver())
if _extra_kwargs:
    graph = graph.with_config(**_extra_kwargs)


# ── quick smoke-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio

    graph.get_graph().print_ascii()

    async def _run():
        cfg = {"configurable": {"thread_id": "test-1"}}
        result = await graph.ainvoke(
            {"input": "Find Python tools for FASTQ processing on Linux", "chat_history": []},
            config=cfg,
        )
        print("\n=== output ===")
        output = result.get("output")
        print(output.content if output else "(no output)")

    asyncio.run(_run())
