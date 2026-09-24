"""
CAIRNS Recommendation API (FastAPI).

Exposes the full Supervisor + KG/QV RAG agent as a simple REST endpoint so
collaborators can call it from their own server/environment.

Endpoints
---------
GET  /            → service metadata
GET  /health      → readiness probe (checks Qdrant / Ollama / KG index)
POST /recommend   → run a question through the agent, return grounded answer + evidence

The agent graph is async; we drive it from a single long-lived event loop in a
background thread (run_coroutine_threadsafe), which is safe under a threaded
web server. The compiled graph is loaded once at startup.
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import config  # noqa: F401  (validates env wiring; from cairns/ on PYTHONPATH)
from agents.route_agentic_graph import graph


# ── async runner (one background loop for the whole process) ───────────────────
_loop = asyncio.new_event_loop()


def _start_loop() -> None:
    asyncio.set_event_loop(_loop)
    _loop.run_forever()


threading.Thread(target=_start_loop, daemon=True).start()


def _run_graph(question: str, chat_history: list, thread_id: str) -> dict:
    async def _go():
        return await graph.ainvoke(
            {"input": question, "chat_history": chat_history},
            config={"configurable": {"thread_id": thread_id}},
        )

    return asyncio.run_coroutine_threadsafe(_go(), _loop).result()


# ── request / response models ─────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    question: str = Field(..., description="Natural-language question about computational tools.")
    chat_history: list[list[str]] = Field(
        default_factory=list,
        description="Prior turns as [[user, assistant], ...] for follow-ups.",
    )
    thread_id: Optional[str] = Field(default=None, description="Optional conversation id.")


class EvidenceCard(BaseModel):
    tool_id: str
    name: str = ""
    source: str = "tooldb"          # tooldb | biomodels | MISM_models | ...
    score: float = 0.0
    snippet: str = ""
    why_matched: list[str] = []
    url: str = ""
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "The complete source record, unfiltered — includes raw_metadata "
            "(the untouched source-API response, e.g. every field BioModels "
            "or MISM returned) for sources that provide it."
        ),
    )


class RecommendResponse(BaseModel):
    answer: str
    evidence: list[EvidenceCard]
    elapsed_seconds: float


# ── app ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CAIRNS Recommendation API",
    description="Evidence-grounded computational-tool recommendations (ToolDB + BioModels).",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


def _card_from(raw: dict) -> EvidenceCard:
    payload = raw.get("payload", {}) if isinstance(raw.get("payload"), dict) else {}
    return EvidenceCard(
        tool_id=str(raw.get("tool_id", "")),
        name=str(raw.get("name") or raw.get("tool_id", "")),
        source=str(payload.get("source", raw.get("source", "tooldb"))),
        score=float(raw.get("score", 0.0)),
        snippet=str(raw.get("snippet", ""))[:400],
        why_matched=[str(w) for w in (raw.get("why_matched") or [])][:3],
        url=str(payload.get("url", "")),
        metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
    )


def _collect_evidence(result: dict) -> list[EvidenceCard]:
    extra = result.get("extra", {}) or {}
    cards = extra.get("evidence_cards")
    if not cards:
        cards = (extra.get("kg_result") or {}).get("evidence_cards", []) + (
            extra.get("qv_result") or {}
        ).get("evidence_cards", [])
    return [_card_from(c) for c in (cards or [])[:25]]


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "CAIRNS Recommendation API",
        "version": "1.0.0",
        "sources": ["tooldb", "biomodels"],
        "endpoints": {"POST /recommend": "ask a question", "GET /health": "readiness"},
        "backend": config.MODEL_BACKEND,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    import requests

    status: dict[str, Any] = {"ok": True, "checks": {}}

    try:
        r = requests.get(f"{config.QDRANT_URL}/collections", timeout=3)
        names = {c["name"] for c in r.json().get("result", {}).get("collections", [])}
        ok = config.TOOLDB_QDRANT_COLLECTION in names
        status["checks"]["qdrant"] = "ok" if ok else "collection-missing"
        status["ok"] &= ok
    except Exception as e:  # noqa: BLE001
        status["checks"]["qdrant"] = f"unreachable: {e.__class__.__name__}"
        status["ok"] = False

    if config.MODEL_BACKEND == "ollama":
        try:
            requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3).raise_for_status()
            status["checks"]["ollama"] = "ok"
        except Exception as e:  # noqa: BLE001
            status["checks"]["ollama"] = f"unreachable: {e.__class__.__name__}"
            status["ok"] = False
    else:
        status["checks"]["llm"] = f"backend={config.MODEL_BACKEND}"

    kg_ok = os.path.exists(config.KG_SQLITE_PATH)
    status["checks"]["kg_index"] = "ok" if kg_ok else "missing"
    status["ok"] &= kg_ok
    return status


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    t0 = time.time()
    thread_id = req.thread_id or f"api-{int(t0*1000)}"
    result = _run_graph(req.question, req.chat_history, thread_id)
    out = result.get("output")
    answer = getattr(out, "content", str(out)) if out is not None else ""
    return RecommendResponse(
        answer=answer,
        evidence=_collect_evidence(result),
        elapsed_seconds=round(time.time() - t0, 2),
    )
