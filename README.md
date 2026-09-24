# CAIRNS Recommendation API

A self-contained REST API that turns a plain-language question into
**evidence-grounded computational-tool recommendations**, drawing from **three
sources**:

1. **ToolDB** — the existing computational-tool catalog (bio.tools / NDE records)
2. **BioModels** — systems-biology models from [biomodels.org](https://www.biomodels.org)
3. **MISM_models** — multiscale simulation models from the [MISM platform](https://mism-dev.renci.org)

It packages the CAIRNS agent (Supervisor + Knowledge-Graph + Vector-Search RAG)
behind a single `POST /recommend` endpoint, so **collaborators can run it in
their own server environment** with Docker — no need to touch the original app.

```
question ──► intent ──► supervisor ──► {KG_lookup, QV_lookup}* ──► synthesize ──► grounded answer + evidence
```

---

## Quick start (Docker — recommended)

```bash
cd cairns_api
cp .env.example .env                 # edit if needed (backend, limits…)

# ToolDB catalog. For a quick trial, use the bundled 50-tool demo:
cp examples/tooldb_demo.json data/tooldb.json
# (For the full catalog, ask the maintainer for the ToolDB JSON and copy it here instead.)

docker compose up -d                 # starts qdrant + ollama + api

# one-time: pull local models (only if MODEL_BACKEND=ollama)
docker compose exec ollama ollama pull llama3.1:8b
docker compose exec ollama ollama pull nomic-embed-text

# one-time: build the three-source index (ToolDB + BioModels + MISM_models)
docker compose run --rm api bash scripts/build_indexes.sh

# ask a question
curl -s localhost:8000/recommend -H 'Content-Type: application/json' \
  -d '{"question":"Recommend tools for RNA-seq differential expression"}' | jq
```

Interactive docs: **http://localhost:8000/docs**

> **GPU note:** the default backend is local Ollama, which is slow on CPU.
> Either give the `ollama` container a GPU (uncomment the `deploy` block in
> `docker-compose.yml`), **or** switch to a cloud LLM in `.env`
> (`MODEL_BACKEND=openai` + `OPENAI_API_KEY`) — then no GPU is needed.

---

## The API

### `POST /recommend`
```jsonc
// request
{ "question": "tools for single-cell immune cell annotation",
  "chat_history": [],            // optional [[user, assistant], ...] for follow-ups
  "thread_id": null }            // optional
// response
{ "answer": "Here are 3 evidence-backed options ... [tool_id] ...",
  "evidence": [
    { "tool_id": "...", "name": "...", "source": "tooldb|biomodels|MISM_models",
      "score": 3.0, "snippet": "...", "why_matched": ["..."], "url": "...",
      "metadata": { "...": "the complete, unfiltered source record" } }
  ],
  "elapsed_seconds": 12.4 }
```
Every recommended tool is cited and appears in `evidence` with its **source**
(`tooldb`, `biomodels`, or `MISM_models`), so results are auditable. `metadata`
carries the full source record — for BioModels/MISM this includes
`raw_metadata`, the untouched API response, so no field is ever dropped.

### `GET /health`
Readiness probe — checks Qdrant, the LLM backend, and the KG index.

### `GET /`
Service metadata.

---

## What each part does

```
cairns_api/
├── app/
│   └── main.py            FastAPI server. Loads the compiled agent graph once,
│                          drives it via a background event loop, exposes
│                          /recommend, /health, /. ← the API surface.
│
├── cairns/                The agent core (importable package; PYTHONPATH root).
│   │                      Copied from the working CAIRNS app, unchanged except
│   │                      for source-tagging (see below).
│   ├── agents/
│   │   ├── route_agentic_graph.py  Builds the LangGraph state machine
│   │   │                           (intent→supervisor→KG/QV→synthesize).
│   │   ├── supervisor.py           Routes each turn to KG, QV, or FINISH.
│   │   ├── intent_agent_graph.py   Classifies query intent + scope.
│   │   └── utils.py                KG/QV retrieval nodes + synthesize node
│   │                               (merge evidence, RRF rank, LLM select, build answer).
│   ├── chains/
│   │   ├── kg_chain.py             Knowledge-graph retrieval (term match → edges → tools).
│   │   ├── question_lookup_chain.py  Vector (semantic) retrieval over Qdrant.
│   │   └── qvkg_chain.py           Merge + Reciprocal-Rank-Fusion + grounded-answer builder.
│   ├── models/
│   │   ├── factory.py              Picks the LLM/embedding backend (ollama|openai).
│   │   ├── ollama_backend.py       Local Ollama chat + embeddings.
│   │   ├── openai_backend.py       Cloud / OpenAI-compatible chat + embeddings.
│   │   └── agent_state.py          Shared graph state (input, extra, output…).
│   ├── databases/
│   │   ├── sqlite_kg.py            Reads the SQLite knowledge graph (KG source of truth).
│   │   └── qdrant.py               Qdrant vector-store access.
│   ├── util/                       Prompt loading, chat-history formatting, helpers.
│   ├── guardrails/input_guard.py   Optional input filtering (off by default).
│   └── config.py                   All settings from environment variables.
│
├── db_builder/            Index-build pipeline (run once to populate the DBs).
│   ├── biomodels_fetch.py     Fetches models from biomodels.org and maps each
│   │                            into the ToolDB record shape, tagged
│   │                            source="biomodels" (keeps the full API
│   │                            response under raw_metadata).
│   ├── mism_fetch.py          Fetches models from the MISM platform and maps
│   │                            each into the ToolDB record shape, tagged
│   │                            source="MISM_models" (keeps the full API
│   │                            response under raw_metadata).
│   ├── merge_sources.py       Combines any number of sources (ToolDB +
│   │                            BioModels + MISM_models, or more later) into
│   │                            one static, source-tagged JSON array.
│   ├── tooldb_prepare_docs.py  Normalizes records → retrieval docs (canonical_text).
│   ├── tooldb_create_embeddings.py  Embeds docs (via the configured backend).
│   ├── qdrant_loader_toodb.py  Loads embeddings into a Qdrant collection (QV index).
│   ├── build_kg_sqlite.py     Builds the SQLite knowledge graph (KG index);
│   │                            stores `source` + full `metadata_json` per tool.
│   └── tooldb_utils.py        Record→document mapping; preserves `source`
│                                and the full record (as `metadata`).
│
├── scripts/
│   └── build_indexes.sh       Orchestrates the whole build: fetch (skipped by
│                              default, reuses the static combined catalog) →
│                              merge → embed → Qdrant → build KG.
│
├── docker/Dockerfile         Image for the API (agent core + FastAPI).
├── docker-compose.yml        api + qdrant + ollama, one command.
├── requirements.txt          Python deps (agent core + FastAPI; no UI deps).
├── .env.example              All configurable settings (copy → .env).
└── data/                     Runtime data (git-ignored): ToolDB json, built
                              indexes, Qdrant storage, Ollama models.
```

---

## The three sources (how it works)

All sources are normalized into the **same schema.org `ComputationalTool`
record shape** and carry a `source` field, so the retrieval layer treats them
uniformly and every evidence card reports where the tool came from.
`merge_sources.py` takes any number of sources (not just these three), so
adding a fourth later is a config change, not a rewrite.

| | ToolDB | BioModels | MISM_models |
|---|--------|-----------|-------------|
| Origin | existing catalog JSON (`data/tooldb.json`) | fetched from biomodels.org | fetched from mism-dev.renci.org |
| Fetched by | (you provide the file) | `db_builder/biomodels_fetch.py` | `db_builder/mism_fetch.py` |
| `source` tag | `tooldb` | `biomodels` | `MISM_models` |
| Built into | same Qdrant collection + same SQLite KG | same Qdrant collection + same SQLite KG | same Qdrant collection + same SQLite KG |

At query time the KG and vector retrievers pull from the combined index; the
`synthesize` step merges, de-duplicates by `tool_id`, ranks with Reciprocal Rank
Fusion, and the LLM selects a grounded subset. Results from all sources are
interleaved and labeled. Every evidence card's `metadata` field carries the
complete, unfiltered source record — for BioModels/MISM this includes
`raw_metadata`, the exact API response, so nothing is lost in the mapping.

Tune the mix in `.env`: `BIOMODELS_LIMIT`, `BIOMODELS_QUERY`, `MISM_LIMIT`,
`MISM_BASE_URL`, `TOOLDB_LIMIT`.

**The combined catalog is static by default.** `build_indexes.sh` writes one
source-tagged snapshot to `data/tools_combined.json` and reuses it on every
subsequent build — it does **not** re-fetch BioModels/MISM over the network
each time. To refresh a source, opt in explicitly:
```bash
# re-merge from whatever's already in data/*.json (no network calls)
FORCE_REBUILD_SOURCES=true docker compose run --rm api bash scripts/build_indexes.sh

# also re-fetch BioModels and/or MISM before merging
FETCH_BIOMODELS=true FETCH_MISM=true FORCE_REBUILD_SOURCES=true \
  docker compose run --rm api bash scripts/build_indexes.sh
```

### What data ships with the repo vs. what you provide

| Data | Included in repo? | How to get it |
|------|-------------------|---------------|
| **BioModels** | not needed | fetched from the public API during the build (opt-in refresh; see above) |
| **MISM_models** | not needed | fetched from the public API during the build (opt-in refresh; see above) |
| **ToolDB — demo (50 tools)** | ✅ yes, `examples/tooldb_demo.json` | just copy it to `data/tooldb.json` — enough to try the API |
| **ToolDB — full catalog** | ❌ no (too large) | ask the maintainer for the full JSON, copy it to `data/tooldb.json` |

Built indexes, `.env`, and the full data are **git-ignored** — the repo carries
code + the small demo dataset only.

---

## Running without Docker (venv)

```bash
cd cairns_api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/cairns"
set -a; source .env; set +a          # or export the vars yourself

# start your own Qdrant + Ollama, then:
bash scripts/build_indexes.sh        # one-time index build
uvicorn app.main:app --host 0.0.0.0 --port 8000 --app-dir "$PWD"
```

---

## Configuration reference

All via environment (see `.env.example`). Key ones:

| Variable | Meaning |
|----------|---------|
| `MODEL_BACKEND` | `ollama` (local, needs GPU) or `openai` (cloud, needs key) |
| `OLLAMA_BASE_URL` / `OLLAMA_CHAT_MODEL` / `OLLAMA_EMBED_MODEL` | local backend |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_CHAT_MODEL` / `OPENAI_EMBED_MODEL` | cloud backend |
| `QDRANT_URL` / `TOOLDB_QDRANT_COLLECTION` | vector store |
| `KG_SQLITE_PATH` | knowledge-graph file |
| `TOOLDB_JSON_PATH` | input ToolDB catalog |
| `TOOLDB_LIMIT` / `BIOMODELS_LIMIT` / `BIOMODELS_QUERY` / `FETCH_BIOMODELS` | ToolDB/BioModels index build inputs |
| `MISM_LIMIT` / `MISM_BASE_URL` / `FETCH_MISM` | MISM_models index build inputs |
| `FORCE_REBUILD_SOURCES` | force a re-merge of the static combined catalog (see above) |

---

## Notes for collaborators

- **Offline / no internet on the build host?** `FETCH_BIOMODELS` and
  `FETCH_MISM` already default to `false` — the build reuses whatever's in
  `data/biomodels.json` / `data/mism_models.json` (produced elsewhere by
  `biomodels_fetch.py` / `mism_fetch.py`) and merges all three sources.
- **No GPU?** Set `MODEL_BACKEND=openai` in `.env` — the API then needs only an
  API key, not a GPU.
- The index build is a **one-time** step; after that just keep the `api`
  container running. Rebuild only when you change the sources or limits.
