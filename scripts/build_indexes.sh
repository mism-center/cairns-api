#!/usr/bin/env bash
# Build the multi-source index (ToolDB + BioModels + MISM_models) → Qdrant + SQLite KG.
#
#   1) fetch BioModels + MISM metadata     (network; skipped by default — see below)
#   2) merge all sources                   → one static JSON array (source-tagged)
#   3) prepare docs → embeddings → Qdrant  (QV index)
#   4) build SQLite knowledge graph        (KG index)
#
# Static by default: once data/tools_combined.json exists, this script reuses
# it as-is and does NOT hit BioModels/MISM over the network again — it's a
# pre-compiled snapshot, not fetched on the fly on every build. To refresh a
# source (or all of them), opt in explicitly:
#   FORCE_REBUILD_SOURCES=true bash scripts/build_indexes.sh   # re-merge from
#                                                               # existing per-source files
#   FETCH_BIOMODELS=true FORCE_REBUILD_SOURCES=true bash ...   # + re-fetch BioModels
#   FETCH_MISM=true FORCE_REBUILD_SOURCES=true bash ...        # + re-fetch MISM
#
# Env (see .env.example): TOOLDB_JSON_PATH, BIOMODELS_LIMIT, TOOLDB_LIMIT,
# MISM_LIMIT, MISM_BASE_URL, QDRANT_URL, MODEL_BACKEND, OLLAMA_*/OPENAI_*,
# KG_SQLITE_PATH, collection name.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
export PYTHONPATH="$HERE/cairns:${PYTHONPATH:-}"
PY="${PYTHON:-python}"

# Prefer the docker-compose bind mount (./data:/data) so every file this
# script writes actually persists on the host; fall back to a local ./data
# for venv/bare-metal runs where /data won't exist.
if [ -d /data ]; then
  DATA="/data"
else
  DATA="$HERE/data"
fi
mkdir -p "$DATA" "$(dirname "${KG_SQLITE_PATH:-$DATA/tooldb_kg.sqlite}")"

: "${TOOLDB_JSON_PATH:?Set TOOLDB_JSON_PATH to the ToolDB catalog JSON}"
BIOMODELS_LIMIT="${BIOMODELS_LIMIT:-200}"
BIOMODELS_QUERY="${BIOMODELS_QUERY:-*}"
MISM_LIMIT="${MISM_LIMIT:-1000}"
MISM_BASE_URL="${MISM_BASE_URL:-https://mism-dev.renci.org/api/v1}"
TOOLDB_LIMIT="${TOOLDB_LIMIT:-300}"
COMBINED="${TOOLS_COMBINED_PATH:-$DATA/tools_combined.json}"
PREP="${TOOLDB_PREPARED_DOCS_PATH:-$DATA/tools_docs.jsonl}"
EMB="${TOOLDB_EMBEDDED_DOCS_PATH:-$DATA/tools_docs_with_embeddings.jsonl}"
KG="${KG_SQLITE_PATH:-$DATA/tooldb_kg.sqlite}"
COLLECTION="${TOOLDB_QDRANT_COLLECTION:-${QDRANT_COLLECTION_NAME:-tooldb_tools}}"

if [[ -f "$COMBINED" && "${FORCE_REBUILD_SOURCES:-false}" != "true" ]]; then
  echo "[1-2/4] using existing static combined catalog (unchanged): $COMBINED"
  echo "        (set FORCE_REBUILD_SOURCES=true to re-merge, add FETCH_BIOMODELS=true /"
  echo "         FETCH_MISM=true to also re-fetch those sources over the network)"
else
  # 1) BioModels (skip with FETCH_BIOMODELS=false — default, reuse existing data/biomodels.json)
  if [[ "${FETCH_BIOMODELS:-false}" == "true" ]]; then
    echo "[1/4] fetching BioModels (query='$BIOMODELS_QUERY' limit=$BIOMODELS_LIMIT)"
    "$PY" db_builder/biomodels_fetch.py \
      --output "$DATA/biomodels.json" --query "$BIOMODELS_QUERY" --limit "$BIOMODELS_LIMIT"
  else
    echo "[1/4] skipping BioModels fetch (FETCH_BIOMODELS=false); reusing $DATA/biomodels.json if present"
  fi

  # 1b) MISM_models (skip with FETCH_MISM=false — default, reuse existing data/mism_models.json)
  if [[ "${FETCH_MISM:-false}" == "true" ]]; then
    echo "[1/4] fetching MISM_models (base_url=$MISM_BASE_URL limit=$MISM_LIMIT)"
    "$PY" db_builder/mism_fetch.py \
      --output "$DATA/mism_models.json" --base-url "$MISM_BASE_URL" --limit "$MISM_LIMIT"
  else
    echo "[1/4] skipping MISM fetch (FETCH_MISM=false); reusing $DATA/mism_models.json if present"
  fi

  # 2) merge all sources into one static, source-tagged JSON array
  echo "[2/4] merging sources (tooldb + biomodels + MISM_models) → $COMBINED"
  "$PY" db_builder/merge_sources.py \
    --tooldb "$TOOLDB_JSON_PATH" \
    --biomodels "$DATA/biomodels.json" \
    --source "MISM_models=$DATA/mism_models.json" \
    --output "$COMBINED" \
    --tooldb-limit "$TOOLDB_LIMIT"
fi

# 3) QV: prepare → embed → load Qdrant
echo "[3/4] building QV index (embeddings → Qdrant '$COLLECTION')"
"$PY" db_builder/tooldb_prepare_docs.py --input "$COMBINED" --output "$PREP" --limit 0
"$PY" db_builder/tooldb_create_embeddings.py --input "$PREP" --output "$EMB"
"$PY" db_builder/qdrant_loader_toodb.py \
  --input "$EMB" --qdrant-url "${QDRANT_URL:-http://127.0.0.1:6333}" \
  --collection "$COLLECTION" --recreate

# 4) KG: build SQLite graph from the combined (source-tagged) records
echo "[4/4] building KG index (SQLite → $KG)"
"$PY" db_builder/build_kg_sqlite.py --input "$COMBINED" --output "$KG" --limit 0 --clear

echo "[done] multi-source index built. QV collection='$COLLECTION'  KG='$KG'  combined='$COMBINED'"
