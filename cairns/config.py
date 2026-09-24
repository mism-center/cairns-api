import os
from pathlib import Path

from langfuse import Langfuse


def _legacy_to_backend_name(server_type: str) -> str:
    normalized = (server_type or "").lower()
    if normalized in {"vllm", "openai"}:
        return "openai"
    if normalized == "ollama":
        return "ollama"
    return normalized or "openai"


# Legacy vars retained for compatibility with existing runtime setup.
LLM_URL = os.getenv("LLM_URL", "https://vllm.apps.renci.org/v1").rstrip("/")
EMBEDDING_URL = os.getenv("EMBEDDING_URL", "http://localhost:11434").rstrip("/")
LLM_SERVER_TYPE = os.getenv("LLM_SERVER_TYPE", "VLLM")

# New backend-router vars.
MODEL_BACKEND = os.getenv("MODEL_BACKEND", _legacy_to_backend_name(LLM_SERVER_TYPE)).lower()
MODEL_TIMEOUT_SECONDS = float(os.getenv("MODEL_TIMEOUT_SECONDS", "120"))

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", os.getenv("GEN_MODEL_NAME", "llama3.1:latest"))
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", os.getenv("EMB_MODEL_NAME", "bge-m3"))
_ollama_embed_dimensions_raw = os.getenv("OLLAMA_EMBED_DIMENSIONS", "").strip()
OLLAMA_EMBED_DIMENSIONS = int(_ollama_embed_dimensions_raw) if _ollama_embed_dimensions_raw else None

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", LLM_URL).rstrip("/")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", os.getenv("GEN_MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct"))
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", os.getenv("EMB_MODEL_NAME", "text-embedding-3-small"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", os.getenv("GEN_API_KEY", "EMPTY"))

GEN_MODEL_NAME = os.getenv("GEN_MODEL_NAME", OPENAI_CHAT_MODEL if MODEL_BACKEND == "openai" else OLLAMA_CHAT_MODEL)
GEN_TEMPERATURE = float(os.getenv("GEN_TEMPERATURE", "0"))
GEN_API_KEY = os.getenv("GEN_API_KEY", OPENAI_API_KEY)
GUARDIAN_MODEL_NAME = os.getenv("GUARDIAN_MODEL_NAME", "llama3.1:latest")
GUARDIAN_MODEL_HOST = os.getenv("GUARDIAN_MODEL_URL", OLLAMA_BASE_URL)
EMB_MODEL_NAME = os.getenv("EMB_MODEL_NAME", OPENAI_EMBED_MODEL if MODEL_BACKEND == "openai" else OLLAMA_EMBED_MODEL)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333").rstrip("/")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "tooldb_tools")
STUDIES_JSON_FILE = os.getenv("STUDIES_JSON_FILE", Path(os.path.dirname(__file__), "..", "data", "99_studies.json"))
LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", Path(os.path.dirname(__file__), "..", "cairns.log"))

LANGFUSE_ENABLED = os.getenv("LANGFUSE_ENABLED", "false").lower() == "true"
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
REDIS_GRAPH_NAME = os.getenv("REDIS_GRAPH_NAME", "tooldb")
KG_BACKEND = os.getenv("KG_BACKEND", "sqlite").lower()
KG_SQLITE_PATH = os.getenv(
    "KG_SQLITE_PATH",
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "db_builder", "output", "tooldb_kg.sqlite"),
)

TMP_DIR = os.getenv("TMP_DIR", os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "tmp"))
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
APP_ID = os.getenv("APP_ID", "QV_KG_NO_ROUTE")
SERVER_ROOT_URL = os.getenv("ROOT_URL", "/agent")
PROMPTS_FILE = os.getenv("PROMPTS_FILE", "")
INPUT_GUARD_ENABLED = os.getenv("INPUT_GUARD_ENABLED", "false").lower() == "true"

# ToolDB builder defaults.
TOOLDB_JSON_PATH = os.getenv("TOOLDB_JSON_PATH", "")
TOOLDB_PREPARED_DOCS_PATH = os.getenv("TOOLDB_PREPARED_DOCS_PATH", "db_builder/output/tooldb_docs.jsonl")
TOOLDB_EMBEDDED_DOCS_PATH = os.getenv("TOOLDB_EMBEDDED_DOCS_PATH", "db_builder/output/tooldb_docs_with_embeddings.jsonl")
TOOLDB_QDRANT_COLLECTION = os.getenv("TOOLDB_QDRANT_COLLECTION", QDRANT_COLLECTION_NAME)
TOOLDB_CHUNK_SIZE = int(os.getenv("TOOLDB_CHUNK_SIZE", "0"))
TOOLDB_QV_TOP_K = int(os.getenv("TOOLDB_QV_TOP_K", "20"))
TOOLDB_EMBED_MAX_CHARS = int(os.getenv("TOOLDB_EMBED_MAX_CHARS", "12000"))
TOOLDB_EMBED_MIN_CHARS = int(os.getenv("TOOLDB_EMBED_MIN_CHARS", "512"))
KG_ENABLE_LLM_PARSING = os.getenv("KG_ENABLE_LLM_PARSING", "false").lower() == "true"
KG_TERM_MATCH_LIMIT = int(os.getenv("KG_TERM_MATCH_LIMIT", "50"))
KG_TOOL_RESULT_LIMIT = int(os.getenv("KG_TOOL_RESULT_LIMIT", "25"))


if LANGFUSE_ENABLED:
    langfuse = Langfuse(secret_key=LANGFUSE_SECRET_KEY,
                        public_key=LANGFUSE_PUBLIC_KEY,
                        host=LANGFUSE_HOST)
else:
    langfuse = None


def configure_langfuse(runnable):
    if LANGFUSE_ENABLED:
        from langchain_core.runnables.config import RunnableConfig
        from langfuse.callback import CallbackHandler

        langfuse_handler = CallbackHandler(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
            metadata={
                "cairns_version": "v1.0.1",
                "guardian_model": GUARDIAN_MODEL_NAME,
                "embedding_model": EMB_MODEL_NAME,
                "generative_model": GEN_MODEL_NAME,
            },
            tags=[
                GEN_MODEL_NAME,
                APP_ID,
                ENVIRONMENT
            ],
            environment=ENVIRONMENT
        )
        langfuse_handler.auth_check()
        runnable_config = RunnableConfig(callbacks=[langfuse_handler])
        return runnable.with_config(runnable_config)
    return runnable
