from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config as app_config
from models.factory import get_embedding_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create embeddings for prepared ToolDB retrieval docs.")
    parser.add_argument("--input", dest="input_path", default=app_config.TOOLDB_PREPARED_DOCS_PATH)
    parser.add_argument("--output", dest="output_path", default=app_config.TOOLDB_EMBEDDED_DOCS_PATH)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=64)
    parser.add_argument("--max-chars", dest="max_chars", type=int, default=app_config.TOOLDB_EMBED_MAX_CHARS)
    parser.add_argument("--min-chars", dest="min_chars", type=int, default=app_config.TOOLDB_EMBED_MIN_CHARS)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    docs = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs


def write_jsonl(path: Path, docs: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for doc in docs:
            stream.write(json.dumps(doc, ensure_ascii=False) + "\n")


def batched(values: list[str], size: int):
    for index in range(0, len(values), size):
        yield index, values[index : index + size]


def is_context_length_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "context length" in msg or "input length exceeds" in msg


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def embed_one_with_backoff(embedding_model, text: str, min_chars: int) -> list[float]:
    candidate = text
    while True:
        try:
            vectors = embedding_model.embed([candidate])
            if not vectors or not isinstance(vectors[0], list):
                raise RuntimeError("Embedding model returned empty vector for single input.")
            return vectors[0]
        except Exception as exc:
            if not is_context_length_error(exc):
                raise
            if len(candidate) <= min_chars:
                raise RuntimeError(
                    f"Failed embedding due to context limit even at {len(candidate)} chars."
                ) from exc
            candidate = candidate[: max(min_chars, len(candidate) // 2)]


def main():
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    docs = load_jsonl(input_path)
    if not docs:
        raise RuntimeError(f"No documents found in {input_path}")

    embedding_model = get_embedding_model(app_config)
    texts = [truncate_text(doc.get("canonical_text", ""), args.max_chars) for doc in docs]
    all_vectors = [None] * len(docs)

    for start_index, batch in batched(texts, args.batch_size):
        try:
            vectors = embedding_model.embed(batch)
        except Exception as exc:
            if not is_context_length_error(exc):
                raise
            print(
                f"Batch context-length failure at {start_index}-{start_index + len(batch) - 1}; "
                "falling back to per-document retry with truncation."
            )
            vectors = [embed_one_with_backoff(embedding_model, text, args.min_chars) for text in batch]
        for offset, vector in enumerate(vectors):
            all_vectors[start_index + offset] = vector
        print(f"Embedded {min(start_index + len(batch), len(texts))}/{len(texts)} docs")

    for idx, vector in enumerate(all_vectors):
        if vector is None:
            raise RuntimeError(f"Missing vector for document index {idx}")
        docs[idx]["vector"] = vector

    write_jsonl(output_path, docs)
    print(f"Embedded docs: {len(docs)}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
