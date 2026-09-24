from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config as app_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load ToolDB embeddings into a Qdrant collection.")
    parser.add_argument("--input", dest="input_path", default=app_config.TOOLDB_EMBEDDED_DOCS_PATH)
    parser.add_argument("--qdrant-url", dest="qdrant_url", default=app_config.QDRANT_URL)
    parser.add_argument("--collection", dest="collection_name", default=app_config.TOOLDB_QDRANT_COLLECTION)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=256)
    parser.add_argument("--recreate", action="store_true")
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


def stable_point_id(doc_id: str) -> int:
    digest = hashlib.sha1(doc_id.encode("utf-8")).hexdigest()
    return int(digest[:15], 16)


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int, recreate: bool):
    vectors_config = VectorParams(size=vector_size, distance=Distance.COSINE)
    if recreate:
        client.recreate_collection(collection_name=collection_name, vectors_config=vectors_config)
        return
    try:
        client.get_collection(collection_name=collection_name)
    except Exception:
        client.create_collection(collection_name=collection_name, vectors_config=vectors_config)


def batched(docs: list[dict], size: int):
    for index in range(0, len(docs), size):
        yield docs[index : index + size]


def main():
    args = parse_args()
    input_path = Path(args.input_path)
    docs = load_jsonl(input_path)
    if not docs:
        raise RuntimeError(f"No embedded documents found in {input_path}")

    first_vector = docs[0].get("vector")
    if not isinstance(first_vector, list) or not first_vector:
        raise RuntimeError("Embedded docs are missing 'vector' values.")

    client = QdrantClient(url=args.qdrant_url)
    ensure_collection(
        client=client,
        collection_name=args.collection_name,
        vector_size=len(first_vector),
        recreate=args.recreate,
    )

    uploaded = 0
    for batch in batched(docs, args.batch_size):
        points = []
        for doc in batch:
            payload = dict(doc.get("payload", {}))
            payload.update(
                {
                    "doc_id": doc.get("doc_id"),
                    "tool_id": doc.get("tool_id"),
                    "tool_name": doc.get("tool_name"),
                    "canonical_text": doc.get("canonical_text"),
                    "chunk_index": doc.get("chunk_index"),
                    "chunk_count": doc.get("chunk_count"),
                }
            )
            points.append(
                PointStruct(
                    id=stable_point_id(str(doc.get("doc_id"))),
                    vector=doc["vector"],
                    payload=payload,
                )
            )

        client.upsert(collection_name=args.collection_name, points=points)
        uploaded += len(points)
        print(f"Uploaded {uploaded}/{len(docs)}")

    print(f"Collection: {args.collection_name}")
    print(f"Qdrant URL: {args.qdrant_url}")
    print(f"Uploaded points: {uploaded}")


if __name__ == "__main__":
    main()
