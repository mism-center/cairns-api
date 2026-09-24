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
from tooldb_utils import build_tool_documents, load_json_array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ToolDB retrieval documents from JSON records.")
    parser.add_argument("--input", dest="input_path", default=app_config.TOOLDB_JSON_PATH, required=False)
    parser.add_argument("--output", dest="output_path", default=app_config.TOOLDB_PREPARED_DOCS_PATH, required=False)
    parser.add_argument("--chunk-size", dest="chunk_size", type=int, default=app_config.TOOLDB_CHUNK_SIZE)
    parser.add_argument("--limit", dest="limit", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.input_path:
        raise ValueError("Missing --input path. Provide ToolDB JSON via --input or TOOLDB_JSON_PATH env var.")

    records = load_json_array(args.input_path)
    if args.limit and args.limit > 0:
        records = records[: args.limit]

    all_docs = []
    for record in records:
        all_docs.extend(build_tool_documents(record, chunk_size=args.chunk_size))

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for doc in all_docs:
            stream.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"Input tools: {len(records)}")
    print(f"Prepared docs: {len(all_docs)}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
