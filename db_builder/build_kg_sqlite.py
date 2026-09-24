from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config as app_config
from tooldb_utils import (
    as_text,
    collect_text_list,
    ensure_list,
    extract_io_data,
    load_json_array,
    normalize_entity,
    normalize_tool_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ToolDB KG into a local SQLite database.")
    parser.add_argument("--input", dest="input_path", default=app_config.TOOLDB_JSON_PATH)
    parser.add_argument("--output", dest="output_path", default=app_config.KG_SQLITE_PATH)
    parser.add_argument("--clear", action="store_true", help="Delete existing db file before loading")
    parser.add_argument("--limit", dest="limit", type=int, default=0)
    return parser.parse_args()


def create_schema(conn: sqlite3.Connection):
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS tool_nodes (
            tool_id TEXT PRIMARY KEY,
            identifier TEXT,
            name TEXT,
            description TEXT,
            url TEXT,
            source TEXT DEFAULT 'tooldb',
            metadata_json TEXT
        );

        CREATE TABLE IF NOT EXISTS term_nodes (
            label TEXT NOT NULL,
            node_key TEXT NOT NULL,
            node_name TEXT,
            url TEXT,
            PRIMARY KEY (label, node_key)
        );

        CREATE TABLE IF NOT EXISTS term_tool_edges (
            tool_id TEXT NOT NULL,
            label TEXT NOT NULL,
            node_key TEXT NOT NULL,
            node_name TEXT,
            relation TEXT NOT NULL,
            PRIMARY KEY (tool_id, label, node_key, relation)
        );

        CREATE INDEX IF NOT EXISTS idx_term_nodes_label_name
            ON term_nodes(label, node_name);
        CREATE INDEX IF NOT EXISTS idx_term_edges_label_node
            ON term_tool_edges(label, node_key);
        CREATE INDEX IF NOT EXISTS idx_term_edges_tool
            ON term_tool_edges(tool_id);
        """
    )


def upsert_tool(
    conn: sqlite3.Connection,
    tool_id: str,
    identifier: str,
    name: str,
    description: str,
    url: str,
    source: str = "tooldb",
    metadata_json: str = "{}",
):
    conn.execute(
        """
        INSERT INTO tool_nodes(tool_id, identifier, name, description, url, source, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(tool_id) DO UPDATE SET
            identifier=excluded.identifier,
            name=excluded.name,
            description=excluded.description,
            url=excluded.url,
            source=excluded.source,
            metadata_json=excluded.metadata_json
        """,
        (tool_id, identifier, name, description, url, source, metadata_json),
    )


def upsert_term(conn: sqlite3.Connection, label: str, node_key: str, node_name: str, url: str):
    conn.execute(
        """
        INSERT INTO term_nodes(label, node_key, node_name, url)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(label, node_key) DO UPDATE SET
            node_name=CASE
                WHEN excluded.node_name IS NOT NULL AND excluded.node_name <> '' THEN excluded.node_name
                ELSE term_nodes.node_name
            END,
            url=CASE
                WHEN excluded.url IS NOT NULL AND excluded.url <> '' THEN excluded.url
                ELSE term_nodes.url
            END
        """,
        (label, node_key, node_name, url),
    )


def add_edge(
    conn: sqlite3.Connection,
    tool_id: str,
    label: str,
    node_key: str,
    node_name: str,
    relation: str,
):
    conn.execute(
        """
        INSERT OR IGNORE INTO term_tool_edges(tool_id, label, node_key, node_name, relation)
        VALUES (?, ?, ?, ?, ?)
        """,
        (tool_id, label, node_key, node_name, relation),
    )


def add_relation(
    conn: sqlite3.Connection,
    tool_id: str,
    label: str,
    node_key: str,
    node_name: str,
    relation: str,
    url: str = "",
):
    if not node_key:
        return
    upsert_term(conn, label=label, node_key=node_key, node_name=node_name or node_key, url=url or "")
    add_edge(conn, tool_id=tool_id, label=label, node_key=node_key, node_name=node_name or node_key, relation=relation)


def load_tool(conn: sqlite3.Connection, record: dict):
    tool_id = normalize_tool_id(record)
    if not tool_id:
        return

    tool_name = as_text(record.get("name")) or tool_id
    tool_description = as_text(record.get("description"))
    identifier = as_text(record.get("identifier"))
    upsert_tool(
        conn,
        tool_id=tool_id,
        identifier=identifier,
        name=tool_name,
        description=tool_description,
        url=as_text(record.get("url")),
        source=str(record.get("source") or "tooldb"),
        # Full source record (includes raw_metadata for sources that provide
        # it), so a tool found only via KG matching still carries everything.
        metadata_json=json.dumps(record, ensure_ascii=False),
    )

    for topic in [normalize_entity(item) for item in ensure_list(record.get("topicCategory"))]:
        node_id = topic["id"] or topic["name"]
        add_relation(
            conn,
            tool_id=tool_id,
            label="EDAM_TOPIC",
            node_key=node_id,
            node_name=topic["name"],
            relation="HAS_TOPIC",
            url=topic["url"],
        )

    for operation in [normalize_entity(item) for item in ensure_list(record.get("featureList"))]:
        node_id = operation["id"] or operation["name"]
        add_relation(
            conn,
            tool_id=tool_id,
            label="EDAM_OPERATION",
            node_key=node_id,
            node_name=operation["name"],
            relation="HAS_OPERATION",
            url=operation["url"],
        )

    io_data = extract_io_data(record)
    for io_item in io_data["inputs"]:
        for fmt in io_item["formats"]:
            fmt_id = fmt["id"] or fmt["name"]
            add_relation(
                conn,
                tool_id=tool_id,
                label="FORMAT",
                node_key=fmt_id,
                node_name=fmt["name"],
                relation="HAS_INPUT_FORMAT",
                url=fmt["url"],
            )

    for io_item in io_data["outputs"]:
        for fmt in io_item["formats"]:
            fmt_id = fmt["id"] or fmt["name"]
            add_relation(
                conn,
                tool_id=tool_id,
                label="FORMAT",
                node_key=fmt_id,
                node_name=fmt["name"],
                relation="HAS_OUTPUT_FORMAT",
                url=fmt["url"],
            )

    for language in collect_text_list(record.get("programmingLanguage")):
        add_relation(
            conn,
            tool_id=tool_id,
            label="LANGUAGE",
            node_key=language,
            node_name=language,
            relation="USES_LANGUAGE",
        )

    for category in collect_text_list(record.get("applicationCategory")):
        add_relation(
            conn,
            tool_id=tool_id,
            label="APP_CATEGORY",
            node_key=category,
            node_name=category,
            relation="HAS_APP_CATEGORY",
        )

    for os_name in collect_text_list(record.get("operatingSystem")):
        add_relation(
            conn,
            tool_id=tool_id,
            label="OS",
            node_key=os_name,
            node_name=os_name,
            relation="RUNS_ON",
        )


def print_verification(conn: sqlite3.Connection):
    tool_count = conn.execute("SELECT count(*) FROM tool_nodes").fetchone()[0]
    term_count = conn.execute("SELECT count(*) FROM term_nodes").fetchone()[0]
    edge_count = conn.execute("SELECT count(*) FROM term_tool_edges").fetchone()[0]
    print(f"Tool nodes: {tool_count}")
    print(f"Term nodes: {term_count}")
    print(f"Edges: {edge_count}")

    print("Relation counts:")
    rows = conn.execute(
        "SELECT relation, count(*) AS cnt FROM term_tool_edges GROUP BY relation ORDER BY cnt DESC"
    ).fetchall()
    for row in rows:
        print(f"  {row[0]}: {row[1]}")


def main():
    args = parse_args()
    if not args.input_path:
        raise ValueError("Missing --input path. Provide ToolDB JSON via --input or TOOLDB_JSON_PATH env var.")

    records = load_json_array(args.input_path)
    if args.limit and args.limit > 0:
        records = records[: args.limit]

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.clear and output_path.exists():
        output_path.unlink()

    conn = sqlite3.connect(str(output_path))
    create_schema(conn)

    for idx, record in enumerate(records, start=1):
        load_tool(conn, record)
        if idx % 500 == 0:
            conn.commit()
            print(f"Loaded tools: {idx}/{len(records)}")

    conn.commit()
    print(f"Loaded tools: {len(records)}")
    print(f"Wrote: {output_path}")
    print_verification(conn)
    conn.close()


if __name__ == "__main__":
    main()

