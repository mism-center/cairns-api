from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List, Tuple


class SQLiteKGDB:
    def __init__(self, config):
        self.db_path = Path(config.KG_SQLITE_PATH)

    def _connect(self) -> sqlite3.Connection | None:
        if not self.db_path.exists():
            return None
        # Use short-lived connections to avoid cross-thread SQLite usage errors in async/server contexts.
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def load_term_rows(self, label: str, key_name: str, name_key: str, limit: int) -> List[Tuple[str, str]]:
        del key_name, name_key  # schema stores normalized keys as node_key/node_name.
        conn = self._connect()
        if conn is None:
            return []
        query = """
            SELECT node_key, node_name
            FROM term_nodes
            WHERE label = ?
            ORDER BY length(node_name) DESC
            LIMIT ?
        """
        try:
            rows = conn.execute(query, (label, max(int(limit), 1))).fetchall()
            return [(str(row["node_key"] or ""), str(row["node_name"] or "")) for row in rows]
        finally:
            conn.close()

    def query_tools_for_term_rows(
        self, label: str, key_name: str, node_key: str, limit: int
    ) -> List[Tuple[str, str, str, str, str, str, str, str, str, str]]:
        del key_name  # schema stores normalized keys as node_key.
        conn = self._connect()
        if conn is None:
            return []
        # COALESCE keeps this backward-compatible with KG DBs built before the
        # `source`/`metadata_json` columns existed.
        query = """
            SELECT
                t.tool_id,
                t.name AS tool_name,
                t.description AS tool_description,
                t.identifier AS tool_identifier,
                t.url AS tool_url,
                e.relation,
                e.node_key,
                e.node_name,
                COALESCE(t.source, 'tooldb') AS tool_source,
                COALESCE(t.metadata_json, '{}') AS tool_metadata_json
            FROM term_tool_edges e
            JOIN tool_nodes t ON t.tool_id = e.tool_id
            WHERE e.label = ? AND e.node_key = ?
            LIMIT ?
        """
        try:
            rows = conn.execute(query, (label, node_key, max(int(limit), 1))).fetchall()
            return [
                (
                    str(row["tool_id"] or ""),
                    str(row["tool_name"] or ""),
                    str(row["tool_description"] or ""),
                    str(row["tool_identifier"] or ""),
                    str(row["tool_url"] or ""),
                    str(row["relation"] or ""),
                    str(row["node_key"] or ""),
                    str(row["node_name"] or ""),
                    str(row["tool_source"] or "tooldb"),
                    str(row["tool_metadata_json"] or "{}"),
                )
                for row in rows
            ]
        finally:
            conn.close()
