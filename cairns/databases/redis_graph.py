from falkordb.asyncio import FalkorDB
from falkordb import FalkorDB as FalkorDB_Sync
from falkordb.graph import QueryResult
from falkordb.node import Node
from falkordb.edge import Edge
from functools import reduce


class RedisGraphDB:
    def __init__(self, config):
        self.db = FalkorDB(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            password=config.REDIS_PASSWORD
        )
        self.db_sync = FalkorDB_Sync(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            password=config.REDIS_PASSWORD
        )
        self.graph = self.db.select_graph(config.REDIS_GRAPH_NAME)
        self.graph_sync = self.db_sync.select_graph(config.REDIS_GRAPH_NAME)

    async def query_graph(self, cypher_str: str):
        return await self.graph.query(cypher_str)

    def query_graph_sync(self, cypher_str: str):
        return self.graph_sync.query(cypher_str)

    def load_term_rows(self, label: str, key_name: str, name_key: str, limit: int):
        query = (
            f"MATCH (n:{label}) "
            f"RETURN n.{key_name} as node_key, n.{name_key} as node_name "
            f"LIMIT {max(int(limit), 1)}"
        )
        results = self.query_graph_sync(query)
        rows = []
        for row in results.result_set:
            rows.append((str(row[0] or ""), str(row[1] or "")))
        return rows

    def query_tools_for_term_rows(self, label: str, key_name: str, node_key: str, limit: int):
        escaped = (node_key or "").replace("\\", "\\\\").replace("'", "\\'")
        query = f"""
            MATCH (t:Tool)-[r]->(n:{label})
            WHERE n.{key_name} = '{escaped}'
            RETURN
                t.tool_id,
                t.name,
                t.description,
                t.identifier,
                t.url,
                type(r),
                n.{key_name},
                n.name
            LIMIT {max(int(limit), 1)}
        """
        results = self.query_graph_sync(query)
        rows = []
        for row in results.result_set:
            rows.append(
                (
                    str(row[0] or ""),
                    str(row[1] or ""),
                    str(row[2] or ""),
                    str(row[3] or ""),
                    str(row[4] or ""),
                    str(row[5] or ""),
                    str(row[6] or ""),
                    str(row[7] or ""),
                )
            )
        return rows

    @staticmethod
    def get_results_as_kgx(results: QueryResult):
        result_set = results.result_set
        every_thing_flat = reduce(lambda x, y: x + y, result_set, [])
        nodes = [x.properties for x in every_thing_flat if isinstance(x, Node)]
        edges = [x.properties for x in every_thing_flat if isinstance(x, Edge)]
        return {
            "nodes": nodes,
            "edges": edges
        }

    @staticmethod
    def get_results_as_table(results: QueryResult):
        table = {}
        for index, header in enumerate(results.header):
            header_label = header[1]
            table[header_label] = [row[index].properties for row in results.result_set]
        return table

