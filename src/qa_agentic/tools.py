from __future__ import annotations
import json
import duckdb
import pandas as pd
from pathlib import Path
from .safety import ensure_read_only

class ToolRegistry:
    def __init__(self, registry_path: str | Path):
        self.registry = json.loads(Path(registry_path).read_text())

    def list_tools(self):
        return self.registry.get("tools", [])

# --- Tool implementations ---

def tool_duckdb_sql(sql: str, paths: dict[str, str] | None = None) -> pd.DataFrame:
    sql = ensure_read_only(sql)
    con = duckdb.connect(database=":memory:")
    try:
        # Register parquet paths as DuckDB views for convenience
        if paths:
            for name, pth in paths.items():
                pth = str(Path(pth))
                con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{pth}')")
        return con.execute(sql).fetchdf()
    finally:
        con.close()


def tool_fetch_schema(resource: str, schema_blob: dict) -> dict:
    for ds in schema_blob.get("datasets", []):
        if ds.get("name") == resource:
            return ds
    raise KeyError(f"Unknown resource: {resource}")