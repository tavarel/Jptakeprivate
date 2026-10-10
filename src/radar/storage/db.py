"""DuckDB helpers: SQL views over the Parquet files, for exploration and the dashboard."""

from __future__ import annotations

import re
from pathlib import Path

import duckdb

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def connect(db_path: Path) -> duckdb.DuckDBPyConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path))


def refresh_views(con: duckdb.DuckDBPyConnection, *dirs: Path) -> list[str]:
    """Create or replace one view per Parquet file found in `dirs`."""
    created: list[str] = []
    for directory in dirs:
        if not directory.exists():
            continue
        for parquet in sorted(directory.glob("*.parquet")):
            name = parquet.stem
            if not _IDENTIFIER.match(name):
                raise ValueError(f"Parquet file name is not a valid SQL identifier: {parquet}")
            path_literal = str(parquet).replace("'", "''")
            con.execute(
                f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{path_literal}')"
            )
            created.append(name)
    return created
