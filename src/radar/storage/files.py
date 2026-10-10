"""File writes that other processes may read at the same time."""

from __future__ import annotations

import os
import re
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Final

import pandas as pd

from radar.storage.paths import ensure_parent


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    handle, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp")
    os.close(handle)
    temporary = Path(name)
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


# Daily marts (features, scores, live snapshots) are written every weekday: zstd makes them
# ~40% smaller than the default snappy for the same read speed.
DAILY_COMPRESSION: Final = "zstd"
_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def expired_dated_files(folder: Path, prefix: str, today: date, keep_days: int) -> list[Path]:
    """Files `<prefix>…<YYYY-MM-DD>…` in `folder` dated more than `keep_days` before `today`."""
    limit = today - timedelta(days=keep_days)
    old = []
    for path in sorted(folder.glob(f"{prefix}*")):
        found = _DATE.search(path.name)
        if found and date.fromisoformat(found.group(1)) < limit:
            old.append(path)
    return old
