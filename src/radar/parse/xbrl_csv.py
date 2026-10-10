"""Read the EDINET type=5 ZIP (XBRL converted to CSV) into one long "facts" table."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

COLUMN_MAP: dict[str, str] = {
    "要素ID": "element_id",
    "項目名": "label",
    "コンテキストID": "context_id",
    "相対年度": "relative_year",
    "連結・個別": "consolidation",
    "期間・時点": "period_type",
    "ユニットID": "unit_id",
    "単位": "unit",
    "値": "value",
}
FACT_COLUMNS = list(COLUMN_MAP.values())


def _decode(raw: bytes) -> str:
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    # UTF-16LE without BOM: every other byte of ASCII text is NUL.
    if raw[:200].count(b"\x00") > 50:
        return raw.decode("utf-16-le")
    return raw.decode("utf-8-sig")


def read_facts_csv(raw: bytes) -> pd.DataFrame:
    """Parse one EDINET CSV file (bytes) into the standard facts columns, all as strings."""
    text = _decode(raw)
    first_line = text.split("\n", 1)[0]
    sep = "\t" if "\t" in first_line else ","
    frame = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, keep_default_na=False)
    frame.columns = [str(c).strip().strip('"') for c in frame.columns]
    missing = [c for c in COLUMN_MAP if c not in frame.columns]
    if missing:
        raise ValueError(f"Unexpected EDINET CSV format, missing columns: {missing}")
    out = frame.rename(columns=COLUMN_MAP)[FACT_COLUMNS]
    # "－" or "-" is used for "not applicable"; treat as empty.
    out["value"] = out["value"].replace({"－": "", "-": ""})
    return out


def read_facts_zip(zip_path: Path, prefix: str = "jpcrp") -> pd.DataFrame:
    """All facts from the main report CSV(s) inside an EDINET type=5 ZIP."""
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(zip_path) as archive:
        for name in sorted(archive.namelist()):
            filename = name.rsplit("/", 1)[-1]
            if filename.lower().endswith(".csv") and filename.startswith(prefix):
                facts = read_facts_csv(archive.read(name))
                facts["source_file"] = filename
                frames.append(facts)
    if not frames:
        raise ValueError(f"No '{prefix}*.csv' file found in {zip_path.name}")
    return pd.concat(frames, ignore_index=True)
