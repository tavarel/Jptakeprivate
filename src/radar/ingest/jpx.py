"""JPX (Tokyo Stock Exchange) capital-plan lists, which have no API."""

from __future__ import annotations

import math
import re
from datetime import date
from pathlib import Path

import pandas as pd

from radar.text_ja import normalize

CAPITAL_PLAN_COLUMNS = [
    "list_month",
    "published_on",
    "sec_code",
    "company_name",
    "market_segment",
    "status",
]
_SHEET_MONTH = re.compile(r"開示企業一覧\s*\((\d{4})年(\d{1,2})月末時点\)")
# Header cell (whitespace removed) → column. The first match wins, so 開示状況 maps to the first
# column under it (要請に基づく開示状況), not to 前月からの開示状況の変更. Do NOT match a bare
# "コード": the sector code (業種コード) sits BEFORE the securities code (証券コード).
_HEADERS = {
    "証券コード": "sec_code",
    "銘柄名": "company_name",
    "市場区分": "market_segment",
    "開示状況": "status",
}
_SEC_CODE = re.compile(r"^[0-9A-Z]{4,5}$")


def list_month_from_sheet(sheet_name: str) -> date | None:
    """First day of the month a sheet describes, or None for unrelated sheets."""
    match = _SHEET_MONTH.search(normalize(sheet_name))
    return date(int(match.group(1)), int(match.group(2)), 1) if match else None


def published_on(list_month: date) -> date:
    """First day of month M+2 for the list of month M (see module docstring)."""
    months = list_month.year * 12 + list_month.month - 1 + 2
    return date(months // 12, months % 12 + 1, 1)


def _to_sec_code(value: object) -> str:
    """JPX 4-character code (7203, 130A) → 5-character EDINET / J-Quants code (72030, 130A0)."""
    text = normalize(str(value)).removesuffix(".0")
    return text + "0" if len(text) == 4 else text


def _cell(value: object) -> str:
    """Cell text without any whitespace ('証券 コード' → '証券コード'); '' for an empty cell."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return re.sub(r"\s+", "", normalize(str(value)))


def _read_sheet(raw: pd.DataFrame, list_month: date) -> pd.DataFrame:
    """One sheet (read without header) → CAPITAL_PLAN_COLUMNS, finding the header row by text."""
    cells = raw.map(_cell)
    header_rows = cells.index[cells.apply(lambda row: "証券コード" in row.to_numpy(), axis=1)]
    if len(header_rows) == 0:
        return pd.DataFrame(columns=CAPITAL_PLAN_COLUMNS)
    columns: dict[int, str] = {}
    for position, name in enumerate(cells.loc[header_rows[0]]):
        if name in _HEADERS and _HEADERS[name] not in columns.values():
            columns[position] = _HEADERS[name]
    body = raw.loc[header_rows[0] + 1 :].iloc[:, list(columns)]
    frame = body.set_axis(list(columns.values()), axis=1)
    frame = frame[
        frame["sec_code"].map(lambda v: bool(_SEC_CODE.match(_cell(v).removesuffix(".0"))))
    ]
    frame = frame.assign(
        sec_code=frame["sec_code"].map(_to_sec_code),
        status=frame["status"].map(_cell),
        list_month=list_month,
        published_on=published_on(list_month),
    )
    return frame.reindex(columns=CAPITAL_PLAN_COLUMNS)


def load_capital_plan_lists(manual_dir: Path) -> pd.DataFrame:
    """Every monthly list in every workbook of `manual_dir`; one row per (list_month, sec_code)."""
    frames: list[pd.DataFrame] = []
    for path in sorted(manual_dir.glob("*.xlsx")):
        sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=object)
        for name, raw in sheets.items():
            month = list_month_from_sheet(str(name))
            if month is not None:
                frames.append(_read_sheet(raw, month))
    non_empty = [f for f in frames if not f.empty]
    if not non_empty:
        return pd.DataFrame(columns=CAPITAL_PLAN_COLUMNS)
    combined = pd.concat(non_empty, ignore_index=True)
    return combined.drop_duplicates(["list_month", "sec_code"], keep="last").reset_index(drop=True)
