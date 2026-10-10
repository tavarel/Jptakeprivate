"""Original text of a few annual-report sections, quoted in fact packs."""

from __future__ import annotations

import re
from collections.abc import Mapping

import pandas as pd

from radar.text_ja import normalize

_CURRENT_YEAR = "CurrentYear"


def report_excerpts(
    facts: pd.DataFrame, elements: Mapping[str, str], max_chars: int
) -> dict[str, str]:
    """{key: text} for each excerpt element present; current-year context first, cut at `max_chars`
    with '…'.
    """
    result: dict[str, str] = {}
    for key, element_id in elements.items():
        rows = facts[facts["element_id"] == element_id]
        if rows.empty:
            continue
        current = rows[rows["context_id"].astype(str).str.startswith(_CURRENT_YEAR)]
        value = str((current if not current.empty else rows).iloc[0]["value"])
        text = re.sub(r"\s+", " ", normalize(value)).strip()
        if text:
            result[key] = text if len(text) <= max_chars else text[:max_chars].rstrip() + "…"
    return result
