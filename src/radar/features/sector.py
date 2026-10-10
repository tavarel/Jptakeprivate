"""Sector-relative valuation: is a company cheap for its industry?"""

from __future__ import annotations

import pandas as pd


def sector_percentiles(frame: pd.DataFrame, column: str, min_peers: int) -> pd.Series:
    """Percentile rank of `column` within each `sector33` (ties share the average rank)."""
    values = pd.to_numeric(frame[column], errors="coerce")
    groups = frame["sector33"]
    peers = values.groupby(groups).transform("count")
    pct = values.groupby(groups).rank(pct=True)
    return pct.where(peers >= min_peers).rename(f"{column}_sector_percentile")
