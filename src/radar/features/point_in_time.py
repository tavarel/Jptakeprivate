"""Point-in-time selection of filings and prices: nothing public on or after `as_of`."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

import pandas as pd

ANNUAL_TYPES = ("120", "130")


def latest_annual_filings(documents: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Latest usable annual report per company as of `as_of` (00:00, i.e. strictly before T)."""
    return annual_filings_history(documents, as_of, periods=1)


def annual_filings_history(documents: pd.DataFrame, as_of: date, periods: int) -> pd.DataFrame:
    """Up to `periods` fiscal years per company, newest first, as known before `as_of`."""
    cutoff = datetime.combine(as_of, time.min)
    usable = documents[
        documents["doc_type_code"].isin(ANNUAL_TYPES)
        & (pd.to_datetime(documents["submit_datetime"]) < cutoff)
        & (documents["withdrawal_status"].astype(str) == "0")
        & documents["period_end"].notna()
    ].copy()
    if usable.empty:
        return usable

    usable["submit_datetime"] = pd.to_datetime(usable["submit_datetime"])
    usable = usable.sort_values(
        ["edinet_code", "period_end", "submit_datetime"], ascending=[True, False, False]
    )
    per_period = usable.drop_duplicates(["edinet_code", "period_end"])
    return per_period.groupby("edinet_code", as_index=False).head(periods).reset_index(drop=True)


def price_as_of(
    prices: pd.DataFrame, as_of: date, price_lag_days: int = 1, max_staleness_days: int = 10
) -> pd.DataFrame:
    """Last ACTUAL close per code on or before `as_of - price_lag_days` (no-trade days skipped)."""
    reference = as_of - timedelta(days=price_lag_days)
    oldest = reference - timedelta(days=max_staleness_days)
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    # Days without a trade come back with an empty close: they are not a price.
    window = frame[
        (frame["date"] <= reference) & (frame["date"] >= oldest) & frame["close"].notna()
    ]
    latest = window.sort_values(["code", "date"]).groupby("code", as_index=False).tail(1)
    return latest[["code", "date", "close"]].reset_index(drop=True)
