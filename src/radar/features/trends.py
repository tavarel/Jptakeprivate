"""Trend features: how long a company has been cheap, and where its cash is going."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from radar.features.valuation import (
    market_cap,
    net_cash,
    price_to_book,
    ratio,
    shares_outstanding,
    split_multiplier,
)

BOOK_VALUE = 1.0  # PBR < 1: the market values the company below its net assets


class CloseHistory:
    """Actual closes per code, for lookups at past dates (fiscal year ends)."""

    def __init__(self, prices: pd.DataFrame, max_staleness_days: int = 10) -> None:
        self._max_staleness = timedelta(days=max_staleness_days)
        self._by_code: dict[str, tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]] = {}
        frame = prices.loc[prices["close"].notna(), ["code", "date", "close"]]
        if frame.empty:
            return
        ids, names = pd.factorize(frame["code"])  # integer ids sort far faster than strings
        days = pd.to_datetime(frame["date"]).to_numpy(dtype="datetime64[D]")
        order = np.lexsort((days, ids))  # one sort, then a slice per code (fast on 5M rows)
        ids, days = ids[order], days[order]
        closes = frame["close"].to_numpy(dtype=float)[order]
        starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
        ends = np.r_[starts[1:], len(ids)]
        self._by_code = {
            str(names[ids[a]]): (days[a:b], closes[a:b]) for a, b in zip(starts, ends, strict=True)
        }

    def close_on_or_before(self, code: str, day: date | None) -> tuple[date, float] | None:
        """(date, close) of the last trade on or before `day`; None if none within staleness or if
        `day` itself is unknown.
        """
        series = self._by_code.get(str(code))
        if series is None or day is None:
            return None
        days, closes = series
        i = int(np.searchsorted(days, np.datetime64(day, "D"), side="right")) - 1
        if i < 0:
            return None
        found = pd.Timestamp(days[i]).date()
        return (found, float(closes[i])) if day - found <= self._max_staleness else None


def year_end_pbr(
    fin: Mapping[str, Any],
    close: tuple[date, float] | None,
    splits: pd.DataFrame | None,
    period_end: date | None,
) -> float | None:
    """PBR at a fiscal year end: year-end shares × the close on or before it."""
    if close is None or period_end is None:
        return None
    price_date, price = close
    shares = shares_outstanding(fin.get("issued"), fin.get("treasury"))
    adjusted = price * split_multiplier(splits, period_end, price_date)
    return price_to_book(market_cap(adjusted, shares), fin.get("net_assets"))


def consecutive_below_book(pbrs: Sequence[float | None]) -> int | None:
    """Years in a row with PBR < 1, newest first."""
    if not pbrs or pbrs[0] is None:
        return None
    count = 0
    for pbr in pbrs:
        if pbr is None or pbr >= BOOK_VALUE:
            break
        count += 1
    return count


def net_cash_change(
    latest: Mapping[str, Any], previous: Mapping[str, Any] | None, mcap: float | None
) -> float | None:
    """(net cash now − net cash one report earlier) / current market cap."""
    if previous is None:
        return None
    now, before = net_cash(latest), net_cash(previous)
    if now is None or before is None:
        return None
    return ratio(now - before, mcap)
