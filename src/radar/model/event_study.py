"""Tender-offer premium and pre-announcement run-up (event study)."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

OUTPUT_COLUMNS = [
    "doc_id",
    "subject_edinet_code",
    "event_date",
    "event_type",
    "is_take_private",
    "offer_price",
    "pre_close",
    "premium",
    "runup_abnormal",
    "runup_short",
    "pre_announced",
]


@dataclass(frozen=True)
class EventStudyConfig:
    runup_sessions: int
    max_stale_days: int
    pre_announced_tolerance: float
    runup_short_sessions: int

    @classmethod
    def from_config(cls, raw: Mapping[str, Any]) -> EventStudyConfig:
        return cls(
            int(raw["runup_sessions"]),
            int(raw["max_stale_days"]),
            float(raw["pre_announced_tolerance"]),
            int(raw["runup_short_sessions"]),
        )


def _window_return(sessions: pd.DataFrame, every_day: pd.DataFrame) -> float | None:
    """Split-adjusted return from the first to the last traded session of `sessions`."""
    if len(sessions) < 2:
        return None
    start, end = sessions.iloc[0], sessions.iloc[-1]
    between = every_day[(every_day["date"] > start["date"]) & (every_day["date"] <= end["date"])]
    factor = float(between["adj_factor"].fillna(1.0).astype(float).to_numpy().prod())
    return float(end["close"]) / (float(start["close"]) * factor) - 1


def _one_offer(
    offer: Mapping[Hashable, Any],
    stock: pd.DataFrame | None,
    topix: pd.Series,
    config: EventStudyConfig,
) -> dict[str, Any]:
    out: dict[str, Any] = {k: offer.get(k) for k in OUTPUT_COLUMNS[:6]}
    out.update(
        pre_close=None, premium=None, runup_abnormal=None, runup_short=None, pre_announced=None
    )
    event: date = offer["event_date"]
    if stock is None or offer.get("offer_price") is None:
        return out
    # Only sessions with an actual trade: J-Quants rows on no-trade days have an empty close.
    before = stock[(stock["date"] < event) & stock["close"].notna()]
    if before.empty:
        return out
    last = before.iloc[-1]
    if (event - last["date"]).days > config.max_stale_days:
        return out  # suspended or delisted: no meaningful reference price
    pre_close, price = float(last["close"]), float(offer["offer_price"])
    out.update(pre_close=pre_close, premium=price / pre_close - 1)
    # Price at or above the offer (premium ≤ tolerance): already public, contested or discounted.
    out["pre_announced"] = out["premium"] <= config.pre_announced_tolerance
    out["runup_abnormal"] = _abnormal(before, stock, topix, config.runup_sessions)
    out["runup_short"] = _abnormal(before, stock, topix, config.runup_short_sessions)
    return out


def _abnormal(
    before: pd.DataFrame, every_day: pd.DataFrame, topix: pd.Series, sessions: int
) -> float | None:
    """Stock return minus TOPIX return over the last `sessions` traded sessions (None if short)."""
    window = before.tail(sessions + 1)
    if len(window) < sessions + 1:
        return None
    stock_return = _window_return(window, every_day)
    first, last_day = window["date"].iloc[0], window["date"].iloc[-1]
    if stock_return is None or first not in topix.index or last_day not in topix.index:
        return None
    return stock_return - (float(topix[last_day]) / float(topix[first]) - 1)


def event_study(
    offers: pd.DataFrame,
    prices: pd.DataFrame,
    topix: pd.DataFrame,
    sec_codes: Mapping[str, str],
    config: EventStudyConfig,
) -> pd.DataFrame:
    """One row per offer with premium and run-up (None when prices are missing)."""
    codes = {sec_codes.get(str(c)) for c in offers["subject_edinet_code"]} - {None}
    relevant = prices[prices["code"].isin(codes)].copy()
    relevant["date"] = pd.to_datetime(relevant["date"]).dt.date
    by_code = {str(c): g.sort_values("date") for c, g in relevant.groupby("code")}
    index = topix.assign(date=pd.to_datetime(topix["date"]).dt.date).set_index("date")["close"]
    rows = [
        _one_offer(o, by_code.get(str(sec_codes.get(str(o["subject_edinet_code"])))), index, config)
        for o in offers.to_dict("records")
    ]
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def volume_ratio(bars: pd.DataFrame, event: date, recent: int, baseline: int) -> float | None:
    """Median trading value of the `recent` sessions before `event` ÷ its median over the `baseline`
    sessions before those.
    """
    before = bars[pd.to_datetime(bars["date"]).dt.date < event].sort_values("date")
    if len(before) < recent + baseline:
        return None
    window = before["trading_value"].tail(recent + baseline)
    base = float(window.head(baseline).median())
    last = float(window.tail(recent).median())
    return last / base if base > 0 else None


def earlier_offer_within(offers: pd.DataFrame, days: int) -> pd.Series:
    """True for an offer preceded by ANOTHER offer on the same target within `days` (bidding war,
    revised or second-step bid): its run-up reflects a public offer, not leakage.
    """
    dates = pd.to_datetime(offers["event_date"])
    result = pd.Series(False, index=offers.index)
    for _, group in offers.assign(_d=dates).groupby("subject_edinet_code"):
        ordered = group.sort_values("_d")
        gaps = ordered["_d"].diff().dt.days
        result.loc[ordered.index] = (gaps <= days).to_numpy()
    return result
