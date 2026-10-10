"""Market-based features from daily prices.

Input `bars`: code, date, close (unadjusted), adj_factor, trading_value (J-Quants `Va`, yen).
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd

LOOKBACK_SESSIONS = 60
YEAR_DAYS = 365
MAX_GAP_DAYS = 10  # a reference price older than this is stale (suspension): treated as unknown
SKIP_DAYS = 30  # momentum 12-1: the most recent month is left out (it tends to reverse)


def _before(bars: pd.DataFrame, as_of: date) -> pd.DataFrame:
    frame = bars.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    return frame[frame["date"] < as_of].sort_values(["code", "date"])


def _traded(bars: pd.DataFrame) -> pd.DataFrame:
    """Sessions with an actual close (J-Quants rows on no-trade days have none)."""
    return bars[bars["close"].notna()]


def price_change_12m(bars: pd.DataFrame, as_of: date) -> dict[str, float]:
    """Split-adjusted 12-month price change per code; codes without a year of history are absent."""
    return _price_change(bars, as_of - timedelta(days=YEAR_DAYS), as_of)


def momentum_12_1(bars: pd.DataFrame, as_of: date) -> dict[str, float]:
    """Split-adjusted change from a year before as_of to the last close before as_of − 30 days."""
    return _price_change(bars, as_of - timedelta(days=YEAR_DAYS), as_of - timedelta(SKIP_DAYS))


def _price_change(bars: pd.DataFrame, target: date, end: date) -> dict[str, float]:
    """Last close before `end` vs the last close on or before `target`, on one share basis."""
    every_day = _before(bars, end)  # split factors: from every row, traded or not
    frame = _traded(every_day)  # prices: only sessions with an actual close
    factors_by_code = dict(tuple(every_day.groupby("code")))
    result: dict[str, float] = {}
    for code, group in frame.groupby("code"):
        past = group[group["date"] <= target]
        if past.empty or (target - past["date"].iloc[-1]).days > MAX_GAP_DAYS:
            continue
        start = past.iloc[-1]
        all_rows = factors_by_code[code]
        later = all_rows[all_rows["date"] > start["date"]]
        factor = float(later["adj_factor"].fillna(1.0).astype(float).to_numpy().prod())
        last = float(group["close"].iloc[-1])
        if start["close"] > 0 and pd.notna(last):
            result[str(code)] = last / (float(start["close"]) * factor) - 1
    return result


def median_trading_value(bars: pd.DataFrame, as_of: date) -> dict[str, float]:
    """Median daily trading value (yen) over the last 60 sessions before `as_of`, per code."""
    if "trading_value" not in bars.columns:
        return {}
    frame = _before(bars, as_of).groupby("code").tail(LOOKBACK_SESSIONS)
    medians = frame.groupby("code")["trading_value"].median().dropna()
    return {str(code): float(value) for code, value in medians.items()}


def officer_stake(officers: pd.DataFrame, outstanding: float | None) -> float | None:
    """Shares held by all directors / officers ÷ shares outstanding (None when unknown)."""
    if officers.empty or "shares_held" not in officers.columns:
        return None
    if outstanding is None or outstanding <= 0:
        return None
    held = officers["shares_held"].dropna()
    return float(held.sum()) / outstanding if not held.empty else None


def unusual_trading(
    bars: pd.DataFrame,
    topix: pd.DataFrame,
    codes: set[str],
    as_of: date,
    *,
    short: int,
    long: int,
    volume_ratio: float,
    excess_move: float,
) -> pd.DataFrame:
    """Followed companies trading unusually in the last `short` sessions before `as_of`: median
    trading value ≥ `volume_ratio` × the `long`-session median, or a split-adjusted move beyond
    ±`excess_move` vs TOPIX.

    Columns out: sec_code, volume_ratio, return_5d, excess_5d.
    """
    stock = _traded(_before(bars[bars["code"].astype(str).isin(codes)], as_of))
    index = topix.assign(date=pd.to_datetime(topix["date"]).dt.date)
    index = index[index["date"] < as_of].sort_values("date")
    market = _window_return(index["close"].to_numpy(), short)
    rows = []
    for code, group in stock.groupby("code"):
        recent = group.tail(long + short)
        values = recent["trading_value"].astype(float)
        base = float(values.iloc[:-short].median()) if len(values) > short else float("nan")
        ratio = float(values.tail(short).median()) / base if base and base > 0 else float("nan")
        # Every close on the latest share basis: × each AdjFactor dated after it (price_history).
        factors = recent["adj_factor"].fillna(1.0).astype(float)
        later = factors.iloc[::-1].cumprod().iloc[::-1].shift(-1, fill_value=1.0)
        adjusted = recent["close"].astype(float) * later
        change = _window_return(adjusted.to_numpy(), short)
        excess = change - market
        if (ratio >= volume_ratio) or abs(excess) >= excess_move:
            rows.append({"sec_code": str(code), "volume_ratio": ratio, "return_5d": change,
                         "excess_5d": excess})  # fmt: skip
    return pd.DataFrame(rows, columns=["sec_code", "volume_ratio", "return_5d", "excess_5d"])


def _window_return(closes: Any, sessions: int) -> float:
    """Return over the last `sessions` sessions (close `sessions` sessions ago → last close)."""
    if len(closes) <= sessions or closes[-sessions - 1] <= 0:
        return float("nan")
    return float(closes[-1] / closes[-sessions - 1] - 1)
