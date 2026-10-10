"""Position sizing for a research portfolio: 1/3-Kelly, capped by liquidity.

Data contract: picks with sec_code, close (yen), median_trading_value_jpy.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from radar.model.inference import newey_west_se


@dataclass(frozen=True)
class AllocationConfig:
    """config/allocation.yaml."""

    kelly_multiplier: float  # 1/3: fraction of full Kelly
    risk_free: float  # annual cash return the basket must beat
    max_invested: float  # cap on the invested fraction of capital
    max_weight: float  # cap per position, fraction of capital
    participation: float  # max share of a day's median trading value we may be
    max_days: int  # days allowed to build a position
    lot_size: int  # 単元株 (shares per trading unit)
    min_dates: int  # fewer past dates than this → no Kelly estimate
    confidence_z: float = 0.0  # μ used = mean − z × Newey-West error (0 = the plain mean)
    nw_lags: int = 0  # overlapping neighbours of each past date (quarterly, 12 months → 3)


def kelly_fraction(
    returns: pd.Series, config: AllocationConfig, benchmark: pd.Series | None = None
) -> float | None:
    """Fraction of capital to put in the basket from its past period returns (one per date); None
    when there are too few dates to estimate.
    """
    known = returns.astype(float)
    hurdle = config.risk_free if benchmark is None else benchmark.astype(float)
    excess = (known - hurdle).dropna()
    if len(excess) < config.min_dates:
        return None
    variance = float(excess.var(ddof=1))
    if variance <= 0:
        return None
    error = newey_west_se(excess, config.nw_lags) if config.confidence_z else 0.0
    full = (float(excess.mean()) - config.confidence_z * error) / variance
    return min(config.max_invested, max(0.0, config.kelly_multiplier * full))


def allocate(
    picks: pd.DataFrame, *, capital: float, invested: float, config: AllocationConfig
) -> pd.DataFrame:
    """One row per pick: target yen, whole-lot shares, days of volume to build, binding cap.

    Columns out: sec_code, close, median_trading_value_jpy, target_jpy, shares, position_jpy,
    days_to_trade, reason.
    """
    equal = capital * invested / len(picks) if len(picks) else 0.0
    rows = [_position(row, capital, equal, config) for row in picks.to_dict("records")]
    columns = [*picks.columns, "target_jpy", "shares", "position_jpy", "days_to_trade", "reason"]
    return pd.DataFrame(rows, columns=columns)


def _position(
    row: Mapping[Hashable, Any], capital: float, equal: float, config: AllocationConfig
) -> dict[Hashable, Any]:
    close = _number(row.get("close"))
    volume = _number(row.get("median_trading_value_jpy"))
    if close is None or close <= 0 or volume is None or volume <= 0:
        reason = "price unknown" if close is None or close <= 0 else "liquidity unknown"
        return {**row, "target_jpy": 0.0, "shares": 0, "position_jpy": 0.0,
                "days_to_trade": math.nan, "reason": reason}  # fmt: skip
    caps = {
        "equal weight": equal,
        "weight cap": capital * config.max_weight,
        "liquidity cap": volume * config.participation * config.max_days,
    }
    reason = min(caps, key=lambda k: caps[k])
    target = caps[reason]
    lot = close * config.lot_size
    shares = int(target // lot) * config.lot_size
    position = shares * close
    return {
        **row,
        "target_jpy": target,
        "shares": shares,
        "position_jpy": position,
        "days_to_trade": position / (volume * config.participation),
        "reason": reason,
    }


def _number(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return None if math.isnan(number) else number
