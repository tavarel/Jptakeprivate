"""Valuation features."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any

import pandas as pd


def _num(record: Mapping[str, float | bool | None], key: str) -> float:
    value = record.get(key)
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0.0


def interest_bearing_debt(fin: Mapping[str, float | bool | None]) -> float:
    keys = (
        "short_term_loans",
        "current_portion_long_term_loans",
        "long_term_loans",
        "bonds",
        "current_portion_bonds",
    )
    return sum(_num(fin, k) for k in keys)


def net_cash(
    fin: Mapping[str, float | bool | None], investment_securities_haircut: float = 1.0
) -> float | None:
    """Cash + short-term securities (+ optionally long-term securities) − interest-bearing debt.

    Returns None when cash itself is not reported (we refuse to guess).
    """
    if fin.get("cash_and_deposits") is None:
        return None
    if not 0.0 <= investment_securities_haircut <= 1.0:
        raise ValueError("haircut must be between 0 and 1")
    cash_like = _num(fin, "cash_and_deposits") + _num(fin, "short_term_securities")
    cash_like += _num(fin, "investment_securities") * (1.0 - investment_securities_haircut)
    return cash_like - interest_bearing_debt(fin)


def shares_outstanding(issued: float | None, treasury: float | None) -> float | None:
    if issued is None or issued <= 0:
        return None
    return issued - (treasury or 0.0)


def treasury_at_filing(
    issued: float | None, at_filing: float | None, treasury: float | None
) -> float | None:
    """Treasury shares on the filing date, from the year-end count (only that one is reported)."""
    if treasury is None or issued is None or at_filing is None or issued <= 0:
        return treasury
    if at_filing == issued:
        return treasury
    ratio = at_filing / issued
    # A whole-number ratio is a split or consolidation (checked first: a 2:1 consolidation of a
    # company holding > 50% treasury would otherwise look like a cancellation).
    for whole in (ratio, 1 / ratio):
        if whole >= 1.5 and abs(whole - round(whole)) < _SPLIT_TOLERANCE:
            return treasury * ratio
    cancelled = issued - at_filing
    if 0 < cancelled <= treasury:
        return treasury - cancelled
    return treasury


_SPLIT_TOLERANCE = 0.01  # 1:3 gives exactly 3.0; a few option exercises do not hit an integer


def _known(fin: Mapping[Any, Any], key: str) -> float | None:
    """A numeric item of a financials record; None/NaN → None (missing stays missing)."""
    value = fin.get(key)
    if isinstance(value, int | float) and not isinstance(value, bool) and not pd.isna(value):
        return float(value)
    return None


def outstanding_at_filing(fin: Mapping[Any, Any]) -> float | None:
    """Shares outstanding on the FILING date, the date of the officers' holdings."""
    issued, at_filing = _known(fin, "issued"), _known(fin, "issued_at_filing")
    treasury = _known(fin, "treasury")
    if at_filing and issued:
        return shares_outstanding(at_filing, treasury_at_filing(issued, at_filing, treasury))
    return shares_outstanding(issued, treasury)


def book_value_per_share(fin: Mapping[Any, Any]) -> float | None:
    """Net assets per share outstanding at the filing date (None if either is unknown)."""
    net_assets, outstanding = _known(fin, "net_assets"), outstanding_at_filing(fin)
    if net_assets is None or not outstanding:
        return None
    return net_assets / outstanding


_NO_MATERIAL_RENTAL = frozenset({"immaterial", "none", "absent"})


def rental_gain_after_tax(fin: Mapping[Any, Any], tax_rate: float) -> float | None:
    """(fair − book value of rental real estate) × (1 − tax), in yen."""
    status = fin.get("rental_note_status")
    if status in _NO_MATERIAL_RENTAL:
        return 0.0
    book, fair = _known(fin, "rental_book_value"), _known(fin, "rental_fair_value")
    if status != "disclosed" or book is None or fair is None:
        return None
    return (fair - book) * (1 - tax_rate)


def adjusted_net_assets(fin: Mapping[Any, Any], tax_rate: float) -> float | None:
    """Net assets + after-tax unrealized gain on rental real estate (the 'adjusted NAV' floor)."""
    net_assets, gain = _known(fin, "net_assets"), rental_gain_after_tax(fin, tax_rate)
    if net_assets is None or gain is None:
        return None
    return net_assets + gain


def market_cap(price: float | None, outstanding: float | None) -> float | None:
    if price is None or outstanding is None or price <= 0 or outstanding <= 0:
        return None
    return price * outstanding


def price_to_book(mcap: float | None, net_assets: float | None) -> float | None:
    """PBR."""
    if mcap is None or net_assets is None or net_assets <= 0:
        return None
    return mcap / net_assets


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


# In Japan a split's ex-rights date (権利落ち日, when AdjFactor hits prices) is one business day
# before the record date, and the new shares exist from the next day: the share COUNT changes
# about two business days after the ex-rights date.
_EFFECTIVE_LAG = pd.offsets.BDay(2)


def split_multiplier(splits: pd.DataFrame | None, since: date, until: date) -> float:
    """How many shares at `until` per share counted at `since`: 1 / Π adj_factor."""
    if splits is None or splits.empty:
        return 1.0
    ex_dates = pd.to_datetime(splits["date"])
    effective = (ex_dates + _EFFECTIVE_LAG).dt.date
    window = splits[(effective > since) & (ex_dates.dt.date <= until)]
    product = float(window["adj_factor"].astype(float).to_numpy().prod())
    return 1.0 / product if product > 0 else 1.0
