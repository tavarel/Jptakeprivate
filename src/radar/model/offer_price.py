"""Likely offer price for a candidate, from past take-privates of the same buyer type.

Data contract: study with event_type, is_take_private, pre_announced, premium, offer_pbr and pre_pbr
(last close before the offer ÷ book value per share). Offers already at or above their price
(pre_announced) and partial offers are excluded, as in the premium study.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class OfferRangeConfig:
    quantiles: tuple[float, ...]
    min_offers: int
    pbr_bands: tuple[float, ...] = ()  # band edges for the book anchor (config/event_study.yaml)

    @classmethod
    def from_config(cls, raw: Mapping[str, Any]) -> OfferRangeConfig:
        return cls(
            quantiles=tuple(float(q) for q in raw["offer_range_quantiles"]),
            min_offers=int(raw["offer_range_min_offers"]),
            pbr_bands=tuple(float(b) for b in raw.get("offer_pbr_bands", [])),
        )


@dataclass(frozen=True)
class OfferRange:
    event_type: str
    offers: int  # comparable take-privates behind the range
    by_premium: tuple[float, ...] | None  # yen per share, one per quantile
    by_book: tuple[float, ...] | None
    book_basis: str = ""  # which offers the book anchor comes from
    book_offers: int = 0  # how many offers the book anchor rests on


def offer_pbr(offer_price: float | None, book_per_share: float | None) -> float | None:
    """Offer price ÷ book value per share; None when the book value is unknown or ≤ 0."""
    if offer_price is None or book_per_share is None or book_per_share <= 0:
        return None
    return offer_price / book_per_share


def likely_offer_range(
    study: pd.DataFrame,
    event_type: str,
    price: float | None,
    book_per_share: float | None,
    config: OfferRangeConfig,
    *,
    pbr: float | None = None,
) -> OfferRange | None:
    """Price ranges for a buyer type; None when fewer than `min_offers` comparable offers."""
    clean = study[study["is_take_private"].eq(True) & study["pre_announced"].eq(False)]
    comparable = clean[clean["event_type"] == event_type]
    if len(comparable) < config.min_offers:
        return None
    book_offers, basis = comparable, f"{event_type} offers"
    if pbr is not None and config.pbr_bands and "pre_pbr" in clean.columns:
        low, high = _band(pbr, config.pbr_bands)
        pre = pd.to_numeric(clean["pre_pbr"], errors="coerce")
        in_band = (pre >= (low if low is not None else -float("inf"))) & (
            pre < (high if high is not None else float("inf"))
        )
        book_offers, basis = clean[in_band], _band_label(low, high)
    by_book = None
    if len(book_offers) >= config.min_offers:
        by_book = _scaled(book_offers["offer_pbr"], book_per_share, lambda b, q: b * q, config)
    return OfferRange(
        event_type=event_type,
        offers=len(comparable),
        by_premium=_scaled(comparable["premium"], price, lambda p, q: p * (1 + q), config),
        by_book=by_book,
        book_basis=basis,
        book_offers=len(book_offers),
    )


def _band(value: float, edges: Sequence[float]) -> tuple[float | None, float | None]:
    low: float | None = None
    for edge in edges:
        if value < edge:
            return low, edge
        low = edge
    return low, None


def _band_label(low: float | None, high: float | None) -> str:
    if low is None:
        return f"PBR < {high:g}"
    if high is None:
        return f"PBR ≥ {low:g}"
    return f"PBR {low:g}–{high:g}"


def _scaled(
    values: pd.Series,
    base: float | None,
    apply: Callable[[float, float], float],
    config: OfferRangeConfig,
) -> tuple[float, ...] | None:
    known = pd.to_numeric(values, errors="coerce").dropna()
    if base is None or pd.isna(base) or base <= 0 or len(known) < _min_known(config):
        return None
    return tuple(apply(base, float(known.quantile(q))) for q in config.quantiles)


def _min_known(config: OfferRangeConfig) -> int:
    # A quantile of fewer values than quantiles asked for says nothing.
    return max(len(config.quantiles), 1)


def describe(result: OfferRange, quantiles: Sequence[float]) -> str:
    """'¥1,275–¥1,425 (median ¥1,350) by premium · …' for the card and the fact pack."""
    parts = []
    book_label = (
        f"PBR paid for targets at {result.book_basis}"
        if result.book_basis.startswith("PBR")
        else "PBR paid"
    )
    for label, values in (("premium paid", result.by_premium), (book_label, result.by_book)):
        if values is None:
            continue
        low, *_, high = values
        middle = values[len(values) // 2]
        parts.append(f"by {label}: ¥{low:,.0f}–¥{high:,.0f} (median ¥{middle:,.0f})")
    lo, hi = int(quantiles[0] * 100), int(quantiles[-1] * 100)
    source = f"{lo}th–{hi}th percentile of {result.offers} past {result.event_type} take-privates"
    return f"{' · '.join(parts)} — {source}"
