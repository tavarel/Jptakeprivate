"""Terms of a third-party tender offer from its registration statement (240, form `jptoo`)."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from radar.parse.elements import to_number
from radar.text_ja import normalize

logger = logging.getLogger(__name__)

_PRICE = re.compile(r"1株につき[、,]?\s*金?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*円")


@dataclass(frozen=True)
class TenderOfferConfig:
    elements: dict[str, str]
    mbo_pattern: re.Pattern[str]
    mbo_negations: tuple[str, ...]
    parent_min_prior_stake: float
    take_private_min_ratio_after: float

    @classmethod
    def from_config(cls, raw: Mapping[str, Any]) -> TenderOfferConfig:
        return cls(
            elements=dict(raw["elements"]),
            mbo_pattern=re.compile(normalize(raw["mbo_pattern"])),
            mbo_negations=tuple(normalize(n) for n in raw["mbo_negations"]),
            parent_min_prior_stake=float(raw["parent_min_prior_stake"]),
            take_private_min_ratio_after=float(raw["take_private_min_ratio_after"]),
        )


def offer_price(text: str) -> float | None:
    """'株券１株につき金7,600円' → 7600.0 (first per-share price stated), None if absent."""
    match = _PRICE.search(normalize(text))
    return float(match.group(1).replace(",", "")) if match else None


def _value(facts: pd.DataFrame, element_id: str) -> str | None:
    rows = facts.loc[facts["element_id"] == element_id, "value"]
    return None if rows.empty else str(rows.iloc[0])


def extract_tender_offer(facts: pd.DataFrame, config: TenderOfferConfig) -> dict[str, Any]:
    e = config.elements
    prior, bidder = _stakes(facts, e)
    purpose = normalize(_value(facts, e["purpose_text"]) or "")
    ratio_after = to_number(_value(facts, e["ratio_after"]))
    return {
        "offer_price": offer_price(_value(facts, e["price_text"]) or ""),
        "prior_stake": prior,
        "bidder_stake": bidder,
        "ratio_after": ratio_after,
        "is_take_private": _is_take_private(ratio_after, config),
        "self_declared_mbo": _declares_mbo(purpose, config),
    }


def _stakes(facts: pd.DataFrame, e: Mapping[str, str]) -> tuple[float | None, float | None]:
    """(prior_stake, bidder_stake) as shares of all voting rights J; (None, None) if unknown."""
    total = to_number(_value(facts, e["all_voting_rights"]))
    if not total:
        return None, None
    # The form is reported: a blank D or G is its dash "―", i.e. none held.
    bidder = to_number(_value(facts, e["owned_by_bidder"])) or 0.0
    related = to_number(_value(facts, e["owned_by_related"])) or 0.0
    if bidder + related > total:
        logger.warning(
            "tender offer stake above 100%% of votes (%.0f / %.0f)", bidder + related, total
        )
        return None, None
    return (bidder + related) / total, bidder / total


def _is_take_private(ratio_after: float | None, config: TenderOfferConfig) -> bool | None:
    """The bidder aims at (nearly) all shares; None when the target holding is not reported."""
    return None if ratio_after is None else ratio_after >= config.take_private_min_ratio_after


def _declares_mbo(purpose: str, config: TenderOfferConfig) -> bool:
    """True if some sentence describes THIS deal as an MBO without negating it."""
    for match in config.mbo_pattern.finditer(purpose):
        end = purpose.find("。", match.end())
        sentence = purpose[match.start() : end if end != -1 else len(purpose)]
        if not any(negation in sentence for negation in config.mbo_negations):
            return True
    return False


def classify_event(terms: Mapping[str, Any], config: TenderOfferConfig) -> str:
    """mbo / parent / third_party, or unknown when the prior stake cannot be read."""
    if terms.get("self_declared_mbo"):
        return "mbo"
    prior = terms.get("prior_stake")
    if prior is None:
        return "unknown"
    return "parent" if prior >= config.parent_min_prior_stake else "third_party"
