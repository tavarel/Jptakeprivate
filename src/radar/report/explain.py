"""Turn a scored row into plain-language reasons."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from radar.model.rules import RuleSet

_PERCENT = {
    "family_stake",
    "largest_listed_holder_stake",
    "top10_concentration_ex_custodian",
    "net_cash_to_market_cap",
    "officer_stake",
    "investment_securities_to_market_cap",
}
_YEN = {"market_cap_jpy", "net_cash_jpy"}
_AGE = {"oldest_representative_age", "ceo_age", "chair_age"}
# Extra context shown next to a reason: who the listed parent / activists are.
_DETAIL = {"listed_parent": "largest_listed_holder_name", "activist_holder": "activist_names"}


@dataclass(frozen=True)
class Reason:
    rule_id: str
    feature: str  # the data field the rule tests (so "rule: value" is never ambiguous)
    value: str  # the company's own value of that field, formatted
    why: str  # the rule's rationale from the YAML
    detail: str | None = None

    def label(self) -> str:
        """'no_capital_plan — capital_plan_disclosed: not disclosed'."""
        return f"{self.rule_id} — {self.feature}: {self.value}"


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _percent(v: Any) -> str:
    return f"{float(v):.1%}"


def _margin(v: Any) -> str:
    """Distance to the tighter TSE tradable-share minimum: '+90% above' / '20% BELOW'."""
    value = float(v)
    return (
        f"+{value:.0%} above the TSE minimum"
        if value >= 0
        else f"{-value:.0%} BELOW the TSE minimum"
    )


_COUNTS = {"parent_prior_offers", "parent_other_holdings", "activist_filers_365d"}


# How each feature is shown; anything not listed is shown as is.
_FORMATS: dict[str, Callable[[Any], str]] = {
    **dict.fromkeys(_PERCENT, _percent),
    **dict.fromkeys(_YEN, lambda v: f"¥{float(v) / 1e9:.1f}bn"),
    **dict.fromkeys(_AGE, lambda v: f"{int(v)} years"),
    "pbr": lambda v: f"{float(v):.2f}×",
    "adjusted_pbr": lambda v: f"{float(v):.2f}×",
    "parent_pbr": lambda v: f"{float(v):.2f}×",
    "parent_net_cash_to_market_cap": _percent,
    "parent_activist_filers_365d": lambda v: str(int(v)),
    "new_5pct_holders_90d": lambda v: str(int(v)),
    "p_any": lambda v: f"{float(v):.1%}",
    "expected_value": lambda v: f"{float(v):+.1%} (estimate)",
    "listed_holder_stake_change": lambda v: f"{float(v) * 100:+.1f} pts",
    "top10_change": lambda v: f"{float(v) * 100:+.1f} pts",
    "rental_unrealized_gain_to_market_cap": lambda v: f"{float(v):+.1%} of market cap",
    "price_change_12m": lambda v: f"{float(v):+.1%}",
    "median_trading_value_jpy": lambda v: f"¥{float(v) / 1e6:.1f}m",
    "capital_plan_disclosed": lambda v: "disclosed" if bool(v) else "not disclosed",
    **dict.fromkeys(_COUNTS, lambda v: str(int(v))),
    "turnover": lambda v: f"{float(v):.3%} of market cap / day",
    "tradable_share_ratio": _percent,
    "listing_criteria_margin": _margin,
    "years_below_book": lambda v: f"{int(v)} years in a row",
    "net_cash_change_to_market_cap": lambda v: f"{float(v):+.1%} of market cap",
    # Percentile p within the sector: share of peers at or below this value.
    "pbr_sector_percentile": lambda v: f"cheaper than {1 - float(v):.0%} of its sector",
    "net_cash_sector_percentile": lambda v: f"more cash than {float(v):.0%} of its sector",
}


def format_value(feature: str, value: Any) -> str:
    """Human formatting; a missing value is 'unknown', never 0."""
    if _missing(value):
        return "unknown"
    return _FORMATS.get(feature, str)(value)


def explain(row: Mapping[str, Any], ruleset: RuleSet) -> list[Reason]:
    """One Reason per triggered rule, in the rule-file order."""
    rules = row.get("triggered_rules")  # a numpy array after a Parquet round trip
    triggered = set(rules) if rules is not None else set()
    reasons = []
    for rule in ruleset.rules:
        if rule.id in triggered:
            detail_column = _DETAIL.get(rule.id)
            detail = row.get(detail_column) if detail_column else None
            reasons.append(
                Reason(
                    rule_id=rule.id,
                    feature=rule.feature,
                    value=format_value(rule.feature, row.get(rule.feature)),
                    why=" ".join(rule.rationale.split()),
                    detail=None if _missing(detail) else str(detail),
                )
            )
    return reasons


def edinet_url(doc_id: str) -> str:
    """Link to the filing in EDINET's public viewer."""
    return f"https://disclosure2.edinet-fsa.go.jp/WZEK0040.aspx?{doc_id},,"


def points(row: Mapping[str, Any], ruleset: RuleSet) -> tuple[float, float]:
    """(points earned, maximum possible): '4.0 of 12.5' reads better than a bare 0.32."""
    rules = row.get("triggered_rules")
    triggered = set(rules) if rules is not None else set()
    earned = sum(rule.weight for rule in ruleset.rules if rule.id in triggered)
    return earned, ruleset.max_score


def top_percent(scores: pd.Series) -> pd.Series:
    """'Top X%' per company (rank among all scored / count); ties share the better rank."""
    ranks = scores.rank(ascending=False, method="min")
    return (100 * ranks / len(scores)).round(1)


def unavailable_rules(scores: pd.DataFrame, ruleset: RuleSet) -> list[str]:
    """Rules that cannot fire for ANY company on this date because their feature is missing
    everywhere (e.g. no prices yet): they explain why even the top score looks low.
    """
    return [
        rule.id
        for rule in ruleset.rules
        if rule.feature not in scores.columns or scores[rule.feature].isna().all()
    ]
