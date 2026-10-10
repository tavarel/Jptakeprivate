"""Fact pack: the facts and sources needed to write the memo of one company."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from radar.model.rules import RuleSet
from radar.report.explain import edinet_url, explain, format_value, points

KEY_FACTS = [
    "p_any", "expected_value",
    "oldest_representative_age", "family_stake", "largest_listed_holder_stake",
    "largest_listed_holder_name", "parent_pbr", "parent_net_cash_to_market_cap",
    "parent_activist_filers_365d", "market_cap_jpy", "pbr", "adjusted_pbr",
    "rental_unrealized_gain_to_market_cap", "pbr_sector_percentile",
    "years_below_book", "net_cash_to_market_cap", "net_cash_change_to_market_cap",
    "officer_stake", "price_change_12m", "median_trading_value_jpy", "turnover",
    "tradable_share_ratio", "listing_criteria_margin", "top10_change",
    "listed_holder_stake_change", "new_5pct_holders_90d", "activist_filers_365d", "activist_names",
    "market_segment",
    "sector33", "capital_plan_disclosed",
]  # fmt: skip
EXCERPT_TITLES = {
    "business": "Business (事業の内容)",
    "main_customers": "Main customers (主要な顧客ごとの情報, ≥ 10% of sales)",
    "revenue_by_region": "Sales by region",
}


@dataclass(frozen=True)
class FactPackInputs:
    as_of: date
    row: Mapping[str, Any]  # one row of marts/scores_<as_of>
    ruleset: RuleSet
    likely_buyer: str
    last_close: float | None
    offer_table: pd.DataFrame | None  # report.views.offer_range_table; None: too few offers
    hit_rates: pd.DataFrame  # report.views.hit_rate_by_band
    holders: pd.DataFrame  # report.views.holders_table
    officers: pd.DataFrame  # report.views.officers_table
    excerpts: Mapping[str, str]  # parse.excerpts.report_excerpts
    recent_filings: pd.DataFrame  # submit_datetime, filer_name, doc_description


def factpack_markdown(inputs: FactPackInputs) -> str:
    row, doc = inputs.row, str(inputs.row["source_doc_id"])
    earned, maximum = points(dict(row), inputs.ruleset)
    lines = [
        f"# Fact pack — {row['company_name']} ({str(row['sec_code'])[:4]}), as of {inputs.as_of}",
        "",
        "> Facts for a memo. Check each number in the filing before",
        "> using it. Research, not investment advice.",
        "",
        f"**Points:** {earned:.1f} of {maximum:.1f} · source report [{doc}]({edinet_url(doc)}),"
        f" submitted {row.get('filing_submitted_at')}",
        "",
        "## Why it screens",
        *[f"- **{r.label()}** — {r.why}" for r in explain(dict(row), inputs.ruleset)],
        "",
        "## Key facts",
        *[f"- {f}: {format_value(f, row.get(f))}" for f in KEY_FACTS if _shown(f, row)],
        "",
        f"## Likely buyer: {inputs.likely_buyer}",
        *_offer_section(inputs),
        "",
        "## Historical hit rate by points band (backtest, complete dates)",
        _table(_as_percent(inputs.hit_rates, "hit_rate")),
        "",
        "## In the filing's own words",
    ]
    for key, title in EXCERPT_TITLES.items():
        text = inputs.excerpts.get(key)
        lines += [f"**{title}**", "", f"> {text}" if text else "_Not in this filing._", ""]
    lines += [
        "## Top-10 shareholders",
        _table(inputs.holders),
        "",
        "## Directors and officers",
        _table(inputs.officers),
        "",
        "## Recent filings about the company (5% reports, offers, opinions)",
        _table(inputs.recent_filings),
        "",
    ]
    return "\n".join(lines)


def _offer_section(inputs: FactPackInputs) -> list[str]:
    if inputs.offer_table is None or inputs.offer_table.empty:
        return ["_Too few comparable offers for a price range._"]
    close = "unknown" if inputs.last_close is None else f"¥{inputs.last_close:,.0f}"
    return [
        f"**Likely offer price** — last close {close}. What buyers of this type paid before,"
        " applied to this company: not a forecast.",
        "",
        _table(inputs.offer_table),
    ]


# A name only means something when its count / stake is positive: with 0 activists or no
# listed holder the name is absent because there is none, not because it is unknown.
_NAME_OF = {
    "activist_names": "activist_filers_365d",
    "largest_listed_holder_name": "largest_listed_holder_stake",
    # The parent's own figures only apply when there is a listed parent.
    "parent_pbr": "largest_listed_holder_stake",
    "parent_net_cash_to_market_cap": "largest_listed_holder_stake",
    "parent_activist_filers_365d": "largest_listed_holder_stake",
}


def _shown(feature: str, row: Mapping[str, Any]) -> bool:
    if feature not in _NAME_OF:
        return True
    value = row.get(_NAME_OF[feature])
    return isinstance(value, int | float) and not pd.isna(value) and value > 0


def _as_percent(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if frame.empty or column not in frame.columns:
        return frame
    return frame.assign(**{column: frame[column].map(lambda v: f"{float(v):.1%}")})


def _table(frame: pd.DataFrame) -> str:
    """DataFrame → Markdown table without extra dependencies; empty → a note."""
    if frame.empty:
        return "_(none before this date)_"
    head = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "---|" * len(frame.columns)
    body = [
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in values) + " |"
        for values in frame.itertuples(index=False)
    ]
    return "\n".join([head, rule, *body])
