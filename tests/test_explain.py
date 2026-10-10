"""Plain-language explanations for the dashboard."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from radar.model.rules import RuleSet
from radar.report.explain import (
    edinet_url,
    explain,
    format_value,
    points,
    top_percent,
    unavailable_rules,
)


@pytest.mark.parametrize(
    ("feature", "value", "expected"),
    [
        ("oldest_representative_age", 74.0, "74 years"),
        ("family_stake", 0.4012, "40.1%"),
        ("largest_listed_holder_stake", 0.246, "24.6%"),
        ("pbr", 0.4567, "0.46×"),
        ("market_cap_jpy", 12_345_000_000.0, "¥12.3bn"),
        ("activist_filers_365d", 2, "2"),
        ("market_segment", "Standard", "Standard"),
        ("pbr", None, "unknown"),
        ("pbr", math.nan, "unknown"),  # missing is shown as unknown, never as 0
        ("officer_stake", 0.0487, "4.9%"),
        ("price_change_12m", 0.1405, "+14.1%"),
        ("price_change_12m", -0.25, "-25.0%"),
        ("median_trading_value_jpy", 3_455_000.0, "¥3.5m"),
        ("capital_plan_disclosed", False, "not disclosed"),
        ("capital_plan_disclosed", True, "disclosed"),
        ("capital_plan_disclosed", None, "unknown"),
        ("parent_prior_offers", 0.0, "0"),  # counts are integers, not "0.0"
        ("parent_other_holdings", 1.0, "1"),
        ("activist_filers_365d", 2.0, "2"),
        ("turnover", 0.00085, "0.085% of market cap / day"),
        ("tradable_share_ratio", 0.476, "47.6%"),
        ("listing_criteria_margin", 0.904, "+90% above the TSE minimum"),
        ("listing_criteria_margin", -0.2, "20% BELOW the TSE minimum"),
        ("years_below_book", 2.0, "2 years in a row"),
        ("adjusted_pbr", 0.4211, "0.42×"),
        ("listed_holder_stake_change", 0.07, "+7.0 pts"),
        ("new_5pct_holders_90d", 2.0, "2"),
        ("p_any", 0.0612, "6.1%"),
        ("expected_value", 0.083, "+8.3% (estimate)"),
        ("rental_unrealized_gain_to_market_cap", 0.183, "+18.3% of market cap"),
        ("years_below_book", 0.0, "0 years in a row"),
        ("net_cash_change_to_market_cap", 0.12, "+12.0% of market cap"),
        ("pbr_sector_percentile", 0.15, "cheaper than 85% of its sector"),
        ("net_cash_sector_percentile", 0.9, "more cash than 90% of its sector"),
    ],
)
def test_format_value(feature: str, value: object, expected: str) -> None:
    assert format_value(feature, value) == expected


def test_explain_lists_triggered_rules_with_values_and_reasons() -> None:
    row = {
        "triggered_rules": ["aging_leader", "listed_parent"],
        "oldest_representative_age": 74,
        "largest_listed_holder_stake": 0.246,
        "largest_listed_holder_name": "トヨタ自動車株式会社",
    }
    reasons = explain(row, RuleSet.load())
    assert [r.rule_id for r in reasons] == ["aging_leader", "listed_parent"]
    assert reasons[0].value == "74 years"
    assert "succession" in reasons[0].why
    assert reasons[1].detail == "トヨタ自動車株式会社"


def test_explain_nothing_triggered() -> None:
    assert explain({"triggered_rules": []}, RuleSet.load()) == []


def test_edinet_url() -> None:
    assert edinet_url("S100VON4") == "https://disclosure2.edinet-fsa.go.jp/WZEK0040.aspx?S100VON4,,"


def test_explain_accepts_numpy_arrays_from_parquet() -> None:
    row = {"triggered_rules": np.array(["aging_leader"]), "oldest_representative_age": 80}
    assert [r.rule_id for r in explain(row, RuleSet.load())] == ["aging_leader"]


def test_points() -> None:
    ruleset = RuleSet.load()
    row = {"triggered_rules": ["aging_leader", "listed_parent"]}
    earned, maximum = points(row, ruleset)
    assert earned == 4.0
    assert maximum == ruleset.max_score


def test_top_percent_ties_share_the_better_rank() -> None:
    scores = pd.Series([0.32, 0.32, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert top_percent(scores).tolist()[:4] == [10.0, 10.0, 30.0, 40.0]


def test_unavailable_rules_lists_rules_without_any_data() -> None:
    scores = pd.DataFrame(
        {
            "oldest_representative_age": [75, None],
            "pbr": [None, None],
            "market_cap_jpy": [None, None],
        }
    )
    missing = unavailable_rules(scores, RuleSet.load())
    assert "below_book" in missing and "small_cap" in missing
    assert "aging_leader" not in missing
    assert "family_control" in missing  # column absent entirely → no data either


def test_reason_names_the_data_behind_the_rule() -> None:
    row = {"triggered_rules": ["no_capital_plan"], "capital_plan_disclosed": False}
    reason = explain(row, RuleSet.load())[0]
    assert reason.feature == "capital_plan_disclosed"
    assert reason.value == "not disclosed"
    assert reason.label() == "no_capital_plan — capital_plan_disclosed: not disclosed"
