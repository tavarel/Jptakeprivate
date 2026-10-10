"""Tables shown by the dashboard's Tender offers and Backtest tabs (pure helpers)."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from radar.config import load_yaml
from radar.features.ownership import HolderPatterns
from radar.model.offer_price import OfferRange, OfferRangeConfig
from radar.model.rules import RuleSet
from radar.report.views import (
    data_freshness,
    help_texts,
    hit_rate_by_band,
    holders_table,
    likely_buyer,
    liquidity_context,
    live_column,
    offer_range_for,
    offer_range_table,
    officers_table,
    ordering_options,
    points_band,
    premium_summary,
    price_axis_domain,
    price_history,
    public_reason,
    runup_cases,
    skill_verdict,
    track_record,
    universe_share_label,
)


def _study() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "doc_id": ["A", "B", "C", "D"],
            "event_type": ["mbo", "mbo", "parent", "parent"],
            "premium": [0.4, 0.6, 0.2, 9.9],
            "runup_abnormal": [0.0, 0.2, -0.1, 9.9],
            "pre_announced": [False, False, False, True],  # D is excluded from every statistic
            "is_take_private": [True, True, True, True],
        }
    )


def test_premium_summary_excludes_pre_announced_offers() -> None:
    summary = premium_summary(_study()).set_index("event_type")
    assert summary.loc["mbo", "offers"] == 2
    assert summary.loc["mbo", "median_premium"] == pytest.approx(0.5)
    assert summary.loc["parent", "offers"] == 1
    assert summary.loc["parent", "median_premium"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({"earlier_offer": True, "activist_before": "X"}, "earlier offer"),
        (
            {"earlier_offer": False, "activist_before": "ストラテジックキャピタル"},
            "activist: ストラテジックキャピタル",
        ),
        ({"earlier_offer": False, "activist_before": None}, "—"),
        ({"earlier_offer": False, "activist_before": float("nan")}, "—"),
    ],
)
def test_public_reason(row: dict[str, object], expected: str) -> None:
    assert public_reason(row) == expected


def test_runup_cases_unexplained_only() -> None:
    review = pd.DataFrame(
        {
            "doc_id": ["A", "B", "C"],
            "event_date": [date(2025, 1, 1)] * 3,
            "company": ["サンプル工業", "サンプル商事", "サンプル電機"],
            "event_type": ["mbo", "parent", "third_party"],
            "runup_abnormal": [0.3, 0.2, 0.15],
            "runup_short": [0.1, 0.0, 0.05],
            "premium": [0.4, 0.3, 0.2],
            "volume_ratio": [3.0, 1.0, None],
            "earlier_offer": [False, True, False],
            "activist_before": [None, None, "Oasis"],
            "flagged": [True, True, True],
        }
    )
    everything = runup_cases(review, unexplained_only=False)
    assert everything["why_public"].tolist() == ["—", "earlier offer", "activist: Oasis"]
    assert everything["filing"].iloc[0].endswith("?A,,")
    assert runup_cases(review, unexplained_only=True)["company"].tolist() == ["サンプル工業"]


def test_premium_summary_excludes_partial_offers() -> None:
    study = _study().assign(is_take_private=[True, False, True, True])  # B: partial (BASE-like)
    summary = premium_summary(study).set_index("event_type")
    assert summary.loc["mbo", "offers"] == 1
    assert summary.loc["mbo", "median_premium"] == pytest.approx(0.4)


# ------------------------------------------------------------------ company card helpers
def test_price_history_is_split_adjusted_and_ends_before_as_of() -> None:
    prices = pd.DataFrame(
        {
            "code": "99990",
            "date": [date(2025, 1, 6), date(2025, 1, 7), date(2025, 1, 8), date(2025, 1, 9)],
            "close": [200.0, 100.0, 104.0, 999.0],
            "adj_factor": [1.0, 0.5, 1.0, 1.0],  # 1:2 split on 1/7
        }
    )
    history = price_history(prices, "99990", as_of=date(2025, 1, 9), days=365)
    assert history["close"].tolist() == [100.0, 100.0, 104.0]  # 200 restated as 100; 1/9 excluded


def test_holders_table_adds_holder_type() -> None:
    holders = pd.DataFrame(
        {
            "doc_id": ["D1", "D1", "D2"],
            "rank": [1, 2, 1],
            "name": ["日本マスタートラスト信託銀行株式会社(信託口)", "見本 太郎", "X"],
            "ratio": [0.1, 0.05, 0.9],
            "is_organization": [True, False, True],
        }
    )
    table = holders_table(holders, "D1", HolderPatterns.from_config(load_yaml("ownership")))
    assert table["type"].tolist() == ["custodian", "person"]
    assert table["holder"].tolist()[1] == "見本 太郎"


def test_officers_table_ages_on_the_screening_date() -> None:
    officers = pd.DataFrame(
        {
            "doc_id": ["D1", "D1"],
            "name": ["見本 太郎", "見本 次郎"],
            "title": ["代表取締役社長", "取締役"],
            "date_of_birth": [date(1950, 7, 2), None],
            "shares_held": [1000.0, None],
            "is_representative": [True, False],
        }
    )
    table = officers_table(officers, "D1", as_of=date(2025, 7, 1))
    assert table["age"].tolist()[0] == 74
    assert pd.isna(table["age"].tolist()[1])  # unknown, not 0
    assert table["representative"].tolist() == [True, False]


def test_hit_rate_by_band_groups_points() -> None:
    panel = pd.DataFrame(
        {
            "score": [0.9, 0.85, 0.5, 0.1, 0.1, 0.0],  # × 10 points max
            "label": [1, 0, 1, 0, 0, 0],
        }
    )
    bands = hit_rate_by_band(panel, max_score=10.0, band_points=2.0).set_index("points")
    assert bands.loc["8–10", "companies"] == 2 and bands.loc["8–10", "hit_rate"] == 0.5
    assert bands.loc["4–6", "hit_rate"] == 1.0
    assert bands.loc["0–2", "hits"] == 0 and bands.loc["0–2", "companies"] == 3
    assert list(bands.index) == ["8–10", "4–6", "0–2"]  # highest band first, empty bands omitted


def test_likely_buyer_uses_the_rule_thresholds() -> None:
    ruleset = RuleSet.load()
    assert likely_buyer({"largest_listed_holder_stake": 0.25}, ruleset) == "parent"
    assert likely_buyer({"family_stake": 0.35}, ruleset) == "mbo"
    both = {"largest_listed_holder_stake": 0.25, "family_stake": 0.35}
    assert likely_buyer(both, ruleset) == "parent"  # a parent's stake decides who can buy
    assert likely_buyer({"family_stake": None}, ruleset) == "third_party"


def test_track_record_lists_past_top_names_that_got_an_offer() -> None:
    panel = pd.DataFrame(
        {
            "as_of": [date(2024, 3, 31)] * 3 + [date(2024, 6, 30)] * 3,
            "company_name": ["サンプルA", "サンプルB", "サンプルC"] * 2,
            "sec_code": ["99010", "99020", "99030"] * 2,
            "score": [0.9, 0.8, 0.1, 0.9, 0.2, 0.8],
            "label": [1, 0, 1, 1, 0, 0],
            "label_mbo": [1, 0, 1, 1, 0, 0],
            "label_parent": [0] * 6,
        }
    )
    record = track_record(panel, top_n=2)
    # サンプルC was a target but ranked 3rd: not a radar "hit".
    assert record["company_name"].tolist() == ["サンプルA", "サンプルA"]
    assert record["event_types"].tolist() == ["mbo", "mbo"]
    assert record["rank"].tolist() == [1, 1]


def test_liquidity_context_compares_with_all_and_with_same_size_peers() -> None:
    scores = pd.DataFrame(
        {
            "market_cap_jpy": [float(i) * 1e9 for i in range(1, 101)],  # 1bn … 100bn
            "median_trading_value_jpy": [float(i) * 1e6 for i in range(1, 101)],
            "turnover": [0.001] * 100,
        }
    )
    row = {"market_cap_jpy": 5e9, "median_trading_value_jpy": 5e6, "turnover": 0.001}
    context = liquidity_context(row, scores)
    assert context["median_trading_value_jpy"].startswith("Higher than 4% of listed companies")
    assert "same size (¥1.0bn–¥10.0bn): ¥5.5m" in context["median_trading_value_jpy"]
    assert "every ~4.1 years" in context["turnover"]  # 1 / 0.1% = 1,000 sessions ÷ 245 a year


def test_liquidity_context_unknown_value_says_so() -> None:
    scores = pd.DataFrame(
        {"market_cap_jpy": [1e9], "median_trading_value_jpy": [1e6], "turnover": [0.001]}
    )
    context = liquidity_context({"market_cap_jpy": None, "turnover": None}, scores)
    assert context == {}  # nothing to compare: the help text stays generic


def test_price_axis_domain_hugs_the_prices() -> None:
    history = pd.DataFrame({"close": [1000.0, 1100.0, 1200.0]})
    low, high = price_axis_domain(history, padding=0.05)
    assert low == pytest.approx(990.0) and high == pytest.approx(1210.0)  # 5% of the range
    flat = price_axis_domain(pd.DataFrame({"close": [500.0, 500.0]}), padding=0.05)
    assert flat[0] < 500.0 < flat[1]  # a flat line still gets a visible band


def test_points_bands_do_not_overlap_when_the_maximum_is_not_a_multiple() -> None:
    # Real rule set: 12.5 points max, 2-point bands → last band is 12–12.5, not 10.5–12.5.
    assert points_band(12.5, max_score=12.5, width=2.0) == "12–12.5"
    assert points_band(11.0, max_score=12.5, width=2.0) == "10–12"
    assert points_band(0.0, max_score=12.5, width=2.0) == "0–2"
    assert points_band(10.0, max_score=10.0, width=2.0) == "8–10"  # the maximum joins the top band
    panel = pd.DataFrame({"score": [1.0, 0.88, 0.0], "label": [1, 0, 0]})
    bands = hit_rate_by_band(panel, max_score=12.5, band_points=2.0)
    assert bands["points"].tolist() == ["12–12.5", "10–12", "0–2"]


def test_offer_range_for_uses_the_likely_buyer_and_price_over_pbr() -> None:
    study = pd.DataFrame(
        {
            "event_type": ["mbo"] * 12,
            "is_take_private": [True] * 12,
            "pre_announced": [False] * 12,
            "premium": [0.4] * 12,
            "offer_pbr": [1.0] * 12,
        }
    )
    row = {"family_stake": 0.5, "pbr": 0.5}  # family ≥ 30% → likely MBO; book = 2 × price
    config = OfferRangeConfig.from_config(load_yaml("event_study"))
    result = offer_range_for(row, study, price=1000.0, ruleset=RuleSet.load(), config=config)
    assert result is not None and result.event_type == "mbo"
    assert result.by_premium == pytest.approx((1400.0, 1400.0, 1400.0))
    assert result.by_book == pytest.approx((2000.0, 2000.0, 2000.0))
    assert offer_range_for({"pbr": None}, study, 1000.0, RuleSet.load(), config) is None


def test_offer_range_table_one_row_per_anchor_with_upside() -> None:
    found = OfferRange(
        event_type="mbo",
        offers=82,
        by_premium=(2901.0, 3150.0, 3420.0),
        by_book=(2899.0, 3459.0, 4000.0),
        book_basis="PBR 0.6–1",
        book_offers=64,
    )
    table = offer_range_table(found, price=2300.0)
    assert table.columns.tolist() == [
        "anchor",
        "low (25%)",
        "median",
        "high (75%)",
        "median vs last close",
    ]
    assert table["anchor"].tolist() == [
        "Premium paid (82 past mbo offers)",
        "PBR paid (64 offers for targets at PBR 0.6–1)",
    ]
    assert table.loc[0, "median"] == "¥3,150"
    assert table.loc[0, "median vs last close"] == "+37%"
    only_premium = OfferRange("mbo", 82, (2901.0, 3150.0, 3420.0), None)
    assert len(offer_range_table(only_premium, 2300.0)) == 1  # no book anchor → no row


def test_help_texts_match_exact_names_then_the_longest_prefix() -> None:
    texts = {"lift_top": "pooled", "lift_top_": "per type", "lift_top_delta": "delta"}
    found = help_texts(["lift_top", "lift_top_mbo", "lift_top_delta", "unknown"], texts)
    assert found == {"lift_top": "pooled", "lift_top_mbo": "per type", "lift_top_delta": "delta"}


def test_universe_share_label_gives_the_size_as_a_share_of_the_ranked_list() -> None:
    assert universe_share_label(50, 4100) == "top 50 (≈1.2%)"
    assert universe_share_label(10, None) == "top 10"


def test_live_column_only_when_the_screen_has_it() -> None:
    mapping = {"v2": "score", "hybrid": None, "return_model": "return_model"}
    columns = ["sec_code", "score"]
    assert live_column("v2", mapping, columns) == "score"
    assert live_column("hybrid", mapping, columns) is None  # no live version of that ranking
    assert live_column("return_model", mapping, columns) is None  # not in this screen yet
    assert live_column("unknown", mapping, columns) is None


def test_skill_verdict_reads_the_deflated_sharpe_in_words() -> None:
    assert skill_verdict(0.97).startswith("strong")
    assert skill_verdict(0.6).startswith("a hint")
    assert skill_verdict(0.0).startswith("luck alone")
    assert skill_verdict(None) == "not computed yet"


def test_data_freshness_counts_business_days_not_weekends() -> None:
    # Friday's data seen on Monday is fresh: only 1 business day old.
    ok = data_freshness(date(2026, 10, 9), date(2026, 10, 9), today=date(2026, 10, 12), max_days=3)
    assert ok["stale"] is False
    stale = data_freshness(
        date(2026, 10, 9), date(2026, 10, 1), today=date(2026, 10, 12), max_days=3
    )
    assert stale["stale"] is True and "prices" in str(stale["text"])


def test_data_freshness_without_data_is_stale() -> None:
    assert data_freshness(None, None, today=date(2026, 10, 12), max_days=3)["stale"] is True


def test_ordering_options_fall_back_for_screens_made_before_a_column_existed() -> None:
    orderings = {"Rules": "score", "MBO likelihood": "p_mbo_prior", "Return": "return_model"}
    fallbacks = {"p_mbo_prior": "p_mbo"}
    new = ordering_options(orderings, fallbacks, ["score", "p_mbo_prior", "p_mbo"])
    assert new == {"Rules": "score", "MBO likelihood": "p_mbo_prior"}
    old = ordering_options(orderings, fallbacks, ["score", "p_mbo"])
    assert old == {"Rules": "score", "MBO likelihood": "p_mbo"}
