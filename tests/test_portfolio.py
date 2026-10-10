"""Top-N pick returns: equal-weight portfolios of the highest scores, held 12 months."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from radar.model.portfolio import (
    AdjustedPrices,
    TradingCosts,
    ValueBasketConfig,
    holding_return,
    live_returns,
    pick_lenses,
    pick_returns,
    sector_neutral,
    strategy_table,
    summarize_picks,
    top_k_breakdown,
    union_rank_score,
    value_basket_score,
    value_catalyst_score,
    value_momentum_score,
)

AS_OF = date(2024, 3, 31)


def _prices() -> pd.DataFrame:
    rows = [
        # A: 1,000 → 1,500 (+50%)
        ("A", date(2024, 3, 29), 1000.0, 1.0), ("A", date(2025, 3, 31), 1500.0, 1.0),
        # B: 1:2 split mid-way; 2,000 → 1,100 post-split = 2,200 pre-split basis (+10%)
        ("B", date(2024, 3, 29), 2000.0, 1.0), ("B", date(2024, 9, 2), 1000.0, 0.5),
        ("B", date(2025, 3, 31), 1100.0, 1.0),
        ("C", date(2024, 3, 29), 1000.0, 1.0), ("C", date(2024, 11, 1), 1300.0, 1.0),
        # D: trap — the close OF as_of is not known in the morning of as_of
        ("D", date(2024, 3, 31), 9999.0, 1.0), ("D", date(2024, 3, 28), 500.0, 1.0),
        ("D", date(2025, 3, 31), 450.0, 1.0),
    ]  # fmt: skip
    return pd.DataFrame(rows, columns=["code", "date", "close", "adj_factor"])


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "as_of": [AS_OF] * 4,
            "sec_code": ["A", "B", "C", "D"],
            "score": [0.9, 0.8, 0.8, 0.1],  # B and C tie: input order breaks the tie
            "label": [0, 0, 1, 0],
        }
    )


def _topix() -> pd.DataFrame:
    return pd.DataFrame({"date": [date(2024, 3, 29), date(2025, 3, 31)], "close": [2000.0, 2200.0]})


def test_adjusted_prices_restate_a_split_to_one_basis() -> None:
    prices = AdjustedPrices(_prices())
    assert prices.close_on_or_before("B", date(2024, 3, 30)) == (date(2024, 3, 29), 1000.0)
    assert prices.close_on_or_before("B", date(2025, 4, 1)) == (date(2025, 3, 31), 1100.0)


def test_pick_returns_equal_weight_with_exits_and_benchmarks() -> None:
    result = pick_returns(
        _panel(), AdjustedPrices(_prices()), _topix(), horizon_days=365, sizes=[2, 3]
    ).set_index("top_n")
    top3 = result.loc[3]
    assert top3["picks"] == 3 and top3["priced"] == 3
    assert top3["mean_return"] == pytest.approx((0.5 + 0.1 + 0.3) / 3)
    assert top3["hit_rate"] == pytest.approx(1 / 3)
    assert top3["exited_early"] == 1  # C stopped trading: sold at its last close
    assert top3["worst_return"] == pytest.approx(0.1)
    assert top3["topix_return"] == pytest.approx(0.1)
    # All ranked companies (A, B, C, D): D bought at 500 (the 9,999 of as_of is ignored), −10%.
    assert top3["universe_mean_return"] == pytest.approx((0.5 + 0.1 + 0.3 - 0.1) / 4)
    assert result.loc[2, "mean_return"] == pytest.approx((0.5 + 0.1) / 2)  # tie → B first


def test_impossible_one_day_jump_voids_the_holding() -> None:
    rows = [
        ("X", date(2024, 3, 28), 4.0, 1.0), ("X", date(2024, 3, 29), 480.0, 1.0),
        ("X", date(2025, 3, 31), 500.0, 1.0),
        ("Y", date(2024, 3, 28), 100.0, 1.0), ("Y", date(2024, 9, 2), 190.0, 1.0),  # +90%: possible
        ("Y", date(2025, 3, 31), 200.0, 1.0),
    ]  # fmt: skip
    prices = AdjustedPrices(
        pd.DataFrame(rows, columns=["code", "date", "close", "adj_factor"]), max_daily_ratio=3.0
    )
    assert holding_return(prices, "X", date(2024, 3, 29), date(2025, 3, 31), 10) is None
    assert prices.suspect_codes == {"X"}
    held = holding_return(prices, "Y", date(2024, 3, 29), date(2025, 3, 31), 10)
    assert held is not None and held[0] == pytest.approx(1.0)


def test_value_basket_takes_small_cheap_cash_rich_companies_by_net_cash() -> None:
    frame = pd.DataFrame(
        {
            "market_cap_jpy": [5e9, 5e9, 50e9, 5e9, 5e9],
            "pbr": [0.6, 0.6, 0.6, 1.2, None],
            "net_cash_to_market_cap": [0.5, 0.35, 0.9, 0.9, 0.9],
        }
    )
    config = ValueBasketConfig(max_market_cap=30e9, max_pbr=1.0, min_net_cash=0.3)
    score = value_basket_score(frame, config)
    assert score.tolist()[:2] == [0.5, 0.35]  # eligible, ranked by net cash
    assert pd.isna(score[2]) and pd.isna(score[3]) and pd.isna(score[4])  # big / pricey / unknown


def test_quality_basket_also_needs_a_known_operating_profit() -> None:
    frame = pd.DataFrame(
        {
            "market_cap_jpy": [5e9, 5e9, 5e9],
            "pbr": [0.6, 0.6, 0.6],
            "net_cash_to_market_cap": [0.5, 0.5, 0.5],
            "operating_income_to_market_cap": [0.05, -0.02, None],
        }
    )
    config = ValueBasketConfig(
        max_market_cap=30e9, max_pbr=1.0, min_net_cash=0.3, min_operating_yield=0.0
    )
    score = value_basket_score(frame, config)
    assert score[0] == 0.5
    assert pd.isna(score[1]) and pd.isna(score[2])  # a loss, or unknown, is not "quality"


def test_sector_neutral_ranks_within_each_sector() -> None:
    score = pd.Series([0.30, 0.20, 0.05, 0.00, 0.10])
    sector = pd.Series(["銀行業", "銀行業", "化学", "化学", None])
    neutral = sector_neutral(score, sector)
    # The banks' common level is removed: the best chemical now ties the best bank.
    assert neutral.tolist()[:4] == pytest.approx([0.05, -0.05, 0.025, -0.025])
    assert neutral[4] == pytest.approx(0.0)  # unknown sector is its own group


def test_summary_across_dates_shows_the_spread_not_only_the_mean() -> None:
    picks = pd.DataFrame(
        {
            "model": ["v0"] * 4,
            "top_n": [25] * 4,
            "as_of": [date(2024, 3, 31), date(2024, 6, 30), date(2024, 9, 30), date(2024, 12, 31)],
            "mean_return": [0.10, 0.00, 0.20, 0.76],
            "universe_mean_return": [0.10, 0.05, 0.15, 0.24],
        }
    )
    summary = summarize_picks(picks).iloc[0]
    assert summary["mean_excess"] == pytest.approx((0.0 - 0.05 + 0.05 + 0.52) / 4)
    assert summary["median_excess"] == pytest.approx(0.025)
    assert summary["worst_excess"] == pytest.approx(-0.05)
    assert summary["mean_excess_without_best"] == pytest.approx(0.0)
    assert summary["dates_beating_universe"] == 2


def test_live_returns_so_far_per_ranking_against_all_companies() -> None:
    snapshot = pd.DataFrame(
        {"sec_code": ["A", "B", "C", "D"], "score": [0.9, 0.8, 0.8, 0.1],
         "value_basket": [None, 0.5, 0.4, None]}
    )  # fmt: skip
    board = live_returns(
        {AS_OF: snapshot}, ["score", "value_basket"], AdjustedPrices(_prices()),
        today=date(2025, 3, 31), top_n=2, stale_days=10,
    ).set_index("ranking")  # fmt: skip
    # score top 2 = A (+50%), B (+10%); basket top 2 = B (+10%), C (+30%); all: (50+10+30−10)/4.
    assert board.loc["score", "top_return_so_far"] == pytest.approx(0.30)
    assert board.loc["value_basket", "top_return_so_far"] == pytest.approx(0.20)
    assert board.loc["score", "universe_return_so_far"] == pytest.approx(0.20)
    assert board.loc["value_basket", "excess_so_far"] == pytest.approx(0.0)


def _scored_two_dates() -> pd.DataFrame:
    """Walk-forward test rows of two dates: a ranking, the prior / isotonic P and the realized
    12-month return (NaN = no price, as for a company with no TSE quote).
    """
    return pd.DataFrame(
        {
            "as_of": [date(2024, 3, 31)] * 4 + [date(2024, 6, 30)] * 4,
            "label": [1, 0, 0, 0, 0, 1, 0, 0],
            "rank_me": [0.9, 0.8, 0.1, None, 0.9, 0.8, 0.7, 0.1],
            "p_prior": [0.20, 0.10, 0.01, 0.01, 0.30, 0.10, 0.05, 0.01],
            "p_isotonic": [0.08, 0.04, 0.01, 0.01, 0.06, 0.04, 0.03, 0.01],
            "realized_return": [0.40, -0.10, 0.10, 0.00, None, 0.30, 0.00, -0.10],
        }
    )


def test_top_k_breakdown_splits_the_return_into_offers_and_the_rest() -> None:
    table = top_k_breakdown(_scored_two_dates(), ["rank_me"], [2]).iloc[0]
    # Top 2: date 1 = rows 0 (hit, +40%), 1 (−10%); date 2 = rows 4 (no price), 5 (hit, +30%).
    assert table["dates"] == 2 and table["picks"] == 4 and table["priced"] == 3
    assert table["hit_rate"] == pytest.approx(0.5)
    assert table["p_prior_mean"] == pytest.approx((0.20 + 0.10 + 0.30 + 0.10) / 4)
    assert table["p_isotonic_mean"] == pytest.approx((0.08 + 0.04 + 0.06 + 0.04) / 4)
    assert table["return_hits"] == pytest.approx(0.35)
    assert table["return_misses"] == pytest.approx(-0.10)
    # Priced picks: 2 hits, 1 miss → 2/3 × 35% from offers, 1/3 × −10% from the rest.
    assert table["from_offers"] == pytest.approx(2 / 3 * 0.35)
    assert table["from_others"] == pytest.approx(1 / 3 * -0.10)
    assert table["mean_return"] == pytest.approx((0.40 - 0.10 + 0.30) / 3)
    # All companies: date 1 mean (0.4 − 0.1 + 0.1 + 0) / 4 = 10%; date 2 (0.3 + 0 − 0.1) / 3.
    assert table["universe_return"] == pytest.approx((0.10 + 0.2 / 3) / 2)
    assert table["excess"] == pytest.approx(table["mean_return"] - table["universe_return"])


def test_top_k_breakdown_ranks_unknown_scores_last_and_skips_missing_rankings() -> None:
    table = top_k_breakdown(_scored_two_dates(), ["rank_me", "absent"], [4])
    assert table["model"].tolist() == ["rank_me"]
    # Date 1 holds 4 rows, one with an unknown score: it is still the 4th pick, never the 1st.
    assert table.iloc[0]["picks"] == 8


def test_top_k_breakdown_without_probabilities_leaves_them_unknown() -> None:
    scored = _scored_two_dates().drop(columns=["p_prior", "p_isotonic"])
    row = top_k_breakdown(scored, ["rank_me"], [1]).iloc[0]
    assert pd.isna(row["p_prior_mean"]) and pd.isna(row["p_isotonic_mean"])
    assert row["hit_rate"] == pytest.approx(0.5)


def _walk_forward_picks() -> pd.DataFrame:
    """Two rankings × top 10 over three dates; all companies return 10% on every date."""
    dates = [date(2024, 3, 31), date(2024, 6, 30), date(2024, 9, 30)]
    rows = []
    for model, returns in {"steady": [0.12, 0.13, 0.14], "lottery": [-0.05, 0.00, 0.65]}.items():
        for as_of, value in zip(dates, returns, strict=True):
            rows.append(
                {"model": model, "top_n": 10, "as_of": as_of, "mean_return": value,
                 "universe_mean_return": 0.10, "topix_return": 0.08, "hit_rate": 0.1,
                 "share_positive": 0.6, "worst_return": value - 0.3}
            )  # fmt: skip
    return pd.DataFrame(rows)


def _top_k() -> pd.DataFrame:
    return pd.DataFrame(
        {"model": ["steady"], "top_n": [10], "p_prior_mean": [0.05], "p_isotonic_mean": [0.03],
         "from_offers": [0.02], "from_others": [0.11]}
    )  # fmt: skip


def test_strategy_table_shows_level_spread_and_luck_per_strategy() -> None:
    table = strategy_table(_walk_forward_picks(), _top_k()).set_index("model")
    steady, lottery = table.loc["steady"], table.loc["lottery"]
    assert steady["mean_excess"] == pytest.approx(0.03)
    assert lottery["mean_excess"] == pytest.approx(0.10)  # higher mean …
    assert lottery["mean_excess_without_best"] == pytest.approx(-0.125)  # … from one date
    assert steady["worst_excess"] == pytest.approx(0.02)
    assert steady["std_excess"] == pytest.approx(0.01)
    assert lottery["dates_positive"] == 1 and steady["dates_positive"] == 3
    assert steady["mean_excess_vs_topix"] == pytest.approx(0.05)
    assert steady["worst_stock"] == pytest.approx(0.12 - 0.3)
    assert steady["from_offers"] == pytest.approx(0.02)
    assert pd.isna(lottery["from_offers"])  # no top-k breakdown for it: unknown, not 0


def test_lenses_prefer_robust_return_and_the_best_worst_date() -> None:
    table = strategy_table(_walk_forward_picks(), _top_k())
    lenses = {
        "return": {"metric": "mean_excess_without_best", "tie_break": "std_excess"},
        "risk": {"metric": "worst_excess", "tie_break": "std_excess"},
    }
    chosen = pick_lenses(table, lenses)
    assert chosen == {"return": ("steady", 10), "risk": ("steady", 10)}


def test_lenses_break_ties_by_the_smaller_spread_and_skip_unknowns() -> None:
    table = pd.DataFrame(
        {"model": ["a", "b", "c"], "top_n": [10, 25, 50], "worst_excess": [0.01, 0.01, None],
         "std_excess": [0.05, 0.02, 0.0]}
    )  # fmt: skip
    chosen = pick_lenses(table, {"risk": {"metric": "worst_excess", "tie_break": "std_excess"}})
    assert chosen == {"risk": ("b", 25)}


COSTS = TradingCosts(tiers=((10e6, 0.02), (None, 0.004)), participation=0.10, days=5)


def _scored_with_costs() -> pd.DataFrame:
    frame = _scored_two_dates()
    frame["median_trading_value_jpy"] = [5e6, 2e8, 1e9, 1e9, 1e9, None, 1e9, 1e9]
    frame["dividend_yield"] = [0.03, None, 0.02, 0.02, 0.01, 0.02, 0.0, 0.0]
    frame["style_return"] = [0.10, 0.00, 0.10, 0.00, 0.10, 0.10, 0.00, 0.00]
    return frame


def test_top_k_net_return_adds_dividends_and_subtracts_costs() -> None:
    row = top_k_breakdown(_scored_with_costs(), ["rank_me"], [2], COSTS).iloc[0]
    # Priced picks: rows 0 (+40%, yield 3%, ¥5m/day → 2%), 1 (−10%, yield unknown → 0,
    # ¥200m/day → 0.4%), 5 (+30%, 2%, liquidity unknown → the highest cost, 2%).
    assert row["dividend_estimate"] == pytest.approx((0.03 + 0.0 + 0.02) / 3)
    assert row["dividend_coverage"] == pytest.approx(2 / 3)
    assert row["cost"] == pytest.approx((0.02 + 0.004 + 0.02) / 3)
    assert row["net_return"] == pytest.approx(
        row["mean_return"] + row["dividend_estimate"] - row["cost"]
    )
    # Style: each pick minus its own size × book bucket's return.
    assert row["style_excess"] == pytest.approx(((0.40 - 0.10) + (-0.10 - 0.0) + (0.30 - 0.10)) / 3)


def test_top_k_capacity_is_set_by_the_least_liquid_pick() -> None:
    row = top_k_breakdown(_scored_with_costs(), ["rank_me"], [2], COSTS).iloc[0]
    # Equal weights: capital / 2 per name ≤ 10% × ¥/day × 5 days for EVERY pick.
    # Date 1: min ¥5m → ¥2.5m × 2 = ¥5m. Date 2: only row 4 known (¥1bn) → ¥1bn. Median: ¥502.5m.
    assert row["capacity_jpy"] == pytest.approx((5e6 + 1e9) / 2)


def test_top_k_without_costs_leaves_the_net_columns_unknown() -> None:
    row = top_k_breakdown(_scored_two_dates(), ["rank_me"], [2]).iloc[0]
    assert pd.isna(row["net_return"]) and pd.isna(row["capacity_jpy"])
    assert pd.isna(row["style_excess"])


def test_strategy_table_reports_overlap_aware_error_and_selection_haircut() -> None:
    table = strategy_table(_walk_forward_picks(), _top_k(), nw_lags=1, z=1.645).set_index("model")
    steady, lottery = table.loc["steady"], table.loc["lottery"]
    assert steady["t_nw"] == pytest.approx(steady["mean_excess"] / steady["excess_se_nw"])
    assert steady["ci_low"] < steady["mean_excess"] < steady["ci_high"]
    assert steady["ci_high"] - steady["mean_excess"] == pytest.approx(
        1.645 * steady["excess_se_nw"]
    )
    # Steady small wins are more credible than one lucky date, after the selection haircut.
    assert 0 <= lottery["deflated_sharpe"] < steady["deflated_sharpe"] <= 1


def _basket_frame() -> pd.DataFrame:
    """Four basket members (small, cheap, cash-rich) and one big company outside it."""
    return pd.DataFrame(
        {
            "market_cap_jpy": [5e9, 5e9, 5e9, 5e9, 50e9],
            "pbr": [0.6, 0.6, 0.6, 0.6, 0.6],
            "net_cash_to_market_cap": [0.9, 0.5, 0.4, 0.35, 0.9],
            "p_parent": [0.01, 0.20, None, 0.02, 0.90],
            "p_mbo": [0.01, 0.05, 0.03, 0.30, 0.90],
            "momentum_12_1": [-0.60, 0.10, None, 0.05, 0.20],
        }
    )


def test_value_catalyst_ranks_the_basket_by_parent_and_mbo_likelihood() -> None:
    config = ValueBasketConfig(max_market_cap=30e9, max_pbr=1.0, min_net_cash=0.3)
    score = value_catalyst_score(_basket_frame(), config)
    order = score.sort_values(ascending=False).dropna().index.tolist()
    assert order[:2] == [3, 1]  # P(parent)+P(MBO) 0.32, 0.25 — not net cash
    assert pd.isna(score[4])  # outside the basket, however likely the offer
    assert score[2] == pytest.approx(0.03)  # unknown part: the known part only, never invented


def test_value_momentum_drops_the_worst_momentum_from_the_basket() -> None:
    config = ValueBasketConfig(max_market_cap=30e9, max_pbr=1.0, min_net_cash=0.3)
    score = value_momentum_score(_basket_frame(), config, worst_share=0.25)
    # Row 0 is in the worst quarter of momentum: out, although it holds the most cash.
    assert pd.isna(score[0])
    assert score[1] == pytest.approx(0.5)
    assert score[2] == pytest.approx(0.4)  # no year of history: not excluded on an unknown


def test_union_rank_takes_the_top_of_each_list_in_turn() -> None:
    frame = pd.DataFrame(
        {
            "v2": [0.9, 0.8, 0.5, 0.5, 0.1, 0.1],
            "p_mbo": [0.0, 0.1, 0.0, 0.0, 0.9, 0.8],
            "p_any": [0.0, 0.0, 0.1, 0.3, 0.0, 0.0],  # breaks the v2 tie between rows 2 and 3
        }
    )
    combo = union_rank_score(frame, {"v2": "p_any", "p_mbo": None})
    top4 = combo.sort_values(ascending=False).index[:4].tolist()
    assert sorted(top4) == [0, 1, 4, 5]  # the top 2 of v2 + the top 2 of MBO
    assert combo[3] > combo[2]  # tie on v2 broken by p_any, not by row order


def test_union_rank_ignores_unknown_scores() -> None:
    frame = pd.DataFrame({"a": [None, 0.5, 0.2], "b": [0.9, None, None]})
    combo = union_rank_score(frame, {"a": None, "b": None})
    assert combo.notna().all() and combo[0] == combo.max()  # top of b although a is unknown
