"""Trend features: consecutive years below book, change in net cash (pure functions)."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

from radar.features.point_in_time import annual_filings_history
from radar.features.sector import sector_percentiles
from radar.features.trends import (
    CloseHistory,
    consecutive_below_book,
    net_cash_change,
    year_end_pbr,
)


def test_consecutive_below_book_counts_from_the_newest_year() -> None:
    assert consecutive_below_book([0.8, 0.9, 1.2, 0.5]) == 2  # stops at the first year ≥ 1
    assert consecutive_below_book([1.1, 0.5]) == 0
    assert consecutive_below_book([0.7, None, 0.5]) == 1  # unknown year: count is a floor
    assert consecutive_below_book([None, 0.5]) is None  # newest unknown → unknown, not 0
    assert consecutive_below_book([]) is None


def test_close_history_returns_last_actual_close_on_or_before() -> None:
    prices = pd.DataFrame(
        {
            "code": ["99010"] * 3,
            "date": [date(2025, 3, 27), date(2025, 3, 28), date(2025, 3, 31)],
            "close": [100.0, 110.0, None],  # 3/31: no trade
        }
    )
    history = CloseHistory(prices)
    assert history.close_on_or_before("99010", date(2025, 3, 31)) == (date(2025, 3, 28), 110.0)
    assert history.close_on_or_before("99010", date(2025, 3, 26)) is None  # before any price
    assert history.close_on_or_before("99010", date(2025, 5, 1)) is None  # stale (> 10 days)
    assert history.close_on_or_before("99999", date(2025, 3, 31)) is None


def test_year_end_pbr_adjusts_a_split_effective_after_the_year_end() -> None:
    fin = {"issued": 10e6, "treasury": 0.0, "net_assets": 20e9}
    assert year_end_pbr(fin, (date(2025, 3, 28), 1000.0), None, date(2025, 3, 31)) == 0.5
    # 1:2 split, ex-rights 3/28 (effective 4/1): the price is post-split, the count is not.
    splits = pd.DataFrame({"date": [date(2025, 3, 28)], "adj_factor": [0.5]})
    pbr = year_end_pbr(fin, (date(2025, 3, 28), 500.0), splits, date(2025, 3, 31))
    assert pbr == pytest.approx(0.5)
    assert year_end_pbr(fin, None, None, date(2025, 3, 31)) is None
    assert year_end_pbr({**fin, "net_assets": None}, (date(2025, 3, 28), 1.0), None,
                        date(2025, 3, 31)) is None  # fmt: skip


def test_net_cash_change_relative_to_market_cap() -> None:
    latest = {"cash_and_deposits": 8e9, "long_term_loans": 1e9}
    previous = {"cash_and_deposits": 5e9, "long_term_loans": 1e9}
    assert net_cash_change(latest, previous, 10e9) == pytest.approx(0.3)
    assert net_cash_change(latest, None, 10e9) is None  # no earlier report
    assert net_cash_change(latest, {"cash_and_deposits": None}, 10e9) is None
    assert net_cash_change(latest, previous, None) is None


def test_annual_history_is_point_in_time_and_resolves_amendments() -> None:
    def doc(doc_id: str, type_: str, period: date, submitted: datetime) -> dict[str, object]:
        return {
            "doc_id": doc_id,
            "edinet_code": "E99001",
            "doc_type_code": type_,
            "period_end": period,
            "submit_datetime": submitted,
            "withdrawal_status": "0",
        }

    documents = pd.DataFrame(
        [
            doc("FY22", "120", date(2023, 3, 31), datetime(2023, 6, 25)),
            doc("FY23", "120", date(2024, 3, 31), datetime(2024, 6, 25)),
            doc("FY23_FIX", "130", date(2024, 3, 31), datetime(2024, 9, 1)),  # amendment
            doc("FY24", "120", date(2025, 3, 31), datetime(2025, 7, 1, 9)),  # trap: ON as_of
        ]
    )
    history = annual_filings_history(documents, date(2025, 7, 1), periods=3)
    assert history["doc_id"].tolist() == ["FY23_FIX", "FY22"]  # newest period first


def test_close_history_without_prices_finds_nothing() -> None:
    empty = pd.DataFrame(columns=["code", "date", "close"])
    assert CloseHistory(empty).close_on_or_before("99010", date(2025, 3, 31)) is None


def test_sector_percentiles_rank_within_each_sector_only() -> None:
    frame = pd.DataFrame(
        {
            "sector33": ["電気機器", "電気機器", "電気機器", "銀行業", None],
            "pbr": [0.5, 1.0, None, 0.3, 0.2],
        }
    )
    pct = sector_percentiles(frame, "pbr", min_peers=2)
    assert pct.tolist()[:2] == [0.5, 1.0]  # cheapest of two ranked companies → 0.5
    assert pd.isna(pct[2])  # missing PBR is not ranked
    assert pd.isna(pct[3])  # a sector with fewer than min_peers values: no comparison
    assert pd.isna(pct[4])  # unknown sector


def test_year_end_pbr_without_a_period_end_is_unknown() -> None:
    fin = {"issued": 10e6, "treasury": 0.0, "net_assets": 20e9}
    assert year_end_pbr(fin, (date(2025, 3, 28), 1000.0), None, None) is None


def test_close_history_with_a_missing_day_finds_nothing() -> None:
    prices = pd.DataFrame({"code": ["99010"], "date": [date(2025, 3, 28)], "close": [1.0]})
    assert CloseHistory(prices).close_on_or_before("99010", None) is None


def test_annual_history_skips_reports_without_a_period() -> None:
    documents = pd.DataFrame(
        {
            "doc_id": ["FY24", "FY24_FIX"],
            "edinet_code": ["E99001", "E99001"],
            "doc_type_code": ["120", "130"],
            "period_end": [date(2025, 3, 31), None],
            "submit_datetime": [datetime(2025, 6, 25), datetime(2025, 8, 1)],
            "withdrawal_status": ["0", "0"],
        }
    )
    history = annual_filings_history(documents, date(2025, 10, 1), periods=3)
    assert history["doc_id"].tolist() == ["FY24"]
