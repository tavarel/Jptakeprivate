from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

from radar.features.family import family_stake, family_surnames
from radar.features.market import (
    median_trading_value,
    momentum_12_1,
    officer_stake,
    price_change_12m,
    unusual_trading,
)
from radar.features.point_in_time import latest_annual_filings, price_as_of
from radar.features.valuation import (
    adjusted_net_assets,
    market_cap,
    net_cash,
    price_to_book,
    shares_outstanding,
    split_multiplier,
    treasury_at_filing,
)


# ------------------------------------------------------------------ family
def _officers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["山田 太郎", "佐藤 花子", "鈴木 一郎"],
            "is_representative": [True, False, True],
        }
    )


def _holders() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": [
                "山田 太郎",
                "株式会社山田興産",
                "日本マスタートラスト信託銀行株式会社(信託口)",
                "山田 次郎",
                "自己株式(山田工業)",
            ],
            "ratio": [0.20, 0.15, 0.08, 0.05, 0.10],
            "is_organization": [False, True, True, False, True],
        }
    )


def test_family_surnames_from_representatives_only() -> None:
    assert family_surnames(_officers()) == frozenset({"山田", "鈴木"})


def test_family_stake_sums_persons_and_family_companies_but_not_treasury() -> None:
    result = family_stake(_holders(), frozenset({"山田"}))
    assert result.stake == pytest.approx(0.40)  # 0.20 + 0.15 + 0.05
    assert result.founder_in_control
    assert "日本マスタートラスト信託銀行株式会社(信託口)" not in result.matched_holders


def test_family_stake_without_surnames_is_missing_not_zero() -> None:
    # ~37% of representatives have no recoverable surname (names spaced per character):
    # "unknown family" must stay None, never become "no family" (0.0).
    result = family_stake(_holders(), frozenset())
    assert result.stake is None
    assert not result.founder_in_control


def test_family_stake_without_shareholders_is_missing() -> None:
    assert family_stake(_holders().iloc[0:0], frozenset({"山田"})).stake is None


def test_family_stake_no_match_is_zero() -> None:
    # Surname known and holders known, but none of them is family: a real 0, not missing.
    assert family_stake(_holders(), frozenset({"高橋"})).stake == 0.0


# ------------------------------------------------------------------ valuation
def test_net_cash_conservative_default() -> None:
    fin = {
        "cash_and_deposits": 1000.0,
        "short_term_securities": 100.0,
        "investment_securities": 500.0,
        "long_term_loans": 300.0,
    }
    assert net_cash(fin) == 800.0  # investment securities ignored by default
    assert net_cash(fin, investment_securities_haircut=0.4) == pytest.approx(1100.0)


def test_net_cash_none_when_cash_missing() -> None:
    assert net_cash({"long_term_loans": 300.0}) is None


def test_market_cap_and_pbr() -> None:
    outstanding = shares_outstanding(1_000_000, 100_000)
    mcap = market_cap(1500.0, outstanding)
    assert mcap == 1_350_000_000
    assert price_to_book(mcap, 2_700_000_000) == pytest.approx(0.5)
    assert price_to_book(mcap, -10.0) is None
    assert market_cap(None, outstanding) is None


# ------------------------------------------------------------------ point in time
def _documents() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # company A: FY2024 report, then an amendment, then FY2025 report filed later
            ("D1", "E1", "120", datetime(2024, 6, 25, 15), date(2024, 3, 31), "0"),
            ("D2", "E1", "130", datetime(2024, 9, 1, 9), date(2024, 3, 31), "0"),
            ("D3", "E1", "120", datetime(2025, 6, 25, 15), date(2025, 3, 31), "0"),
            # company B: only a withdrawn report
            ("D4", "E2", "120", datetime(2024, 6, 20, 15), date(2024, 3, 31), "2"),
        ],
        columns=[
            "doc_id",
            "edinet_code",
            "doc_type_code",
            "submit_datetime",
            "period_end",
            "withdrawal_status",
        ],
    )


@pytest.mark.parametrize(
    ("as_of", "expected_doc"),
    [
        (date(2024, 6, 25), None),  # filed at 15:00 ON as_of → not usable yet
        (date(2024, 6, 26), "D1"),
        (date(2024, 9, 2), "D2"),  # amendment public → replaces original
        (date(2025, 6, 26), "D3"),  # newer fiscal year wins
    ],
)
def test_latest_annual_filings_has_no_look_ahead(as_of: date, expected_doc: str | None) -> None:
    result = latest_annual_filings(_documents(), as_of)
    company_a = result[result["edinet_code"] == "E1"]
    if expected_doc is None:
        assert company_a.empty
    else:
        assert company_a.iloc[0]["doc_id"] == expected_doc
    assert "E2" not in set(result["edinet_code"])  # withdrawn filings never count


def test_price_as_of_uses_lag_and_drops_stale_prices() -> None:
    prices = pd.DataFrame(
        {
            "code": ["1000", "1000", "1000", "2000"],
            "date": [date(2025, 3, 27), date(2025, 3, 28), date(2025, 3, 31), date(2025, 1, 10)],
            "close": [100.0, 101.0, 102.0, 50.0],
        }
    )
    result = price_as_of(prices, as_of=date(2025, 3, 31), price_lag_days=1)
    assert result.set_index("code").loc["1000", "close"] == 101.0  # close of 3/31 not yet known
    assert "2000" not in set(result["code"])  # stale → excluded


def test_split_multiplier_counts_splits_after_the_report_only() -> None:
    splits = pd.DataFrame(
        {
            "date": [date(2025, 3, 26), date(2025, 5, 1), date(2025, 8, 1)],
            "adj_factor": [0.5, 0.5, 0.25],
        }
    )
    assert split_multiplier(splits, date(2025, 3, 31), date(2025, 6, 30)) == pytest.approx(2.0)
    assert split_multiplier(splits, date(2025, 6, 1), date(2025, 6, 30)) == 1.0
    assert split_multiplier(None, date(2025, 3, 31), date(2025, 6, 30)) == 1.0
    reverse = pd.DataFrame({"date": [date(2025, 5, 1)], "adj_factor": [10.0]})  # 10:1 consolidation
    assert split_multiplier(reverse, date(2025, 3, 31), date(2025, 6, 30)) == pytest.approx(0.1)


# ------------------------------------------------------------------ market features (v1 candidates)
def _bars() -> pd.DataFrame:
    days = pd.bdate_range("2024-06-03", "2025-06-30")
    frame = pd.DataFrame({"date": [d.date() for d in days]})
    frame["code"] = "99990"
    frame["close"] = 1000.0
    frame["adj_factor"] = 1.0
    frame["trading_value"] = 5_000_000.0
    frame.loc[frame["date"] >= date(2025, 1, 6), "close"] = 400.0  # after a 1:2 split + a fall
    frame.loc[frame["date"] == date(2025, 1, 6), "adj_factor"] = 0.5
    return frame


def test_price_change_12m_is_split_adjusted() -> None:
    # 1000 a year ago = 500 on today's share basis (1:2 split); now 400 → -20%, not -60%.
    change = price_change_12m(_bars(), date(2025, 6, 30))
    assert change["99990"] == pytest.approx(-0.20)


def test_price_change_12m_needs_a_year_of_history() -> None:
    assert price_change_12m(_bars(), date(2024, 9, 30)) == {}  # not enough history: unknown


def test_median_trading_value_uses_the_last_60_sessions_before_as_of() -> None:
    bars = _bars()
    bars.loc[bars["date"] == date(2025, 6, 30), "trading_value"] = 9e12  # ON as_of: ignored
    assert median_trading_value(bars, date(2025, 6, 30))["99990"] == 5_000_000.0


def test_officer_stake() -> None:
    officers = pd.DataFrame({"shares_held": [1_000.0, 500.0, None]})
    assert officer_stake(officers, outstanding=10_000.0) == pytest.approx(0.15)
    assert officer_stake(officers.iloc[0:0], outstanding=10_000.0) is None  # no table: unknown
    assert officer_stake(officers, outstanding=None) is None


def test_split_with_ex_date_before_year_end_but_effective_after_it() -> None:
    splits = pd.DataFrame({"date": [date(2026, 3, 30)], "adj_factor": [0.25]})
    assert split_multiplier(splits, date(2026, 3, 31), date(2026, 9, 30)) == pytest.approx(4.0)
    # A price dated before the ex-rights date is still pre-split: no adjustment.
    assert split_multiplier(splits, date(2026, 3, 31), date(2026, 3, 27)) == 1.0


def test_price_as_of_skips_days_without_a_trade() -> None:
    prices = pd.DataFrame(
        {
            "code": ["1000", "1000", "1000"],
            "date": [date(2025, 9, 26), date(2025, 9, 29), date(2025, 9, 30)],
            "close": [101.0, 102.0, float("nan")],  # no trade on 9/30
        }
    )
    result = price_as_of(prices, as_of=date(2025, 10, 1), price_lag_days=1)
    assert result.set_index("code").loc["1000", "close"] == 102.0
    assert result.set_index("code").loc["1000", "date"] == date(2025, 9, 29)


def test_price_change_keeps_a_split_factor_from_a_no_trade_day() -> None:
    bars = _bars()
    bars.loc[bars["date"] == date(2025, 1, 6), "close"] = float("nan")  # ex-rights day, no trade
    assert price_change_12m(bars, date(2025, 6, 30))["99990"] == pytest.approx(-0.20)


def test_family_stake_unifies_kanji_variants_and_skips_employee_plans() -> None:
    # 日本アイ・エス・ケイ (廣澤 / 広沢) and 今村証券 (今村証券社員持株会).
    holders = pd.DataFrame(
        {
            "name": ["廣澤興産(有)", "広沢 かほる", "広沢(有)", "広沢証券社員持株会", "佐藤 一郎"],
            "ratio": [0.23, 0.09, 0.065, 0.05, 0.02],
            "is_organization": [True, False, True, True, False],
        }
    )
    result = family_stake(holders, frozenset({"廣澤"}), excluded=("持株会",))
    assert result.stake == pytest.approx(0.385)


@pytest.mark.parametrize(
    ("issued", "at_filing", "treasury", "expected"),
    [
        (100.0, 100.0, 10.0, 10.0),  # nothing changed
        (7_778_000.0, 6_928_000.0, 2_170_000.0, 1_320_000.0),
        (100.0, 300.0, 10.0, 30.0),  # 1:3 split: treasury shares split too
        (300.0, 100.0, 30.0, 10.0),  # 3:1 consolidation (more shares cancelled than held)
        (29_900_000.0, 32_022_000.0, 31_700.0, 31_700.0),  # new shares (IPO): treasury unchanged
        (1000.0, 500.0, 600.0, 300.0),  # 2:1 consolidation with > 50% treasury: not a cancellation
        (100.0, 100.0, None, None),
    ],
)
def test_treasury_at_filing(
    issued: float, at_filing: float, treasury: float | None, expected: float | None
) -> None:
    result = treasury_at_filing(issued, at_filing, treasury)
    assert result == (None if expected is None else pytest.approx(expected))


@pytest.mark.parametrize(
    ("status", "book", "fair", "expected"),
    [
        ("disclosed", 2e9, 5e9, 10e9 + 3e9 * (1 - 0.3)),  # gain taxed before it is added
        ("disclosed", 5e9, 4e9, 10e9 - 1e9 * (1 - 0.3)),  # a loss lowers the adjusted book
        ("immaterial", None, None, 10e9),  # omitted as not material → gain ≈ 0
        ("none", None, None, 10e9),  # 該当事項はありません
        ("absent", None, None, 10e9),  # the note is only required when material
        ("unreadable", None, None, None),  # never guessed
        (None, None, None, None),  # IFRS / not parsed
    ],
)
def test_adjusted_net_assets(
    status: str | None, book: float | None, fair: float | None, expected: float | None
) -> None:
    fin = {"net_assets": 10e9, "rental_note_status": status,
           "rental_book_value": book, "rental_fair_value": fair}  # fmt: skip
    result = adjusted_net_assets(fin, tax_rate=0.3)
    assert result == (None if expected is None else pytest.approx(expected))


def test_momentum_12_1_skips_the_last_month() -> None:
    # Classic 12-1 momentum: a year ago → one month ago. A jump in the last month (and the
    # close ON as_of, a look-ahead trap) must not count; the 1:2 split is still restated.
    bars = _bars()
    bars.loc[bars["date"] >= date(2025, 6, 2), "close"] = 800.0
    bars.loc[bars["date"] == date(2025, 6, 30), "close"] = 9999.0
    assert momentum_12_1(bars, date(2025, 6, 30))["99990"] == pytest.approx(-0.20)
    assert price_change_12m(bars, date(2025, 6, 30))["99990"] == pytest.approx(0.60)


def test_momentum_12_1_needs_a_year_of_history() -> None:
    assert momentum_12_1(_bars(), date(2024, 9, 30)) == {}


def _quiet_then_busy() -> tuple[pd.DataFrame, pd.DataFrame]:
    days = pd.bdate_range("2025-03-03", "2025-06-30")
    bars = pd.DataFrame({"date": [d.date() for d in days]})
    bars["code"], bars["close"], bars["adj_factor"] = "99990", 1000.0, 1.0
    bars["trading_value"] = 5_000_000.0
    last5 = bars["date"] >= date(2025, 6, 23)
    bars.loc[last5, "trading_value"] = 20_000_000.0  # 4× the usual volume
    bars.loc[last5, "close"] = 1150.0  # +15% in the last week
    bars.loc[bars["date"] == date(2025, 6, 30), ["close", "trading_value"]] = [9999.0, 9e12]
    topix = pd.DataFrame({"date": [d.date() for d in days], "close": 2000.0})
    return bars, topix


def test_unusual_trading_flags_volume_and_price_before_as_of_only() -> None:
    bars, topix = _quiet_then_busy()
    found = unusual_trading(bars, topix, {"99990"}, date(2025, 6, 30),
                            short=5, long=60, volume_ratio=3.0, excess_move=0.10)  # fmt: skip
    row = found.iloc[0]
    assert row["sec_code"] == "99990"
    assert row["volume_ratio"] == pytest.approx(4.0)  # the as_of session (9e12) is ignored
    assert row["excess_5d"] == pytest.approx(0.15)
    calm = unusual_trading(bars, topix, {"99990"}, date(2025, 6, 20),
                           short=5, long=60, volume_ratio=3.0, excess_move=0.10)  # fmt: skip
    assert calm.empty  # nothing unusual before the busy week


def test_unusual_trading_ignores_a_stock_split() -> None:
    bars, topix = _quiet_then_busy()
    bars["trading_value"], bars["close"] = 5_000_000.0, 1000.0
    bars.loc[bars["date"] >= date(2025, 6, 25), "close"] = 500.0  # 1:2 split, no real move
    bars.loc[bars["date"] == date(2025, 6, 25), "adj_factor"] = 0.5
    found = unusual_trading(bars, topix, {"99990"}, date(2025, 6, 30),
                            short=5, long=60, volume_ratio=3.0, excess_move=0.10)  # fmt: skip
    assert found.empty
