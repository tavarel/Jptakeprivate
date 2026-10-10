"""Premium and pre-announcement run-up per tender offer (synthetic prices)."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from radar.model.event_study import (
    EventStudyConfig,
    earlier_offer_within,
    event_study,
    volume_ratio,
)

CONFIG = EventStudyConfig(
    runup_sessions=3, max_stale_days=10, pre_announced_tolerance=0.03, runup_short_sessions=2
)
DAYS = [date(2025, 6, d) for d in (2, 3, 4, 5, 6, 9)]


def _prices(closes: list[float], factors: list[float] | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "code": "99990",
            "date": DAYS,
            "close": closes,
            "adj_factor": factors or [1.0] * len(DAYS),
        }
    )


def _topix(closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"date": DAYS, "close": closes})


def _offers(price: float, event_date: date = date(2025, 6, 10)) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "doc_id": ["T1"],
            "subject_edinet_code": ["E99001"],
            "event_date": [event_date],
            "offer_price": [price],
            "event_type": ["mbo"],
            "is_take_private": [True],
        }
    )


SEC = {"E99001": "99990"}


def test_premium_and_market_adjusted_runup() -> None:
    prices = _prices([100, 100, 100, 110, 115, 120])  # rises into the announcement
    topix = _topix([1000, 1000, 1000, 1010, 1020, 1050])
    row = event_study(_offers(180.0), prices, topix, SEC, CONFIG).iloc[0]
    assert row["pre_close"] == 120.0  # last close BEFORE the filing date
    assert row["premium"] == pytest.approx(0.5)  # 180 / 120 − 1
    # Window = the 3 sessions before the last close: 6/4 (100) → 6/9 (120): +20%; TOPIX +5%.
    assert row["runup_abnormal"] == pytest.approx(0.20 - 0.05)
    assert not row["pre_announced"]


def test_runup_is_split_adjusted() -> None:
    # A 1:2 split on 6/5 halves the price; without adjusting it would look like a crash.
    prices = _prices([200, 200, 200, 100, 105, 110], [1, 1, 1, 0.5, 1, 1])
    row = event_study(_offers(150.0), prices, _topix([1000] * 6), SEC, CONFIG).iloc[0]
    assert row["runup_abnormal"] == pytest.approx(110 / (200 * 0.5) - 1)  # +10%


def test_pre_announced_offer_is_flagged() -> None:
    prices = _prices([100, 100, 100, 179, 180, 181])  # already trading at the offer price
    row = event_study(_offers(182.0), prices, _topix([1000] * 6), SEC, CONFIG).iloc[0]
    assert row["pre_announced"]


def test_missing_or_stale_prices_give_unknown_not_zero() -> None:
    late = event_study(
        _offers(150.0, date(2025, 7, 30)), _prices([100] * 6), _topix([1000] * 6), SEC, CONFIG
    )
    assert pd.isna(late.iloc[0]["premium"])  # last close is 7 weeks old (suspended?)
    no_code = event_study(_offers(150.0), _prices([100] * 6), _topix([1000] * 6), {}, CONFIG)
    assert pd.isna(no_code.iloc[0]["premium"])


def test_short_history_gives_premium_but_no_runup() -> None:
    prices = _prices([100, 100, 100, 100, 100, 100]).tail(2)
    row = event_study(_offers(150.0), prices, _topix([1000] * 6), SEC, CONFIG).iloc[0]
    assert row["premium"] == pytest.approx(0.5)
    assert pd.isna(row["runup_abnormal"])


def test_volume_ratio_flags_heavy_trading_before_the_announcement() -> None:
    days = [d.date() for d in pd.bdate_range("2025-01-01", periods=30)]
    bars = pd.DataFrame({"code": "99990", "date": days, "close": 100.0, "adj_factor": 1.0})
    bars["trading_value"] = [1e6] * 25 + [5e6] * 5  # the last 5 sessions trade 5× normal
    ratio = volume_ratio(bars, days[-1] + pd.Timedelta(days=1), recent=5, baseline=20)
    assert ratio == pytest.approx(5.0)
    assert volume_ratio(bars, days[3], recent=5, baseline=20) is None  # not enough history


def test_earlier_offer_marks_bidding_wars() -> None:
    # A second bidder's run-up reflects the first,
    # already public offer — not leakage.
    offers = pd.DataFrame(
        {
            "doc_id": ["A", "B", "C", "D"],
            "subject_edinet_code": ["E1", "E1", "E2", "E1"],
            "event_date": [date(2025, 1, 10), date(2025, 3, 1), date(2025, 3, 1), date(2026, 1, 5)],
        }
    )
    marked = earlier_offer_within(offers, days=180)
    assert marked.tolist() == [False, True, False, False]  # D: >180 days after B


def test_price_above_the_offer_is_flagged_like_pre_announced() -> None:
    # A contest or an earlier announcement pushed the
    # price ABOVE the offer → negative premium; it must leave the statistics too.
    prices = _prices([100, 100, 100, 150, 160, 170])
    row = event_study(_offers(155.0), prices, _topix([1000] * 6), SEC, CONFIG).iloc[0]
    assert row["premium"] < 0
    assert row["pre_announced"]


def test_short_window_runup() -> None:
    prices = _prices([100, 100, 100, 100, 100, 120])  # all of it in the last session
    row = event_study(_offers(180.0), prices, _topix([1000] * 6), SEC, CONFIG).iloc[0]
    assert row["runup_short"] == pytest.approx(0.20)  # 2-session window: 6/6 (100) → 6/9 (120)


def test_pre_close_skips_a_day_without_trade() -> None:
    # Same no-trade bug as price_as_of: the last row before the filing had no close.
    prices = _prices([100, 100, 100, 100, 120, float("nan")])
    row = event_study(_offers(180.0), prices, _topix([1000] * 6), SEC, CONFIG).iloc[0]
    assert row["pre_close"] == 120.0
    assert row["premium"] == pytest.approx(0.5)


def test_split_on_a_no_trade_day_is_still_applied() -> None:
    # The ex-rights row (AdjFactor 0.5) has no trade; its factor must not be dropped with it.
    prices = _prices([200, 200, 200, float("nan"), 100, 110], [1, 1, 1, 0.5, 1, 1])
    row = event_study(_offers(150.0), prices, _topix([1000] * 6), SEC, CONFIG).iloc[0]
    # Window = last 3 traded sessions + start: 6/3 (200) → 6/9 (110), split in between: +10%.
    assert row["runup_abnormal"] == pytest.approx(110 / (200 * 0.5) - 1)
