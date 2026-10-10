"""Decision journal: what was decided, and what happened afterwards."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from radar.report.journal import journal_outcomes, parse_journal


def _raw() -> dict[str, object]:
    return {
        "entries": [
            {"date": date(2025, 1, 6), "code": "99990", "action": "buy", "price": 1000,
             "reason": "サンプル: listed parent at 40%"},
            {"date": "2025-03-03", "code": 99980, "action": "pass", "reason": "too illiquid"},
        ]
    }  # fmt: skip


def test_parse_journal_reads_dates_codes_and_optional_price() -> None:
    entries = parse_journal(_raw())
    assert entries["date"].tolist() == [date(2025, 1, 6), date(2025, 3, 3)]
    assert entries["code"].tolist() == ["99990", "99980"]  # a code typed as a number still works
    assert pd.isna(entries.loc[1, "price"])  # not given ≠ 0


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({"date": "2025-01-06", "code": "99990", "action": "hold"}, "action"),
        ({"code": "99990", "action": "buy"}, "date"),
        ({"date": "not a date", "code": "99990", "action": "buy"}, "date"),
    ],
)
def test_parse_journal_explains_a_bad_entry(entry: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_journal({"entries": [entry]})


def test_empty_journal_is_an_empty_table() -> None:
    assert parse_journal({}).empty and parse_journal({"entries": None}).empty


def test_outcomes_restate_splits_and_flag_a_later_offer() -> None:
    prices = pd.DataFrame(
        {
            "code": ["99990"] * 3 + ["99980"] * 2,
            "date": [date(2025, 1, 6), date(2025, 6, 2), date(2025, 9, 1),
                     date(2025, 3, 3), date(2025, 9, 1)],
            "close": [1000.0, 550.0, 600.0, 200.0, 180.0],
            "adj_factor": [1.0, 0.5, 1.0, 1.0, 1.0],
        }
    )  # fmt: skip
    offers = pd.DataFrame({"sec_code": ["99990", "99990"],
                           "event_date": [date(2024, 12, 1), date(2025, 8, 1)]})  # fmt: skip
    ranks = pd.Series({"99990": 3})
    table = journal_outcomes(parse_journal(_raw()), prices, offers, ranks, today=date(2025, 9, 2))
    first, second = table.iloc[0], table.iloc[1]
    assert first["return_since"] == pytest.approx(0.20)  # 1000 → 600 × 2 after the split
    assert first["offer_after"] == date(2025, 8, 1)  # the 2024 offer came BEFORE the decision
    assert first["rank_now"] == 3
    assert second["return_since"] == pytest.approx(-0.10)
    assert pd.isna(second["offer_after"]) and pd.isna(second["rank_now"])


def test_outcome_without_prices_is_unknown() -> None:
    empty = pd.DataFrame(columns=["code", "date", "close", "adj_factor"])
    offers = pd.DataFrame(columns=["sec_code", "event_date"])
    table = journal_outcomes(parse_journal(_raw()), empty, offers, pd.Series(dtype=float),
                             today=date(2025, 9, 2))  # fmt: skip
    assert table["return_since"].isna().all()


def test_four_character_codes_as_people_write_them() -> None:
    prices = pd.DataFrame({"code": ["71220", "71220"], "date": [date(2025, 1, 6), date(2025, 9, 1)],
                           "close": [100.0, 150.0], "adj_factor": [1.0, 1.0]})  # fmt: skip
    entries = parse_journal(
        {"entries": [{"date": "2025-01-06", "code": "7122", "action": "watch"}]}
    )
    offers = pd.DataFrame(columns=["sec_code", "event_date"])
    table = journal_outcomes(entries, prices, offers, pd.Series({"71220": 12}),
                             today=date(2025, 9, 2))  # fmt: skip
    assert table.loc[0, "return_since"] == pytest.approx(0.5)
    assert table.loc[0, "rank_now"] == 12
