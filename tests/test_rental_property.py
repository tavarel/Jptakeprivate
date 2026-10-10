"""賃貸等不動産 note: book value and fair value of rental real estate (adjusted PBR)."""

from __future__ import annotations

import pytest

from radar.config import load_yaml
from radar.parse.rental_property import RentalConfig, parse_rental_note

CONFIG = RentalConfig.from_config(load_yaml("trends"))

TWO_COLUMNS = (
    "(賃貸等不動産関係)当社は賃貸用住居を有しております。(単位:千円)前連結会計年度当連結会計年度"
    "連結貸借対照表計上額期首残高2,456,2953,075,505期中増減額619,209△1,087,765"
    "期末残高3,075,5051,987,739期末時価7,179,1456,622,057(注)1.連結貸借対照表計上額は…"
)
ONE_COLUMN = (
    "(単位:百万円)連結貸借対照表計上額当連結会計年度期首残高4,756当連結会計年度増減額△28"
    "当連結会計年度末残高4,728当連結会計年度末の時価9,248(注)1 …"
)


def test_two_glued_columns_take_the_current_year() -> None:
    note = parse_rental_note(TWO_COLUMNS, CONFIG)
    assert note.status == "disclosed"
    assert note.book_value == 1_987_739_000  # 千円 → yen, current-year column
    assert note.fair_value == 6_622_057_000


def test_one_column_in_millions() -> None:
    note = parse_rental_note(ONE_COLUMN, CONFIG)
    assert (note.book_value, note.fair_value) == (4_728_000_000, 9_248_000_000)


@pytest.mark.parametrize(
    "text",
    [
        "(賃貸等不動産関係)賃貸等不動産の総資産に占める割合が僅少であるため、記載を省略しております。",
        "(賃貸等不動産関係)賃貸等不動産の総額に重要性が乏しいため記載を省略しております。",
    ],
)
def test_omitted_note_is_immaterial(text: str) -> None:
    note = parse_rental_note(text, CONFIG)
    assert note.status == "immaterial" and note.book_value is None


def test_absent_and_ambiguous_notes() -> None:
    assert parse_rental_note(None, CONFIG).status == "absent"
    # Digits without thousands separators cannot be split between the two columns.
    ambiguous = "(単位:千円)期末残高512190698540期末時価900120950210"
    assert parse_rental_note(ambiguous, CONFIG).status == "unreadable"
    no_unit = "期末残高3,075,5051,987,739期末時価7,179,1456,622,057"
    assert parse_rental_note(no_unit, CONFIG).status == "unreadable"  # unit unknown: no guess


def test_implausible_ratio_is_unreadable() -> None:
    text = "(単位:千円)期末残高1,000期末時価900,000"  # 900× book: a misread, not a gem
    assert parse_rental_note(text, CONFIG).status == "unreadable"


def test_small_and_large_numbers_glued_split_after_full_groups() -> None:
    text = "(単位:百万円)期末残高9991,000期末時価1,2003,400"  # 999 | 1,000 and 1,200 | 3,400
    note = parse_rental_note(text, CONFIG)
    assert (note.book_value, note.fair_value) == (1_000e6, 3_400e6)


def test_label_must_be_followed_by_its_number() -> None:
    # 「期末時価」 first appears in prose, then in the table.
    text = (
        "当該賃貸等不動産の連結貸借対照表計上額、期中増減額及び期末時価は、以下のとおりであります。"
        "(単位:百万円)期首残高5,5815,968期中増減額387725期末残高5,9686,694期末時価9,99310,889(注)"
    )
    note = parse_rental_note(text, CONFIG)
    assert (note.book_value, note.fair_value) == (6_694e6, 10_889e6)


def test_horizontal_table_is_unreadable_not_misread() -> None:
    # Headers in a row, then one row of numbers — no label next to its value.
    text = (
        "連結貸借対照表計上額(百万円)連結決算日における時価(百万円)当連結会計年度期首残高"
        "当連結会計年度増減額当連結会計年度末残高10,718△62810,08911,572(注)2当連結会計年度末の時価は…"
    )
    assert parse_rental_note(text, CONFIG).status == "unreadable"


def test_no_rental_property_at_all() -> None:
    assert parse_rental_note("(賃貸等不動産関係)該当事項はありません。", CONFIG).status == "none"
