from __future__ import annotations

from datetime import date

import pytest

from radar.text_ja import (
    age_on,
    is_organization,
    join_spaced_characters,
    parse_japanese_date,
    split_surname,
    surname_dictionary,
    unify_kanji_variants,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1950-01-02", date(1950, 1, 2)),
        ("1950年1月2日", date(1950, 1, 2)),
        ("１９５０年１月２日", date(1950, 1, 2)),  # full-width digits
        ("昭和25年1月2日", date(1950, 1, 2)),
        ("平成元年1月8日", date(1989, 1, 8)),
        ("令和元年5月1日", date(2019, 5, 1)),
        ("昭和64年1月7日", date(1989, 1, 7)),
    ],
)
def test_parse_japanese_date(raw: str, expected: date) -> None:
    assert parse_japanese_date(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "不明", "1950年13月40日"])
def test_parse_japanese_date_returns_none_on_bad_input(raw: str | None) -> None:
    assert parse_japanese_date(raw) is None


def test_age_on_counts_completed_years() -> None:
    birth = date(1950, 6, 15)
    assert age_on(birth, date(2025, 6, 14)) == 74
    assert age_on(birth, date(2025, 6, 15)) == 75
    assert age_on(None, date(2025, 1, 1)) is None


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("山田 太郎", "山田"),
        ("山田　太郎", "山田"),  # full-width space
        ("山田 太郎(注)1", "山田"),
        ("山田太郎", None),  # ambiguous without a separator
        ("伊　藤　浩　一", None),  # ambiguous: must not return "伊"
        ("林 誠", "林"),  # two single characters are still surname + given name
        ("伊　藤　浩　一(注)1", None),  # annotation must not stop the characters being joined
    ],
)
def test_split_surname(name: str, expected: str | None) -> None:
    assert split_surname(name) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("伊 藤 浩 一", "伊藤浩一"),
        ("山田 太郎", "山田 太郎"),
    ],
)
def test_join_spaced_characters(name: str, expected: str) -> None:
    assert join_spaced_characters(name) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("日本マスタートラスト信託銀行株式会社(信託口)", True),
        ("株式会社ヤマダ興産", True),
        ("（株）ヤマダ", True),
        ("THE MASTER TRUST BANK OF JAPAN", True),
        ("NORTHERN TRUST CO. (AVFC) RE FIDELITY FUNDS", True),
        ("山田 太郎", False),
        ("NAGANO TARO", False),  # "AG" inside a word must not count
        ("サンプル従業員持株会", True),
    ],
)
def test_is_organization(name: str, expected: bool) -> None:
    assert is_organization(name) is expected


@pytest.mark.parametrize(
    ("name", "known", "expected"),
    [
        ("伊藤浩一", {"伊藤"}, "伊藤"),
        ("伊　藤　浩　一", {"伊藤"}, "伊藤"),  # spaced per character, then joined
        ("長谷川一郎", {"長谷", "長谷川"}, "長谷川"),  # longest known prefix wins
        ("伊藤浩一", {"伊"}, None),  # one-character surnames are too ambiguous to guess
        ("伊藤浩一郎太", {"伊藤"}, None),  # given names longer than 3 characters: no guess
        ("伊藤", {"伊藤"}, None),  # nothing left for a given name
        ("山田 太郎", {"伊藤"}, "山田"),  # a separator always wins over the dictionary
    ],
)
def test_split_surname_with_dictionary(name: str, known: set[str], expected: str | None) -> None:
    assert split_surname(name, frozenset(known)) == expected


def test_surname_dictionary_from_spaced_names_only() -> None:
    names = ["山田 太郎", "伊　藤　浩　一", "林 誠", "佐藤花子", "サンプル 太郎"]
    # 林: single character (ambiguous as a prefix); 伊藤浩一 / 佐藤花子: no separator.
    assert surname_dictionary(names) == frozenset({"山田", "サンプル"})


def test_unify_kanji_variants_old_and_new_forms() -> None:
    assert unify_kanji_variants("廣澤") == unify_kanji_variants("広沢") == "広沢"
    assert unify_kanji_variants("髙橋") == "高橋"
    assert unify_kanji_variants("渡邉") == unify_kanji_variants("渡邊") == "渡辺"
    assert unify_kanji_variants("斉藤") != unify_kanji_variants("斎藤")  # distinct surnames
