"""Ownership features: holder types, listed parent (親子上場), concentration."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

from radar.config import load_yaml
from radar.features.ownership import (
    HolderPatterns,
    classify_holder,
    company_key,
    holder_portfolio_counts,
    listed_company_codes,
    listed_company_keys,
    listing_criteria_margin,
    ownership_changes,
    ownership_features,
    parent_features,
    parent_prior_offers,
    tradable_share_ratio,
)

AS_OF = date(2025, 7, 1)


@pytest.fixture
def patterns() -> HolderPatterns:
    """The real patterns from config/ — tests fail loudly if the config shape changes."""
    return HolderPatterns.from_config(load_yaml("ownership"))


@pytest.mark.parametrize(
    ("name", "is_org", "expected"),
    [
        ("自己株式", True, "treasury"),
        ("サンプル従業員持株会", True, "employee_plan"),
        ("日本マスタートラスト信託銀行株式会社(信託口)", True, "custodian"),  # not "financial"
        ("STATE STREET BANK 505001(常任代理人 株式会社みずほ銀行)", True, "custodian"),
        ("公益財団法人 サンプル財団", True, "foundation"),
        ("株式会社サンプル銀行", True, "financial"),
        ("サンプル生命保険相互会社", True, "financial"),
        ("サンプル工業株式会社", True, "company"),
        ("見本 太郎", False, "person"),
    ],
)  # fmt: skip
def test_classify_holder(patterns: HolderPatterns, name: str, is_org: bool, expected: str) -> None:
    assert classify_holder(name, is_org, patterns) == expected


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("トヨタ自動車株式会社", "トヨタ自動車株式会社"),
        ("明治ホールディングス(株)", "明治ホールディングス株式会社"),
        ("株式会社 サンプル", "株式会社サンプル"),
        ("ＳＥＭＩＴＥＣ株式会社", "SEMITEC株式会社"),
    ],
)
def test_company_key_ignores_legal_form_and_width(a: str, b: str) -> None:
    assert company_key(a) == company_key(b)


def _documents() -> pd.DataFrame:
    rows = [
        ("トヨタ自動車株式会社", "72030", datetime(2025, 6, 18, 15)),
        ("サンプル親会社株式会社", "99990", datetime(2025, 7, 1, 9)),  # trap: first filing ON as_of
        ("サンプル投信株式会社", None, datetime(2025, 6, 1, 9)),  # filer, but not listed
    ]
    return pd.DataFrame(rows, columns=["filer_name", "sec_code", "submit_datetime"])


def test_listed_company_keys_are_point_in_time() -> None:
    keys = listed_company_keys(_documents(), AS_OF)
    assert company_key("トヨタ自動車株式会社") in keys
    assert company_key("サンプル親会社株式会社") not in keys  # not known before as_of
    assert company_key("サンプル投信株式会社") not in keys  # no sec_code
    assert company_key("サンプル親会社株式会社") in listed_company_keys(
        _documents(), date(2025, 7, 2)
    )


def _holders() -> pd.DataFrame:
    rows = [
        ("トヨタ自動車株式会社", 0.246, True),
        ("サンプル親会社株式会社", 0.30, True),  # listed only after as_of
        ("日本マスタートラスト信託銀行株式会社(信託口)", 0.12, True),
        ("株式会社サンプル銀行", 0.05, True),
        ("見本 太郎", 0.02, False),
    ]
    return pd.DataFrame(rows, columns=["name", "ratio", "is_organization"])


def test_ownership_features(patterns: HolderPatterns) -> None:
    keys = listed_company_keys(_documents(), AS_OF)
    result = ownership_features(_holders(), keys, patterns)
    assert result["largest_listed_holder_stake"] == pytest.approx(0.246)
    assert result["largest_listed_holder_name"] == "トヨタ自動車株式会社"
    # Everything except the custodian: 0.246 + 0.30 + 0.05 + 0.02.
    assert result["top10_concentration_ex_custodian"] == pytest.approx(0.616)


def test_ownership_without_listed_holder_is_zero(patterns: HolderPatterns) -> None:
    result = ownership_features(_holders().iloc[2:], frozenset(), patterns)
    assert result["largest_listed_holder_stake"] == 0.0  # holders known, none listed: a real 0
    assert result["largest_listed_holder_name"] is None


def test_ownership_without_shareholder_table_is_missing(patterns: HolderPatterns) -> None:
    result = ownership_features(_holders().iloc[0:0], frozenset(), patterns)
    assert all(value is None for value in result.values())


def test_ownership_ignores_missing_ratios(patterns: HolderPatterns) -> None:
    holders = _holders().assign(ratio=[None, None, 0.12, 0.05, 0.02])
    result = ownership_features(holders, frozenset({company_key("トヨタ自動車株式会社")}), patterns)
    assert result["largest_listed_holder_stake"] is None  # listed holder, unknown ratio
    assert result["top10_concentration_ex_custodian"] == pytest.approx(0.07)


# ------------------------------------------------------------------ parent track record
def _index() -> pd.DataFrame:
    rows = [
        ("P1", "E90001", "72030", "トヨタ自動車株式会社", "120", None, datetime(2024, 6, 18, 15)),
        (
            "P2",
            "E90002",
            "99990",
            "フリージア・マクロス株式会社",
            "120",
            None,
            datetime(2024, 6, 25, 15),
        ),
        ("T1", "E90001", "72030", "トヨタ自動車株式会社", "240", "E99005", datetime(2025, 3, 1, 9)),
        (
            "T2",
            "E90001",
            "72030",
            "トヨタ自動車株式会社",
            "240",
            "E99006",
            datetime(2025, 7, 1, 9),
        ),  # ON as_of
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "doc_id", "edinet_code", "sec_code", "filer_name", "doc_type_code",
            "subject_edinet_code", "submit_datetime",
        ],
    ).assign(withdrawal_status="0")  # fmt: skip


def test_parent_prior_offers_point_in_time() -> None:
    codes = listed_company_codes(_index(), AS_OF)
    toyota = codes[company_key("トヨタ自動車株式会社")]
    assert toyota == "E90001"
    assert parent_prior_offers(_index(), toyota, AS_OF) == 1  # T2 is filed ON as_of: not yet
    freesia = codes[company_key("フリージア・マクロス株式会社")]
    assert parent_prior_offers(_index(), freesia, AS_OF) == 0  # holds, never bid


def test_holder_portfolio_counts_stakes_of_20pct_or_more() -> None:
    holders = pd.DataFrame(
        {
            "edinet_code": ["A", "B", "C", "D"],
            "name": ["フリージア・マクロス株式会社"] * 3 + ["トヨタ自動車株式会社"],
            "ratio": [0.475, 0.354, 0.189, 0.246],
        }
    )
    counts = holder_portfolio_counts(holders, min_stake=0.20)
    assert counts[company_key("フリージア・マクロス株式会社")] == 2  # 18.9% is below the line
    assert counts[company_key("トヨタ自動車株式会社")] == 1


# ------------------------------------------------------------------ tradable shares (流通株式)
def test_tradable_share_ratio_like_kyowa(patterns: HolderPatterns) -> None:
    # Shaped on 協和コンサルタンツ: a 47.5% holder, a custodian above 10% (NOT excluded: it holds
    # for many investors), officers 4.9%, no treasury shares.
    holders = pd.DataFrame(
        {
            "name": [
                "サンプル・マクロス株式会社",
                "日本マスタートラスト信託銀行株式会社(信託口)",
                "見本 太郎",
            ],
            "ratio": [0.475, 0.12, 0.03],
            "is_organization": [True, True, False],
        }
    )
    officers = pd.DataFrame({"name": ["見本 太郎", "見本 次郎"], "shares_held": [30.0, 19.0]})
    ratio = tradable_share_ratio(
        holders, officers, outstanding=1000.0, issued=1000.0, major_holder=0.10, patterns=patterns
    )
    assert ratio == pytest.approx(1 - 0.475 - 0.049)


def test_tradable_share_ratio_does_not_double_count_an_officer_who_is_a_major_holder(
    patterns: HolderPatterns,
) -> None:
    holders = pd.DataFrame({"name": ["見本 太郎"], "ratio": [0.30], "is_organization": [False]})
    officers = pd.DataFrame({"name": ["見本　太郎"], "shares_held": [300.0]})  # same person
    ratio = tradable_share_ratio(
        holders, officers, outstanding=1000.0, issued=1000.0, major_holder=0.10, patterns=patterns
    )
    assert ratio == pytest.approx(0.70)


def test_tradable_share_ratio_counts_treasury_shares_as_not_tradable(
    patterns: HolderPatterns,
) -> None:
    holders = pd.DataFrame(
        {"name": ["サンプル興産株式会社"], "ratio": [0.0], "is_organization": [True]}
    )
    officers = pd.DataFrame({"name": ["見本 太郎"], "shares_held": [0.0]})
    # 10% treasury: tradable = 90% of listed shares.
    kwargs = {"outstanding": 900.0, "issued": 1000.0, "major_holder": 0.10, "patterns": patterns}
    assert tradable_share_ratio(holders, officers, **kwargs) == pytest.approx(0.9)
    assert tradable_share_ratio(holders.iloc[0:0], officers, **kwargs) is None


def test_listing_criteria_margin() -> None:
    criteria = {"Standard": {"ratio": 0.25, "value": 1e9}}
    # 47.6% float on a ¥4.1bn company: ratio 1.90× and value 1.95× the minimum → margin +0.90.
    assert listing_criteria_margin(0.476, 4.1e9, "Standard", criteria) == pytest.approx(
        0.904, abs=1e-3
    )
    assert listing_criteria_margin(0.20, 4.1e9, "Standard", criteria) == pytest.approx(
        -0.2
    )  # fails
    assert listing_criteria_margin(0.5, 4.1e9, "Growth", criteria) is None  # no criteria given
    assert listing_criteria_margin(None, 4.1e9, "Standard", criteria) is None


def test_tradable_share_ratio_with_no_officers_listed(patterns: HolderPatterns) -> None:
    holders = pd.DataFrame(
        {"name": ["サンプル興産株式会社"], "ratio": [0.4], "is_organization": [True]}
    )
    officers = pd.DataFrame(
        {"name": pd.Series([], dtype=object), "shares_held": pd.Series([], dtype=float)}
    )
    ratio = tradable_share_ratio(
        holders, officers, outstanding=100.0, issued=100.0, major_holder=0.10, patterns=patterns
    )
    assert ratio == pytest.approx(0.6)


def test_curated_homonym_is_not_the_listed_company(patterns: HolderPatterns) -> None:
    holders = pd.DataFrame(
        {
            "rank": [1, 2],
            "name": ["株式会社平和堂", "伊藤忠商事株式会社"],
            "ratio": [0.27, 0.05],
            "is_organization": [True, True],
            "address": ["名古屋市千種区今池五丁目5-6", "東京都港区北青山2丁目5番1号"],
        }
    )
    keys = frozenset({company_key("株式会社平和堂"), company_key("伊藤忠商事株式会社")})
    homonyms = {company_key("株式会社平和堂"): ("名古屋市",)}
    result = ownership_features(holders, keys, patterns, homonyms=homonyms)
    assert result["largest_listed_holder_name"] == "伊藤忠商事株式会社"  # kept despite 大阪 HQ
    # The same name at another address is still the listed company.
    elsewhere = holders.assign(address=["滋賀県彦根市西今町1番地", None])
    assert ownership_features(elsewhere, keys, patterns, homonyms=homonyms)[
        "largest_listed_holder_stake"
    ] == pytest.approx(0.27)


def test_parent_features_come_from_the_parents_own_row_on_the_same_date() -> None:
    table = pd.DataFrame(
        {
            "edinet_code": ["E_CHILD", "E_PARENT", "E_ALONE"],
            "largest_listed_holder_name": ["サンプル親会社株式会社", None, "未上場商事株式会社"],
            "pbr": [0.6, 1.8, 0.9],
            "net_cash_to_market_cap": [0.4, 0.05, 0.1],
            "activist_filers_365d": [0, 2, 0],
        }
    )
    codes = {company_key("サンプル親会社株式会社"): "E_PARENT"}  # listed before as_of
    result = parent_features(table, codes).set_index(table["edinet_code"])
    assert result.loc["E_CHILD", "parent_pbr"] == 1.8
    assert result.loc["E_CHILD", "parent_net_cash_to_market_cap"] == 0.05
    assert result.loc["E_CHILD", "parent_activist_filers_365d"] == 2  # pressure on the parent
    assert result.loc["E_ALONE"].isna().all()  # holder not listed → unknown, not 0
    assert result.loc["E_PARENT"].isna().all()  # no listed holder at all


def test_ownership_changes_against_the_previous_report(patterns: HolderPatterns) -> None:
    keys = frozenset({company_key("サンプル親会社株式会社")})
    now = pd.DataFrame(
        {"rank": [1, 2], "name": ["サンプル親会社株式会社", "山田 太郎"], "ratio": [0.45, 0.10],
         "is_organization": [True, False]}
    )  # fmt: skip
    before = now.assign(ratio=[0.38, 0.12])
    changes = ownership_changes(now, before, keys, patterns)
    assert changes["listed_holder_stake_change"] == pytest.approx(0.07)  # the parent bought
    assert changes["top10_change"] == pytest.approx(0.05)  # 0.55 → 0.50
    assert ownership_changes(now, None, keys, patterns) == {
        "listed_holder_stake_change": None, "top10_change": None
    }  # fmt: skip
    newcomer = before.assign(name=["別の会社株式会社", "山田 太郎"])
    assert ownership_changes(now, newcomer, keys, patterns)["listed_holder_stake_change"] is None
