"""`build_feature_table`: three synthetic companies, each feature, and look-ahead traps."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

from radar.config import load_yaml
from radar.features.build import (
    FEATURE_COLUMNS,
    TradableConfig,
    audit_features,
    build_feature_table,
)
from radar.features.ownership import HolderPatterns

AS_OF = date(2025, 7, 1)


def _documents() -> pd.DataFrame:
    rows = [
        # E99001 サンプル工業: usable FY2024 report.
        ("D1", "E99001", "99010", "サンプル工業株式会社", "120", datetime(2025, 6, 20, 15), "0"),
        # E99002 サンプル商事: an old report, and a NEWER one submitted ON as_of (trap: unusable).
        ("D2", "E99002", "99020", "サンプル商事株式会社", "120", datetime(2024, 6, 20, 15), "0"),
        ("D3", "E99002", "99020", "サンプル商事株式会社", "120", datetime(2025, 7, 1, 9), "0"),
        # E99003 サンプル電機: IFRS filer, representative without a recoverable surname.
        ("D4", "E99003", "99030", "サンプル電機株式会社", "120", datetime(2025, 6, 25, 15), "0"),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "doc_id",
            "edinet_code",
            "sec_code",
            "filer_name",
            "doc_type_code",
            "submit_datetime",
            "withdrawal_status",
        ],
    )
    frame["period_end"] = [
        date(2025, 3, 31),
        date(2024, 3, 31),
        date(2025, 3, 31),
        date(2025, 3, 31),
    ]
    return frame


def _officers() -> pd.DataFrame:
    rows = [
        ("D1", "サンプル 太郎", True, "chair", date(1950, 7, 2)),  # 74 on as_of (birthday after)
        ("D1", "サンプル 次郎", True, "president", date(1980, 1, 1)),
        ("D1", "鈴木 一郎", False, "other", date(1940, 1, 1)),  # oldest, but not a representative
        ("D2", "見本 花子", True, "president", date(1960, 1, 1)),
        ("D3", "見本 三郎", True, "president", date(1990, 1, 1)),  # in the trap filing
        ("D4", "伊藤浩一", True, "president", None),  # birth date missing
    ]
    return pd.DataFrame(
        rows, columns=["doc_id", "name", "is_representative", "role", "date_of_birth"]
    )


def _shareholders() -> pd.DataFrame:
    rows = [
        ("D1", "株式会社サンプル興産", 0.30, True),
        ("D1", "サンプル 太郎", 0.10, False),
        ("D1", "日本マスタートラスト信託銀行株式会社(信託口)", 0.08, True),
        ("D2", "トヨタ自動車株式会社", 0.25, True),
        ("D4", "伊藤興産株式会社", 0.40, True),
    ]
    return pd.DataFrame(rows, columns=["doc_id", "name", "ratio", "is_organization"])


def _financials() -> pd.DataFrame:
    base = {
        "cash_and_deposits": None,
        "short_term_securities": None,
        "investment_securities": None,
        "short_term_loans": None,
        "current_portion_long_term_loans": None,
        "long_term_loans": None,
        "bonds": None,
        "current_portion_bonds": None,
        "net_assets": None,
        "operating_income": None,
        "issued": None,
        "treasury": None,
        "dividend_per_share": None,
        "is_consolidated": True,
        "is_jgaap": True,
    }
    rows = [
        {**base, "doc_id": "D1", "cash_and_deposits": 6e9, "long_term_loans": 1e9,
         "net_assets": 20e9, "issued": 10e6, "treasury": 0.0, "operating_income": 0.8e9,
         "dividend_per_share": 20.0},
        {**base, "doc_id": "D2", "cash_and_deposits": 1e9, "net_assets": 5e9, "issued": 1e6},
        {**base, "doc_id": "D3", "cash_and_deposits": 99e9, "net_assets": 99e9, "issued": 1e6},
        {**base, "doc_id": "D4", "issued": 1e6, "is_jgaap": False},
    ]  # fmt: skip
    return pd.DataFrame(rows)


def _prices() -> pd.DataFrame:
    rows = [
        ("99010", date(2025, 6, 30), 1000.0),
        ("99010", date(2025, 7, 1), 9999.0),  # trap: close of as_of is not known yet
        ("99020", date(2025, 6, 30), 4000.0),
        ("99030", date(2025, 6, 30), 500.0),
    ]
    return pd.DataFrame(rows, columns=["code", "date", "close"])


def _listings() -> pd.DataFrame:
    rows = [
        ("99010", date(2025, 6, 30), "Standard"),
        ("99010", date(2025, 7, 1), "Prime"),  # trap: segment change effective on as_of
        ("99020", date(2025, 6, 30), "Prime"),
    ]
    return pd.DataFrame(rows, columns=["code", "date", "market_segment"])


def _capital_plans() -> pd.DataFrame:
    return pd.DataFrame(
        [("99020", date(2025, 6, 15)), ("99010", date(2025, 7, 1))],  # 2nd: published ON as_of
        columns=["sec_code", "published_on"],
    )


@pytest.fixture
def table() -> pd.DataFrame:
    result = build_feature_table(
        AS_OF,
        documents=_documents(),
        officers=_officers(),
        shareholders=_shareholders(),
        financials=_financials(),
        prices=_prices(),
        listings=_listings(),
        capital_plans=_capital_plans(),
    )
    return result.set_index("edinet_code")


def test_columns_and_one_row_per_company(table: pd.DataFrame) -> None:
    assert set(table.reset_index().columns) == set(FEATURE_COLUMNS)
    assert sorted(table.index) == ["E99001", "E99002", "E99003"]
    assert (table["as_of"] == AS_OF).all()


def test_filing_submitted_on_as_of_is_ignored(table: pd.DataFrame) -> None:
    assert table.loc["E99002", "source_doc_id"] == "D2"
    assert table.loc["E99002", "ceo_age"] == 65  # from D2 (born 1960), not D3


def test_ages_use_representatives_only(table: pd.DataFrame) -> None:
    row = table.loc["E99001"]
    assert row["oldest_representative_age"] == 74  # not the 85-year-old non-representative
    assert row["chair_age"] == 74
    assert row["ceo_age"] == 45
    assert pd.isna(table.loc["E99003", "oldest_representative_age"])  # missing ≠ 0


def test_family_features(table: pd.DataFrame) -> None:
    assert table.loc["E99001", "family_stake"] == pytest.approx(0.40)
    assert table.loc["E99001", "founder_in_control"]
    assert table.loc["E99002", "family_stake"] == 0.0  # surname known, no family holder
    # 伊藤浩一 has no recoverable surname → unknown, never "no family".
    assert pd.isna(table.loc["E99003", "family_stake"])
    assert pd.isna(table.loc["E99003", "founder_in_control"])


def test_valuation_uses_lagged_price(table: pd.DataFrame) -> None:
    row = table.loc["E99001"]
    assert row["market_cap_jpy"] == pytest.approx(10e9)  # 1000 × 10M, not the as_of close
    assert row["pbr"] == pytest.approx(0.5)
    assert row["net_cash_jpy"] == pytest.approx(5e9)
    assert row["net_cash_to_market_cap"] == pytest.approx(0.5)


def test_operating_yield_is_operating_income_over_market_cap(table: pd.DataFrame) -> None:
    assert table.loc["E99001", "operating_income_to_market_cap"] == pytest.approx(0.08)
    assert pd.isna(table.loc["E99002", "operating_income_to_market_cap"])  # unreported ≠ 0


def test_dividend_yield_is_last_year_dividends_over_market_cap(table: pd.DataFrame) -> None:
    # ¥20 × 10M shares = ¥200M paid for the year / ¥10bn market cap = 2%.
    assert table.loc["E99001", "dividend_yield"] == pytest.approx(0.02)
    assert pd.isna(table.loc["E99002", "dividend_yield"])  # unreported ≠ 0


def test_ifrs_filer_has_no_financial_features(table: pd.DataFrame) -> None:
    row = table.loc["E99003"]
    assert not row["is_jgaap"]
    assert pd.isna(row["pbr"]) and pd.isna(row["net_cash_jpy"])
    assert row["market_cap_jpy"] == pytest.approx(500e6)  # price × shares still works


def test_segment_and_capital_plan_are_point_in_time(table: pd.DataFrame) -> None:
    assert table.loc["E99001", "market_segment"] == "Standard"  # change on as_of ignored
    assert not table.loc["E99001", "capital_plan_disclosed"]  # list published on as_of ignored
    assert table.loc["E99002", "capital_plan_disclosed"]
    assert pd.isna(table.loc["E99003", "market_segment"])


def test_without_optional_inputs_features_are_missing_not_false() -> None:
    result = build_feature_table(
        AS_OF,
        documents=_documents(),
        officers=_officers(),
        shareholders=_shareholders(),
        financials=_financials(),
        prices=_prices(),
    )
    assert result["capital_plan_disclosed"].isna().all()
    assert result["market_segment"].isna().all()


def test_no_filings_before_as_of_gives_empty_table() -> None:
    result = build_feature_table(
        date(2020, 1, 1),
        documents=_documents(),
        officers=_officers(),
        shareholders=_shareholders(),
        financials=_financials(),
        prices=_prices(),
    )
    assert result.empty
    assert list(result.columns) == FEATURE_COLUMNS


def test_sole_representative_without_president_title_is_ceo() -> None:
    officers = _officers()
    officers.loc[officers["doc_id"] == "D2", "role"] = "other"
    result = build_feature_table(
        AS_OF,
        documents=_documents(),
        officers=officers,
        shareholders=_shareholders(),
        financials=_financials(),
        prices=_prices(),
    ).set_index("edinet_code")
    assert result.loc["E99002", "ceo_age"] == 65
    assert result.loc["E99001", "ceo_age"] == 45  # unchanged when a 社長 exists


def test_universe_is_listed_companies_with_a_parsed_report() -> None:
    documents = _documents()
    fund = documents.iloc[[0]].assign(doc_id="D8", edinet_code="E99008", sec_code=None)
    unparsed = documents.iloc[[0]].assign(doc_id="D9", edinet_code="E99009", sec_code="99090")
    result = build_feature_table(
        AS_OF,
        documents=pd.concat([documents, fund, unparsed], ignore_index=True),
        officers=_officers(),
        shareholders=_shareholders(),
        financials=_financials(),
        prices=_prices(),
    )
    assert sorted(result["edinet_code"]) == ["E99001", "E99002", "E99003"]


def test_ownership_features_with_patterns() -> None:
    documents = pd.concat(
        [
            _documents(),
            pd.DataFrame(  # トヨタ自動車 filed before as_of with a securities code → listed
                [
                    (
                        "D7",
                        "E99007",
                        "72030",
                        "トヨタ自動車株式会社",
                        "350",
                        datetime(2025, 6, 1),
                        "0",
                    )
                ],
                columns=_documents().columns[:-1],
            ),
        ],
        ignore_index=True,
    )
    result = build_feature_table(
        AS_OF,
        documents=documents,
        officers=_officers(),
        shareholders=_shareholders(),
        financials=_financials(),
        prices=_prices(),
        holder_patterns=HolderPatterns.from_config(load_yaml("ownership")),
    ).set_index("edinet_code")
    assert result.loc["E99002", "largest_listed_holder_stake"] == pytest.approx(0.25)
    assert result.loc["E99002", "largest_listed_holder_name"] == "トヨタ自動車株式会社"
    assert result.loc["E99001", "largest_listed_holder_stake"] == 0.0
    assert result.loc["E99001", "top10_concentration_ex_custodian"] == pytest.approx(0.40)


def test_ownership_features_missing_without_patterns(table: pd.DataFrame) -> None:
    assert table["largest_listed_holder_stake"].isna().all()


def test_joined_names_use_surnames_seen_elsewhere() -> None:
    officers = pd.concat(
        [
            _officers(),
            pd.DataFrame([("D1", "伊藤 二郎", False, "other", None)], columns=_officers().columns),
        ],
        ignore_index=True,
    )
    result = build_feature_table(
        AS_OF,
        documents=_documents(),
        officers=officers,
        shareholders=_shareholders(),
        financials=_financials(),
        prices=_prices(),
    ).set_index("edinet_code")
    assert result.loc["E99003", "family_stake"] == pytest.approx(0.40)
    assert result.loc["E99003", "founder_in_control"]


def test_market_cap_carries_share_count_through_a_split() -> None:
    prices = pd.concat(
        [
            _prices().assign(adj_factor=1.0),
            pd.DataFrame(
                {
                    "code": ["99010"],
                    "date": [date(2025, 6, 2)],
                    "close": [500.0],
                    "adj_factor": [0.5],
                }
            ),
        ],
        ignore_index=True,
    )
    prices.loc[(prices["code"] == "99010") & (prices["date"] == date(2025, 6, 30)), "close"] = 500.0
    result = build_feature_table(
        AS_OF,
        documents=_documents(),
        officers=_officers(),
        shareholders=_shareholders(),
        financials=_financials(),
        prices=prices,
    ).set_index("edinet_code")
    assert result.loc["E99001", "market_cap_jpy"] == pytest.approx(10e9)  # 500 × 20M
    assert result.loc["E99001", "pbr"] == pytest.approx(0.5)


def test_split_after_the_price_date_is_ignored() -> None:
    # A split dated ON as_of is not known when scoring (look-ahead trap).
    prices = pd.concat(
        [
            _prices().assign(adj_factor=1.0),
            pd.DataFrame(
                {"code": ["99010"], "date": [AS_OF], "close": [500.0], "adj_factor": [0.5]}
            ),
        ],
        ignore_index=True,
    )
    result = build_feature_table(
        AS_OF,
        documents=_documents(),
        officers=_officers(),
        shareholders=_shareholders(),
        financials=_financials(),
        prices=prices,
    ).set_index("edinet_code")
    assert result.loc["E99001", "market_cap_jpy"] == pytest.approx(10e9)  # 1000 × 10M


def test_capital_plan_missing_before_the_first_list_is_published() -> None:
    # JPX's first list came out in January 2024: before any list is public the feature is
    # unknown, not "did not disclose".
    result = build_feature_table(
        AS_OF,
        documents=_documents(),
        officers=_officers(),
        shareholders=_shareholders(),
        financials=_financials(),
        prices=_prices(),
        capital_plans=_capital_plans().assign(published_on=date(2025, 8, 1)),
    )
    assert result["capital_plan_disclosed"].isna().all()


def test_activism_features() -> None:
    documents = _documents().assign(issuer_edinet_code=None)
    report = pd.DataFrame(
        [
            ("A1", "E99100", None, "株式会社ストラテジックキャピタル", "350",
             datetime(2025, 3, 1, 15), "0", None, "E99002"),
            ("A0", "E99100", None, "サンプル", "120", datetime(2024, 1, 4, 9), "0", None, None),
        ],
        columns=[*_documents().columns, "issuer_edinet_code"],
    )  # fmt: skip
    result = build_feature_table(
        AS_OF,
        documents=pd.concat([documents, report], ignore_index=True),
        officers=_officers(),
        shareholders=_shareholders(),
        financials=_financials(),
        prices=_prices(),
        activist_patterns=tuple(load_yaml("activists")["activist_filers"]),
        index_start=date(2024, 1, 4),
    ).set_index("edinet_code")
    assert result.loc["E99002", "activist_filers_365d"] == 1
    assert result.loc["E99002", "activist_names"] == "ストラテジックキャピタル"
    assert result.loc["E99001", "activist_filers_365d"] == 0  # index covers the year: a real 0


def test_activism_missing_when_index_too_short(table: pd.DataFrame) -> None:
    assert table["activist_filers_365d"].isna().all()  # no patterns given


def test_capital_plan_unknown_outside_the_requested_segments() -> None:
    listings = pd.concat(
        [
            _listings(),
            pd.DataFrame(
                {"code": ["99030"], "date": [date(2025, 6, 30)], "market_segment": ["Growth"]}
            ),
        ],
        ignore_index=True,
    )
    result = build_feature_table(
        AS_OF,
        documents=_documents(),
        officers=_officers(),
        shareholders=_shareholders(),
        financials=_financials(),
        prices=_prices(),
        listings=listings,
        capital_plans=_capital_plans(),
        capital_plan_segments=("Prime", "Standard"),
    ).set_index("edinet_code")
    assert pd.isna(result.loc["E99003", "capital_plan_disclosed"])  # Growth: never asked
    assert result.loc["E99002", "capital_plan_disclosed"]  # Prime, on the list
    assert not result.loc["E99001", "capital_plan_disclosed"]  # Standard, not on the list


def test_parent_track_record() -> None:
    # トヨタ自動車 (listed, 25% of E99002) filed one tender offer before as_of → a parent that buys.
    toyota = ("E99007", "72030", "トヨタ自動車株式会社")
    extra = pd.DataFrame(
        [
            ("D7", *toyota, "120", datetime(2025, 6, 1), "0", None),
            ("D8", *toyota, "240", datetime(2025, 3, 1), "0", "E99050"),
        ],
        columns=[*_documents().columns[:-1], "subject_edinet_code"],
    )  # fmt: skip
    documents = pd.concat([_documents().assign(subject_edinet_code=None), extra], ignore_index=True)
    result = build_feature_table(
        AS_OF,
        documents=documents,
        officers=_officers(),
        shareholders=_shareholders(),
        financials=_financials(),
        prices=_prices(),
        holder_patterns=HolderPatterns.from_config(load_yaml("ownership")),
        portfolio_min_stake=0.20,
    ).set_index("edinet_code")
    assert result.loc["E99002", "parent_prior_offers"] == 1
    assert result.loc["E99002", "parent_other_holdings"] == 0  # Toyota holds ≥20% only here
    assert pd.isna(result.loc["E99001", "parent_prior_offers"])  # no listed parent: unknown


def test_audit_features_catches_impossible_rows() -> None:
    features = pd.DataFrame(
        {
            "edinet_code": ["OK", "LOOKAHEAD", "GROWTH", "RATIO", "AGE"],
            "as_of": [AS_OF] * 5,
            "filing_submitted_at": [datetime(2025, 6, 1)] * 2 + [datetime(2025, 6, 1)] * 3,
            "market_segment": ["Standard", "Standard", "Growth", "Prime", "Prime"],
            "capital_plan_disclosed": [False, True, False, True, True],
            "family_stake": [0.3, 0.3, 0.3, 1.4, 0.3],
            "oldest_representative_age": [70, 70, 70, 70, 140],
        }
    )
    features.loc[1, "filing_submitted_at"] = datetime(2025, 7, 1, 9)  # ON as_of: look-ahead
    issues = audit_features(features, ("Prime", "Standard"))
    found = set(zip(issues["edinet_code"], issues["check"], strict=True))
    assert found == {
        ("LOOKAHEAD", "filing_not_before_as_of"),
        ("GROWTH", "capital_plan_outside_request"),
        ("RATIO", "stake_out_of_range"),
        ("AGE", "implausible_age"),
    }


def test_officer_stake_uses_shares_at_the_filing_date() -> None:
    financials = _financials()
    financials["issued_at_filing"] = None
    financials.loc[financials["doc_id"] == "D1", "issued_at_filing"] = 30e6  # 1:3 split
    officers = _officers().assign(shares_held=[9e6, 6e6, 0.0, None, None, None])
    result = build_feature_table(
        AS_OF,
        documents=_documents(),
        officers=officers,
        shareholders=_shareholders(),
        financials=financials,
        prices=_prices(),
    ).set_index("edinet_code")
    assert result.loc["E99001", "officer_stake"] == pytest.approx(0.5)  # 15M / 30M, not 1.5


def test_audit_features_ignores_missing_values() -> None:
    features = pd.DataFrame(
        {
            "edinet_code": ["NAN"],
            "as_of": [AS_OF],
            "filing_submitted_at": [datetime(2025, 6, 1)],
            "market_segment": ["Prime"],
            "capital_plan_disclosed": [None],
            "family_stake": [float("nan")],
            "officer_stake": [float("nan")],
            "oldest_representative_age": [float("nan")],
        }
    )
    assert audit_features(features, ("Prime", "Standard")).empty


def test_turnover_and_tradable_share_features() -> None:
    prices = _prices().assign(trading_value=5e6)
    result = build_feature_table(
        AS_OF,
        documents=_documents(),
        officers=_officers().assign(shares_held=[1e6, 0.0, 0.0, None, None, None]),
        shareholders=_shareholders(),
        financials=_financials(),
        prices=prices,
        listings=_listings(),
        tradable=TradableConfig(
            criteria={"Standard": {"ratio": 0.25, "value": 1e9}},
            major_holder=0.10,
            patterns=HolderPatterns.from_config(load_yaml("ownership")),
        ),
    ).set_index("edinet_code")
    row = result.loc["E99001"]  # ¥10bn market cap, 10M shares
    assert row["turnover"] == pytest.approx(5e6 / 10e9)
    # Not tradable: 株式会社サンプル興産 (30%) and サンプル 太郎 (10%, at the ≥10% line). Tarō is
    # also the officer holding 1M shares — counted once. The custodian (8%) stays tradable.
    assert row["tradable_share_ratio"] == pytest.approx(1 - 0.30 - 0.10)
    assert row["listing_criteria_margin"] == pytest.approx(min(0.6 / 0.25, 0.6 * 10e9 / 1e9) - 1)


def test_market_cap_uses_filing_date_shares_around_a_year_end_split() -> None:
    financials = _financials()
    financials["issued_at_filing"] = None
    financials.loc[financials["doc_id"] == "D1", "issued_at_filing"] = 50e6  # 10M × 5
    prices = _prices().assign(adj_factor=1.0)
    prices.loc[prices["code"] == "99010", "close"] = 200.0  # post-split price
    split = pd.DataFrame(
        {"code": ["99010"], "date": [date(2024, 12, 27)], "close": [200.0], "adj_factor": [0.2]}
    )
    result = build_feature_table(
        AS_OF,
        documents=_documents(),
        officers=_officers(),
        shareholders=_shareholders(),
        financials=financials,
        prices=pd.concat([prices, split], ignore_index=True),
    ).set_index("edinet_code")
    assert result.loc["E99001", "market_cap_jpy"] == pytest.approx(200.0 * 50e6)  # ¥10bn, not ¥2bn


def test_newest_report_not_parsed_falls_back_to_the_previous_one() -> None:
    documents = pd.concat(
        [
            _documents(),
            pd.DataFrame(
                [
                    (
                        "D9",
                        "E99001",
                        "99010",
                        "サンプル工業株式会社",
                        "120",
                        datetime(2025, 6, 28, 15),
                        "0",
                    )
                ],
                columns=_documents().columns[:-1],
            ).assign(period_end=date(2025, 3, 31)),
        ],
        ignore_index=True,
    )  # D9: newer than D1, not parsed (absent from financials)
    result = build_feature_table(
        AS_OF,
        documents=documents,
        officers=_officers(),
        shareholders=_shareholders(),
        financials=_financials(),
        prices=_prices(),
    ).set_index("edinet_code")
    assert result.loc["E99001", "source_doc_id"] == "D1"


def test_trend_features_use_only_reports_and_prices_before_as_of() -> None:
    # E99001: FY2023 report (D0) + FY2024 report (D1); both year ends priced below book.
    documents = pd.concat(
        [
            _documents(),
            pd.DataFrame(
                [{"doc_id": "D0", "edinet_code": "E99001", "sec_code": "99010",
                  "filer_name": "サンプル工業株式会社", "doc_type_code": "120",
                  "submit_datetime": datetime(2024, 6, 20, 15), "withdrawal_status": "0",
                  "period_end": date(2024, 3, 31)}]
            ),
        ],
        ignore_index=True,
    )  # fmt: skip
    financials = pd.concat(
        [
            _financials(),
            pd.DataFrame(
                [{"doc_id": "D0", "cash_and_deposits": 3e9, "long_term_loans": 1e9,
                  "net_assets": 18e9, "issued": 10e6, "treasury": 0.0, "is_jgaap": True}]
            ),
        ],
        ignore_index=True,
    )  # fmt: skip
    prices = pd.concat(
        [
            _prices(),
            pd.DataFrame(
                {
                    "code": ["99010", "99010", "99020"],
                    "date": [date(2024, 3, 29), date(2025, 3, 31), date(2024, 3, 29)],
                    "close": [900.0, 800.0, 100.0],
                }
            ),
        ],
        ignore_index=True,
    )
    result = build_feature_table(
        AS_OF,
        documents=documents,
        officers=_officers(),
        shareholders=_shareholders(),
        financials=financials,
        prices=prices,
    ).set_index("edinet_code")
    # PBR 0.5 (9bn / 18bn) at FY2023, 0.4 (8bn / 20bn) at FY2024 → two years below book.
    assert result.loc["E99001", "years_below_book"] == 2
    # Net cash 5bn → 2bn one report earlier: +3bn / market cap 10bn.
    assert result.loc["E99001", "net_cash_change_to_market_cap"] == pytest.approx(0.3)
    # E99002: only D2 is usable (D3 was submitted ON as_of) → no earlier report to compare.
    assert pd.isna(result.loc["E99002", "net_cash_change_to_market_cap"])
    assert result.loc["E99002", "years_below_book"] == 1  # FY2023: 100 × 1M / 5bn = 0.02


def test_sector_is_point_in_time_and_percentiles_need_peers() -> None:
    listings = _listings().assign(sector33="電気機器")
    listings.loc[listings["date"] == date(2025, 7, 1), "sector33"] = "銀行業"  # trap: on as_of
    result = build_feature_table(
        AS_OF,
        documents=_documents(),
        officers=_officers(),
        shareholders=_shareholders(),
        financials=_financials(),
        prices=_prices(),
        listings=listings,
        sector_min_peers=2,
    ).set_index("edinet_code")
    assert result.loc["E99001", "sector33"] == "電気機器"
    # E99001 PBR 0.5 vs E99002 PBR 0.8 in the same sector → 0.5 and 1.0.
    assert result.loc["E99001", "pbr_sector_percentile"] == 0.5
    assert result.loc["E99002", "pbr_sector_percentile"] == 1.0
    assert pd.isna(result.loc["E99003", "sector33"])  # not in the listings


def test_adjusted_pbr_is_added_and_pbr_unchanged() -> None:
    financials = _financials()
    financials["rental_note_status"] = ["disclosed", "absent", "absent", None]
    financials["rental_book_value"] = [1e9, None, None, None]
    financials["rental_fair_value"] = [11e9, None, None, None]
    result = build_feature_table(
        AS_OF,
        documents=_documents(),
        officers=_officers(),
        shareholders=_shareholders(),
        financials=financials,
        prices=_prices(),
        rental_tax_rate=0.3,
    ).set_index("edinet_code")
    # E99001: mcap 10bn, net assets 20bn → PBR 0.5; +10bn gain × 0.7 → adjusted book 27bn.
    assert result.loc["E99001", "pbr"] == pytest.approx(0.5)  # untouched
    assert result.loc["E99001", "adjusted_pbr"] == pytest.approx(10 / 27)
    assert result.loc["E99001", "rental_unrealized_gain_to_market_cap"] == pytest.approx(0.7)
    assert result.loc["E99002", "adjusted_pbr"] == pytest.approx(result.loc["E99002", "pbr"])
    assert pd.isna(result.loc["E99003", "adjusted_pbr"])  # IFRS: no net assets, no guess
