"""Officers, major shareholders and financials extraction from a synthetic facts table."""

from __future__ import annotations

from datetime import date

import pytest

from radar.parse.elements import ElementMap, to_number
from radar.parse.financials import extract_financials
from radar.parse.officers import classify_role, extract_officers, is_representative
from radar.parse.shareholders import extract_major_shareholders, normalize_ratio

from conftest import make_facts

NAME = "jpcrp_cor:NameInformationAboutDirectorsAndCorporateAuditors"
TITLE = "jpcrp_cor:OfficialTitleOrPositionInformationAboutDirectorsAndCorporateAuditors"
DOB = "jpcrp_cor:DateOfBirthInformationAboutDirectorsAndCorporateAuditors"
P1 = "FilingDateInstant_jpcrp030000-asr_E99001-000YamadaTaroMember"
P2 = "FilingDateInstant_jpcrp030000-asr_E99001-000SatoHanakoMember"
STANDARD = "jpdei_cor:AccountingStandardsDEI"
CONSOLIDATED = "jpdei_cor:WhetherConsolidatedFinancialStatementsArePreparedDEI"


def test_extract_officers(elements: ElementMap) -> None:
    facts = make_facts(
        [
            (NAME, P1, "山田　太郎"),
            (TITLE, P1, "代表取締役会長"),
            (DOB, P1, "1948-03-01"),
            (NAME, P2, "佐藤 花子"),
            (TITLE, P2, "取締役"),
            (DOB, P2, "昭和45年7月7日"),
        ]
    )
    officers = extract_officers(facts, elements).set_index("person_key")
    assert officers.loc[P1, "name"] == "山田 太郎"
    assert officers.loc[P1, "is_representative"]
    assert officers.loc[P1, "role"] == "chair"
    assert officers.loc[P1, "date_of_birth"] == date(1948, 3, 1)
    assert not officers.loc[P2, "is_representative"]
    assert officers.loc[P2, "date_of_birth"] == date(1970, 7, 7)


def test_extract_officers_merges_executive_officers(elements: ElementMap) -> None:
    exec_name = "jpcrp_cor:NameInformationAboutExecutiveDirectors"
    exec_title = "jpcrp_cor:OfficialTitleOrPositionInformationAboutExecutiveDirectors"
    facts = make_facts(
        [
            (NAME, P1, "山田　太郎"),
            (TITLE, P1, "取締役"),
            (DOB, P1, "1960-01-01"),
            (exec_name, P1, "山田　太郎"),
            (exec_title, P1, "代表執行役最高経営責任者(ＣＥＯ)"),
            (exec_name, P2, "佐藤　花子"),
            (exec_title, P2, "執行役"),
        ]
    )
    officers = extract_officers(facts, elements).set_index("person_key")
    assert len(officers) == 2
    assert officers.loc[P1, "is_representative"]
    assert officers.loc[P1, "role"] == "president"
    assert officers.loc[P1, "date_of_birth"] == date(1960, 1, 1)
    assert not officers.loc[P2, "is_representative"]


def test_extract_officers_skips_rows_without_a_name(elements: ElementMap) -> None:
    # The officers' total shareholding is a fact on the
    # bare FilingDateInstant context (no name, no title), and AGM nominees (…Proposal elements)
    # get their own context with no current-officer facts. Neither is a current officer.
    shares = (
        "jpcrp_cor:NumberOfSharesHeldOrdinarySharesInformationAboutDirectorsAndCorporateAuditors"
    )
    nominee = "FilingDateInstant_jpcrp030000-asr_E99001-000SuzukiIchiroMember"
    facts = make_facts(
        [
            (NAME, P1, "山田　太郎"),
            (TITLE, P1, "代表取締役社長"),
            (shares, "FilingDateInstant", "50308"),
            (NAME + "Proposal", nominee, "鈴木 一郎"),
            (TITLE + "Proposal", nominee, "取締役"),
        ]
    )
    officers = extract_officers(facts, elements)
    assert officers["person_key"].tolist() == [P1]
    assert officers["is_representative"].tolist() == [True]


def test_extract_officers_missing_elements_returns_empty(elements: ElementMap) -> None:
    assert extract_officers(make_facts([]), elements).empty


@pytest.mark.parametrize(
    ("title", "role"),
    [
        ("代表取締役社長", "president"),
        ("代表取締役会長兼社長", "chair"),
        ("常務取締役", "other"),
        (None, "other"),
        (float("nan"), "other"),  # missing title from a pandas pivot
        # Spaces inside words, and 副社長 contains 社長.
        ("代　表取締役社　長執行役員", "president"),
        ("取締役副社長執行役員", "other"),
        ("代表取締役副会長", "other"),
    ],
)
def test_classify_role(title: str | float | None, role: str) -> None:
    assert classify_role(title) == role


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("代表取締役社長", True),
        ("代　表取締役社　長執行役員", True),  # full-width space inside the word
        ("取締役 / 代表執行役CEO", True),
        ("取締役副社長執行役員", False),
        (None, False),
    ],
)
def test_is_representative(title: str | None, expected: bool) -> None:
    assert is_representative(title) is expected


def test_extract_major_shareholders(elements: ElementMap) -> None:
    facts = make_facts(
        [
            (
                "jpcrp_cor:NameMajorShareholders",
                "CurrentYearInstant_No2MajorShareholdersMember",
                "株式会社ヤマダ興産",
            ),
            ("jpcrp_cor:ShareholdingRatio", "CurrentYearInstant_No2MajorShareholdersMember", "0.2"),
            (
                "jpcrp_cor:NameMajorShareholders",
                "CurrentYearInstant_No1MajorShareholdersMember",
                "山田 太郎",
            ),
            (
                "jpcrp_cor:ShareholdingRatio",
                "CurrentYearInstant_No1MajorShareholdersMember",
                "25.5",
            ),
        ]
    )
    holders = extract_major_shareholders(facts, elements)
    assert list(holders["rank"]) == [1, 2]
    assert holders.loc[0, "ratio"] == pytest.approx(0.255)
    assert bool(holders.loc[1, "is_organization"])


def test_extract_major_shareholders_skips_total_row(elements: ElementMap) -> None:
    first = "CurrentYearInstant_No1MajorShareholdersMember"
    facts = make_facts(
        [
            ("jpcrp_cor:NameMajorShareholders", first, "サンプル興産株式会社"),
            ("jpcrp_cor:ShareholdingRatio", first, "0.1732"),
            ("jpcrp_cor:NumberOfSharesHeld", "CurrentYearInstant", "1240000"),
            ("jpcrp_cor:ShareholdingRatio", "CurrentYearInstant", "0.5269"),
        ]
    )
    holders = extract_major_shareholders(facts, elements)
    assert holders["rank"].tolist() == [1]


def test_normalize_ratio() -> None:
    assert normalize_ratio(0.15) == 0.15
    assert normalize_ratio(15.0) == pytest.approx(0.15)
    assert normalize_ratio(None) is None


@pytest.mark.parametrize(
    ("raw", "value"), [("1,234", 1234.0), ("△500", -500.0), ("", None), ("abc", None)]
)
def test_to_number(raw: str, value: float | None) -> None:
    assert to_number(raw) == value


def test_extract_financials_prefers_consolidated(elements: ElementMap) -> None:
    facts = make_facts(
        [
            ("jppfs_cor:CashAndDeposits", "CurrentYearInstant", "1000"),
            ("jppfs_cor:CashAndDeposits", "CurrentYearInstant_NonConsolidatedMember", "400"),
            ("jppfs_cor:NetAssets", "CurrentYearInstant_NonConsolidatedMember", "5000"),
            ("jppfs_cor:OperatingIncome", "CurrentYearDuration", "300"),
            (STANDARD, "FilingDateInstant", "Japan GAAP"),
            (CONSOLIDATED, "FilingDateInstant", "true"),
        ]
    )
    fin = extract_financials(facts, elements)
    assert fin["cash_and_deposits"] == 1000  # consolidated wins
    assert fin["net_assets"] == 5000  # falls back to non-consolidated
    assert fin["operating_income"] == 300
    assert fin["long_term_loans"] is None  # not reported ≠ zero
    assert fin["is_consolidated"] is True
    assert fin["is_jgaap"] is True


def test_extract_financials_ifrs_filer_has_no_jgaap_items(elements: ElementMap) -> None:
    facts = make_facts(
        [
            ("jppfs_cor:CashAndDeposits", "CurrentYearInstant_NonConsolidatedMember", "400"),
            ("jpigp_cor:CashAndCashEquivalentsIFRS", "CurrentYearInstant", "9000"),
            (
                "jpcrp_cor:TotalNumberOfIssuedSharesSummaryOfBusinessResults",
                "FilingDateInstant",
                "100",
            ),
            (STANDARD, "FilingDateInstant", "IFRS"),
        ]
    )
    fin = extract_financials(facts, elements)
    assert fin["is_jgaap"] is False
    assert fin["cash_and_deposits"] is None
    assert fin["issued"] == 100  # share counts do not depend on the accounting standard


def test_extract_financials_unknown_standard_is_none(elements: ElementMap) -> None:
    facts = make_facts([("jppfs_cor:NetAssets", "CurrentYearInstant", "5000")])
    fin = extract_financials(facts, elements)
    assert fin["is_jgaap"] is None  # missing ≠ False
    assert fin["net_assets"] == 5000


def test_extract_financials_non_consolidated_filer(elements: ElementMap) -> None:
    facts = make_facts(
        [
            ("jpcrp_cor:NumberOfEmployees", "CurrentYearInstant", "120"),
            ("jppfs_cor:CashAndDeposits", "CurrentYearInstant_NonConsolidatedMember", "400"),
            (STANDARD, "FilingDateInstant", "Japan GAAP"),
            (CONSOLIDATED, "FilingDateInstant", "false"),
        ]
    )
    fin = extract_financials(facts, elements)
    assert fin["is_consolidated"] is False
    assert fin["cash_and_deposits"] == 400


def test_extract_financials_unknown_consolidation_is_none(elements: ElementMap) -> None:
    facts = make_facts([("jppfs_cor:NetAssets", "CurrentYearInstant", "5000")])
    assert extract_financials(facts, elements)["is_consolidated"] is None


def test_extract_officers_falls_back_to_agm_nominees_without_a_representative(
    elements: ElementMap,
) -> None:
    nominee = "FilingDateInstant_jpcrp030000-asr_E99001-000SuzukiIchiroMember"
    facts = make_facts(
        [
            (NAME, P2, "佐藤　花子"),
            (TITLE, P2, "取締役 (監査等委員)"),
            (NAME + "Proposal", nominee, "鈴木　一郎"),
            (TITLE + "Proposal", nominee, "代表取締役社長"),
            (DOB + "Proposal", nominee, "1950-05-05"),
            (NAME + "Proposal", P2, "佐藤　花子"),  # also a nominee: must not be duplicated
            (TITLE + "Proposal", P2, "取締役 (監査等委員)"),
        ]
    )
    officers = extract_officers(facts, elements).set_index("name")
    assert sorted(officers.index) == ["佐藤 花子", "鈴木 一郎"]
    assert officers.loc["鈴木 一郎", "is_representative"]
    assert officers.loc["鈴木 一郎", "date_of_birth"] == date(1950, 5, 5)


def test_extract_officers_ignores_nominees_when_a_representative_exists(
    elements: ElementMap,
) -> None:
    facts = make_facts(
        [
            (NAME, P1, "山田　太郎"),
            (TITLE, P1, "代表取締役社長"),
            (NAME + "Proposal", P2, "佐藤　花子"),
            (TITLE + "Proposal", P2, "取締役"),
        ]
    )
    assert extract_officers(facts, elements)["name"].tolist() == ["山田 太郎"]


def test_extract_financials_reads_issued_shares_at_filing_date(elements: ElementMap) -> None:
    facts = make_facts(
        [
            ("jpcrp_cor:TotalNumberOfIssuedSharesSummaryOfBusinessResults",
             "CurrentYearInstant_NonConsolidatedMember", "4054600"),
            ("jpcrp_cor:NumberOfIssuedSharesAsOfFilingDateIssuedSharesTotalNumberOfSharesEtc",
             "FilingDateInstant", "12163800"),
        ]
    )  # fmt: skip
    fin = extract_financials(facts, elements)
    assert fin["issued"] == 4054600
    assert fin["issued_at_filing"] == 12163800


def test_extract_major_shareholders_drops_a_named_total_row(elements: ElementMap) -> None:
    name, ratio = "jpcrp_cor:NameMajorShareholders", "jpcrp_cor:ShareholdingRatio"
    first = "CurrentYearInstant_No1MajorShareholdersMember"
    second = "CurrentYearInstant_No2MajorShareholdersMember"
    facts = make_facts(
        [
            (name, first, "サンプル自動車株式会社"),
            (ratio, first, "1.0000"),
            (name, second, "計"),
            (ratio, second, "1.0000"),
        ]
    )
    holders = extract_major_shareholders(facts, elements)
    assert holders["name"].tolist() == ["サンプル自動車株式会社"]


def test_issued_shares_prefer_the_share_table_over_the_summary(elements: ElementMap) -> None:
    summary = "jpcrp_cor:TotalNumberOfIssuedSharesSummaryOfBusinessResults"
    table = "jpcrp_cor:NumberOfIssuedSharesAsOfFiscalYearEndIssuedSharesTotalNumberOfSharesEtc"
    facts = make_facts(
        [
            (summary, "CurrentYearInstant_NonConsolidatedMember", "134380"),
            (table, "FilingDateInstant", "134380704"),
        ]
    )
    assert extract_financials(facts, elements)["issued"] == 134380704


def test_issued_shares_fall_back_to_the_summary(elements: ElementMap) -> None:
    summary = "jpcrp_cor:TotalNumberOfIssuedSharesSummaryOfBusinessResults"
    facts = make_facts([(summary, "CurrentYearInstant_NonConsolidatedMember", "1000")])
    assert extract_financials(facts, elements)["issued"] == 1000


def test_issued_shares_count_common_shares_only(elements: ElementMap) -> None:
    table = "jpcrp_cor:NumberOfIssuedSharesAsOfFiscalYearEndIssuedSharesTotalNumberOfSharesEtc"
    at_filing = "jpcrp_cor:NumberOfIssuedSharesAsOfFilingDateIssuedSharesTotalNumberOfSharesEtc"
    facts = make_facts(
        [
            (table, "FilingDateInstant", "33400115"),
            (table, "FilingDateInstant_OrdinaryShareMember", "30650115"),
            (table, "FilingDateInstant_ClassTwoPreferredSharesMember", "2000000"),
            (at_filing, "FilingDateInstant", "36148647"),
            (at_filing, "FilingDateInstant_OrdinaryShareMember", "36148647"),
        ]
    )
    fin = extract_financials(facts, elements)
    assert fin["issued"] == 30650115
    assert fin["issued_at_filing"] == 36148647


def test_shareholder_address_and_headquarters_address(elements: ElementMap) -> None:
    ctx = "CurrentYearInstant_No1MajorShareholdersMember"
    facts = make_facts(
        [
            ("jpcrp_cor:NameMajorShareholders", ctx, "株式会社サンプル興産"),
            ("jpcrp_cor:ShareholdingRatio", ctx, "27.32"),
            ("jpcrp_cor:AddressMajorShareholders", ctx, "名古屋市千種区今池五丁目５－６"),
            (
                elements.dei["hq_address"],
                "FilingDateInstant",
                "愛知県名古屋市千種区内山三丁目18番10号",
            ),
        ]
    )
    holders = extract_major_shareholders(facts, elements)
    assert holders.loc[0, "address"] == "名古屋市千種区今池五丁目5-6"  # NFKC: full-width digits
    assert (
        extract_financials(facts, elements)["hq_address"]
        == "愛知県名古屋市千種区内山三丁目18番10号"
    )


def test_dividend_per_share_is_read_for_every_accounting_standard(elements: ElementMap) -> None:
    dps = "jpcrp_cor:DividendPaidPerShareSummaryOfBusinessResults"
    facts = make_facts(
        [
            (dps, "Prior1YearDuration_NonConsolidatedMember", "300.00"),
            (dps, "CurrentYearDuration_NonConsolidatedMember", "76.00"),
            (STANDARD, "FilingDateInstant", "IFRS"),
        ]
    )
    assert extract_financials(facts, elements)["dividend_per_share"] == 76.0


def test_dividend_per_share_blank_is_unknown_not_zero(elements: ElementMap) -> None:
    dps = "jpcrp_cor:DividendPaidPerShareSummaryOfBusinessResults"
    facts = make_facts([(dps, "CurrentYearDuration_NonConsolidatedMember", "")])
    assert extract_financials(facts, elements)["dividend_per_share"] is None
