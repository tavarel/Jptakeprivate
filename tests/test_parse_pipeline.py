"""`radar parse`: downloaded annual reports → staging tables, plus the quality report."""

from __future__ import annotations

import zipfile
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from radar import pipeline
from radar.config import Settings
from radar.ingest.edinet import EdinetError
from radar.parse.quality import quality_report, sanity_checks
from radar.storage.files import expired_dated_files, write_parquet_atomic
from radar.storage.paths import (
    edinet_doc_path,
    ensure_parent,
    mart_table_path,
    staging_table_path,
)

HEADER = "要素ID\t項目名\tコンテキストID\t相対年度\t連結・個別\t期間・時点\tユニットID\t単位\t値"
NAME = "jpcrp_cor:NameInformationAboutDirectorsAndCorporateAuditors"
TITLE = "jpcrp_cor:OfficialTitleOrPositionInformationAboutDirectorsAndCorporateAuditors"
DOB = "jpcrp_cor:DateOfBirthInformationAboutDirectorsAndCorporateAuditors"
PERSON = "FilingDateInstant_jpcrp030000-asr_E99001-000SampleTaroMember"
HOLDER = "CurrentYearInstant_No1MajorShareholdersMember"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path, _env_file=None)  # type: ignore[call-arg]


def _write_zip(settings: Settings, doc_id: str, rows: list[tuple[str, str, str]]) -> None:
    """A synthetic EDINET type=5 ZIP: UTF-16 with BOM, tab-separated (the real format)."""
    lines = [HEADER] + [f"{e}\t\t{c}\t\t\t\t\t\t{v}" for e, c, v in rows]
    path = ensure_parent(edinet_doc_path(settings.raw_dir, doc_id, 5))
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"XBRL_TO_CSV/jpcrp030000-asr-001_{doc_id}.csv", "\n".join(lines).encode("utf-16")
        )


def _index(settings: Settings, rows: list[tuple[str, str, str]]) -> None:
    """Document index with (doc_id, edinet_code, doc_type_code) rows."""
    docs = pd.DataFrame(
        [
            {
                "doc_id": doc_id,
                "edinet_code": code,
                "sec_code": "99990",
                "filer_name": "サンプル工業株式会社",
                "doc_type_code": doc_type,
                "doc_description": "有価証券報告書",
                "submit_datetime": datetime(2025, 6, 27, 15, 0),
                "period_end": date(2025, 3, 31),
                "issuer_edinet_code": None,
                "subject_edinet_code": None,
                "parent_doc_id": None,
                "withdrawal_status": "0",
                "xbrl_flag": True,
                "csv_flag": True,
            }
            for doc_id, code, doc_type in rows
        ],
        columns=pipeline.documents_frame([]).columns,
    )
    docs.to_parquet(ensure_parent(staging_table_path(settings.staging_dir, "documents")))


GOOD_FILING = [
    (NAME, PERSON, "サンプル　太郎"),
    (TITLE, PERSON, "代表取締役社長"),
    (DOB, PERSON, "1950-04-01"),
    ("jpcrp_cor:NameMajorShareholders", HOLDER, "サンプル興産株式会社"),
    ("jpcrp_cor:ShareholdingRatio", HOLDER, "0.42"),
    ("jpcrp_cor:AddressMajorShareholders", HOLDER, "愛知県名古屋市千種区"),
    ("jpcrp_cor:AddressOfRegisteredHeadquarterCoverPage", "FilingDateInstant", "滋賀県彦根市"),
    (
        "jpcrp_cor:NotesRealEstateForLeaseEtcConsolidatedFinancialStatementsTextBlock",
        "CurrentYearDuration",
        "(単位:百万円)期末残高5,9686,694期末時価9,99310,889(注)",
    ),
    ("jppfs_cor:CashAndDeposits", "CurrentYearInstant_NonConsolidatedMember", "1000"),
    ("jpdei_cor:AccountingStandardsDEI", "FilingDateInstant", "Japan GAAP"),
]


def test_parse_writes_staging_tables_with_lineage(settings: Settings) -> None:
    _index(settings, [("S100AAA1", "E99001", "120"), ("S100BBB2", "E99002", "350")])
    _write_zip(settings, "S100AAA1", GOOD_FILING)
    _write_zip(settings, "S100BBB2", GOOD_FILING)  # not an annual report: must be ignored

    result = pipeline.parse_downloaded_filings(settings)

    assert result.parsed == 1 and result.failed == []
    officers = pd.read_parquet(staging_table_path(settings.staging_dir, "officers"))
    holders = pd.read_parquet(staging_table_path(settings.staging_dir, "shareholders"))
    financials = pd.read_parquet(staging_table_path(settings.staging_dir, "financials"))
    assert officers[["doc_id", "edinet_code"]].values.tolist() == [["S100AAA1", "E99001"]]
    assert officers["date_of_birth"].tolist() == [date(1950, 4, 1)]
    assert holders["ratio"].tolist() == [0.42]
    assert financials.loc[0, "cash_and_deposits"] == 1000
    assert financials.loc[0, "period_end"] == date(2025, 3, 31)
    assert pd.isna(financials.loc[0, "long_term_loans"])  # not reported stays missing
    assert financials["long_term_loans"].dtype == "float64"  # all-None column keeps its type
    assert bool(financials.loc[0, "is_jgaap"])
    assert holders["address"].tolist() == ["愛知県名古屋市千種区"]
    assert financials.loc[0, "hq_address"] == "滋賀県彦根市"
    # Rental real estate note (adjusted PBR): current-year column, in yen.
    assert financials.loc[0, "rental_note_status"] == "disclosed"
    assert financials.loc[0, "rental_book_value"] == 6_694e6
    assert financials.loc[0, "rental_fair_value"] == 10_889e6


def test_parse_skips_documents_not_downloaded(settings: Settings) -> None:
    _index(settings, [("S100AAA1", "E99001", "120"), ("S100CCC3", "E99003", "120")])
    _write_zip(settings, "S100AAA1", GOOD_FILING)
    assert pipeline.parse_downloaded_filings(settings).parsed == 1


def test_parse_logs_broken_file_and_continues(settings: Settings) -> None:
    _index(settings, [("S100AAA1", "E99001", "120"), ("S100BAD9", "E99009", "120")])
    _write_zip(settings, "S100AAA1", GOOD_FILING)
    bad = ensure_parent(edinet_doc_path(settings.raw_dir, "S100BAD9", 5))
    bad.write_bytes(b"not a zip")

    result = pipeline.parse_downloaded_filings(settings)

    assert result.parsed == 1
    assert result.failed == ["S100BAD9"]


def test_parse_skips_fund_report_filed_as_annual_report(settings: Settings) -> None:
    _index(settings, [("S100AAA1", "E99001", "120"), ("S100FND1", "E99002", "120")])
    _write_zip(settings, "S100AAA1", GOOD_FILING)
    fund = ensure_parent(edinet_doc_path(settings.raw_dir, "S100FND1", 5))
    with zipfile.ZipFile(fund, "w") as archive:
        archive.writestr("XBRL_TO_CSV/jpaud-aar-cn-001_G99001-000.csv", "x".encode("utf-16"))

    result = pipeline.parse_downloaded_filings(settings)

    assert result.parsed == 1
    assert result.failed == ["S100FND1"]


def test_parse_is_idempotent(settings: Settings) -> None:
    _index(settings, [("S100AAA1", "E99001", "120"), ("S100AAA2", "E99002", "120")])
    _write_zip(settings, "S100AAA1", GOOD_FILING)
    _write_zip(settings, "S100AAA2", GOOD_FILING)
    path = staging_table_path(settings.staging_dir, "officers")

    pipeline.parse_downloaded_filings(settings)
    first = pd.read_parquet(path)
    pipeline.parse_downloaded_filings(settings)
    pd.testing.assert_frame_equal(first, pd.read_parquet(path))


def test_parse_with_no_annual_reports_writes_empty_tables(settings: Settings) -> None:
    _index(settings, [])
    result = pipeline.parse_downloaded_filings(settings)
    assert result.parsed == 0
    assert pd.read_parquet(staging_table_path(settings.staging_dir, "officers")).empty


# ------------------------------------------------------------------ quality report
def test_quality_report_counts_per_filing() -> None:
    officers = pd.DataFrame(
        {
            "doc_id": ["D1", "D1", "D2", "D3"],
            "is_representative": [True, False, True, False],
            "date_of_birth": [date(1950, 1, 1), None, None, date(1960, 1, 1)],
        }
    )
    holders = pd.DataFrame({"doc_id": ["D1", "D2"], "ratio": [0.3, None]})
    financials = pd.DataFrame(
        {"doc_id": ["D1", "D2", "D3", "D4"], "is_jgaap": [True, False, None, True]}
    )
    report = quality_report(officers, holders, financials)
    assert report.filings == 4
    assert report.with_representative == pytest.approx(0.50)  # D1, D2
    assert report.representative_with_birth_date == pytest.approx(0.25)  # D1 only
    assert report.with_shareholder_ratios == pytest.approx(0.25)  # D1 only
    assert report.jgaap == pytest.approx(0.50)  # None is not J-GAAP


def test_quality_report_empty() -> None:
    empty = pd.DataFrame(
        columns=["doc_id", "is_representative", "date_of_birth", "ratio", "is_jgaap"]
    )
    report = quality_report(empty, empty, empty)
    assert report.filings == 0
    assert report.with_representative == 0.0


# ------------------------------------------------------------------ sanity checks
def test_sanity_checks_flag_impossible_values() -> None:
    officers = pd.DataFrame(
        {
            "doc_id": ["D1", "D2", "D3"],
            "is_representative": [True, False, True],
            "date_of_birth": [date(1950, 1, 1), date(1960, 1, 1), date(2015, 1, 1)],
        }
    )
    holders = pd.DataFrame(
        {
            "doc_id": ["D1", "D1", "D2", "D2"],
            "rank": [1, 1, 1, 2],
            "ratio": [0.7, 0.5, 0.2, 1.5],
        }
    )
    financials = pd.DataFrame(
        {
            "doc_id": ["D1", "D2", "D3"],
            "period_end": [date(2025, 3, 31)] * 3,
            "issued": [100.0, 100.0, 100.0],
            "treasury": [10.0, 200.0, None],
            "net_assets": [5.0, 5.0, None],
            "is_jgaap": [True, True, True],
        }
    )
    issues = sanity_checks(officers, holders, financials)
    found = set(zip(issues["doc_id"], issues["check"], strict=True))
    assert ("D1", "top10_ratio_over_100pct") in found
    assert ("D1", "duplicate_rank") in found
    assert ("D2", "ratio_out_of_range") in found
    assert ("D2", "no_representative") in found
    assert ("D2", "treasury_exceeds_issued") in found
    assert ("D3", "implausible_age") in found  # born 2015: a parsing error, not a director
    assert ("D3", "jgaap_without_net_assets") in found
    assert ("D1", "implausible_age") not in found


def test_sanity_checks_clean_data_has_no_issues() -> None:
    officers = pd.DataFrame(
        {"doc_id": ["D1"], "is_representative": [True], "date_of_birth": [date(1960, 1, 1)]}
    )
    holders = pd.DataFrame({"doc_id": ["D1"], "rank": [1], "ratio": [0.3]})
    financials = pd.DataFrame(
        {
            "doc_id": ["D1"],
            "period_end": [date(2025, 3, 31)],
            "issued": [100.0],
            "treasury": [1.0],
            "net_assets": [5.0],
            "is_jgaap": [True],
        }
    )
    assert sanity_checks(officers, holders, financials).empty


def test_sanity_checks_tolerate_holders_without_rank() -> None:
    officers = pd.DataFrame(
        {"doc_id": ["D1"], "is_representative": [True], "date_of_birth": [date(1960, 1, 1)]}
    )
    holders = pd.DataFrame(
        {"doc_id": ["D1", "D1", "D1"], "rank": [1, None, 1], "ratio": [0.1, 0.1, 0.1]}
    )
    financials = pd.DataFrame(
        {
            "doc_id": ["D1"],
            "period_end": [date(2025, 3, 31)],
            "issued": [100.0],
            "treasury": [1.0],
            "net_assets": [5.0],
            "is_jgaap": [True],
        }
    )
    issues = sanity_checks(officers, holders, financials)
    assert issues["check"].tolist() == ["duplicate_rank"]


def test_incremental_parse_equals_a_full_rebuild(settings: Settings) -> None:
    # `radar update` only reads new reports; the result must match `radar parse` from scratch.
    _index(settings, [("S100AAA1", "E99001", "120"), ("S100AAA2", "E99002", "120")])
    _write_zip(settings, "S100AAA2", GOOD_FILING)
    pipeline.parse_downloaded_filings(settings)  # only AAA2 exists so far
    _write_zip(settings, "S100AAA1", GOOD_FILING)  # a new report arrives

    result = pipeline.parse_downloaded_filings(settings, only_new=True)
    assert result.parsed == 1  # AAA2 was not read again
    incremental = {
        t: pd.read_parquet(staging_table_path(settings.staging_dir, t))
        for t in ("officers", "shareholders", "financials")
    }

    pipeline.parse_downloaded_filings(settings)
    for table, frame in incremental.items():
        full = pd.read_parquet(staging_table_path(settings.staging_dir, table))
        pd.testing.assert_frame_equal(frame, full)


def test_write_parquet_atomic_replaces_in_one_step(tmp_path: Path) -> None:
    path = tmp_path / "staging" / "documents.parquet"
    write_parquet_atomic(pd.DataFrame({"a": [1]}), path)
    write_parquet_atomic(pd.DataFrame({"a": [1, 2]}), path)
    assert pd.read_parquet(path)["a"].tolist() == [1, 2]
    assert [p.name for p in path.parent.iterdir()] == ["documents.parquet"]  # no temp left behind


class _FlakyEdinet:
    """Fails on some documents like a network blip; or always, like an outage."""

    def __init__(self, failing: set[str] | None = None) -> None:
        self.failing = failing  # None = every call fails

    def download(self, doc_id: str, fetch_type: int = 5) -> Path:
        if self.failing is None or doc_id in self.failing:
            raise EdinetError("ConnectError: Temporary failure in name resolution")
        return Path(f"/tmp/{doc_id}.zip")


def test_download_skips_a_network_blip_and_continues() -> None:
    docs = pd.DataFrame({"doc_id": ["A", "B", "C"]})
    paths = pipeline.download_documents(_FlakyEdinet({"B"}), docs)  # type: ignore[arg-type]
    assert [p.stem for p in paths] == ["A", "C"]


def test_download_stops_on_a_real_outage() -> None:
    docs = pd.DataFrame({"doc_id": [f"D{i}" for i in range(30)]})
    with pytest.raises(EdinetError, match="consecutive"):
        pipeline.download_documents(_FlakyEdinet(), docs)  # type: ignore[arg-type]


def test_discover_activists_reads_purposes_of_downloaded_reports(settings: Settings) -> None:
    _index(settings, [("S100LVH1", "E99101", "350"), ("S100LVH2", "E99102", "350")])
    docs = pd.read_parquet(staging_table_path(settings.staging_dir, "documents"))
    docs["doc_description"] = "大量保有報告書"
    docs["filer_name"] = ["サンプル・パートナーズ株式会社", "サンプル信託銀行株式会社"]
    docs.to_parquet(staging_table_path(settings.staging_dir, "documents"))
    purposes = {
        "S100LVH1": "純投資及び状況に応じて重要提案行為等を行うこと。",
        "S100LVH2": "信託財産の運用として保有している。",
    }
    for doc_id, purpose in purposes.items():
        line = f"jplvh_cor:PurposeOfHolding\t\tFilingDateInstant\t\t\t\t\t\t{purpose}"
        path = ensure_parent(edinet_doc_path(settings.raw_dir, doc_id, 5))
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(
                f"XBRL_TO_CSV/jplvh010000-lvh-001_{doc_id}.csv",
                "\n".join([HEADER, line]).encode("utf-16"),
            )

    candidates = pipeline.discover_activists(settings, edinet=None)  # files already on disk

    assert candidates["filer_name"].tolist() == ["サンプル・パートナーズ株式会社"]
    assert mart_table_path(settings.marts_dir, "activist_candidates").exists()


def test_expired_dated_files_only_matches_the_prefix_and_old_dates(tmp_path: Path) -> None:
    for name in ("features_2026-01-01.parquet", "features_2026-10-01.parquet",
                 "scores_2026-01-01.parquet", "factpack_2026-01-01_7122.md"):  # fmt: skip
        (tmp_path / name).write_text("x")
    old = expired_dated_files(tmp_path, "features_", date(2026, 10, 10), keep_days=14)
    assert [p.name for p in old] == ["features_2026-01-01.parquet"]  # scores are never pruned
