"""Activist signal: distinct activist filers of 350 reports in the year before as_of."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

from radar.config import load_yaml
from radar.features.activism import (
    DiscoveryConfig,
    active_patterns,
    activism_window_covered,
    activist_candidates,
    activist_features,
    all_patterns,
    declares_proposals,
    initial_reports_per_filer,
    is_activist,
    new_large_holders,
    recent_offers,
)
from radar.parse.large_holding import holding_purposes

from conftest import make_facts

AS_OF = date(2025, 7, 1)
PATTERNS = all_patterns(load_yaml("activists")["activist_filers"])  # dated entries → names


def _documents() -> pd.DataFrame:
    rows = [
        ("A1", "350", "株式会社ストラテジックキャピタル", "E99001", datetime(2025, 3, 1, 15)),
        ("A2", "350", "株式会社ストラテジックキャピタル", "E99001", datetime(2025, 5, 1, 15)),
        ("A3", "350", "ＥＦＦＩＳＳＩＭＯ　ＣＡＰＩＴＡＬ", "E99001", datetime(2024, 9, 1, 15)),
        ("A4", "350", "野村證券株式会社", "E99001", datetime(2025, 6, 1, 15)),  # passive
        ("A5", "350", "エリオット・マネジメント", "E99002", datetime(2025, 7, 1, 9)),  # ON as_of
        ("A6", "350", "エリオット・マネジメント", "E99002", datetime(2024, 6, 30, 9)),  # >365 days
        ("A7", "120", "株式会社ストラテジックキャピタル", "E99002", datetime(2025, 6, 1, 9)),
        ("A0", "120", "サンプル工業株式会社", None, datetime(2024, 1, 4, 9)),  # index start
    ]
    return pd.DataFrame(
        rows,
        columns=["doc_id", "doc_type_code", "filer_name", "issuer_edinet_code", "submit_datetime"],
    )


def test_is_activist_matches_width_and_case() -> None:
    assert is_activist("株式会社シティインデックスイレブンス", PATTERNS)
    assert is_activist("ｅｆｆｉｓｓｉｍｏ capital", PATTERNS)
    assert not is_activist("野村證券株式会社", PATTERNS)
    assert not is_activist("村上 三郎", PATTERNS)  # a bare common surname is not a signal


def test_activist_features_point_in_time() -> None:
    result = activist_features(_documents(), AS_OF, PATTERNS)
    by_company = result.set_index("edinet_code")
    # E99001: two distinct activists in the year before as_of (repeat filings count once).
    assert by_company.loc["E99001", "activist_filers_365d"] == 2
    assert (
        by_company.loc["E99001", "activist_names"] == "EFFISSIMO CAPITAL / ストラテジックキャピタル"
    )
    # E99002: one report filed ON as_of (not yet public) and one older than 365 days.
    assert "E99002" not in by_company.index


def test_activism_window_covered() -> None:
    assert activism_window_covered(date(2024, 1, 1), AS_OF)
    assert not activism_window_covered(date(2024, 1, 4), date(2024, 12, 31))
    assert not activism_window_covered(None, AS_OF)


DISCOVERY = DiscoveryConfig.from_config(load_yaml("activists")["discovery"])


def test_holding_purposes_reads_every_joint_holder() -> None:
    element = DISCOVERY.purpose_element
    facts = make_facts(
        [
            (element, "Row1", "純投資及び状況に応じて重要提案行為等を行うこと。"),
            (element, "Row2", "信託財産の運用として保有している。"),
        ]
    )
    assert holding_purposes(facts, DISCOVERY.purpose_element) == [
        "純投資及び状況に応じて重要提案行為等を行うこと。",
        "信託財産の運用として保有している。",
    ]


@pytest.mark.parametrize(
    ("purpose", "expected"),
    [
        ("純投資及び状況に応じて重要提案行為等を行うこと。", True),  # ストラテジックキャピタル
        ("純投資。", False),  # Effissimo: activist in practice, silent on paper
        ("重要提案行為等を行うことは予定しておりません。", False),
        ("純投資であり、重要提案行為等を行う予定はない。", False),
        ("証券業務に係わる商品在庫として保有している。", False),
    ],
)
def test_declares_proposals(purpose: str, expected: bool) -> None:
    assert declares_proposals(purpose, DISCOVERY) is expected


def test_initial_reports_one_per_filer_latest_first() -> None:
    documents = pd.DataFrame(
        {
            "doc_id": ["OLD", "NEW", "CHANGE", "SPECIAL", "OTHER"],
            "edinet_code": ["E99101", "E99101", "E99101", "E99102", "E99103"],
            "doc_type_code": ["350"] * 5,
            "doc_description": [
                "大量保有報告書", "大量保有報告書", "変更報告書",
                "大量保有報告書（特例対象株券等）", "大量保有報告書",
            ],
            "submit_datetime": pd.to_datetime(
                ["2024-01-05", "2025-03-01", "2025-06-01", "2025-01-01", "2024-05-01"]
            ),
            "withdrawal_status": ["0"] * 5,
        }
    )  # fmt: skip
    chosen = initial_reports_per_filer(documents)
    # Change reports and 特例 (institutional, passive by regime) are not initial declarations.
    assert sorted(chosen["doc_id"]) == ["NEW", "OTHER"]


def test_activist_candidates_exclude_known_activists() -> None:
    purposes = pd.DataFrame(
        {
            "filer_edinet_code": ["E99101", "E99102", "E99103"],
            "filer_name": [
                "株式会社ストラテジックキャピタル", "サンプル・パートナーズ", "サンプル信託銀行",
            ],
            "purpose": [
                "純投資及び状況に応じて重要提案行為等を行うこと。",
                "純投資及び重要提案行為等を行うこと。",
                "信託財産の運用として保有している。",
            ],
        }
    )  # fmt: skip
    candidates = activist_candidates(purposes, ["ストラテジックキャピタル"], DISCOVERY)
    assert candidates["filer_name"].tolist() == ["サンプル・パートナーズ"]


@pytest.mark.parametrize(
    "purpose",
    [
        # Real 350s of tender-offer vehicles after the offer: a squeeze-out, not activism.
        "提出者は、発行者の非公開化を目的とした重要提案行為等を行うことを予定しております。"
        "具体的には、会社法第180条に基づき株式の併合を行うことを要請する予定です。",
        "提出者は、発行者の完全子会社化を目的とした重要提案行為等を行う予定です。",
        # Strategic partner: synergy, not pressure on management.
        "資本関係を構築した上で事業シナジーを創出し、状況に応じて重要提案行為等を行うこと",
        "政策投資、重要提案行為等を行うこと",  # 商船三井 (corporate holding)
    ],
)
def test_deal_vehicles_and_partners_are_not_activist_candidates(purpose: str) -> None:
    purposes = pd.DataFrame(
        {
            "filer_edinet_code": ["E99101"],
            "filer_name": ["サンプルSPC株式会社"],
            "purpose": [purpose],
        }
    )
    assert activist_candidates(purposes, [], DISCOVERY).empty


def test_activists_count_only_from_the_year_they_were_known() -> None:
    # An activist known only from 2025 must not count on a 2024 screening (look-ahead).
    entries = ["ストラテジックキャピタル", {"name": "エリオット", "since": 2025}]
    assert active_patterns(entries, date(2024, 12, 31)) == ("ストラテジックキャピタル",)
    assert active_patterns(entries, date(2025, 1, 1)) == ("ストラテジックキャピタル", "エリオット")
    assert all_patterns(entries) == ("ストラテジックキャピタル", "エリオット")


def test_new_large_holders_in_the_window_before_as_of() -> None:
    documents = pd.DataFrame(
        {
            "doc_type_code": ["350"] * 5,
            "doc_description": [
                "大量保有報告書", "大量保有報告書", "変更報告書",
                "大量保有報告書（特例対象株券等）", "大量保有報告書",
            ],
            "filer_name": ["サンプル投資", "サンプル商事", "サンプル投資", "大手信託", "遅い投資"],
            "edinet_code": ["E9", "E8", "E9", "E7", "E6"],
            "issuer_edinet_code": ["E1"] * 5,
            "submit_datetime": [
                datetime(2025, 5, 1), datetime(2025, 6, 20), datetime(2025, 6, 25),
                datetime(2025, 6, 1), datetime(2025, 7, 1, 9),
            ],
        }
    )  # fmt: skip
    counts = new_large_holders(documents, date(2025, 7, 1), window_days=90)
    # Initial non-特例 reports in [as_of − 90 d, as_of): サンプル投資 and サンプル商事. The change
    # report, the passive 特例 filer and the report ON as_of do not count.
    assert counts == {"E1": 2}
    assert new_large_holders(documents, date(2025, 7, 1), window_days=30) == {"E1": 1}


def test_recent_offers_mark_companies_already_in_play() -> None:
    documents = pd.DataFrame(
        {
            "doc_type_code": ["240", "240", "240", "240"],
            "edinet_code": ["E90", "E91", "E1", "E92"],  # E1 → E1: a self-tender, not an offer
            "subject_edinet_code": ["E1", "E2", "E1", "E3"],
            "withdrawal_status": ["0", "0", "0", "0"],
            "submit_datetime": [
                datetime(2026, 5, 13), datetime(2025, 9, 1), datetime(2026, 6, 1),
                datetime(2026, 10, 9, 9),  # ON as_of: not known yet
            ],
        }
    )  # fmt: skip
    found = recent_offers(documents, date(2026, 10, 9), window_days=365)
    assert found == {"E1": date(2026, 5, 13)}  # E2's offer is older than a year
