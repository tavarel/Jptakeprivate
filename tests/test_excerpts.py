"""Original text of a few 有報 sections for the fact pack (no generation, no summarising)."""

from __future__ import annotations

from radar.parse.elements import ElementMap
from radar.parse.excerpts import report_excerpts

from conftest import make_facts

ELEMENTS = ElementMap.load()


def test_excerpts_keep_the_filing_text_and_prefer_the_current_year() -> None:
    customers = ELEMENTS.excerpts["main_customers"]
    facts = make_facts(
        [
            (
                ELEMENTS.excerpts["business"],
                "FilingDateInstant",
                "３ 【事業の内容】\n当社グループは、鉄道車両関連事業及び不動産賃貸事業に事業活動を"
                "展開しております。",
            ),
            (customers, "Prior1YearDuration", "主要な顧客ごとの情報 前期の顧客"),
            (customers, "CurrentYearDuration", "主要な顧客ごとの情報 近畿日本鉄道株式会社5,887"),
        ]
    )
    excerpts = report_excerpts(facts, ELEMENTS.excerpts, max_chars=40)
    assert excerpts["business"].startswith("3 【事業の内容】 当社グループは")  # NFKC, one line
    assert excerpts["business"].endswith("…")  # cut at max_chars
    assert excerpts["main_customers"] == "主要な顧客ごとの情報 近畿日本鉄道株式会社5,887"
    assert "revenue_by_region" not in excerpts  # absent in the filing → absent, not ""
