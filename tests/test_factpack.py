"""Fact pack markdown: facts for a memo, never the memo itself."""

from __future__ import annotations

from datetime import date

import pandas as pd

from radar.model.rules import RuleSet
from radar.report.factpack import FactPackInputs, factpack_markdown


def test_factpack_has_every_section_and_links_the_filing() -> None:
    row = {
        "company_name": "サンプル工業株式会社", "sec_code": "99990", "source_doc_id": "S100AAA1",
        "filing_submitted_at": "2025-06-27", "triggered_rules": ["aging_leader"],
        "oldest_representative_age": 75, "family_stake": None, "pbr": 0.6,
        "activist_filers_365d": 0, "activist_names": None,
        "largest_listed_holder_stake": 0.0, "largest_listed_holder_name": None,
    }  # fmt: skip
    inputs = FactPackInputs(
        as_of=date(2025, 7, 1),
        row=row,
        ruleset=RuleSet.load(),
        likely_buyer="mbo",
        last_close=1000.0,
        offer_table=pd.DataFrame(
            {"anchor": ["Premium paid (82 past mbo offers)"], "low (25%)": ["¥1,275"],
             "median": ["¥1,350"], "high (75%)": ["¥1,425"], "median vs last close": ["+35%"]}
        ),
        hit_rates=pd.DataFrame(
            {"points": ["2–4"], "companies": [10], "hits": [1], "hit_rate": [0.1]}
        ),
        holders=pd.DataFrame(
            {"rank": [1], "holder": ["サンプル興産"], "type": ["company"], "stake": [0.4]}
        ),
        officers=pd.DataFrame({"name": ["サンプル 太郎"], "age": [75]}),
        excerpts={"business": "3 【事業の内容】 当社は…"},
        recent_filings=pd.DataFrame(columns=["submit_datetime", "filer_name", "doc_description"]),
    )  # fmt: skip
    text = factpack_markdown(inputs)
    assert text.startswith("# Fact pack — サンプル工業株式会社 (9999), as of 2025-07-01")
    assert "S100AAA1" in text and "edinet-fsa.go.jp" in text
    for heading in ("## Why it screens", "## Key facts", "## Likely buyer: mbo",
                    "## In the filing's own words", "## Top-10 shareholders"):  # fmt: skip
        assert heading in text
    assert "family_stake: unknown" in text  # missing shown as unknown, never 0
    assert "| Premium paid (82 past mbo offers) | ¥1,275 | ¥1,350 | ¥1,425 | +35% |" in text
    assert "last close ¥1,000" in text
    assert "_(none before this date)_" in text  # empty table says so
    # Zero activists / no listed holder: "none", not "unknown" (the count is known to be 0).
    assert "activist_filers_365d: 0" in text and "activist_names" not in text
    assert "largest_listed_holder_name" not in text
    assert "parent_pbr" not in text  # no listed parent: its figures do not apply
    assert "| 2–4 | 10 | 1 | 10.0% |" in text  # hit rate as a percentage
