"""臨時報告書 (doc 180): the reason comes from WHICH text blocks the report contains."""

from __future__ import annotations

from radar.config import load_yaml
from radar.parse.extraordinary_report import ExtraordinaryConfig, classify_extraordinary

from conftest import make_facts

CONFIG = ExtraordinaryConfig.from_config(load_yaml("extraordinary_reports"))
REASON = "jpcrp-esr_cor:ReasonForFilingTextBlock"
CTX = "FilingDateInstant"


def _report(*blocks: str, reason: str = "１【提出理由】") -> dict[str, object]:
    rows = [(REASON, CTX, reason)]
    rows += [(f"jpcrp-esr_cor:{b}TextBlock", CTX, "２【報告内容】…") for b in blocks]
    rows.append(("jpcrp-esr_cor:PlaceForPublicInspectionCoverPageTextBlock", CTX, "東京証券取引所"))
    return classify_extraordinary(make_facts(rows), CONFIG)


def test_major_shareholder_change_is_an_alert() -> None:
    result = _report("ChangesInMajorShareholder")
    assert result["reasons"] == ["major_shareholder"]
    assert result["alert"] is True
    assert result["labels"] == "主要株主の異動"


def test_parent_change_after_a_tender_offer_is_told_apart_from_a_new_subsidiary() -> None:
    # 「…公開買付けの結果、当社の親会社及び主要株主に異動がありますので…」
    parent = _report(
        "ChangesInParentCompaniesOrSpecifiedSubsidiaries", "ChangesInMajorShareholder",
        reason="公開買付けの結果、当社の親会社及び主要株主に異動がありますので",
    )  # fmt: skip
    assert parent["reasons"] == ["parent_change", "major_shareholder"]
    assert parent["tender_offer"] is True and parent["alert"] is True
    subsidiary = _report(
        "ChangesInParentCompaniesOrSpecifiedSubsidiaries",
        reason="当社において特定子会社の異動がありましたので",
    )
    assert subsidiary["reasons"] == ["specified_subsidiary"]
    assert subsidiary["alert"] is False  # the company bought something: not about its control


def test_share_consolidation_is_matched_before_company_split() -> None:
    # "…ForPurposeOfReverseStockSplit" also contains "Split": the order in the config decides.
    result = _report("DecisionOnHoldingShareholdersMeetingForPurposeOfReverseStockSplit")
    assert result["reasons"] == ["share_consolidation"] and result["alert"] is True


def test_routine_reports_are_kept_but_not_alerted() -> None:
    result = _report("ResolutionOfShareholdersMeeting")
    assert result["reasons"] == ["agm_votes"] and result["alert"] is False


def test_unknown_block_is_other_and_the_cover_page_is_ignored() -> None:
    assert _report("SomethingNewTheTaxonomyAdded")["reasons"] == ["other"]
    assert _report()["reasons"] == []


def test_excerpt_is_the_start_of_the_reason_text() -> None:
    result = _report("ChangesInMajorShareholder", reason="当社の主要株主に異動がありました。" * 20)
    assert str(result["excerpt"]).startswith("当社の主要株主")
    assert len(str(result["excerpt"])) <= CONFIG.excerpt_chars + 1  # + the ellipsis
    assert _report("ChangesInMajorShareholder", reason="")["excerpt"] == ""


def test_report_without_a_reason_text_still_classifies() -> None:
    facts = make_facts([("jpcrp-esr_cor:ChangesInMajorShareholderTextBlock", CTX, "…")])
    result = classify_extraordinary(facts, CONFIG)
    assert result["reasons"] == ["major_shareholder"] and result["tender_offer"] is False
