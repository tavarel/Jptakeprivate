"""Tender-offer terms from the 240 filing (premium and event typing)."""

from __future__ import annotations

import pytest

from radar.config import load_yaml
from radar.parse.tender_offer import (
    TenderOfferConfig,
    classify_event,
    extract_tender_offer,
    offer_price,
)

from conftest import make_facts

CONFIG = TenderOfferConfig.from_config(load_yaml("tender_offers"))
E = CONFIG.elements
CTX = "FilingDateInstant"


def _facts(price_text: str, purpose: str, d: str, g: str, j: str) -> object:
    return make_facts(
        [
            (E["price_text"], CTX, price_text),
            (E["purpose_text"], CTX, purpose),
            (E["owned_by_bidder"], CTX, d),
            (E["owned_by_related"], CTX, g),
            (E["all_voting_rights"], CTX, j),
            (E["ratio_after"], CTX, "1.0000"),
        ]
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("株券１株につき金7,600円新株予約権証券―", 7600.0),
        ("普通株式１株につき、金1,234.5円", 1234.5),
        ("株券―", None),  # no price stated → unknown
    ],
)
def test_offer_price(text: str, expected: float | None) -> None:
    assert offer_price(text) == expected


def test_extract_mbo_like_the_real_gerbera_filing() -> None:
    # Shaped on S100Y4AZ: SPC bidder owns nothing (D blank), management/founders 10,370 votes.
    purpose = (
        "本公開買付けは、いわゆるマネジメント・バイアウト（ＭＢＯ）のための取引の一環として…"
        "上場廃止となる"
    )
    terms = extract_tender_offer(
        _facts("株券１株につき金7,600円", purpose, "", "10370", "28574"), CONFIG
    )
    assert terms["offer_price"] == 7600.0
    assert terms["prior_stake"] == pytest.approx(10370 / 28574)
    assert terms["ratio_after"] == 1.0
    assert terms["is_take_private"] is True
    assert terms["self_declared_mbo"] is True
    assert classify_event(terms, CONFIG) == "mbo"


def test_parent_and_third_party() -> None:
    parent = extract_tender_offer(
        _facts("１株につき金500円", "当社の連結子会社である対象者…", "30000", "100", "50000"),
        CONFIG,
    )
    assert parent["prior_stake"] == pytest.approx(0.602)
    assert classify_event(parent, CONFIG) == "parent"
    third = extract_tender_offer(
        _facts("１株につき金500円", "対象者を完全子会社化", "", "", "50000"), CONFIG
    )
    assert third["prior_stake"] == 0.0  # dashes on a reported form = none held
    assert classify_event(third, CONFIG) == "third_party"


def test_mbo_mention_in_a_comparison_is_not_an_mbo() -> None:
    purpose = "過去のマネジメント・バイアウト（ＭＢＯ）事例におけるプレミアム水準を参考に…"
    terms = extract_tender_offer(_facts("１株につき金500円", purpose, "", "", "50000"), CONFIG)
    assert terms["self_declared_mbo"] is False


def test_missing_voting_rights_total_means_unknown_stake() -> None:
    terms = extract_tender_offer(_facts("１株につき金500円", "", "", "", ""), CONFIG)
    assert terms["prior_stake"] is None
    assert classify_event(terms, CONFIG) == "unknown"  # not a guess


@pytest.mark.parametrize(
    "sentence",
    [
        "本公開買付けを含む本取引は、いわゆるマネジメント・バイアウト(MBO)取引にも該当いたしません。",
        "本公開買付けは、いわゆるマネジメント・バイアウト取引(注1)に該当するものではございません。",
        "本取引は、いわゆるマネジメント・バイアウト(MBO)には該当せず、",
    ],
)
def test_a_negated_mbo_sentence_is_not_an_mbo(sentence: str) -> None:
    # Real phrasings from 2024–2026 filings by strategic and parent bidders.
    terms = extract_tender_offer(_facts("１株につき金500円", sentence, "", "", "50000"), CONFIG)
    assert terms["self_declared_mbo"] is False


def test_affirmative_mbo_sentences() -> None:
    for sentence in [
        "本取引は、いわゆるマネジメント・バイアウト(MBO)に該当し、武部氏は…",
        "本取引成立後も継続して経営にあたることを予定しているため、本取引は、いわゆるマネジメント・バイアウト(MBO)に該当いたします。",
    ]:
        terms = extract_tender_offer(_facts("１株につき金500円", sentence, "", "", "50000"), CONFIG)
        assert terms["self_declared_mbo"] is True, sentence


def test_partial_offer_is_not_a_take_private() -> None:
    facts = _facts(
        "１株につき金400円",
        "本公開買付けは、対象者株式の上場廃止を企図するものではありません。",
        "",
        "",
        "50000",
    )
    facts.loc[facts["element_id"] == E["ratio_after"], "value"] = "0.3000"
    terms = extract_tender_offer(facts, CONFIG)
    assert terms["is_take_private"] is False


def test_full_offer_is_a_take_private() -> None:
    terms = extract_tender_offer(
        _facts("１株につき金500円", "上場廃止となる見込みです。", "", "", "50000"), CONFIG
    )
    assert terms["is_take_private"] is True  # ratio_after 1.0000 in the fixture


def test_bidder_stake_separates_the_bidder_from_related_parties() -> None:
    # A parent SELLING to a fund appears in G (tender agreement), not in D (日立金属 / BCJ-52).
    terms = extract_tender_offer(_facts("１株につき金500円", "", "", "26700", "50000"), CONFIG)
    assert terms["bidder_stake"] == 0.0
    assert terms["prior_stake"] == pytest.approx(0.534)


def test_impossible_stake_is_unknown_not_used() -> None:
    terms = extract_tender_offer(_facts("１株につき金500円", "", "72900", "19000", "48208"), CONFIG)
    assert terms["prior_stake"] is None
    assert terms["bidder_stake"] is None
    assert classify_event(terms, CONFIG) == "unknown"
