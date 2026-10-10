"""Likely offer price from past take-privates of the same buyer type (premium and offer PBR)."""

from __future__ import annotations

import pandas as pd
import pytest

from radar.config import load_yaml
from radar.features.valuation import book_value_per_share, outstanding_at_filing
from radar.model.offer_price import OfferRangeConfig, describe, likely_offer_range, offer_pbr

CONFIG = OfferRangeConfig.from_config(load_yaml("event_study"))


def _study() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_type": ["mbo", "mbo", "mbo", "mbo", "parent", "mbo"],
            "is_take_private": [True, True, True, True, True, False],  # last: partial offer
            "pre_announced": [False, False, False, False, False, False],
            "premium": [0.2, 0.3, 0.4, 0.5, 0.1, 0.9],
            "offer_pbr": [0.8, 1.0, 1.2, None, 0.5, 3.0],
        }
    )


def test_real_config_has_ordered_quantiles() -> None:
    assert list(CONFIG.quantiles) == sorted(CONFIG.quantiles)
    assert CONFIG.min_offers >= 3


def test_likely_offer_range_by_premium_and_by_book_value() -> None:
    config = OfferRangeConfig(quantiles=(0.25, 0.5, 0.75), min_offers=3)
    result = likely_offer_range(_study(), "mbo", price=1000.0, book_per_share=1500.0, config=config)
    assert result is not None
    assert result.offers == 4  # partial offer excluded; other types excluded
    assert result.by_premium == pytest.approx((1275.0, 1350.0, 1425.0))  # premium 27.5/35/42.5%
    assert result.by_book == pytest.approx((1350.0, 1500.0, 1650.0))  # PBR 0.9/1.0/1.1 (3 known)


def test_likely_offer_range_needs_enough_offers_and_inputs() -> None:
    config = OfferRangeConfig(quantiles=(0.25, 0.5, 0.75), min_offers=3)
    assert likely_offer_range(_study(), "parent", 1000.0, 1500.0, config) is None  # 1 offer only
    result = likely_offer_range(_study(), "mbo", 1000.0, None, config)
    assert result is not None and result.by_book is None  # no book value → no PBR-based range
    assert likely_offer_range(_study(), "mbo", None, 1500.0, config) is not None


def test_offer_pbr_and_book_value_per_share() -> None:
    assert offer_pbr(1200.0, 1000.0) == pytest.approx(1.2)
    assert offer_pbr(1200.0, None) is None
    assert offer_pbr(1200.0, -5.0) is None  # negative equity: meaningless
    fin = {"issued": 10e6, "issued_at_filing": 10e6, "treasury": 0.0, "net_assets": 20e9}
    assert outstanding_at_filing(fin) == 10e6
    assert book_value_per_share(fin) == pytest.approx(2000.0)
    assert book_value_per_share({**fin, "net_assets": None}) is None


def test_describe_reads_like_a_sentence() -> None:
    config = OfferRangeConfig(quantiles=(0.25, 0.5, 0.75), min_offers=3)
    result = likely_offer_range(_study(), "mbo", 1000.0, 1500.0, config)
    assert result is not None
    text = describe(result, config.quantiles)
    assert text.startswith("by premium paid: ¥1,275–¥1,425 (median ¥1,350)")
    assert text.endswith("25th–75th percentile of 4 past mbo take-privates")


def test_book_anchor_uses_offers_for_targets_priced_like_the_company() -> None:
    study = pd.DataFrame(
        {
            "event_type": ["mbo", "parent", "third_party", "mbo", "mbo", "parent"],
            "is_take_private": [True] * 6,
            "pre_announced": [False] * 6,
            "premium": [0.4] * 6,
            "offer_pbr": [0.6, 0.7, 0.8, 5.0, 6.0, 7.0],
            "pre_pbr": [0.40, 0.45, 0.50, 3.5, 4.0, 4.5],
        }
    )
    config = OfferRangeConfig(quantiles=(0.25, 0.5, 0.75), min_offers=3, pbr_bands=(0.6, 3.0))
    result = likely_offer_range(study, "mbo", 1000.0, 2000.0, config, pbr=0.5)
    assert result is not None
    assert result.by_book == pytest.approx((1300.0, 1400.0, 1500.0))  # 0.65/0.7/0.75 × ¥2,000
    assert result.book_basis == "PBR < 0.6"
    assert "by PBR paid for targets at PBR < 0.6" in describe(result, config.quantiles)


def test_book_anchor_reports_how_many_offers_it_rests_on() -> None:
    config = OfferRangeConfig(quantiles=(0.25, 0.5, 0.75), min_offers=3)
    result = likely_offer_range(_study(), "mbo", 1000.0, 1500.0, config)
    assert result is not None and result.book_offers == 4
