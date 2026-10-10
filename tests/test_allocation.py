"""1/3-Kelly sizing and liquidity-capped positions."""

from __future__ import annotations

import pandas as pd
import pytest

from radar.model.allocation import AllocationConfig, allocate, kelly_fraction

CONFIG = AllocationConfig(
    kelly_multiplier=1 / 3,
    risk_free=0.0,
    max_invested=1.0,
    max_weight=0.10,
    participation=0.10,
    max_days=5,
    lot_size=100,
    min_dates=5,
)


def test_kelly_is_a_third_of_mean_over_variance() -> None:
    returns = pd.Series([0.10, 0.30, 0.10, 0.30, 0.20])  # mean 0.2, variance 0.01
    assert kelly_fraction(returns, CONFIG) == pytest.approx(min(1.0, (0.2 / 0.01) / 3))
    noisy = pd.Series([0.6, -0.4, 0.5, -0.3, 0.1])  # mean 0.1, variance 0.205
    assert kelly_fraction(noisy, CONFIG) == pytest.approx(0.1 / 0.205 / 3)


def test_kelly_never_shorts_and_needs_enough_dates() -> None:
    assert kelly_fraction(pd.Series([-0.1, -0.2, 0.0, -0.1, 0.05]), CONFIG) == 0.0
    assert kelly_fraction(pd.Series([0.2, 0.3]), CONFIG) is None  # too few dates to say


def test_allocate_equal_weights_capped_by_liquidity_in_lots() -> None:
    picks = pd.DataFrame(
        {
            "sec_code": ["1001", "1002", "1003"],
            "close": [1000.0, 500.0, 2000.0],
            "median_trading_value_jpy": [1e9, 1e6, None],
        }
    )
    plan = allocate(picks, capital=3_000_000, invested=0.9, config=CONFIG).set_index("sec_code")
    # Target ¥900k each, capped at 10% of capital (¥300k); 1002 trades ¥1m a day → 10% × 5 days.
    assert plan.loc["1001", "target_jpy"] == pytest.approx(300_000)
    assert plan.loc["1002", "target_jpy"] == pytest.approx(300_000)  # ¥500k liquidity cap > ¥300k
    assert plan.loc["1001", "shares"] == 300  # whole 100-share lots: ¥300k / ¥1,000
    assert plan.loc["1002", "shares"] == 600
    assert plan.loc["1002", "days_to_trade"] == pytest.approx(300_000 / (0.10 * 1e6))
    # Unknown liquidity is not "unlimited": no position.
    assert plan.loc["1003", "shares"] == 0 and plan.loc["1003", "reason"] == "liquidity unknown"


def test_illiquid_stock_is_capped_by_days_of_volume() -> None:
    picks = pd.DataFrame(
        {"sec_code": ["2001"], "close": [100.0], "median_trading_value_jpy": [200_000.0]}
    )
    plan = allocate(picks, capital=10_000_000, invested=1.0, config=CONFIG)
    # 10% of ¥200k a day for 5 days = ¥100k, far below the ¥1m weight cap.
    assert plan.loc[0, "target_jpy"] == pytest.approx(100_000)
    assert plan.loc[0, "shares"] == 1000
    assert plan.loc[0, "reason"] == "liquidity cap"


def test_kelly_on_excess_measures_the_edge_not_the_market() -> None:
    # A bull market: the basket +20% a year, but all companies +18% → the edge is 2 pp.
    basket = pd.Series([0.10, 0.30, 0.10, 0.30, 0.20])
    market = basket - pd.Series([0.01, 0.03, 0.01, 0.03, 0.02])
    raw = kelly_fraction(basket, CONFIG)
    edge = kelly_fraction(basket, CONFIG, benchmark=market)
    assert raw == pytest.approx(1.0)  # the market alone fills the cap
    excess = basket - market
    assert edge == pytest.approx(min(1.0, excess.mean() / excess.var() / 3))


def test_kelly_with_a_confidence_haircut_invests_nothing_on_weak_evidence() -> None:
    careful = AllocationConfig(**{**CONFIG.__dict__, "confidence_z": 1.0, "nw_lags": 3})
    # Overlapping windows drift together: mean +4.3%, Newey-West error ≈ 7% → no evidence.
    drifting = pd.Series([0.30, 0.25, 0.10, -0.20, -0.25, -0.05, 0.15])
    assert kelly_fraction(drifting, careful) == 0.0
    steady = pd.Series([0.10, 0.12, 0.11, 0.13, 0.12])
    assert kelly_fraction(steady, careful) == pytest.approx(1.0)  # strong evidence: capped
