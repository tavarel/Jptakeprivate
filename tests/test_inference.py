"""Honest statistics for few, overlapping dates: Newey-West, deflated Sharpe, style benchmark."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from radar.model.inference import deflated_sharpe, newey_west_se, style_benchmark


def test_newey_west_of_a_constant_series_is_zero() -> None:
    assert newey_west_se(pd.Series([0.05] * 7), lags=3) == pytest.approx(0.0)


def test_newey_west_widens_the_error_of_an_overlapping_series() -> None:
    # Overlapping 12-month windows move together: a smooth series has positive autocorrelation,
    # so its true error is LARGER than the naive s/√n.
    smooth = pd.Series([0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12])
    naive = float(smooth.std(ddof=1)) / len(smooth) ** 0.5
    assert newey_west_se(smooth, lags=0) == pytest.approx(naive * ((6 / 7) ** 0.5))
    assert newey_west_se(smooth, lags=3) > newey_west_se(smooth, lags=0)


def test_newey_west_needs_two_points() -> None:
    assert pd.isna(newey_west_se(pd.Series([0.1]), lags=3))


def test_deflated_sharpe_falls_as_more_strategies_were_tried() -> None:
    few = deflated_sharpe(0.8, n_obs=7, n_trials=2, trials_variance=0.1, skew=0.0, kurtosis=3.0)
    many = deflated_sharpe(0.8, n_obs=7, n_trials=60, trials_variance=0.1, skew=0.0, kurtosis=3.0)
    assert 0.0 <= many < few <= 1.0


def test_deflated_sharpe_of_one_trial_is_the_plain_probability_of_skill() -> None:
    # One trial → no selection haircut, only the probabilistic Sharpe ratio:
    # Φ(SR·√(T−1) / √(1 − skew·SR + (kurt − 1)/4·SR²)) = Φ(0.5·2 / √1.125) = Φ(0.943).
    value = deflated_sharpe(0.5, n_obs=5, n_trials=1, trials_variance=0.0, skew=0.0, kurtosis=3.0)
    assert value == pytest.approx(0.8271, abs=1e-3)


def test_style_benchmark_is_the_mean_of_the_same_size_and_book_bucket() -> None:
    day = date(2024, 3, 31)
    frame = pd.DataFrame(
        {
            "as_of": [day] * 4,
            "market_cap_jpy": [1e9, 1.1e9, 9e10, 1e11],
            "pbr": [0.5, 0.6, 2.0, 2.5],
            "realized_return": [0.10, 0.30, -0.10, None],
        }
    )
    bench = style_benchmark(frame, buckets=2)
    # Rows 0 and 1 share the small / cheap bucket: benchmark = their mean (unpriced rows ignored).
    assert bench.tolist()[:2] == pytest.approx([0.20, 0.20])
    assert bench[2] == pytest.approx(-0.10) and bench[3] == pytest.approx(-0.10)


def test_style_benchmark_puts_unknown_characteristics_in_their_own_bucket() -> None:
    frame = pd.DataFrame(
        {
            "as_of": [date(2024, 3, 31)] * 3,
            "market_cap_jpy": [1e9, 2e9, None],
            "pbr": [0.5, 0.5, 0.5],
            "realized_return": [0.1, 0.1, 0.4],
        }
    )
    assert style_benchmark(frame, buckets=2)[2] == pytest.approx(0.4)
