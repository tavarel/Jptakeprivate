"""Calibrated probabilities and expected value per company."""

from __future__ import annotations

import pandas as pd
import pytest

from radar.model.expected_value import (
    apply_isotonic,
    brier,
    expected_value,
    fit_isotonic,
    reliability_table,
)
from radar.model.logistic import prior_correct


def test_prior_correction_undoes_balanced_class_weights() -> None:
    # Balanced weights train as if half the companies were targets; the real rate is 2%.
    p = pd.Series([0.5, 0.9])
    corrected = prior_correct(p, base_rate=0.02)
    # odds 1 × 0.02/0.98 → 0.02; odds 9 × 0.0204 → 0.155
    assert corrected.tolist() == pytest.approx([0.02, 9 * 0.02 / 0.98 / (1 + 9 * 0.02 / 0.98)])
    assert (corrected <= p).all()


def test_expected_value_mixes_offer_premiums_and_the_no_offer_return() -> None:
    probabilities = pd.DataFrame({"mbo": [0.10, None], "parent": [0.05, 0.05]})
    premiums = {"mbo": 0.40, "parent": 0.30}
    ev = expected_value(probabilities, premiums, no_offer_return=0.05)
    assert ev[0] == pytest.approx(0.10 * 0.40 + 0.05 * 0.30 + 0.85 * 0.05)
    assert pd.isna(ev[1])  # an unknown probability → unknown value, never 0


def test_reliability_table_compares_predicted_and_observed_by_decile() -> None:
    p = pd.Series([0.01] * 50 + [0.10] * 50)
    observed = pd.Series([0] * 50 + [1] * 5 + [0] * 45)
    table = reliability_table(p, observed, bins=2)
    assert table["predicted"].tolist() == pytest.approx([0.01, 0.10])
    assert table["observed"].tolist() == pytest.approx([0.0, 0.10])


def test_isotonic_calibration_maps_scores_to_observed_rates() -> None:
    # Held-out year: the raw model says 40% where only 5% happened, 2% where 1% happened.
    raw = pd.Series([0.40] * 100 + [0.02] * 100)
    observed = pd.Series([1] * 5 + [0] * 95 + [1] * 1 + [0] * 99)
    calibrator = fit_isotonic(raw, observed)
    mapped = apply_isotonic(calibrator, pd.Series([0.40, 0.02, 0.90]))
    assert mapped.tolist()[:2] == pytest.approx([0.05, 0.01])
    assert mapped.iloc[2] == pytest.approx(0.05)  # Beyond the clipped, not invented


def test_brier_score() -> None:
    assert brier(pd.Series([1.0, 0.0]), pd.Series([1, 0])) == 0.0
    assert brier(pd.Series([0.5, 0.5]), pd.Series([1, 0])) == pytest.approx(0.25)


def test_prior_correction_keeps_a_certain_model_certain() -> None:
    # A model fitted on one date can be sure (p = 1.0 or 0.0): the odds form gave ∞/∞ = NaN.
    corrected = prior_correct(pd.Series([1.0, 0.0, 0.5]), base_rate=0.02)
    assert corrected.tolist() == pytest.approx([1.0, 0.0, 0.02])


def test_isotonic_skips_unknown_probabilities_instead_of_crashing() -> None:
    p = pd.Series([0.1, None, 0.3, 0.9])
    observed = pd.Series([0, 1, 0, 1])
    calibrator = fit_isotonic(p, observed)  # the unknown row is left out of the fit
    applied = apply_isotonic(calibrator, pd.Series([0.2, None]))
    assert applied.iloc[0] == pytest.approx(0.0)
    assert pd.isna(applied.iloc[1])  # unknown stays unknown, never 0
