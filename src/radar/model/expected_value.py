"""Expected 12-month value of holding a company."""

from __future__ import annotations

import logging
from collections.abc import Mapping

import pandas as pd
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)


def expected_value(
    probabilities: pd.DataFrame, premiums: Mapping[str, float], no_offer_return: float
) -> pd.Series:
    """probabilities: one column per deal type (calibrated P), one row per company."""
    kinds = [k for k in probabilities.columns if k in premiums]
    p = probabilities[kinds].astype(float)
    gain = sum(p[k] * premiums[k] for k in kinds)
    none = 1 - p.sum(axis=1, min_count=len(kinds))
    ev = gain + none * no_offer_return
    return pd.Series(ev, index=probabilities.index, name="expected_value").where(
        p.notna().all(axis=1)
    )


def reliability_table(probability: pd.Series, observed: pd.Series, bins: int) -> pd.DataFrame:
    """Mean predicted vs observed offer rate per probability bin (calibration check only)."""
    frame = pd.DataFrame({"p": probability.astype(float), "y": observed.astype(float)}).dropna()
    frame["bin"] = pd.qcut(frame["p"].rank(method="first"), bins, labels=False)
    table = frame.groupby("bin").agg(
        companies=("y", "size"), predicted=("p", "mean"), observed=("y", "mean")
    )
    return table.reset_index(drop=True)


def fit_isotonic(probability: pd.Series, observed: pd.Series) -> IsotonicRegression:
    """Monotone map from a model's probability to the rate observed on a HELD-OUT period."""
    known = probability.astype(float).notna()
    if not known.all():
        logger.warning("isotonic fit: %d rows without a probability left out", int((~known).sum()))
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(
        probability[known].astype(float).to_numpy(), observed[known].astype(float).to_numpy()
    )
    return calibrator


def apply_isotonic(calibrator: IsotonicRegression, probability: pd.Series) -> pd.Series:
    """Calibrated probability; an unknown input stays unknown (NaN), never 0."""
    values = probability.astype(float)
    known = values.notna()
    result = pd.Series(float("nan"), index=probability.index, name=probability.name)
    if known.any():
        result[known] = calibrator.predict(values[known].to_numpy())
    return result


def brier(probability: pd.Series, observed: pd.Series) -> float:
    """Mean squared error of a probability (lower is better; 0 = perfect)."""
    return float(((probability.astype(float) - observed.astype(float)) ** 2).mean())
