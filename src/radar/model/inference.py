"""Statistics for few, overlapping test dates: Newey-West, deflated Sharpe, style benchmark."""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

EULER_GAMMA = 0.5772156649
STYLE_COLUMNS = ("market_cap_jpy", "pbr")


def newey_west_se(values: pd.Series, lags: int) -> float:
    """Standard error of the mean of `values` with Newey-West (Bartlett) autocorrelation terms."""
    x = values.dropna().astype(float).to_numpy()
    n = len(x)
    if n < 2:
        return float("nan")
    centred = x - x.mean()
    variance = float(centred @ centred) / n
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1 - lag / (lags + 1)
        variance += 2 * weight * float(centred[lag:] @ centred[:-lag]) / n
    return math.sqrt(max(variance, 0.0) / n)


def deflated_sharpe(
    sharpe: float,
    *,
    n_obs: int,
    n_trials: int,
    trials_variance: float,
    skew: float,
    kurtosis: float,
) -> float:
    """P(true Sharpe > the best Sharpe expected from `n_trials` lucky tries)."""
    normal = NormalDist()
    expected_max = 0.0
    if n_trials > 1 and trials_variance > 0:
        expected_max = math.sqrt(trials_variance) * (
            (1 - EULER_GAMMA) * normal.inv_cdf(1 - 1 / n_trials)
            + EULER_GAMMA * normal.inv_cdf(1 - 1 / (n_trials * math.e))
        )
    spread = 1 - skew * sharpe + (kurtosis - 1) / 4 * sharpe**2
    if n_obs < 2 or spread <= 0 or not math.isfinite(sharpe):
        return float("nan")
    return normal.cdf((sharpe - expected_max) * math.sqrt(n_obs - 1) / math.sqrt(spread))


def style_benchmark(frame: pd.DataFrame, buckets: int) -> pd.Series:
    """Mean `realized_return` of the companies in the same size × price-to-book bucket on the same
    date (quantiles within the date).

    Columns in: as_of, market_cap_jpy, pbr, realized_return. Aligned with `frame`.
    """
    keys = [frame["as_of"]]
    for column in STYLE_COLUMNS:
        pct = frame.groupby("as_of")[column].rank(pct=True)
        keys.append(np.ceil(pct * buckets).fillna(-1))  # -1 = unknown characteristic
    return frame.groupby(keys)["realized_return"].transform("mean")
