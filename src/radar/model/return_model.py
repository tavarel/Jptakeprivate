"""Return model (portfolio question): predict the 12-month return in EXCESS of the all-companies
mean, instead of the chance of an offer.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def fit_return_model(
    train: pd.DataFrame, excess: pd.Series, columns: Sequence[str], alpha: float, clip: float
) -> Pipeline:
    """Ridge on `columns` → excess return, targets clipped at the `clip` / 1 − `clip` quantiles."""
    target = excess.astype(float)
    low, high = float(target.quantile(clip)), float(target.quantile(1 - clip))
    model = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )
    model.fit(train[list(columns)].astype(float), target.clip(low, high).to_numpy())
    return model


def predict_return(model: Pipeline, frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    return np.asarray(model.predict(frame[list(columns)].astype(float)), dtype=float)
