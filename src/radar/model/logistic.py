"""Statistical model (v1): regularised logistic regression on the backtest panel.

Data contract: in, a panel with the feature columns of config/model.yaml (+ `market_segment`) and
`label`; out, a probability per row (`predict`) and one coefficient per design column
(`coefficients`, on standardised inputs so magnitudes are comparable).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class LogisticConfig:
    features: list[str]
    log_features: list[str]
    segments: list[str]
    c: float
    train_end: date
    test_start: date

    @classmethod
    def from_config(cls, raw: Mapping[str, Any]) -> LogisticConfig:
        return cls(
            features=list(raw["features"]),
            log_features=list(raw["log_features"]),
            segments=list(raw["segments"]),
            c=float(raw["C"]),
            train_end=raw["train_end"],
            test_start=raw["test_start"],
        )


def design_matrix(frame: pd.DataFrame, config: LogisticConfig) -> pd.DataFrame:
    """Feature table → float columns: raw features, log10 sizes, segment flags."""
    x = pd.DataFrame(
        {f: pd.to_numeric(frame[f], errors="coerce").astype(float) for f in config.features}
    )
    for column in config.log_features:
        values = pd.to_numeric(frame[column], errors="coerce").astype(float)
        x[f"log_{column}"] = np.log10(values.where(values > 0))
    for segment in config.segments:
        x[f"segment_{segment}"] = (frame["market_segment"] == segment).astype(float)
    return x.reset_index(drop=True)


def fit(train: pd.DataFrame, config: LogisticConfig) -> Pipeline:
    """Imputation (+ indicators) → scaling → L2 logistic regression, balanced classes."""
    x = design_matrix(train, config)
    model = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("logit", LogisticRegression(C=config.c, class_weight="balanced", max_iter=1000)),
        ]
    )
    model.fit(x, train["label"].to_numpy())
    return model


def predict(model: Pipeline, frame: pd.DataFrame, config: LogisticConfig) -> pd.Series:
    """Probability of becoming a target within the label horizon, aligned with `frame`."""
    proba = model.predict_proba(design_matrix(frame, config))[:, 1]
    return pd.Series(proba, index=frame.index, name="probability")


def coefficients(model: Pipeline) -> pd.DataFrame:
    """One row per design column (incl. `missing_<feature>` indicators), largest effect first."""
    imputer: SimpleImputer = model.named_steps["impute"]
    names = [str(n) for n in imputer.get_feature_names_out()]
    names = [n.replace("missingindicator_", "missing_") for n in names]
    coefs = model.named_steps["logit"].coef_[0]
    table = pd.DataFrame({"feature": names, "coefficient": coefs})
    order = table["coefficient"].abs().sort_values(ascending=False).index
    return table.loc[order].reset_index(drop=True)


def fit_by_type(
    train: pd.DataFrame, config: LogisticConfig, event_types: list[str]
) -> dict[str, Pipeline]:
    """One model per deal type, each on its own `label_<type>`."""
    return {kind: fit(train.assign(label=train[f"label_{kind}"]), config) for kind in event_types}


def combine(probabilities: Mapping[str, pd.Series]) -> pd.Series:
    """P(any offer) = 1 − Π(1 − p_type), treating the deal types as separate events."""
    none = pd.Series(1.0, index=next(iter(probabilities.values())).index)
    for p in probabilities.values():
        none = none * (1 - p)
    return (1 - none).rename("probability")


def prior_correct(probability: pd.Series, base_rate: float) -> pd.Series:
    """Undo `class_weight="balanced"`."""
    # Same as odds × π/(1 − π) back to a probability, written so that p = 1 stays 1 (the odds
    # form gives ∞/∞ = NaN for a model sure of itself, e.g. one fitted on a single date).
    weighted = probability * base_rate
    corrected: pd.Series = weighted / (weighted + (1 - probability) * (1 - base_rate))
    corrected.name = probability.name
    return corrected
