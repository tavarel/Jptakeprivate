"""Logistic regression: design matrix, fit/predict, coefficients (synthetic panel)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from radar.config import load_yaml
from radar.model.logistic import (
    LogisticConfig,
    coefficients,
    combine,
    design_matrix,
    fit,
    fit_by_type,
    predict,
)

CONFIG = LogisticConfig(
    features=["age", "noise"],
    log_features=["market_cap_jpy"],
    segments=["Standard"],
    c=1.0,
    train_end=date(2023, 12, 31),
    test_start=date(2025, 1, 1),
)


def _panel(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.uniform(40, 85, n)
    return pd.DataFrame(
        {
            "age": age,
            "noise": rng.normal(size=n),
            "market_cap_jpy": rng.uniform(1e9, 1e11, n),
            "market_segment": rng.choice(["Prime", "Standard"], n),
            "label": (age + rng.normal(scale=5, size=n) > 75).astype(int),
        }
    )


def test_real_config_loads() -> None:
    config = LogisticConfig.from_config(load_yaml("model"))
    assert (config.test_start - config.train_end).days >= 365  # gap ≥ label horizon
    assert "pbr" in config.features


def test_design_matrix_keeps_missing_as_nan() -> None:
    frame = pd.DataFrame(
        {
            "age": [70.0, None],
            "noise": [0.0, 1.0],
            "market_cap_jpy": [1e10, 0.0],  # 0 cannot be logged → missing, not -inf
            "market_segment": ["Standard", None],
        }
    )
    x = design_matrix(frame, CONFIG)
    assert list(x.columns) == ["age", "noise", "log_market_cap_jpy", "segment_Standard"]
    assert pd.isna(x.loc[1, "age"])
    assert x.loc[0, "log_market_cap_jpy"] == pytest.approx(10.0)
    assert pd.isna(x.loc[1, "log_market_cap_jpy"])
    assert x["segment_Standard"].tolist() == [1.0, 0.0]


def test_fit_learns_the_signal_and_ranks_targets_first() -> None:
    train, test = _panel(seed=0), _panel(seed=1)
    model = fit(train, CONFIG)
    proba = predict(model, test, CONFIG)
    assert proba.between(0, 1).all()
    assert proba[test["label"] == 1].mean() > proba[test["label"] == 0].mean()
    coefs = coefficients(model).set_index("feature")["coefficient"]
    assert coefs["age"] > 0  # older leader → higher probability, as generated
    assert abs(coefs["age"]) > abs(coefs["noise"])


def test_missing_values_get_an_indicator_coefficient() -> None:
    train = _panel()
    train.loc[train.index[:50], "age"] = None
    coefs = coefficients(fit(train, CONFIG))
    assert "missing_age" in set(coefs["feature"])


def test_one_model_per_deal_type_finds_its_own_driver() -> None:
    rng = np.random.default_rng(3)
    n = 600
    frame = pd.DataFrame(
        {
            "age": rng.uniform(40, 85, n),  # drives MBOs
            "noise": rng.normal(size=n),  # drives parent deals in this synthetic panel
            "market_cap_jpy": rng.uniform(1e9, 1e11, n),
            "market_segment": "Standard",
        }
    )
    frame["label_mbo"] = (frame["age"] + rng.normal(scale=4, size=n) > 78).astype(int)
    frame["label_parent"] = (frame["noise"] + rng.normal(scale=0.3, size=n) > 1.3).astype(int)
    models = fit_by_type(frame, CONFIG, ["mbo", "parent"])
    mbo = coefficients(models["mbo"]).set_index("feature")["coefficient"]
    parent = coefficients(models["parent"]).set_index("feature")["coefficient"]
    assert abs(mbo["age"]) > abs(mbo["noise"])
    assert abs(parent["noise"]) > abs(parent["age"])

    per_type = {kind: predict(model, frame, CONFIG) for kind, model in models.items()}
    combined = combine(per_type)
    assert combined.between(0, 1).all()
    assert (combined >= per_type["mbo"] - 1e-12).all() and (
        combined >= per_type["parent"] - 1e-12
    ).all()


def test_combine_is_the_probability_of_any_offer() -> None:
    combined = combine({"a": pd.Series([0.5, 0.0]), "b": pd.Series([0.5, 0.2])})
    assert combined.tolist() == pytest.approx([0.75, 0.2])  # 1 − (1 − a)(1 − b)
