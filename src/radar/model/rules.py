"""Explainable rule-based score (v0)."""

from __future__ import annotations

import operator
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression

from radar.config import CONFIG_DIR, load_yaml

_OPS: dict[str, Callable[[Any, Any], bool]] = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "==": operator.eq,
    "!=": operator.ne,
}


class Rule(BaseModel):
    id: str
    feature: str
    op: str
    threshold: float | int | str | bool
    weight: float
    rationale: str = ""

    def triggered(self, value: Any) -> bool:
        """False when the feature is missing — missing data never adds points."""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return False
        try:
            return bool(_OPS[self.op](value, self.threshold))
        except TypeError:
            return False


class RuleSet(BaseModel):
    version: int
    rules: list[Rule]

    @classmethod
    def load(cls, name: str = "scoring_v0", config_dir: Path = CONFIG_DIR) -> RuleSet:
        ruleset = cls.model_validate(load_yaml(name, config_dir))
        unknown = {r.op for r in ruleset.rules} - _OPS.keys()
        if unknown:
            raise ValueError(f"Unknown operators in {name}: {unknown}")
        return ruleset

    @classmethod
    def default(cls, config_dir: Path = CONFIG_DIR) -> RuleSet:
        """The rule set used by the daily screen (config/scoring.yaml `default_ruleset`)."""
        return cls.load(default_ruleset_name(config_dir), config_dir)

    @property
    def min_score(self) -> float:
        """Points of a company firing every NEGATIVE rule (0 for hand-set rule sets)."""
        return sum(min(r.weight, 0.0) for r in self.rules)

    @property
    def max_score(self) -> float:
        """Points of a company firing every positive rule."""
        return sum(max(r.weight, 0.0) for r in self.rules)


def default_ruleset_name(config_dir: Path = CONFIG_DIR) -> str:
    """Name of the rule set chosen as default."""
    return str(load_yaml("scoring", config_dir)["default_ruleset"])


def score(features: pd.DataFrame, ruleset: RuleSet) -> pd.DataFrame:
    """Add `score` (0–1, share of the maximum) and `triggered_rules` (list of rule ids)."""
    out = features.copy()
    triggered: list[list[str]] = []
    raw: list[float] = []
    for _, row in out.iterrows():
        hits = [r for r in ruleset.rules if r.triggered(row.get(r.feature))]
        triggered.append([r.id for r in hits])
        raw.append(sum(r.weight for r in hits))
    out["score"] = [s / ruleset.max_score for s in raw] if ruleset.max_score else 0.0
    out["triggered_rules"] = triggered
    return out.sort_values("score", ascending=False).reset_index(drop=True)


def audit_scores(scores: pd.DataFrame, ruleset: RuleSet, tolerance: float = 1e-9) -> pd.DataFrame:
    """Recompute every rule and the score from the feature columns of a scores table."""
    issues: list[tuple[str, str, str]] = []
    for row in scores.to_dict("records"):
        rules = row.get("triggered_rules")  # a numpy array after a Parquet round trip
        stored = {str(r) for r in rules} if rules is not None else set()
        fresh = {r.id for r in ruleset.rules if r.triggered(row.get(r.feature))}
        code = str(row.get("edinet_code"))
        if stored != fresh:
            detail = (
                f"stored only: {sorted(stored - fresh)}, recomputed only: {sorted(fresh - stored)}"
            )
            issues.append((code, "triggered_rules", detail))
        expected = sum(r.weight for r in ruleset.rules if r.id in fresh) / ruleset.max_score
        if abs(float(row["score"]) - expected) > tolerance:
            issues.append((code, "score", f"stored {row['score']:.4f} vs {expected:.4f}"))
    return pd.DataFrame(issues, columns=["edinet_code", "check", "detail"])


def rule_flags(features: pd.DataFrame, ruleset: RuleSet) -> pd.DataFrame:
    """0/1 per rule (column = rule id): the rules as inputs of a model."""
    flags = {}
    for rule in ruleset.rules:
        if rule.feature in features.columns:
            flags[rule.id] = [int(rule.triggered(v)) for v in features[rule.feature]]
        else:
            flags[rule.id] = [0] * len(features)
    return pd.DataFrame(flags, index=features.index)


def fit_rule_model(panel: pd.DataFrame, ruleset: RuleSet, label: str, c: float) -> Any:
    """L2 logistic on the 0/1 rule flags."""
    model = LogisticRegression(C=c, class_weight="balanced", max_iter=1000)
    model.fit(rule_flags(panel, ruleset).to_numpy(), panel[label].to_numpy())
    return model


def with_weights(ruleset: RuleSet, weights: dict[str, float]) -> RuleSet:
    """A copy of `ruleset` whose weights are replaced (thresholds and rationales unchanged)."""
    rules = [r.model_copy(update={"weight": float(weights[r.id])}) for r in ruleset.rules]
    return RuleSet(version=ruleset.version, rules=rules)


def bootstrap_rule_weights(
    panel: pd.DataFrame, ruleset: RuleSet, label: str, *, c: float, draws: int, seed: int
) -> pd.DataFrame:
    """How stable are the learned weights?"""
    rng = np.random.default_rng(seed)
    dates = np.array(sorted(panel["as_of"].unique()))
    by_date = {d: group for d, group in panel.groupby("as_of")}
    samples = []
    for _ in range(draws):
        chosen = rng.choice(dates, size=len(dates), replace=True)
        sample = pd.concat([by_date[d] for d in chosen], ignore_index=True)
        if sample[label].nunique() < 2:
            continue
        samples.append(fit_rule_model(sample, ruleset, label, c).coef_[0])
    weights = pd.DataFrame(samples, columns=[r.id for r in ruleset.rules])
    return pd.DataFrame(
        {
            "rule": weights.columns,
            "p5": weights.quantile(0.05).to_numpy(),
            "p50": weights.quantile(0.50).to_numpy(),
            "p95": weights.quantile(0.95).to_numpy(),
            "share_positive": (weights > 0).mean().to_numpy(),
            "draws": len(weights),
        }
    )
