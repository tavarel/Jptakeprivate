"""Founding-family control signals."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from radar.text_ja import normalize, split_surname, unify_kanji_variants


@dataclass(frozen=True)
class FamilyStake:
    stake: float | None  # None = unknown (no surname or no shareholder table), never 0
    surnames: frozenset[str]
    matched_holders: list[str] = field(default_factory=list)

    @property
    def founder_in_control(self) -> bool:
        """True if at least one representative's surname appears among the top shareholders."""
        return bool(self.matched_holders)


def family_surnames(
    officers: pd.DataFrame, known_surnames: frozenset[str] = frozenset()
) -> frozenset[str]:
    reps = officers[officers["is_representative"].astype(bool)]
    surnames = {split_surname(name, known_surnames) for name in reps["name"].dropna()}
    return frozenset(s for s in surnames if s)


def family_stake(
    shareholders: pd.DataFrame,
    surnames: frozenset[str],
    known_surnames: frozenset[str] = frozenset(),
    excluded: tuple[str, ...] = (),
) -> FamilyStake:
    """Sum of ownership ratios of top shareholders that look family-controlled."""
    if not surnames or shareholders.empty:
        return FamilyStake(stake=None, surnames=surnames)

    unified = {unify_kanji_variants(s) for s in surnames}
    total = 0.0
    matched: list[str] = []
    for row in shareholders.to_dict("records"):
        name = normalize(str(row["name"]))
        if "自己株" in name or any(e in name for e in excluded):
            continue
        if row["is_organization"]:
            is_family = any(s in unify_kanji_variants(name) for s in unified)
        else:
            surname = split_surname(name, known_surnames)
            is_family = surname is not None and unify_kanji_variants(surname) in unified
        ratio = row["ratio"]
        if is_family and ratio is not None and not pd.isna(ratio):
            total += float(ratio)
            matched.append(name)
    return FamilyStake(stake=round(total, 6), surnames=surnames, matched_holders=matched)
