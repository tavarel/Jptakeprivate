# ADR 0004 — Explainable rules before machine learning

Status: accepted · 2026-10-07

## Context
Positives are rare (~1–3%/year), labels are noisy, and the audience (M&A, PE, funds) needs to
understand *why* a company is flagged.

## Decision
v0 is a weighted rule set in YAML. A model (logistic regression) comes only after v0 is
backtested, and must beat v0 on the same out-of-time test set to replace it.

## Consequences
A transparent baseline from week 6; ML effort is justified by measured lift, not by fashion.
