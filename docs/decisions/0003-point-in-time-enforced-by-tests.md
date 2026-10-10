# ADR 0003 — Point-in-time correctness is enforced by tests

Status: accepted · 2026-10-07

## Context
The value of the project is the backtest. Look-ahead bias (using data not yet public at the
scoring date) would make results look excellent and be worthless.

## Decision
All date filtering goes through `features/point_in_time.py` (filings strictly before T, prices
lagged ≥ 1 day) and `labels/` (events strictly after T). Each of these rules has a unit test with
a deliberate "trap" row dated exactly at T.

## Consequences
Slightly more conservative features (e.g. a filing published in the morning of T is ignored
until T+1). Accepted.
