# ADR 0005 — Prices from J-Quants (Free plan), unadjusted close + split factor

Status: accepted · 2026-10-08

## Context
Valuation features (market cap, PBR, net cash / market cap) need daily prices and, for the
backtest, prices of companies that were later **taken private and delisted** — those are the
positives. The project is a prototype, so the data must be free.

Options considered:
- **J-Quants Free plan** (JPX, official): ~2 years of history, 12 weeks delayed, 5 requests/min.
  Verified window on 2026-10-08: 2024-07-16 ~ 2026-07-16. Delisted stocks remain queryable for
  dates when they were listed.
- **Unofficial free sources** (Yahoo Finance via `yfinance`, Stooq): longer history, but no
  guarantee they keep delisted tickers. Losing exactly the stocks that were bought out would make
  the valuation rules never fire on the positives and bias the backtest — the worst possible
  failure for this project. Terms of use are also unclear.
- **J-Quants Light plan** (paid): longer history; rejected for now (prototype).

Separately, the share count comes from the annual report (as of fiscal year end), while the price
is from a later day. A stock split in between would halve the market cap if ignored.

## Decision
1. Use J-Quants Free only. The backtest covers the plan's window: quarter-ends 2024-09-30 →
   2026-06-30, of which those with as_of + 365 days ≤ today have complete labels.
2. Store the **unadjusted** close (`C`) and the split factor (`AdjFactor`). Market cap =
   close × report shares × 1 / Π AdjFactor over (period_end, price date]
   (`features.valuation.split_multiplier`). `AdjC` is NOT used: it is restated to today's share
   basis, which would mix in splits that happened after `as_of` (look-ahead).
3. Market segment comes from month-end snapshots of `/equities/master?date=`, which list the
   companies listed on that date (point-in-time, delistings included).

## Revisit when
The backtest needs more years. First measure, on the J-Quants window, how many tender-offer
targets an unofficial source still has prices for; adopt it only as a supplement if coverage of
delisted targets is close to complete, else buy the Light plan.

## Consequences
Few backtest dates (5 complete quarter-ends): results will be indicative, not conclusive, and must
be reported as such. Share buybacks after the report are still ignored.
