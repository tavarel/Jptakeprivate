# Architecture

## Goals that drive the design
1. **Trustworthy backtest** — no look-ahead bias, reproducible from raw data.
2. **Explainable output** — every score comes with the rules/signals that produced it.
3. **Cheap to run** — a laptop, free APIs, no servers. DuckDB + Parquet instead of a database server.
4. **Easy to extend** — new signals = new pure function + config entry + test.

## Layers

```
cli.py            argument parsing, printing                         (presentation)
pipeline.py       orchestration: read storage → call pure code → write (application)
ingest/           HTTP clients, raw downloads, caching                  (infrastructure)
parse/            raw files → tidy DataFrames                           (pure)
features/         point-in-time signals per company                     (pure)
labels/           ground-truth events and labels                        (pure)
model/            scoring rules, statistical models, backtest metrics   (pure)
report/           dashboard, memo template                              (presentation)
storage/          paths and DuckDB views                                (infrastructure)
config.py, log.py settings, YAML config, logging
```

Dependencies point downwards only. Pure modules never touch the network, the file system or
settings — that is what makes them trivially testable and lets the same code run in the CLI,
a notebook or a scheduled job.

## Data flow and storage (medallion layout)

| Layer | Location | Content | Rule |
|---|---|---|---|
| Bronze | `data/raw/` | exact API responses (JSON lists, ZIPs) | immutable, write-once cache |
| Silver | `data/staging/*.parquet` | typed tables: documents, officers, shareholders, financials, prices, capital_plans | rebuildable from bronze |
| Gold | `data/marts/*.parquet` | events, features_<as_of>, scores_<as_of>, backtest results | rebuildable from silver |
| — | `data/radar.duckdb` | SQL views over silver + gold | disposable |

Every silver/gold row keeps its lineage (`doc_id`, `edinet_code`, `as_of`) so any number on the
dashboard can be traced back to the filing it came from.

## Point-in-time contract
- Filing usable at T ⇔ `submit_datetime < T 00:00 JST` and not withdrawn. Amendment (130) replaces
  its original only once the amendment itself is public.
- Price used at T = last close ≤ T − 1 day, and not older than 10 days (suspended stocks drop out).
- JPX lists: the latest list dated before T.
- Label at T = 1 if a third-party tender offer for the company is filed in (T, T + 365 days].
- Evaluation: thresholds chosen on early years, metrics reported on later years only
  (`model.backtest.time_split`, with a gap ≥ the label horizon).

## External constraints
- EDINET: one document list per day (no bulk endpoint); key passed as URL query parameter;
  missing documents return HTTP 200 with a JSON error body; CSV files are UTF-16 TSV.
- J-Quants V2: `x-api-key` header; free plan ≈ 5 requests/minute, 12-week delay, 2 years history.
- Be polite: default 0.5 requests/second, exponential backoff on 429/5xx only.

See `docs/decisions/` for the reasoning behind the main choices.
