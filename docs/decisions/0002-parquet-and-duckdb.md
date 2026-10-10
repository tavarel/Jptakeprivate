# ADR 0002 — Parquet files + DuckDB, no database server

Status: accepted · 2026-10-07

## Context
~4,000 companies × a few years of filings and daily prices: millions of rows at most. Single user,
laptop, free.

## Decision
Store tables as Parquet in a bronze/silver/gold layout; query them with DuckDB views.

## Consequences
Zero infrastructure, fast analytics, files are easy to inspect and rebuild. No concurrent writes
(not needed). If the project ever becomes a shared service, migrate gold tables to Postgres.
