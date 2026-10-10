# Data dictionary

Keep this file in sync with the code: any new table or column must be documented here
in the same change.

## staging/documents.parquet — EDINET document index
| column | type | description |
|---|---|---|
| doc_id | str | EDINET document ID (`S100…`), primary key |
| edinet_code | str | filer's EDINET code (`E#####`) |
| sec_code | str? | 5-digit securities code (listed companies only; ticker = first 4 digits) |
| filer_name | str | filer name (Japanese) |
| doc_type_code | str | 120 annual, 130 amended annual, 240 tender offer, 350 large holding… |
| doc_description | str | free-text title |
| submit_datetime | datetime | submission time (JST, minute precision) — **point-in-time key** |
| period_end | date? | fiscal period end (annual reports) |
| issuer_edinet_code | str? | issuer (for some filings by third parties) |
| subject_edinet_code | str? | target company (tender offers, large-shareholding reports) |
| parent_doc_id | str? | original document of an amendment |
| withdrawal_status | str | "0" active, "1" withdrawal notice, "2" withdrawn |
| xbrl_flag, csv_flag | bool | whether XBRL / CSV versions exist |

## staging/officers.parquet — 役員の状況
| column | type | description |
|---|---|---|
| doc_id, edinet_code | str | lineage |
| person_key | str | XBRL context ID of the person within the filing (one row per person; directors and 執行役 merged) |
| name | str | NFKC-normalised name (`山田 太郎`); per-character spacing removed (`伊 藤 浩 一` → `伊藤浩一`) |
| title | str? | job title (`代表取締役社長`); a person in both tables gets both, joined by ` / ` |
| date_of_birth | date? | parsed from ISO, Gregorian or era format |
| shares_held | float? | shares held by the officer |
| is_representative | bool | 代表取締役 / 代表執行役 |
| role | str | `chair` / `president` / `other` |

## staging/shareholders.parquet — 大株主の状況
| column | type | description |
|---|---|---|
| doc_id, edinet_code | str | lineage |
| rank | int? | 1–10 |
| name | str | normalised holder name |
| shares_held | float? | shares |
| ratio | float? | fraction of shares outstanding (0–1, excluding treasury) |
| is_organization | bool | heuristic: company/fund vs. natural person |
| address | str? | holder's address as filed (NFKC); matched against the checked `homonyms` list |

## staging/financials.parquet
One row per filing. Columns = keys of `financials` and `shares` in `config/xbrl_elements.yaml`
(JPY or share counts, `None` if not reported) plus `is_consolidated`, `is_jgaap`, `doc_id`,
`edinet_code`, `period_end`.
`is_consolidated` (bool?) comes from `jpdei_cor:WhetherConsolidatedFinancialStatementsArePreparedDEI`.
`is_jgaap` (bool?) comes from `jpdei_cor:AccountingStandardsDEI`; `None` if the filing does not
state its standard. When `False` (IFRS / US GAAP) every `financials` item is `None` on purpose:
those filers' jppfs_cor facts are parent-only. Share counts are kept.
All amount / share-count columns are float64 (NaN = not reported), even when a whole column is
empty. `issued` is at fiscal year end; `issued_at_filing` is on the filing date (they differ after
a split in between) — officers' holdings are as of the filing date, so `officer_stake` uses it.
Treasury shares are only reported at year end; `features.valuation.treasury_at_filing` carries
them to the filing date (cancellation one for one, split by the ratio, new shares unchanged).
`hq_address` (str?) is the head-office address from the cover page (本店の所在の場所).
`rental_book_value`, `rental_fair_value` (float?, yen, current year) and `rental_note_status`
(disclosed / immaterial / none / absent / unreadable; None for non-J-GAAP) come from the
賃貸等不動産 text note (`parse/rental_property.py`); unreadable = horizontal tables, unknown unit
or ambiguous digits — never guessed.
`period_end` (date) comes from the document index; amendments (130) have none.
`dividend_per_share` (float?, yen) is 1株当たり配当額 of the current year from the five-year summary
(`per_share` in config/xbrl_elements.yaml), the FILER's own figure (non-consolidated context
first), read for every accounting standard, on that year's share basis; an empty cell is None
(non-payer and unreported look the same). Added 2026-10-10.

`radar parse` rebuilds officers / shareholders / financials from `data/raw` on every run (rows
sorted by `doc_id`), so re-running gives identical files. Unreadable ZIPs are listed, not fatal.

## staging/prices.parquet
| column | type | description |
|---|---|---|
| code | str | 4/5-digit security code |
| date | date | trading day |
| close | float | UNADJUSTED close (`C`); NaN if no trade that day |
| adj_factor | float | J-Quants `AdjFactor`: 0.5 on the ex-date of a 1:2 split, 1.0 otherwise (ADR 0005) |

Source: `radar prices` (J-Quants `/equities/bars/daily?date=`, all stocks per weekday).

`code` must equal `documents.sec_code` (5 digits) for the feature join.
Optional until M3: without this table the valuation features are missing.

## staging/listings.parquet — optional
`code` (str), `date` (date of the snapshot), `company_name` (str), `market_segment` (str:
Prime / Standard / Growth, other names kept as given), `sector33` (str: TSE 33-sector name,
J-Quants `S33Nm`). Source: `radar listings`, month-end snapshots of
J-Quants `/equities/master?date=`. Features use the latest row with `date < as_of`.

## staging/capital_plans.parquet — optional
| column | type | description |
|---|---|---|
| list_month | date | first day of the month the list describes (state at month end) |
| published_on | date | first day of month + 2: conservative availability date (published ~15th of M+1) |
| sec_code | str | 5 characters (JPX `7203` → `72030`) |
| company_name, market_segment | str | as written by JPX |
| status | str | `開示済` (disclosed) or `検討中` (under consideration) |

Only `開示済` rows published before `as_of` count. Before any list is public, or without the
table, `capital_plan_disclosed` is missing (not False). Source: `radar capital-plans`.

## marts/events_tender_offers.parquet
| column | type | description |
|---|---|---|
| subject_edinet_code | str | target company |
| event_date | date | filing date of the tender offer registration (240) |
| doc_id | str | the 240 filing |
| acquirer_name | str | filer (bidder) |

## marts/features_<as_of>.parquet
One row per listed company (has `sec_code`) whose latest annual report submitted before
`as_of` was parsed. Missing inputs give NaN/None, never 0 or False.

| column | type | description |
|---|---|---|
| as_of | date | screening date; every input is filtered to data public before it |
| edinet_code, sec_code, company_name | str | identity (from the document index) |
| source_doc_id, filing_submitted_at | str, datetime | the annual report used (lineage) |
| oldest_representative_age | int? | oldest 代表取締役 / 代表執行役, age on `as_of` |
| ceo_age, chair_age | int? | representative titled 社長 / 会長 (a sole representative titled only 代表取締役 counts as CEO) |
| family_stake | float? | top-10 holders matching the representatives' surnames, kanji variants unified (廣澤 = 広沢), employee plans (持株会) excluded (None = surname unknown) |
| founder_in_control | bool? | family_stake found ≥1 matching holder (None when unknown) |
| market_cap_jpy | float? | close lagged ≥1 day × common shares outstanding on the filing date (year end if not reported) × splits after it (counted from their effective date). Matches J-Quants within ±1% for 92% of companies |
| pbr | float? | market cap / net assets (J-GAAP only; None for equity ≤ 0) |
| adjusted_pbr | float? | market cap / (net assets + after-tax unrealized gain on rental real estate); = pbr when the note is absent / immaterial / none, None when unreadable. Added 2026-10-09, `pbr` unchanged |
| rental_unrealized_gain_to_market_cap | float? | (fair − book) × (1 − tax, config/trends.yaml) / market cap; 0 without material rental property |
| net_cash_jpy, net_cash_to_market_cap | float? | cash + short-term securities − interest-bearing debt |
| operating_income_to_market_cap | float? | 営業利益 (operating income, J-GAAP, latest annual report) / market cap — the "operating yield"; None when unreported. Added 2026-10-10 (quality value basket) |
| investment_securities_to_market_cap | float? | 投資有価証券 (largely cross-holdings) / market cap |
| officer_stake | float? | shares held by directors and officers / common shares at the filing date |
| price_change_12m | float? | split-adjusted change of the last actual close over 12 months |
| momentum_12_1 | float? | split-adjusted change from a year before as_of to the last close before as_of − 30 days (classic 12-1 momentum). Not used by the offer models. Added 2026-10-10 |
| dividend_yield | float? | last fiscal year's dividends (1株当たり配当額 × year-end shares outstanding; unknown treasury → issued) / market cap; None when the dividend is unreported. Added 2026-10-10 |
| median_trading_value_jpy | float? | median daily trading value over the recent window (config) |
| turnover | float? | median trading value / market cap, per day |
| tradable_share_ratio | float? | approx. 流通株式: 1 − ≥10% holders (excl. custodians) − officers, rescaled for treasury (upper bound) |
| listing_criteria_margin | float? | tradable ratio / value vs the segment's TSE minimum (config/jpx.yaml); < 0 = below |
| years_below_book | int? | consecutive fiscal year ends with PBR < 1, newest first; None if the newest year is unknown; a floor when an older year is unknown (prices start 2021-10) |
| net_cash_change_to_market_cap | float? | (net cash − net cash one report earlier) / market cap; None without an earlier report public before as_of |
| largest_listed_holder_stake | float? | biggest top-10 holder that is a listed company (親子上場): name match, minus the human-checked homonyms of config/ownership.yaml; 0 = none listed, None = no table |
| largest_listed_holder_name | str? | that holder, for the explanation |
| parent_prior_offers | float? | tender offers (240) filed by that holder before as_of |
| parent_other_holdings | float? | other listed companies where that holder has ≥ `portfolio_min_stake` |
| document_index_start | date? | first day of the contiguous EDINET index (coverage of the counts above) |
| top10_concentration_ex_custodian | float? | sum of top-10 ratios excluding custodians and treasury (free-float proxy) |
| activist_filers_365d | int? | distinct activists (`config/activists.yaml`) filing a 350 on the company in [as_of − 365 d, as_of); None if the fetched index does not cover the window |
| activist_names | str? | those activists, ` / `-joined |
| market_segment | str? | Prime / Standard / Growth (optional input) |
| sector33 | str? | TSE 33-sector name from listings before as_of |
| pbr_sector_percentile | float? | percentile of PBR within the sector (low = cheaper); None with < `sector_min_peers` values (config/trends.yaml) |
| net_cash_sector_percentile | float? | percentile of net cash / market cap within the sector (high = more cash) |
| capital_plan_disclosed | bool? | on a JPX capital-plan list published before `as_of` (optional input) |
| is_jgaap | bool? | from the filing's DEI |

## marts/scores_<as_of>.parquet
The feature columns plus `score` (share of the maximum positive rule weight of the DEFAULT rule
set, config/scoring.yaml — can be negative with learned weights) and `triggered_rules`, sorted by
score; `score_v0` (hand-set v0, kept for comparison); with saved live models (`radar model`):
`p_<type>` (probability per deal type), `p_any` (P of any offer — isotonic-calibrated when the
live bundle holds a calibrator; per-type P scaled to the same level), `p_any_prior`
(prior-corrected, the ORDER the walk-forward validated: isotonic ties the top), `p_<type>_prior`
(same, per deal type — `value_catalyst` ranks on these), `expected_value`.
`combo_v2_mbo` (2026-10-10): best percentile rank of the default score (ties broken by
`p_any_prior`) and of `p_mbo_prior` — its first 2k names are the top k of each list (the
"top 7 + top 7"); tracked live and in the walk-forward, not a decision candidate.
Round 3 (2026-10-10): `value_momentum` (value basket without the worst 10% of 12-1 momentum,
ranked by net cash) and `value_catalyst` (value basket ranked by p_parent + p_mbo). Portfolio
rankings (higher = buy first, NaN = not eligible): `value_basket` (net cash / market cap of
small, below-book, cash-rich companies), `value_basket_quality` (same AND operating income > 0),
and with a saved return model (`radar walkforward`) `return_model` (predicted 12-month return in
excess of all companies) and `return_model_sector_neutral` (that minus its 33-sector median).
Feature `offer_filed_365d` (date?): a third-party tender offer was filed on the company in the
year before as_of — already in play; hidden from the default ranking and the fact packs.

## marts/backtest_panel_<rules>.parquet, marts/backtest_metrics_<rules>.parquet
`backtest_panel_<rules>`: the scores table for every quarter-end `as_of`, plus `label` (1 if the
company became the target of a take-private tender offer in (as_of, as_of + horizon], see
`config/backtest.yaml`; partial and withdrawn (260) offers excluded) and `label_<type>` for
`mbo`, `parent`, `third_party`, `unknown` (offer not read).
`backtest_metrics_<rules>`: one row per `as_of` with `label_complete` (bool: the whole label
window is covered by the document index — False means late offers are not yet known and count as
misses), `companies`, `positives`, `precision_at_<k>`, `recall_top`, `lift_top` (top 10% of
scores), and `positives_<type>`, `recall_top_<type>`, `lift_top_<type>` per event type.

## marts/ablation_<rules>.parquet
One row per variant: `without` = `(none)` (full rule set) or the id of the rule removed, then the
backtest metrics averaged over the as_of dates with a complete label window, `lift_top_<type>`
per event type, and `lift_top_delta` (variant lift − full lift; negative → the removed rule was
helping).

## marts/model_comparison_<rules>.parquet, marts/logistic_coefficients_<rules>.parquet
`model_comparison`: one row per (test `as_of`, `model` ∈ {<rules>, logistic}) with the same
metrics as `backtest_metrics`. Train/test split in `config/model.yaml` (gap ≥ horizon).
`logistic_coefficients`: `feature` (design column, incl. `log_*`, `segment_*`, `missing_*`
indicators), `coefficient` (on standardised inputs), largest |coefficient| first.

## marts/live/scores_<as_of>.parquet and marts/live_scoreboard.parquet (live track record)
`live/scores_<as_of>`: a copy of scores_<as_of> frozen by `radar update` (or `radar scoreboard
--freeze`), written once and never rewritten. `live_scoreboard`: one row per frozen ranking —
`as_of`, `days_elapsed`, `window_complete`, `companies`, `targets_so_far`, `hits_top_fraction`,
`hits_top_n`, `lift_so_far`, and for each portfolio ranking `top_return_so_far`,
`universe_return_so_far`, `excess_so_far` (top-N bought the morning of the snapshot, valued at
the last close). Source: `radar scoreboard`.

Per-type models (`radar model`): `model_comparison_<rules>` also has rows `logistic_<type>`,
`logistic_by_type` (1 − Π(1 − p_type)) and `hybrid` (rank of the rules + rank of the parent
model); coefficients per type in `logistic_coefficients_<rules>_<type>`.

## Walk-forward and learned weights (2026-10-10)
`walk_forward_<rules>` (one row per test date and candidate, same metrics as the backtest),
`walk_forward_picks_<rules>` (top-N returns per candidate), `walk_forward_choice_<rules>`
(`model`, `mean_metric` = mean lift@top10%, `wins_vs_baseline`, `eligible`). `config/scoring_v2.yaml` is written by
`radar learn-weights`; `backtest_panel_scoring_v2` is the v0 panel re-scored with it and
`backtest_metrics_scoring_v2` only flags `in_sample` (no metrics by design).
`pick_returns_<rules>`: per (as_of, top_n) hit_rate, mean/median/worst return, share_positive,
exited_early, topix_return, universe_mean_return, excess_vs_universe.
`model_reliability_<rules>`: predicted vs observed offer rate per probability decile.
`walk_forward_picks_summary_<rules>`: per (model, top_n) the excess return ACROSS dates —
`mean_excess`, `median_excess`, `worst_excess`, `mean_excess_without_best`,
`dates_beating_universe`. `walk_forward_return_choice_<rules>` / `walk_forward_return_round2_<rules>`:
the return rule's table (`mean_metric` here = mean top-25 excess return, `wins_vs_baseline` vs the
value basket). `walk_forward_calibration_<rules>`: `method` (`p_prior` / `p_isotonic`), `companies`,
predicted vs observed rate per decile, `brier` (pooled over the test rows both methods cover).
`walk_forward_top_k_<rules>`: small portfolios (sizes in config/model.yaml `small_portfolios`),
one row per (model, top_n) pooled over the test dates — `dates`, `picks`, `priced`, `hits`,
`hit_rate`, `p_prior_mean` / `p_isotonic_mean` (mean predicted P(any offer) of the picks),
`return_hits` / `return_misses` (mean 12-month return of priced hits / non-hits),
`from_offers` + `from_others` = `mean_return` (priced picks), `universe_return`, `excess`.
`walk_forward_top_k_<rules>` also has (2026-10-10): `style_excess` (pick minus its size × PBR
quintile on the same date), `dividend_estimate` / `dividend_coverage`, `cost` (round-trip, by
liquidity: config/backtest.yaml `cost_tiers`), `net_return`, `net_excess` (vs all companies with
their dividends), `capacity_jpy` (largest equal-weight portfolio at 10% of daily trading value
over 5 days, median over dates). `walk_forward_return_round3_<rules>`: the round-3 rule's table.
`rule_weights_bootstrap_<rules>` (`radar weight-stability`): per rule `p5` / `p50` / `p95` of the
learned weight over resamples of whole dates, `share_positive`, `draws`.
`models/live_models.joblib` holds `calibrator` (isotonic; None = prior correction only):
models fitted on complete dates ≥ 1 year before the last, the map fitted on that last year.
`models/return_model.joblib`: the live ridge return model (refitted on every complete date).

## marts/activist_candidates.parquet
`filer_edinet_code`, `filer_name`, `doc_id` (the initial non-特例 350 read), `submit_datetime`,
`purpose` (保有目的 declaring 重要提案行為, not negated). Filers already matched by
`config/activists.yaml` are excluded. For human review only. Source: `radar discover-activists`.

## staging/topix.parquet (premium study)
`date` (date), `close` (float): TOPIX daily close (J-Quants Light). Source: `radar topix`.

## staging/tender_offers.parquet (premium study and labels)
`aims_delisting` was REMOVED (text heuristic, unreliable); use `is_take_private`.
One row per third-party tender offer whose 240 filing (form `jptoo`) was downloaded.
| column | type | description |
|---|---|---|
| subject_edinet_code, event_date, doc_id, acquirer_name | | as in `marts/events_tender_offers` |
| offer_price | float? | yen per share, first price stated in 買付け等の価格 |
| prior_stake | float? | (bidder D + special-interest parties G) / all voting rights J; blank D/G = none; None if D + G > J (filing error) |
| bidder_stake | float? | D / J: the bidder alone (G also holds a parent SELLING under a tender agreement) |
| ratio_after | float? | bidder's target holding after the offer (1.0 = 100%) |
| is_take_private | bool? | target holding after the offer ≥ 95% (partial offers are not take-privates) |
| self_declared_mbo | bool | purpose describes THIS deal as a マネジメント・バイアウト |
| event_type | str | `mbo`, `parent` (prior stake ≥ 20%: a ≥ 20% blockholder is on the deal side, buyer or seller), `third_party`, `unknown` |
Source: `radar tender-offers`. Config: `config/tender_offers.yaml`.

## marts/event_study.parquet (premium study)
Per offer: `doc_id`, `subject_edinet_code`, `event_date`, `event_type`, `offer_price`,
`pre_close` (last close before the filing date), `premium` (offer / pre_close − 1),
`runup_abnormal` (split-adjusted stock return − TOPIX return over the 60 sessions before),
`pre_announced` (pre_close within 3% of the offer: announced earlier, excluded from summaries),
`book_per_share` (target's net assets per share from its latest annual report before the offer,
split-adjusted to the offer date), `offer_pbr` (offer price ÷ book_per_share) and `pre_pbr`
(pre_close ÷ book_per_share). None when the book value is unknown (IFRS, no prior report).
Source: `radar event-study`. Config: `config/event_study.yaml`.

## data/marts/factpack_<as_of>_<code>.md
Markdown, one per company: reasons, key facts, likely buyer and offer-price range
(`model/offer_price.py`), hit rate by points band, excerpts of the 有報 (事業の内容, main
customers, sales by region; `config/xbrl_elements.yaml` `excerpts`), top-10 holders, officers,
recent 240/290/350 filings. Facts for a memo, not a memo. Source: `radar factpack`.

## data/manual/journal.yaml — decision journal (by hand; never committed)
`entries`: list of {`date` (YYYY-MM-DD), `code` (4- or 5-character securities code), `action`
(watch · buy · sell · pass), `price` (optional, yen), `reason` (text)}. The Journal tab adds
`return_since` (split-adjusted, last close on or before `date` → last close before today),
`offer_after` (first take-private offer filed after `date`) and `rank_now` (default ranking of
the latest screen). A bad entry is reported by number and field, never silently skipped.

## 臨時報告書 (docTypeCode 180) classification — config/extraordinary_reports.yaml
Not stored as a table: `radar update` downloads the reports of followed companies (top 50 +
watchlist, last `lookback_days`) into data/raw, and alerts / the card classify them on the fly
(`parse/extraordinary_report.classify_extraordinary`): `reasons` (keys: major_shareholder,
parent_change, specified_subsidiary, representative, share_consolidation, share_exchange,
shareholder_agreement, merger, company_split, acquisition, agm_votes, other), `labels`
(Japanese), `alert` (control-related), `tender_offer` (the 提出理由 mentions 公開買付け),
`excerpt` (start of the 提出理由).

## Alerts note extras and retention (2026-10-10)
`alerts_<date>.md` may start with "Data coverage dropped" (config/report.yaml `coverage_drift`:
a key feature known for > 5 pp fewer companies than on the previous screen — broken data) and
"Unusual trading on followed companies" (`unusual_trading`: last-5-session median trading
value ≥ 3× the 60-session median, or a ±10% move vs TOPIX; split-adjusted, sessions before
the date). Thresholds not validated: reasons to look, not signals.
`radar update` deletes `features_<date>` older than 14 days (scores keep the same columns) and
fact packs older than 180 days (`retention`); daily marts are written with zstd compression.
Raw data, scores and the live record are never pruned.
