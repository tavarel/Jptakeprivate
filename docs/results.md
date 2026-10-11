# Results — backtest of the take-private radar (updated 2026-10-10)

Everything below is reproducible with:

```bash
uv run radar backtest --start 2022-09-30 --end 2026-06-30            # rules v0
uv run radar ablation
uv run radar backtest --start 2022-09-30 --end 2026-06-30 --rules scoring_v1
uv run radar ablation --rules scoring_v1
uv run radar model                                                   # logistic vs v0
uv run radar walkforward                                             # walk-forward choices
uv run radar weight-stability                                        # bootstrap of v2 weights
```

`make refit` runs the whole sequence on the newest complete data.

## Summary (current numbers)

| | Value |
|---|---|
| Base rate: listed companies taken private within 12 months | 1.8% |
| Hand-set rules, lift of the top 10%, 13 complete dates | 1.69× |
| Learned weights (v2, default), walk-forward lift on 7 unseen dates | **2.22×** (7 / 7 wins) |
| v2 on parent buyouts / on MBOs | 4.4× / 0.3× |
| Per-type logistic models on MBOs | 1.9× |
| Portfolio default (pre-registered return rule) | value basket — beat all listed companies on 6 / 7 dates |
| Strategies with ≥ 95% "chance of skill" (deflated Sharpe, 72 tried) | none |
| Capacity of the small-cap strategies | ¥5–16m |

The sections below are in the order the work was done: each decision rule is written before
its result. Later sections update earlier numbers (label hygiene, the dated activist list).

## Setup

- **Universe**: every listed company with a parsed annual report filed before the screening
  date — about 3,900–4,200 companies per date.
- **Screening dates**: quarter-ends from 2022-09-30 to 2026-06-30. The **13 dates up to 2025-09-30
  have a complete 12-month label window**, and all averages below use only those. The later dates
  are shown, but their offers are not all known yet.
  The start is 2022-09-30, not 2022-03-31: EDINET no longer serves the annual reports filed in
  June 2021 (5-year inspection period), so earlier dates would miss most March-year-end companies.
- **Label**: the company became the target of a **take-private** tender offer (240 filing,
  bidder's target holding ≥ 95%) within (as_of, as_of + 365 days]. Partial and withdrawn (260)
  offers are excluded. On average 81 targets per date: a base rate of **2.0%**.
- **Event types** (from the 240 filing): `mbo` (the deal describes itself as an MBO), `parent`
  (a ≥ 20% blockholder is on the deal side, as buyer or as seller), `third_party`.
- **Point-in-time**: filings with `submit_datetime < as_of`, prices lagged ≥ 1 day, listings
  before as_of. This is enforced by tests (ADR 0003).
- **Metrics**: lift@top10% = hit rate in the top 10% of scores ÷ base rate (1.0 = chance);
  recall@top10% = share of all targets that were in the top 10%; precision@k.

## Rule set v0 (`config/scoring_v0.yaml`)

| Mean over 13 complete dates | value |
|---|---|
| lift@top10% | **1.64×** |
| recall@top10% | 16.5% of all targets |
| precision@20 / @50 | 2.7% / 2.0% (base rate 2.0%) |
| lift by type: MBO / parent / third party | 1.72× / 1.56× / 1.65× |

| Screening year | lift@top10% | recall@top10% |
|---|---|---|
| 2022 (2 dates) | 2.64× | 26.4% |
| 2023 | 1.69× | 16.9% |
| 2024 | 1.22× | 12.2% |
| 2025 (3 dates) | 1.48× | 14.9% |

**Reading**: the top decile holds about 1.6 times its share of future targets — better than
chance on every complete date (lowest: 1.0× on 2024-12-31). But **the very top is not sharper**.
Precision@50 equals the base rate, and the hit rate by points band is flat near the top:

| Points (max 12.5) | company-dates | targeted within 12 months |
|---|---|---|
| 8–10 | 311 | 2.3% |
| 6–8 | 2,348 | 3.0% |
| 4–6 | 10,359 | 2.7% |
| 2–4 | 20,626 | 2.1% |
| 0–2 | 18,745 | 1.4% |

Track record: 4 companies ranked in a past top 50 later received an offer: 大正製薬HD,
寺岡製作所, 理研コランダム and 杉田エース, all MBOs.

### Ablation — which rules carry the lift? (v0)

`lift Δ` = lift without the rule − lift with it; **negative means the rule helps**.

| Rule removed | lift | Δ | MBO lift | parent lift | third-party lift |
|---|---|---|---|---|---|
| (none — full v0) | 1.64 | — | 1.72 | 1.56 | 1.65 |
| listed_parent | 0.94 | **−0.71** | 1.93 | 0.15 | 1.35 |
| standard_market | 1.43 | −0.22 | 1.84 | 1.17 | 1.40 |
| activist_holder | 1.50 | −0.14 | 1.67 | 1.37 | 1.51 |
| below_book | 1.53 | −0.12 | 1.66 | 1.35 | 1.65 |
| small_cap | 1.60 | −0.05 | 1.72 | 1.51 | 1.54 |
| cash_rich | 1.65 | +0.002 (≈ 0) | 1.83 | 1.58 | 1.46 |
| no_capital_plan | 1.64 | 0.00 (never fires) | 1.72 | 1.56 | 1.65 |
| family_control | 1.69 | +0.04 | **1.33** | 1.92 | 1.42 |
| aging_leader | 2.04 | **+0.39** | 1.59 | 2.23 | 2.09 |

- **listed_parent carries most of the result**: without it, parent deals are no better than
  chance (0.15×).
- **aging_leader and family_control lower the pooled lift**, but family_control is what finds
  MBOs (1.72× → 1.33× without it). These rules separate *types* of deals. Per ADR 0004, v0 is not
  re-weighted on the data that judged it. Re-weighting is a v2 hypothesis to
  test on later dates.
- **no_capital_plan never fires** on complete dates: the JPX lists only exist from 2025-10, so
  this rule is **untested**.

## Rule set v1: v0 + candidates

v1 = v0 + insider_stake, cross_holdings_rich, price_slump, low_turnover, near_listing_criteria,
acquisitive_parent. Mean lift **1.27×**, which is **worse than v0** (1.64×). The extra points
dilute listed_parent, and parent-deal lift falls to 0.71×.

| v1 candidate | Δ in the v1 ablation | decision |
|---|---|---|
| low_turnover | −0.10 | keep (helps) |
| near_listing_criteria | −0.08 | keep |
| cross_holdings_rich | −0.06 | keep |
| acquisitive_parent | −0.06 | keep — the parent's track record does help |
| price_slump | −0.02 | keep (marginal) |
| insider_stake | **+0.16** | **removed** (not re-tuned) |

Without insider_stake, v1 would reach 1.42× on the same dates, which is still below v0.
**v0 stays the default rule set.** That 1.42× is measured on the data used to make the decision,
so it is not an independent result.

## Logistic regression

L2 logistic regression with balanced classes, median imputation with missing-value
indicators, `C = 0.1` set beforehand (not tuned). Trained on as_of ≤ 2023-12-31 and tested on
2025-03-31, 2025-06-30 and 2025-09-30: dates it never saw, after a 12-month gap.

| Same 3 test dates | logistic | rules v0 |
|---|---|---|
| lift@top10% | **2.66×** | 1.48× |
| precision@20 | **18.3%** | 3.3% |
| precision@50 | **9.3%** | 2.0% |
| recall@top10% | 26.7% | 14.9% |
| lift MBO / parent / third party | 0.34× / **4.90×** / 1.09× | **2.21×** / 1.43× / 0.72× |

- The model is much better overall, **because it finds parent buyouts**. The largest
  interpretable coefficients: large listed holder (+), low tradable-share ratio (−), low
  officer stake (−), low PBR (−), activist (+).
- It is **worse than chance on MBOs** (0.34×), where the rules work (2.2×). The two are
  complementary, which argues for per-type models once there are enough MBOs (~25 a year).
- The "missing" indicators have large coefficients that offset each other. They come from
  regional-exchange-only companies (no TSE price: market cap, turnover and PBR all missing
  together), whose target rate (1.6%) is below average. This is collinearity, not leakage. Do not
  read those coefficients one by one.
- Only 3 test dates and ~300 targets: this is a strong hint, not proof. The next honest test is
  the screenings `radar update` produces from now on.

## Models per deal type (2026-10-09)

The same logistic set-up (inputs, split, `C`) trained once per deal type on `label_<type>`,
then combined as P(any offer) = 1 − Π(1 − p_type). A **hybrid** adds the rules' rank (good at
MBOs) to the parent model's rank, within each date. Mean over the same three 2025 test dates:

| Model | lift@top10% | MBO | parent | third party | precision@20 |
|---|---|---|---|---|---|
| logistic, parent only | **3.00×** | 0.58× | **5.62×** | 0.70× | 11.7% |
| logistic, pooled | 2.66× | 0.34× | 4.90× | 1.09× | 18.3% |
| hybrid (rules + parent model) | 2.29× | 0.92× | 3.72× | 1.12× | 6.7% |
| logistic, per type combined | 2.22× | 1.42× | 3.38× | 0.93× | 6.7% |
| rules v0 | 1.48× | 2.21× | 1.43× | 0.72× | 3.3% |
| logistic, MBO only | 1.12× | **2.37×** | 0.64× | 0.56× | 3.3% |
| logistic, third party only | 1.41× | 0.93× | 1.66× | 1.50× | 3.3% |

- **The MBO model learned what the rules encode**: family stake (+), no large listed holder
  (−), low tradable share (−). On MBOs it matches the rules (2.4× vs 2.2×), which supports
  family_control / aging_leader for this deal type.
- **The parent model** is driven by a low officer stake, a low tradable-share ratio, a large
  listed holder and a low PBR. Parent deals are the most frequent type, so models that find
  them win on the pooled lift.
- **No winner is picked from these 3 dates.** Choosing the best of 7 models on the dates that
  judged them would overfit. The rules stay the default and explain each flag; the models are
  reported. The deciding test is the live scoreboard (`radar scoreboard`), which starts
  2026-10-09.

## Walk-forward choice of the default ranking — decision rule, fixed BEFORE running (2026-10-10)

Written before any walk-forward result existed, so the result cannot shape the rule.

- **Splits**: expanding window. For each complete test date T from 2024-03-31 on, train only on
  dates ≤ T − 365 days (the label horizon). That gives 7 test dates (2024-03 → 2025-09).
- **Candidates**: (1) v0, hand-set weights; (2) v2, the same 9 rules with weights learned by an
  L2 logistic on the 0/1 rule flags; (3) v2 per deal type, combined 1 − Π(1 − p); (4) the logistic
  per type on all features, combined; (5) the hybrid; (6) the expected value.
- **Main metric**: mean lift@top10% over the 7 test dates. Tie-break: mean top-25 return minus
  the all-companies return.
- **A candidate replaces v0 only if** it beats v0 on lift in **at least 5 of the 7 dates**.
  Among candidates within 0.1× of the best mean lift, the simplest wins: v0 > v2 > v2 per type >
  models (v0 counts as the simplest: a change must gain at least 0.1× of mean lift).
- The winner's weights / models are then refitted on ALL complete dates for live use, and the
  live scoreboard confirms or rejects it on future offers.

### Walk-forward result (2026-10-10) — run AFTER the rule above was written

Label hygiene first: 89 offers were already public on their screening date (the price
already sat at the offer); they leave the evaluation. v0's mean lift on the 13 complete dates
went from 1.64× to **1.76×**: those deals had been counted as misses although no ranking could
have predicted them (once announced, the price jumps and the company stops looking cheap).

Mean over the 7 walk-forward test dates (2024-03 → 2025-09), each trained on data ≥ 1 year older:

| Ranking | lift@top10% | wins vs v0 | MBO lift | parent lift | precision@20 | top-25 excess return* | top-50 excess return* |
|---|---|---|---|---|---|---|---|
| **v2 — same rules, learned weights** | **2.29×** | 7 / 7 | 0.29× | **4.49×** | 5.7% | +2.1 pp | −0.5 pp |
| hybrid (rules rank + parent model) | 2.29× | 7 / 7 | 1.00× | 3.78× | 8.6% | +2.3 pp | +5.1 pp |
| v2 per deal type | 2.25× | 7 / 7 | 0.72× | 3.88× | 5.0% | +1.1 pp | +2.3 pp |
| expected value | 2.08× | 7 / 7 | 1.31× | 3.15× | 9.3% | −8.2 pp | +0.4 pp |
| logistic per type | 1.92× | 6 / 7 | 1.88× | 2.30× | 11.4% | −7.2 pp | −2.9 pp |
| v0 — hand-set weights | 1.44× | — | 1.88× | 1.30× | 2.1% | **+7.0 pp** | **+6.3 pp** |

\* 12-month equal-weight price return of the top N minus the all-companies mean (15.6%;
TOPIX 19.0% over the same dates).

**Decision by the pre-registered rule: v2** (beats v0 on 7 of 7 dates; v2, the hybrid and v2 per
type are within 0.1× of the best mean lift, and v2 is the simplest). `radar learn-weights`
refitted its weights on all 13 complete dates → `config/scoring_v2.yaml`, and
`config/scoring.yaml` makes it the default of the daily screen. Learned weights (log-odds):
listed_parent +1.39, activist_holder +0.85, small_cap +0.29, cash_rich +0.23,
standard_market +0.15, family_control +0.13, below_book +0.08, no_capital_plan 0.00 (no data
before 2025-10), **aging_leader −0.24**.

What the table also says — read it before investing on the ranking:
- **Predicting WHO gets an offer is not picking WHAT makes money.** v2 finds parent buyouts,
  which pay the lowest premiums (~32%) and are often larger, pricier companies; its picks did
  not beat the market of small caps. v0 favours small, cheap, family companies (MBOs pay ~40%)
  and its top 25/50 returned +6–7 pp over all listed companies. For a portfolio, v0 or the
  hybrid have the better record so far; the live scoreboard tracks all of them.
- **v2 almost stops finding MBOs** (0.29×). Keep v0's order or the per-type models in view when
  looking for MBOs.
- **Calibration is still poor**: in the top decile of the prior-corrected model the predicted
  rate was 14% vs 4.7% observed; low deciles are under-predicted. The expected value therefore
  ranks poorly and must not be read as a number yet — Kelly sizing needs a better calibration
  first (e.g. isotonic on a held-out year).
- We have looked at these dates many times; the rule was fixed first and 7 dates beat 3, but
  only the live scoreboard is clean evidence.

## Portfolio default — return criterion, fixed BEFORE running (2026-10-10)

Question changed from "who gets an offer" to "which ranking makes money". Written before any
return walk-forward with the new candidates existed:

- **Same 7 walk-forward dates and training cut-off** (each candidate trained only on data ≥ 1
  year older). 12-month equal-weight price return, holdings with an impossible one-day quote
  (> ×3 / < ÷3 after split adjustment, 値幅制限) excluded as data errors.
- **Candidates**: (1) **value basket** — small cap (≤ ¥30bn) AND PBR < 1 AND net cash ≥ 30% of
  market cap, ranked by net cash / market cap: the simple yardstick the radar must beat;
  (2) v0; (3) hybrid; (4) v2; (5) a **return model** — the same feature pipeline, ridge
  regression on the 12-month return in excess of the all-companies mean (targets winsorized at
  the 1st / 99th percentile), trained walk-forward.
- **Main metric**: top-25 return minus the all-companies mean, averaged over the 7 dates.
  Also reported: top 50, median across dates, worst date, mean without the best date.
- **A candidate becomes the portfolio default only if** it beats the value basket on at least 5
  of 7 dates AND by ≥ 1 pp on the mean; otherwise the value basket stays (simplest first:
  value basket > v0 > hybrid > v2 > return model).
- Not modelled: dividends (+2–3%/yr for these stocks), costs and market impact (−1 to −1.5
  pp/yr assumed), taxes. The live scoreboard then tracks the chosen portfolio's return.

### Portfolio result (2026-10-10) — run AFTER the return criterion above was written

Data fix first: J-Quants has a few impossible one-day quotes with AdjFactor 1 (¥4 → ¥480, ¥1 →
¥150; TSE daily limits make > ×3 impossible). Holdings that contain one are now excluded. The
+75.9% top-25 date (2024-12-31) is REAL: エス・サイエンス, a ¥21 penny stock, rose to ¥238 within
the daily limits — and it alone carried most of v0's portfolio edge.

Top 25, 12-month return minus the all-companies mean, 7 walk-forward dates:

| Ranking | mean | median | worst date | mean without best date | dates > all companies | dates > basket |
|---|---|---|---|---|---|---|
| return model (ridge, walk-forward) | **+10.1 pp** | **+6.6 pp** | −2.7 pp | **+6.6 pp** | 5 / 7 | 3 / 7 |
| v0 (hand-set rules) | +7.0 pp | +0.9 pp | −6.6 pp | −0.6 pp | 4 / 7 | 3 / 7 |
| **value basket** (yardstick) | +3.7 pp | +1.1 pp | −5.2 pp | +2.1 pp | **6 / 7** | — |
| hybrid | +2.2 pp | +4.9 pp | −9.4 pp | +1.4 pp | 5 / 7 | 3 / 7 |
| v2 (default for offers) | +2.1 pp | −0.3 pp | −3.9 pp | −0.5 pp | 3 / 7 | 3 / 7 |

Top 50: return model +17.0 pp (without best date +7.8 pp, 6 / 7 dates above all companies),
v0 +6.3 pp, hybrid +5.1 pp, value basket +2.8 pp, v2 −0.5 pp.

**Decision by the pre-registered return rule: the value basket stays the portfolio default** —
no candidate beat it on 5 of 7 dates. The return model is the most promising challenger
(+6.6 pp even without its best date) but won only 3 of 7 dates against the basket; it is
tracked live. v0's portfolio edge was one date. And the activist look-ahead does NOT explain
v2's offer accuracy: without the activist rule, lift is 2.35× vs 2.29×.

## Round 2 — return candidates and calibration, fixed BEFORE running (2026-10-10)

Honest caveat first: these candidates were proposed AFTER seeing round 1 (the return model's
live top picks were regional banks; the value basket held loss-makers such as 出前館, 千趣会).
Testing ideas suggested by the same data is weaker evidence than round 1, so a round-2 winner
only becomes the portfolio default after the live scoreboard confirms it.

- **Candidates** (nothing else is tuned; thresholds set now):
  (1) **value basket** (unchanged yardstick);
  (2) **quality value basket** — the same basket AND operating income / market cap > 0
      (営業利益 from the same 有価証券報告書; unreported → not eligible);
  (3) **sector-neutral return model** — the round-1 ridge prediction minus its 33-sector median
      on each date (keeps stock picking, removes a whole-sector bet like the banks).
- **Rule**: identical to round 1 — top-25 excess over all companies; beat the value basket on ≥ 5
  of 7 dates AND by ≥ 1 pp on the mean; simplest first (basket > quality basket > sector-neutral).
- **Calibration**: P(any offer) from the per-type models, prior-corrected (current) vs
  isotonic regression fitted on the last training year (models fitted on the older training
  dates). Isotonic replaces the prior correction in the live P and expected value only if its
  pooled Brier score on the walk-forward test rows is lower. Calibration rescales probabilities
  monotonically, so rankings and lift cannot change — only the levels used by expected value
  and position sizing.

### Round 2 result (2026-10-10) — run AFTER the round-2 rule above was written

Top 25, 12-month excess over all companies, same 7 walk-forward dates:

| Ranking | mean | dates > basket | eligible |
|---|---|---|---|
| sector-neutral return model | **+9.6 pp** | 4 / 7 | no (needs 5) |
| **value basket** (yardstick) | +3.7 pp | — | stays |
| quality value basket | +3.2 pp | 4 / 7 | no |

**Decision by the rule: the value basket stays the portfolio default.** The sector-neutral
return model is the closest challenger yet: removing the sector bet (the regional banks) kept
almost all of the return model's edge (+9.6 vs +10.1 pp), so that edge is mostly stock picking,
not one sector. It also lifted offer-finding (lift 1.40× vs 0.92×). The quality filter did not
add return in the top 25 but cut the worst date (−2.6 pp vs −5.2 pp).

### Calibration result

Pooled over the walk-forward test rows both methods cover (20,615 company-dates):

| Method | Brier (lower = better) | top decile: predicted → observed |
|---|---|---|
| prior correction (current) | 0.0256 | 15.1% → 5.0% |
| **isotonic on the last training year** | **0.0227** | 4.4% → 4.1% |

**Decision by the rule: isotonic replaces the prior correction** for the live P and expected
value. Not deployed yet: the live bundle (`radar model`) must hold models fitted on all dates
but the last complete year plus an isotonic map fitted on that year — a change to plan first. Rankings and lift do not change (monotone map); only the levels.
Seen at the very top (next section): even isotonic flattens there — the top 10 of the
per-type model predicted ~4% and 14% got an offer — so a P for a single top name stays rough.

## Round 3 and robustness — fixed BEFORE running (2026-10-10)

Prompted by an external review that saw only these results (not the code). What it raised, and
what was kept after checking it against the code and the data:

- **Kept — "the excess may be size + value beta, not skill".** A factor regression is impossible
  on 7 yearly observations (more parameters than points). The honest equivalent: a **style
  benchmark** — each pick minus the mean return of the companies in its own size × PBR quintile
  on the same date (`style_excess`). If the value basket's edge is pure style, this is ≈ 0.
- **Kept — "few, overlapping dates; many strategies tried".** Every strategy now carries a
  **Newey-West** error (3 overlapping neighbours), a 90% interval, and a **deflated Sharpe
  ratio** (Bailey & López de Prado) counting every ranking × size in the table as a trial.
- **Kept — "capacity" and "net of costs and dividends".** Round-trip cost by liquidity
  (< ¥10m/day 2.0%, < ¥100m 1.0%, above 0.4%; config/backtest.yaml), last year's dividend yield
  from the 有報 (1株当たり配当額; unknown counts as 0 and the coverage is shown), and the
  largest equal-weight portfolio every pick can absorb at 10% of its daily trading value over
  5 days.
- **Kept, as round-3 candidates** (two only, thresholds set now):
  (1) **value + momentum** — the value basket without the worst 10% of 12-1 momentum (a year
  ago → a month ago) of the whole list: cheap stocks that keep falling are the classic value trap;
  (2) **value + catalyst** — the value basket ranked by P(parent) + P(MBO) instead of by cash:
  the take-private signal as a tilt inside a base that already works. Parent and MBO only: their
  prices barely run up before the announcement (event study, 2026-10-08), so the premium is still
  there to collect.
- **Not kept**: monthly screening dates (overlapping 12-month windows add no independent
  years: 2022-09 → 2025-09 stays ~3 independent years); purged CV (the walk-forward already
  trains only on dates ≥ 365 days before each test date — that IS the purge); low-volatility,
  more quality metrics, and a wider catalyst label (buybacks, PBR crossing 1) — backlog, larger
  projects; a third weight set shrunk towards v0 (one more strategy tried on the same dates) —
  replaced by a **bootstrap of the v2 weights over dates** as a diagnostic.
- **Rule for round 3**: identical to rounds 1–2 (top-25 excess over all companies; beat the
  value basket on ≥ 5 of 7 dates AND by ≥ 1 pp; simplest first: basket > value + momentum >
  value + catalyst). Round 3 is weaker evidence than round 1: proposed after seeing the data.

### Round 3 and robustness result (2026-10-10) — run AFTER the rule above was written

Rounds 1 and 2 reproduced exactly after the full re-parse (dividends added) and the rebuilt
backtest panel (momentum added).

**Round 3** (top 25, excess over all companies, 7 dates):

| Ranking | mean | dates > basket | eligible |
|---|---|---|---|
| value + momentum | +3.7 pp | 4 / 7 | no |
| **value basket** | +3.7 pp | — | stays |
| value + catalyst | −1.5 pp | 2 / 7 | no |

**Decision by the rule: the value basket stays.** The momentum filter changes little at 25
names (+9.2 pp vs +8.3 pp at 15). The catalyst tilt HURT: ranking the basket by P(parent) +
P(MBO) found FEWER offers than ranking it by cash (hit 2.3% vs 4.0% in the top 25) and lost on
the last three dates — inside a small, cheap, cash-rich basket the offer models have little
left to separate. The take-private signal stays a separate list, not a tilt of this basket.

**Robustness of every strategy** (10 rankings × 6 sizes; the Portfolio tab shows all of them):

| Strategy | mean excess | 90% interval | chance it's skill* | beyond size + value | net excess** | capacity*** |
|---|---|---|---|---|---|---|
| return model, sector-neutral, top 25 | +9.6 pp | +5.9 … +13.3 | **81%** | +3.9 pp | +6.0 pp | ¥5m |
| return model, top 25 | +10.1 pp | +1.1 … +19.1 | 6% | +2.6 pp | +7.1 pp | ¥7m |
| value + momentum, top 15 | +9.2 pp | +3.9 … +14.5 | 42% | +5.9 pp | +6.4 pp | ¥5m |
| value basket, top 15 | +8.3 pp | +2.5 … +14.1 | 30% | +5.0 pp | +5.5 pp | ¥5m |
| value basket, top 25 | +3.7 pp | −0.0 … +7.3 | 7% | +0.5 pp | +1.1 pp | ¥8m |
| v0, top 100 | +6.6 pp | — | 65% | — | — | — |
| v2 (offer default), top 25 | +2.1 pp | −0.7 … +4.8 | 0% | −2.2 pp | −0.6 pp | ¥16m |
| per-type logistic, top 25 | −7.2 pp | −22.8 … +8.3 | 0% | −4.3 pp | −10.4 pp | ¥8m |

\* deflated Sharpe ratio: P(the edge is real) after counting all 72 strategies tried.
\*\* after estimated round-trip costs (1–2%: these are illiquid stocks) and last year's dividends
(≈ 2.2%; 76–90% of picks have a known dividend), minus all companies with their dividends.
\*\*\* largest equal-weight portfolio every pick absorbs at 10% of its daily trading value in 5
days (median over dates).

What it says:
- **No strategy reaches strong evidence (≥ 95%).** The best, the sector-neutral return model at
  25 names, is at 81%: a real candidate, not proof. Everything else is within what luck and the
  number of tries can produce. The live scoreboard is still the test that counts.
- **The value basket at 25 names is almost pure size + value** (+0.5 pp beyond its style
  bucket). At 15 names, and with the momentum filter, +5–6 pp remain beyond style — but on
  7 dates.
- **Costs and dividends roughly cancel** (dividends ≈ 2.2%, costs 1–2%), so net results are
  close to gross. Rankings driven by offers (v2, logistic) lose after style AND after costs.
- **Capacity is tiny: ¥5–16m.** Above that the picks' own trading volume becomes the limit —
  the backtest returns cannot be scaled, as the external review warned.
- **Kelly (now on the excess with one Newey-West error off the mean):** v2 top 25 → 23%;
  the value basket, value + momentum and the sector-neutral return model still reach the 100%
  cap (their excess is steady across the 7 dates). With the deflated Sharpe below 95% for all,
  read the cap as "the history does not argue against it", not as a sizing signal.
- **Which v2 weights are real?** Bootstrap over whole dates (200 resamples, `radar
  weight-stability`), 5th–95th percentile of each learned weight: listed_parent +1.27 … +1.56,
  activist_holder +0.70 … +1.01, small_cap +0.19 … +0.41, cash_rich +0.17 … +0.31 and
  aging_leader −0.47 … −0.11 never change sign — stable. standard_market (+0.03 … +0.27) is
  thin; family_control (−0.02 … +0.31) and below_book (−0.01 … +0.17) straddle zero: for
  predicting offers overall they are noise (family_control still matters for MBOs, see the
  ablation). no_capital_plan stays 0 (no data before 2025-10).
- **Calibration is deployed**: the live P uses the isotonic map (models fitted on
  2022-09 → 2024-09, map fitted on 2024-12 → 2025-09); ranking stays on the prior-corrected P.

## Activist list revision — refit (2026-10-10)

The activist list was reviewed by hand: Palliser dated from 2023 (public campaign, not the
private 2021 engagement), new funds added with the year their activism went public (Reno, Taiyo
Fund, LIM bare; Ariake 2024, MIRI 2024, UGS 2024, 植島 2023, DOE5パーセント 2025). Fewer
activists now count on early dates, which removes part of the list's look-ahead. `make refit`
re-ran everything on the same 13 complete dates:

| | before | after |
|---|---|---|
| v2 weight of activist_holder | +0.85 | **+0.77** (bootstrap 5–95%: +0.61 … +0.95, still always > 0) |
| other v2 weights | — | within ±0.01 |
| v2 walk-forward lift@top10% | 2.29× | **2.22×** (still 7 / 7 wins vs v0 → stays the offer default) |
| v2 without the activist rule | 2.35× | 2.32× |
| top-25 excess: return model / v0 / v2 / value basket | +10.1 / +7.0 / +2.1 / +3.7 pp | +12.5 / +9.0 / +4.0 / +3.7 pp |
| round 2 / round 3 decision | value basket | value basket (unchanged) |
| best "chance it's skill" | 81% (sector-neutral return model, 25) | 71% (v0 top 200); sector-neutral 25 → 58% |

Reading: the activist signal survives the stricter, dated list (smaller but stable), so the
look-ahead was a real but small part of it. The portfolio numbers moved by 2 pp for the rankings
that use the activist flag — a reminder of how sensitive 7 dates are to one input. Still no
strategy reaches 95%. The robustness table above keeps the earlier run for the record; the
Portfolio tab shows the current numbers.

## Combined list "top 7 offer + top 7 MBO" (2026-10-10, proposed after seeing the data)

One ranking that takes the top of the v2 offer ranking (ties broken by the model P) and of the
MBO model in turn. Walk-forward, same 7 dates, excess over all companies:

| Names | mean | 90% range | without best date | worst date | dates > all | got an offer | net** |
|---|---|---|---|---|---|---|---|
| 10 | +6.8 pp | −6.6 … +20.2 | +2.3 | −17.1 | 4 / 7 | 15.7% | +2.1 pp |
| 15 | +9.1 pp | −4.7 … +22.9 | +2.4 | −15.4 | 4 / 7 | 11.4% | +5.3 pp |
| 25 | +6.1 pp | +0.1 … +12.1 | +1.9 | −18.8 | 5 / 7 | 8.0% | +2.9 pp |

It finds offers 4–7× the base rate, and the return comes mostly from its MBO half. "Chance it's
skill" ≤ 5% (counted among all strategies tried; designed after seeing these dates): tracked
on the live scoreboard, not a default.

## Small portfolios — top 10 / top 15 per ranking (2026-10-10)

Question: "if I only buy the top 10 or 15 of each ranking, how often does an offer come,
is the predicted probability right AT THE TOP, and what return should I expect?"
Written BEFORE the numbers (`radar walkforward` → `marts/walk_forward_top_k_<rules>`):

- **Descriptive only — no default changes on these numbers.** Re-training a model to be best
  at the top 10 of these same 7 dates would fit their noise (10 picks × 7 dates = 70 holdings,
  a few hits). The defaults stay as decided above; the live scoreboard judges the top 10/15.
- **Same 7 walk-forward test dates**, each ranking trained only on data ≥ 1 year older; picks
  pooled over the dates (70 or 105 holdings per ranking). Unknown scores rank last.
- **Calibration at the top**: the mean predicted P(any offer) of the picks — prior-corrected
  (`p_prior`) and isotonic (`p_isotonic`) — next to the observed hit rate.
- **Where the return comes from**: mean 12-month return of the priced picks = share of hits ×
  their return (`from_offers`) + share of non-hits × theirs (`from_others`). Price return only.

### Small portfolios result (2026-10-10)

Excess = 12-month return minus all companies (mean 15.7%), mean over 7 dates; "w/o best" drops
the luckiest date; hit = share of picks with a take-private offer within 12 months (base 2.2%);
P = mean predicted P(any offer) of the picks (prior / isotonic); offers + rest = the mean return
of the picks split between picks that got an offer and the others (pp of return).

**Top 10**

| Ranking | excess | w/o best | worst date | > all | hit | P prior / iso | offers + rest |
|---|---|---|---|---|---|---|---|
| value basket | +8.6 | +6.2 | −12.2 | 5/7 | 5.7% | 10% / 2.8% | +8.2 + 16.1 |
| quality value basket | +8.8 | +6.5 | −4.2 | 4/7 | 2.9% | 7% / 2.6% | +0.7 + 23.8 |
| return model | +7.0 | +1.7 | −4.9 | 3/7 | 5.7% | 14% / 2.6% | +3.1 + 19.0 |
| return model, sector-neutral | +3.2 | −9.3 | **−82.5** | 4/7 | 4.3% | 12% / 2.0% | +3.0 + 21.6 |
| hybrid | +0.1 | −2.5 | −19.4 | 3/7 | 10.0% | 29% / 4.3% | +6.2 + 9.6 |
| v0 | −0.4 | −2.5 | −11.7 | 3/7 | 2.9% | 8% / 3.0% | +2.0 + 13.2 |
| v2 by type | −0.4 | −4.0 | −19.2 | 4/7 | 10.0% | 20% / 3.1% | +5.0 + 10.2 |
| expected value | +0.9 | −7.5 | −42.8 | 3/7 | **18.6%** | 84% / 4.6% | +12.1 + 3.3 |
| logistic per type | −4.2 | −10.7 | −54.9 | 4/7 | 14.3% | 78% / 4.3% | +11.2 − 0.9 |
| v2 (offer default) | −5.5 | −7.2 | −15.2 | 2/7 | 8.6% | 18% / 3.0% | +4.6 + 5.1 |

**Top 15**

| Ranking | excess | w/o best | worst date | > all | hit | P prior / iso | offers + rest |
|---|---|---|---|---|---|---|---|
| return model, sector-neutral | **+18.2** | **+10.7** | 0.0 | **7/7** | 5.7% | 12% / 2.0% | +3.5 + 24.5 |
| value basket | +8.3 | +6.6 | −8.9 | 6/7 | 4.8% | 9% / 2.6% | +5.6 + 18.4 |
| return model | +7.9 | +3.9 | −6.9 | 6/7 | 5.7% | 10% / 2.3% | +3.1 + 19.5 |
| quality value basket | +4.5 | +3.4 | −3.0 | 5/7 | 2.9% | 7% / 2.5% | +0.7 + 19.6 |
| hybrid | +0.9 | −0.5 | −10.2 | 4/7 | 7.6% | 25% / 4.2% | +5.2 + 11.4 |
| v0 | +0.2 | −2.0 | −8.8 | 3/7 | 2.9% | 8% / 3.1% | +1.7 + 14.2 |
| v2 (offer default) | −0.9 | −2.6 | −9.1 | 3/7 | 6.7% | 14% / 3.0% | +4.2 + 10.2 |
| expected value | −2.0 | −7.4 | −35.8 | 3/7 | 12.4% | 79% / 4.6% | +7.9 + 5.3 |
| logistic per type | −3.3 | −8.4 | −45.8 | 4/7 | 12.4% | 70% / 4.4% | +8.2 + 3.5 |
| v2 by type | −3.6 | −6.7 | −18.4 | 1/7 | 6.7% | 16% / 2.9% | +3.3 + 8.4 |

What this says (descriptive; no default changes, as fixed above):
- **The rankings that find the most offers lose money in small portfolios.** The expected-value
  and per-type logistic top 10 got offers 14–19% of the time (7–8× the base rate) and the offers
  added +11–12 pp — but the other picks returned ~0, and one date lost 43–55 pp: they buy
  companies whose price already reflects the story, plus fragile names.
- **The probabilities are not usable at the top.** The prior-corrected P of the top 10 says
  78–84% where 14–19% happened; isotonic says ~4–5%. Neither is right for a single name.
- **What made money is the non-offer part**: value basket and return models earn +16–25 pp
  from the picks that did NOT get an offer. A take-private is a bonus, not the engine.
- **Top 10 vs top 15 is noise-sensitive**: the sector-neutral return model is the best top 15
  (+10.7 pp without its best date, 7 of 7 dates) and the worst top 10 (one date −82.5 pp).
  With 7 dates, a single holding can flip a 10-name portfolio. Prefer 15–25 names.
- Caveat: comparing 10 rankings × 6 sizes on the same 7 dates invites picking a lucky cell.
  The Portfolio tab shows every cell; only the live scoreboard can confirm a choice.

The **Portfolio construction** tab of the dashboard (`make app`) shows these numbers for every
ranking × size (top 10 → 200), the risk × return map, each date, and today's list.

## Limitations

- **Event date** = filing of the 240, not the announcement. A pre-announced offer can be public
  weeks earlier (the announcement is in TDnet, which is paid).
- **"parent" mixes a parent that buys with a parent that sells to a fund** under a tender
  agreement (both sit in the 特別関係者 G of the cover page). `bidder_stake` separates them for
  the 113 deals where the bidder itself holds ≥ 20%.
- **The activist list is curated today** (config/activists.yaml). Funds known as activist in
  2026 are applied to 2022, which is a look-ahead risk for activist_holder.
- **Ownership comes from the top-10 table only**: group stakes split across affiliates or held
  through pension trusts (近畿車輛: 30% in a 退職給付信託口 of 近畿日本鉄道) are not summed.
- **Homonyms** are excluded only when checked by hand (config/ownership.yaml). An automatic
  address rule dropped real parents and was abandoned.
- **Prices start 2021-10-08** (J-Quants Light), so years_below_book is a floor on early dates.
