# Week 11 Evaluation Report

**Project:** Football Recruitment Decision-Support Tool · **Decision-maker:** Head of Recruitment
**KPIs (unchanged from Week 10):** KPI 1 market value at the next yearly valuation · KPI 2 passes per 90 minutes next season
**Reproduce:** `cd Predictive_Model && python code/11_evaluate_models.py` (about 1.5 minutes on the machine we tested; needs pandas, numpy, scipy, scikit-learn only). Tables are in `Predictive_Model/outputs/week11/`.
**Data:** all synthetic (no real players or clubs).

---

## 1. Evaluation design and data-leakage checks

**Split.** Time-ordered, matching how the tool is used (a forecast is made before its outcome exists).

| KPI | Development (choices made here) | Backtest (final evaluation) | Live (no outcome yet) |
|---|---|---|---|
| 1 Value | June 2023 and 2024 forecasts: 352 rows, 186 players, ages 16.5 to 32.9 | June 2025: 195 players, ages 16.8 to 33.9 | June 2026: 200 players |
| 2 Passes | Seasons 2023 and 2024: 334 rows, 175 players | Season 2025: 187 players | Season 2026: 198 players |

Inside development, models are compared by 5-fold cross-validation repeated 3 times, **split by player** so a player's rows are never on both sides. Folds come from `evaluation_utils.group_folds`, not scikit-learn's `GroupKFold`, whose fold assignment changed between versions and moved the Week 10 CV numbers by about 0.2 passes per 90.

**Development vs final evaluation.** Everything chosen (model, improvements, adjustment factors) was decided on development data using rules written at the top of `11_evaluate_models.py` before results were looked at. The 2025 backtest was run once per model version.

> **Caveat.** The 2025 backtest numbers were already visible in Week 10, and the pattern of errors there prompted some of the improvement ideas. We then required each idea to pass on development data. It is therefore *not a pristine test set*. The first untouched test is the January 2027 outcome of the June 2026 forecasts.

**Leakage checks (`outputs/week11/leakage_checks.csv`): 14 pass, 1 finding, 0 fail.**
- KPI 1: latest valuation and season stats were published before the forecast date (21 June); predictors recomputed independently from the raw valuation history match the model table; the target valuation was not yet published; age is measured at the forecast date; development outcomes were known before the backtest forecasts; the chosen model uses only value, age and last year's change (the asking price and performance statistics are not predictors).
- KPI 2: last season's and the season-before statistics were published before the June run; the target season's were not; age comes from the date of birth; Week 5 pass figures and the target season are not predictors.
- **Finding:** missing carries values were filled with position medians computed over *all* rows (including backtest and live). Only predictor values are involved, never outcomes. Re-running with development-only medians changes the backtest RMSE from 13.479 to 13.478, so we kept the Week 10 code and documented it.
- Players appear in both development and backtest (186 of 195 for KPI 1). That is normal forecasting, and cross-validation splits by player so it cannot inflate development scores.

## 2. Metrics and why

| KPI | Metrics | Why they matter to the Head of Recruitment |
|---|---|---|
| 1 | Typical error (median absolute % error); RMSE on the log scale; share within 20%; **share of forecasts more than 20% too high** (overpaying risk); coverage of the 80% range; **how often the "pay no more than" price held** | Errors are proportional to value; overpaying is the costly direction; the tool gives a price and a chance, so those must be honest |
| 2 | Typical error (MAE) and RMSE in passes per 90; R²; share within 5 passes; coverage of the 80% range; **correct call on "reaches the need's target style (54.3)"** | The decision is fit to a playing style, so a threshold call matters more than the average error |

## 3. Performance against simple baselines

**KPI 1** (lower is better; 2025 backtest, 195 players)

| Model | Typical error | RMSE (log) | Within 20% | Forecast >20% too high |
|---|---|---|---|---|
| No change in value | 28.6% | 0.488 | 39% | 28.7% |
| Week 6 rule (fixed age adjustment) | 22.6% | 0.438 | 42% | 27.7% |
| **Week 10 age-curve model** | **12.1%** | **0.277** | **72%** | **11.8%** |

RMSE gain over "no change" 0.211 (95% interval 0.134 to 0.289); over the Week 6 rule 0.161 (0.087 to 0.237). Intervals resample whole players and exclude zero, so the improvement is meaningful. The price marked **75% sure held for 82%** of players, the 90% price for 94%, the 50% price for 54%: the prices are reliable, slightly conservative.

**KPI 2** (2025 backtest, 187 players)

| Model | MAE | RMSE | R² | Within 5 passes |
|---|---|---|---|---|
| No change from last season | 12.29 | 17.27 | 0.09 | 32% |
| Position average | 11.57 | 15.98 | 0.22 | 29% |
| **Week 10 gradient boosting** | **9.28** | **13.48** | **0.44** | **40%** |

MAE reduction versus "no change" 3.01 passes (1.81 to 4.26); versus position average 2.30 (1.37 to 3.22). Meaningful.

**Decision view (midfielders, target 54.3 passes per 90, 53 players, 13 who reached it).** The model calls the target right 77% of the time (72% for "no change"), but finds only **23%** of the players who really reached it (46% for "no change"), and 60% of those it calls a fit really fit. It is conservative: a forecast below the target should not by itself rule a player out. Groups are small, so read these percentages as rough.

## 4. Where the models fail (error analysis)

Full tables: `kpi1_segments_*.csv`, `kpi2_segments_*.csv`.

**KPI 1.**
- **Players older than anyone in training (11 of 195, aged 33 or more).** Typical error 56%; all 11 forecast too low; the 80% range held for 0 of 11. Development data cannot show this (it contains no one that old). The yearly age curve is estimated from one or two players at each such age.
- This is *not* a general 30+ problem: for 30+ players inside the training ages the typical error is 10% and the range held for 81%.
- Low-value players: forecast slightly too low (bias +0.13 log) and the range held 75%.
- Players without a valuation from a year ago (9): wider uncertainty (the model already uses a larger spread for them).

**KPI 2.**
- Defenders (bias -5.7, range held 66%) and ages 26 to 29 (bias -4.4, range held 63%) were under-forecast in 2025. This is not present in development data. Defenders' average passing was about 4 passes per 90 higher in 2025 than in 2023 and 2024, a one-position shift the model cannot see.
- High passers (over 50 per 90): largest errors (12.0), range held least (69% test, 65% development), exactly where the target style sits.
- Goalkeepers: MAE 4.6, but their range was far too wide (98% held) because one pooled error distribution was used.

**Implications for users.** Treat KPI 1 forecasts for players older than the training ages as guesses; treat KPI 2 as a rough guide, least reliable for high passers and for defenders; do not read a below-target KPI 2 forecast as a rejection.

## 5. Improvements attempted and what was retained

Rules (written before running): a point-forecast change needs at least a 2% lower CV RMSE and a win in at least 2 of 3 CV repeats; a range change needs coverage closer to 80% across groups *and* a lower interval score; the extrapolation factor is used only if a development stress test shows ranges more than 1.5 times too narrow.

| KPI | Attempt | Evidence (development only) | Decision |
|---|---|---|---|
| 1 | **A.** Pool the oldest part of the age curve into one straight line | CV RMSE 0.2122 vs 0.2121; wins 1 of 3 | **Rejected** |
| 1 | **B.** Widen the range for players older than the training ages | Stress test (train on younger players, test on older: 169 rows, 43 players): ranges were **2.2 times too narrow** (95% CI 1.75 to 2.61); in-range value 1.01 | **Adopted** (x2.21 for the June 2025 model, x2.61 for the live model, estimated on development plus backtest) |
| 2 | **A.** Separate ranges for goalkeepers and outfield players | Cross-fitted by player halves: worst coverage gap from 80% fell from 16.7 to 1.0 points; interval score 38.8 to 35.2 | **Adopted** |
| 2 | **B.** Add "change since season before" and "passes vs position average" | CV RMSE 12.325 vs 12.316 | **Rejected** |
| 2 | **C1.** Simpler model (depth 1, 300 trees) | -1.2% RMSE, wins 3 of 3, below the 2% threshold | **Rejected** (promising; retest with more data) |
| 2 | **C2.** Predict the change from last season | +1.6% worse, wins 0 of 3 | **Rejected** |
| 2 | Model choice re-run under the new folds | Random forest ranks 1st, gradient boosting 4th, gap 0.1 passes (0.8%) is about the size of run-to-run noise (0.09 to 0.11) | Kept gradient boosting |

**One-off backtest, Week 10 vs retained (point forecasts are identical in both):**

| | Week 10 | Week 11 retained |
|---|---|---|
| KPI 1: 80% range held, all players | 80.0% | 82.1% |
| KPI 1: 80% range held, 11 oldest players | 0% | 36% |
| KPI 1: 95% range held, 11 oldest players | 18% | 55% |
| KPI 2: 80% range held, goalkeepers / defenders / midfielders / forwards | 98 / 66 / 77 / 77% | 84 / 74 / 83 / 82% |
| KPI 2: typical 80% range width, goalkeepers | 26.8 passes | 13.6 passes |

For information only (not used to decide): the pooled-tail model (attempt A) gave backtest RMSE 0.260 vs 0.277 but the same typical error (12.1%). The evidence from 11 players is too thin to overturn the development result.

**Retained models:** KPI 1 = Week 10 age-curve regression with widened ranges outside trained ages. KPI 2 = Week 10 gradient boosting with group-specific ranges. Week 10 model files are untouched; the app applies `data/model/week11_settings.json` on top.

## 6. Application changes (Week 11)

- **Validation and limits** screen in plain language; trust strip on each what-if tab.
- **Which inputs can I change?** Each input is tagged *You decide* (asking price), *External condition* or *Fixed characteristic*; the summary says whether a scenario is something the club could do.
- **Input checks (`code/input_checks.py`).** *Block* (no forecast): negative or missing values, age outside 16 to 40, a player too young for the history entered, a 10-fold jump in a year, an asking price 20 times the value, more than 3,420 minutes, impossible passes. *Do not rely*: age outside the training ages; fewer than 900 minutes. *Use with caution*: values outside the 1st to 99th percentile of training data.
- **Baseline and scenario side by side**, with input changes, predictions and differences, and an optional pinned second scenario.
- **Automatic summary** now covers the comparison, trade-offs, reliability and limitations, and states that differences are model-based, not proven effects or guaranteed outcomes.

## 7. User testing

Full record: `docs/USER_TEST_RESULTS.md` (protocol: `docs/USER_TEST_PROTOCOL.md`).

**Participant:** one football fan who understands positions, transfers and fees but has no recruitment experience. This is a **substitute** for the Head of Recruitment, so the test speaks to usability and wording more than to decision value.

**Findings:**
- The participant navigated the screens, changed the asking price, read the predicted market value and found the warning for unrealistic inputs without help.
- They were unsure why the asking price changed "Chance fee holds" but not the forecast.
- They hesitated at the Pin button.
- They found the amount of information overwhelming at first.
- They saw the tool as useful for discussing whether an asking price is reasonable, and wanted simpler explanations of probabilities and comparisons.

**Changes made:**
- Their suggested sentence now sits under the asking-price input.
- A one-line instruction now sits next to the Pin button.

**Not changed:** a simplified first view, which needs more evidence from a real recruiter.

## 8. Remaining limitations and priorities

1. **Synthetic data.** Effects are illustrative; refit both models on real valuations and event data.
2. **Few old players** (11 in the test, 53 aged 30+ in development). The oldest part of the age curve is KPI 1's weakest point; collect more older players before relying on it.
3. **Short history** (two years to build, one to test): whole-position shifts like the 2025 defenders' rise cannot be learned.
4. **Omitted drivers.** KPI 1: injuries, contract length, form, buying club. KPI 2: the new team's style.
5. **Information load** (user test): a simplified first view for non-experts, to be tested with a real recruiter.
6. **Under-calling high passers** (23% found). Consider a classification model for "reaches target", or a calibrated threshold.
7. **Untouched test.** Score the June 2026 forecasts in January 2027 without changing them first.
8. **No cost-of-error score.** Errors are counted in % and passes, not in the money a wrong bid would cost.
9. The widening factor (2.2) rests on 169 rows from 43 players; its interval is wide (1.75 to 2.61), and it still does not rescue the 11 oldest backtest players.
