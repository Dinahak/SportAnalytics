# Week 11 Submission Checklist

Every requirement of the Week 11 brief, where it is covered, and what is still to do. Items marked **TO DO** need real people or real recordings and cannot be done in advance.

## 1. Evaluate predictive performance (both KPIs)
| Item | Where |
|---|---|
| a. Data split and why it fits the use | Report section 1; `outputs/week11/split_design.csv`; app screen 6, tab "How it was tested" |
| b. Data-leakage checks | Report section 1; `outputs/week11/leakage_checks.csv` (14 pass, 1 finding); app screen 6 |
| c. Metrics and their relevance | Report section 2 |
| d. Baseline comparison table, meaningful improvement | Report section 3; `kpi1_baseline_comparison.csv`, `kpi2_baseline_comparison.csv` (with 95% intervals); app screen 6 |
| e. Error analysis and implications | Report section 4; `kpi*_segments_*.csv`; app screen 6, tab "Where it fails" |
| Development kept apart from final evaluation | Rules at the top of `code/11_evaluate_models.py`; report section 1 (including the caveat about the 2025 test) |

## 2. Improve the models (both KPIs)
| Item | Where |
|---|---|
| a. Improvement ideas from the evaluation | Report section 5 (KPI 1: 2 attempts; KPI 2: 4 attempts plus model-choice re-check) |
| b. Same comparison procedure | Same player-grouped folds and pre-set rules for every attempt |
| c. Original vs revised, which kept and why | Report section 5; `kpi*_improvements.csv`, `kpi*_before_after_backtest.csv` |
| d. Remaining limitations | Report section 8; app screen 6, tab "What is still missing" |

## 3. Explanations and what-if
| Item | Where |
|---|---|
| a. Performance and limits in plain language | App screen 6 (summary cards) and the "How much to trust this" box on each what-if tab |
| b. Individual explanation vs overall importance | App screen 4 (Week 10) and contribution bars on screen 5 |
| c. Changeable vs fixed vs external inputs | Screen 5: tags on every input and the "Which inputs can I change?" panel; `explain.VARIABLE_ROLES` |
| d. Input checks, ranges, constraints, handling | `code/input_checks.py`; screen 5 shows Block / Do not rely / Use with caution with reasons |
| e. Baseline vs scenario side by side | Screen 5: "Baseline and scenario side by side" table, plus "Pin this scenario" for a second scenario |
| f. Summary with comparison, trade-offs, limitations, no causal claims | Screen 5 automatic summary (`explain.value_summary`, `explain.perf_summary`) |

## 4. User test  **TO DO**
- Run `docs/USER_TEST_PROTOCOL.md` with at least one person outside the group.
- Fill `docs/USER_TEST_RESULTS.md` with what really happened; remove every `[...]`.
- Make at least one app change from the findings (or explain why none) and record it in that file.
- Complete report section 7 (two or three sentences, pointing to the results file).

## 5. Deliverables
| Item | Status |
|---|---|
| Revised application | Done (`Predictive_Model/app/app.py`) |
| Shareable data and access instructions for restricted files | Done (README, "Data and access") |
| Setup and dependency instructions | Done (README, `requirements.txt`, `docs/SETUP_ON_YOUR_LAPTOP.md`) |
| Evaluation report | Done except section 7 (`docs/WEEK11_EVALUATION_REPORT.md`) |
| 7-minute demo video | **TO DO**: follow `docs/VIDEO_SCRIPT_WEEK11.md`, upload, put the link in the README |
| Updated contributions | **TO DO**: fill the Week 11 table in `docs/CONTRIBUTIONS.md` |
| Week 10 preserved | Done in `docs/week10/`; **TO DO** run the `git tag` command in the setup guide before pushing |
| Link shared with instructor and all members | **TO DO** |
