# Week 11 change log (Week 10 to Week 11)

The Week 10 submission is preserved in three ways: the git tag `week10-submission` (see README), the copies in `docs/week10/` (README, progress report, contributions, video outline, and `app_week10.py`), and the fact that **no Week 10 model file or data file was modified**. Week 11 settings are layered on top in `data/model/week11_settings.json`.

| File | Change | Why |
|---|---|---|
| `code/11_evaluate_models.py` (new) | Split design, leakage checks, baselines with bootstrap intervals, error analysis, improvement trials, one-off backtest, settings output | Assignment parts 1 and 2 |
| `code/evaluation_utils.py` (new) | Version-independent player-grouped folds, cluster bootstrap, interval score | Reproducible numbers |
| `code/input_checks.py` (new) | Block / caution / do-not-rely rules, thresholds from training data | Part 3d |
| `code/value_model.py` | Optional `extrap_inflation` (default 1.0, so Week 10 JSON loads and predicts identically) | Widen ranges outside trained ages |
| `code/performance_model.py` | Optional per-group (goalkeeper / outfield) ranges via `set_group_ranges` (no change unless set) | Goalkeeper and defender range calibration |
| `code/explain.py` | Variable roles, side-by-side tables, rewritten summaries (comparison, trade-offs, reliability, limitations) | Parts 3b, 3c, 3e, 3f |
| `app/app.py` | Screen 6; trust strips; input checks; roles panel; comparison table and pin; applies Week 11 settings | Part 3 |
| `code/run_all.py` | Runs step 11 | Reproducibility |
| `tests/test_week11.py` (new) | 14 tests (input checks, backward compatibility, folds, settings) | Quality |
| `outputs/week11/*`, `data/model/week11_*` (new) | Evaluation tables, settings, revised backtest predictions | Evidence |
| `docs/*` | Evaluation report, user-test protocol and results template, video outline, contributions | Part 5 |

Not changed: `data/raw`, `data/final`, `data/withheld`, Week 10 model files, `statbomb_data/`.
