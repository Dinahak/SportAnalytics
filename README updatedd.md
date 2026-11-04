# SportAnalytics

Football recruitment decision-support tool. It helps a club's Head of Recruitment decide whether to bid for a player and up to what price, using two predictive KPIs:

| KPI | What it predicts | Model |
|---|---|---|
| 1 · Market value | Player's market value at the next yearly valuation | Log-scale regression with an age curve |
| 2 · Passes per 90 | Player's passes per 90 minutes next season | Gradient boosting (explained with SHAP) |

The app shows each model's accuracy and limits, overall predictor importance, per-player contributions, and a what-if tool with input checks, a baseline-versus-scenario table, automatic summaries and a downloadable scenario log.

**Status:** Week 11 (validation and usefulness). The Week 10 submission is preserved: git tag `week10-submission` and `docs/week10/`.

## Setup

Requires **Python 3.10 or newer**.

```bash
git clone https://github.com/MaurineTanui/SportAnalytics.git
cd SportAnalytics/Predictive_Model
pip install -r requirements.txt
streamlit run app/app.py
```

The app opens at http://localhost:8501. All model outputs are already in the repo, so you can run it right away.

Rebuild everything from the raw data (steps 01-10 took about 2 minutes in Week 10 and step 11 about 1.5; steps 01-10 need statsmodels and shap, step 11 needs only pandas, numpy, scipy and scikit-learn):

```bash
cd Predictive_Model
python code/run_all.py            # steps 01-11
python code/11_evaluate_models.py # only the Week 11 evaluation
python -m unittest discover tests -v   # 14 tests
```

**Dependencies:** `Predictive_Model/requirements.txt` (streamlit, pandas, numpy, plotly, scipy, scikit-learn, statsmodels, shap). The Week 11 evaluation was run with Python 3.12, scikit-learn 1.8, numpy 2.4, pandas 3.0, scipy 1.17. Its folds do not depend on the scikit-learn version, but tree-model numbers can differ in the last digits across versions.

## Repository layout

```
Predictive_Model/
  app/app.py              Streamlit application (6 screens)
  code/                   Pipeline steps 01-11 and model code
    value_model.py        KPI 1 model
    performance_model.py  KPI 2 model
    explain.py            Importance, contributions, roles, comparison tables, summaries
    input_checks.py       Week 11: plausible ranges, constraints, block / caution rules
    evaluation_utils.py   Week 11: deterministic folds, bootstrap, interval score
    11_evaluate_models.py Week 11: leakage checks, baselines, error analysis, improvements
  tests/test_week11.py    Unit tests
  data/raw/               Week 5 synthetic data (unchanged)
  data/final/             Cleaned tables the app reads
  data/model/             Model files, training tables, forecasts, week11_settings.json
  data/withheld/          Simulated outcomes held back for validation
  outputs/                Model comparison, backtests, validation results
  outputs/week11/         Week 11 evaluation tables and summary
statbomb_data/            Scripts and docs used to calibrate the synthetic data
docs/                     Reports, user test, contributions, video outline
  week10/                 Preserved copies of the Week 10 submission
```

## Data and access

| Data | Shared? | How to get it |
|---|---|---|
| All synthetic tables in `Predictive_Model/data/` (raw, final, model, withheld) | **Yes, included in full** | In this repo. No real players or clubs. |
| StatsBomb Open Data (used only to calibrate the synthetic data) | No: its user agreement forbids redistribution | `python statbomb_data/01_fetch_statsbomb.py --out <folder>`; see `statbomb_data/LICENSE.pdf` |
| Transfermarkt market values (`transfermkt_data/`, git-ignored) | No: Transfermarkt's terms of use | Authorized users can request the files from the group. Nothing in the app or the evaluation needs them. |

## Deliverables

**Week 11**
- Evaluation report: [`docs/WEEK11_EVALUATION_REPORT.md`](docs/WEEK11_EVALUATION_REPORT.md)
- User test: [`docs/USER_TEST_PROTOCOL.md`](docs/USER_TEST_PROTOCOL.md), [`docs/USER_TEST_RESULTS_Nov3.md`](docs/USER_TEST_RESULTS_Nov3.md)
- What changed: [`docs/CHANGELOG_WEEK11.md`](docs/CHANGELOG_WEEK11.md)
- Checklist: [`docs/WEEK11_SUBMISSION_CHECKLIST.md`](docs/WEEK11_SUBMISSION_CHECKLIST.md) · Laptop setup: [`docs/SETUP_ON_YOUR_LAPTOP.md`](docs/SETUP_ON_YOUR_LAPTOP.md)
- Contributions: [`docs/CONTRIBUTIONS_week_11_new.md`](docs/CONTRIBUTIONS_week_11_new.md)
- Demo video (7 min): [demo video](https://drive.google.com/file/d/1eX31Hnl_FvlV6cS-5l0x7f7OTD05QmNT/view?usp=sharing); outline in [`docs/VIDEO_SCRIPT_WEEK11.md`](docs/VIDEO_SCRIPT_WEEK11.md)

**Week 10 (preserved)**: [`docs/week10/`](docs/week10/) and git tag `week10-submission`.
