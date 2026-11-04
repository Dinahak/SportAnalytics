"""
11_evaluate_models.py
Week 11. Validates the two KPI models, tests improvements, and writes what the application shows.
Needs only pandas, numpy, scipy and scikit-learn (no statsmodels or shap), and never reads data/withheld/.

DESIGN (fixed before the results were looked at)
  Split             Time-ordered. Development = forecasts made in June 2023 and June 2024 (outcomes known by January 2025 /
                    the 2024 season). Backtest = forecasts made in June 2025 (outcomes January 2026 / season 2025).
                    The live June 2026 forecasts have no outcome yet, so they are never scored.
                    Inside development, models are compared by 5-fold cross-validation repeated 3 times, split by player
                    (a player's rows are never in the training and test part of the same fold). Folds come from
                    evaluation_utils.group_folds so they do not depend on the scikit-learn version.
  Model development Everything that is chosen or tuned (model, settings, improvements, adjustment factors) is decided on
                    development data only, with the decision rules below.
  Final evaluation  The 2025 backtest is run ONCE for the Week 10 model and once for the retained revised model.
                    CAVEAT: the 2025 backtest numbers were already shown in Week 10, and the team saw the pattern of errors
                    there, so the 2025 backtest is not a pristine test set. The first genuinely untouched test is the
                    January 2027 outcome of the June 2026 forecasts (see the report, 'priorities for future work').

DECISION RULES (written before running)
  Point-forecast change   Adopt only if the repeated-CV RMSE improves by at least 2% AND the new version wins in at least
                          2 of the 3 CV repeats (same folds for both versions).
  Prediction-range change Adopt only if, in cross-fitted development data, it makes the 80% range coverage of the groups
                          closer to 80% AND lowers the mean 80% interval score.
  Extrapolation factor    Adopt a widening factor for out-of-range ages only if a stress test on development data (train on
                          younger players, test on older ones) gives a root-mean-square standardised error above 1.5.
  Retained model          The version that satisfies its rule. If none does, the Week 10 model is kept.

Outputs: outputs/week11/*.csv, outputs/week11/week11_summary.json, data/model/week11_settings.json,
         data/model/week11_value_backtest_predictions.csv, data/model/week11_perf_backtest_predictions.csv
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

import config as C
from evaluation_utils import band, cluster_bootstrap, group_folds, interval_score, segment_table
from performance_model import FEATURES, POSITIONS, QUANTILE_GRID, PerformanceForecastModel, make_estimator
from value_model import ValueForecastModel, interval, point, price_for_confidence

warnings.filterwarnings("ignore")
OUT = C.OUTPUTS / "week11"
OUT.mkdir(parents=True, exist_ok=True)
REPS, FOLDS = 3, 5
ADOPT_MIN_REL_GAIN = 0.02          # 2% lower CV RMSE
EXTRAP_RULE_THRESHOLD = 1.5
SUMMARY = {"environment": {"python": sys.version.split()[0], "scikit_learn": sklearn.__version__, "numpy": np.__version__,
                           "pandas": pd.__version__}}


def save(df, name, digits=4):
    df.round(digits).to_csv(OUT / name, index=False)
    return df


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


# ====================================================================== load
vt = pd.read_csv(C.MODEL / "value_model_table.csv")
pt = pd.read_csv(C.MODEL / "performance_model_table.csv")
va = pd.read_csv(C.FINAL / "valuation_history.csv", parse_dates=["valuation_date", "available_from_timestamp"])
stt = pd.read_csv(C.FINAL / "player_season_stats.csv", parse_dates=["available_from_timestamp"])
players = pd.read_csv(C.FINAL / "player.csv", parse_dates=["date_of_birth"])
need = pd.read_csv(C.FINAL / "squad_need.csv")
vdev, vbt, vlive = (vt[vt.split == s].reset_index(drop=True) for s in ("development", "backtest", "live"))
pdev, pbt, plive = (pt[pt.split == s].reset_index(drop=True) for s in ("development", "backtest", "live"))
pj = json.loads((C.MODEL / "performance_model.json").read_text())
PNAME, PPARAMS = pj["name"], pj["params"]
print(f"[11] KPI 1 rows: development {len(vdev)}, backtest {len(vbt)}, live {len(vlive)} | "
      f"KPI 2 rows: development {len(pdev)}, backtest {len(pbt)}, live {len(plive)}")

# ====================================================================== 1a. split design
design = pd.DataFrame([
    {"kpi": "KPI 1 Market value", "part": "Development (model choice, improvements)", "forecast_origins": "21 Jun 2023, 21 Jun 2024",
     "rows": len(vdev), "players": vdev.player_id.nunique(), "outcome_period": "Jan 2024, Jan 2025",
     "age_range": f"{vdev.age_at_origin.min():.1f} to {vdev.age_at_origin.max():.1f}"},
    {"kpi": "KPI 1 Market value", "part": "Backtest (final evaluation, run once)", "forecast_origins": "21 Jun 2025",
     "rows": len(vbt), "players": vbt.player_id.nunique(), "outcome_period": "Jan 2026",
     "age_range": f"{vbt.age_at_origin.min():.1f} to {vbt.age_at_origin.max():.1f}"},
    {"kpi": "KPI 1 Market value", "part": "Live (not scorable yet)", "forecast_origins": "21 Jun 2026", "rows": len(vlive),
     "players": vlive.player_id.nunique(), "outcome_period": "Jan 2027 (unknown)",
     "age_range": f"{vlive.age_at_origin.min():.1f} to {vlive.age_at_origin.max():.1f}"},
    {"kpi": "KPI 2 Passes per 90", "part": "Development (model choice, improvements)", "forecast_origins": "seasons 2023, 2024 (June runs)",
     "rows": len(pdev), "players": pdev.player_id.nunique(), "outcome_period": "seasons 2023, 2024",
     "age_range": f"{pdev.age.min():.1f} to {pdev.age.max():.1f}"},
    {"kpi": "KPI 2 Passes per 90", "part": "Backtest (final evaluation, run once)", "forecast_origins": "season 2025 (June 2025 run)",
     "rows": len(pbt), "players": pbt.player_id.nunique(), "outcome_period": "season 2025",
     "age_range": f"{pbt.age.min():.1f} to {pbt.age.max():.1f}"},
    {"kpi": "KPI 2 Passes per 90", "part": "Live (not scorable yet)", "forecast_origins": "season 2026 (June 2026 run)", "rows": len(plive),
     "players": plive.player_id.nunique(), "outcome_period": "season 2026 (unknown)",
     "age_range": f"{plive.age.min():.1f} to {plive.age.max():.1f}"},
])
save(design, "split_design.csv")

# ====================================================================== 1b. leakage checks
checks = []


def check(kpi, name, ok, detail):
    checks.append({"kpi": kpi, "check": name, "result": "PASS" if ok else "FAIL", "detail": detail})
    print(f"[11] leakage check {'PASS' if ok else 'FAIL'} · {kpi} · {name}")


K1 = "KPI 1"
origin_v = pd.to_datetime(vt.forecast_origin)
check(K1, "Latest valuation was published before the forecast origin", bool((pd.to_datetime(vt.value_available_ts) < origin_v).all()),
      "value_available_ts < forecast_origin for every row")
has_st = vt.stats_available_ts.notna()
check(K1, "Season statistics were published before the forecast origin", bool((pd.to_datetime(vt.stats_available_ts[has_st]) < origin_v[has_st]).all()),
      "stats_available_ts < forecast_origin for every row that has statistics")
check(K1, "Target valuation is dated after the forecast origin", bool((pd.to_datetime(vt.target_value_date) > origin_v).all()),
      "target_value_date > forecast_origin for every row")
# independent recomputation of the predictors straight from the raw valuation history
bad_now = bad_prev = bad_target_pub = 0
for r in vt.itertuples():
    origin = pd.Timestamp(r.forecast_origin)
    pv = va[(va.player_id == r.player_id) & (va.available_from_timestamp < origin)].sort_values("valuation_date")
    if abs(pv.iloc[-1].market_value_eur - r.market_value_eur) > 0.01:
        bad_now += 1
    prev = pv[pv.valuation_date.dt.year == r.origin_year - 1]
    if r.has_history and (len(prev) == 0 or abs(prev.market_value_eur.iloc[0] - r.prev_value_eur) > 0.01):
        bad_prev += 1
    tv = va[(va.player_id == r.player_id) & (va.valuation_date.dt.year == r.origin_year + 1)]
    if len(tv) and not (tv.available_from_timestamp > origin).all():
        bad_target_pub += 1
check(K1, "Predictors recomputed from the raw valuation history match the model table", bad_now == 0 and bad_prev == 0,
      f"{bad_now} mismatches in 'value now', {bad_prev} in 'value one year ago', out of {len(vt)} rows")
check(K1, "The target valuation was not yet published at the origin", bad_target_pub == 0, f"{bad_target_pub} rows where it was")
dob = players.set_index("player_id").date_of_birth
age_chk = np.abs(((origin_v - vt.player_id.map(dob)).dt.days / 365.25) - vt.age_at_origin).max()
check(K1, "Age is measured at the forecast origin, not at the outcome date", bool(age_chk < 0.01), f"largest difference {age_chk:.4f} years")
dev_label_date = pd.to_datetime(vdev.target_value_date).max()
bt_origin = pd.to_datetime(vbt.forecast_origin).min()
check(K1, "Development outcomes were already known when the backtest forecasts were made", bool(dev_label_date <= bt_origin),
      f"last development outcome {dev_label_date.date()} <= first backtest origin {bt_origin.date()}")
final_names = json.loads((C.MODEL / "value_model.json").read_text())["names"]
check(K1, "The chosen model uses only value now, age and last year's change (no statistics, no asking price)",
      all(n.startswith("age_") or n == "momentum" for n in final_names), f"model terms: {len(final_names)} age points + momentum")
overlap = len(set(vbt.player_id) & set(vdev.player_id))
checks.append({"kpi": K1, "check": "INFO: players in both development and backtest", "result": "INFO",
               "detail": f"{overlap} of {len(vbt)} backtest players also appear in development. This is how forecasting works (their earlier "
                         "outcomes were known), and cross-validation splits by player so it cannot inflate the development scores."})

K2 = "KPI 2"
ps = stt.set_index(["player_id", "season_year"])
orig2 = pd.to_datetime(pt.origin_year.astype(str) + "-" + C.FORECAST_ORIGIN_MONTH_DAY)
bad_last = bad_before = bad_target = 0
for r, o in zip(pt.itertuples(), orig2):
    if (r.player_id, r.forecast_season - 1) not in ps.index or ps.loc[(r.player_id, r.forecast_season - 1)].available_from_timestamp >= o:
        bad_last += 1
    if r.has_before and ((r.player_id, r.forecast_season - 2) not in ps.index or ps.loc[(r.player_id, r.forecast_season - 2)].available_from_timestamp >= o):
        bad_before += 1
    if (r.player_id, r.forecast_season) in ps.index and ps.loc[(r.player_id, r.forecast_season)].available_from_timestamp < o:
        bad_target += 1
check(K2, "Last season's statistics were published before the June run", bad_last == 0, f"{bad_last} violations in {len(pt)} rows")
check(K2, "The season-before statistics were published before the June run", bad_before == 0, f"{bad_before} violations")
check(K2, "The target season's statistics were NOT yet published at the June run", bad_target == 0, f"{bad_target} rows where they were")
age2 = np.abs(((pd.to_datetime(pt.forecast_season.astype(str) + "-08-01") - pt.player_id.map(dob)).dt.days / 365.25) - pt.age).max()
check(K2, "Age is the age at the start of the forecast season (known from the date of birth)", bool(age2 < 0.05), f"largest difference {age2:.3f} years")
check(K2, "Development outcomes were already known when the backtest forecasts were made",
      bool(stt[stt.season_year == 2024].available_from_timestamp.max() <= pd.Timestamp("2025-06-21")),
      "2024 season statistics were published on 1 June 2025, before the 21 June 2025 backtest run")
check(K2, "The model never sees the target season, the Week 5 pass figures or the asking price",
      set(FEATURES) <= set(pt.columns) and not any("week5" in f or "target" in f for f in FEATURES), f"predictors: {', '.join(FEATURES)}")

# one real (minor) finding: missing carries are filled with the position median computed on ALL rows
med_all = pt.groupby("position").carries_last_raw.median()
med_dev = pdev.groupby("position").carries_last_raw.median()
changed_rows = int(((pt.carries_missing == 1) & (pt.split.isin(["backtest", "live"]))).sum())
checks.append({"kpi": K2, "check": "FINDING: carries imputation uses position medians from all rows", "result": "WARN",
               "detail": f"Week 10 filled {int(pt.carries_missing.sum())} missing carries values with medians taken over development, backtest and live rows "
                         f"({changed_rows} of them in backtest/live rows). Only predictor values are involved, never outcomes, so it is a minor "
                         "leak of distribution information. Effect on the backtest is measured below."})
save(pd.DataFrame(checks), "leakage_checks.csv")

# ====================================================================== KPI 1
print("\n[11] ===== KPI 1: market value =====")


def week6_adjustment(age):
    for limit, adj in C.FMV_AGE_ADJUSTMENT:
        if age <= limit:
            return adj
    return C.FMV_AGE_ADJUSTMENT[-1][1]


def fit_v(df, make=lambda: ValueForecastModel(1, True, False)):
    return make().fit(df.market_value_eur, df.age_at_origin, df.prev_value_eur, df.target_value_eur)


def pred_v(m, df):
    return m.predict(df.market_value_eur, df.age_at_origin, df.prev_value_eur)


class TailValueModel(ValueForecastModel):
    """Improvement attempt A (KPI 1): the age curve keeps one free point per year up to `tail_from`, and from there on a
    single straight line, so that the oldest ages are estimated from every older player instead of one or two."""

    def __init__(self, tail_from=30):
        super().__init__(1, True, False)
        self.tail_from = tail_from

    def _design(self, value_now, age, prev_value):
        a = np.clip(age, self.age_min, self.age_max)
        X = [np.clip(1 - np.abs(np.minimum(a, self.tail_from)[:, None] - self.knots[None, :]) / self.knot_step, 0, None),
             np.clip(a - self.tail_from, 0, None)[:, None]]
        mom = np.where(np.isnan(prev_value), 0.0, np.log(value_now / np.where(np.isnan(prev_value), value_now, prev_value)))
        return np.hstack(X + [mom[:, None]])

    def fit(self, value_now, age, prev_value, value_next):
        value_now, age, value_next = self._arr(value_now), self._arr(age), self._arr(value_next)
        prev_value = self._arr(prev_value, len(age))
        self.age_min, self.age_max = float(age.min()), float(age.max())
        self.knots = np.arange(np.floor(self.age_min), self.tail_from + 1e-9, 1.0)
        self.level_centre = float(np.log(value_now).mean())
        X, y = self._design(value_now, age, prev_value), np.log(value_next / value_now)
        hist = ~np.isnan(prev_value)
        self.n, self.p = X.shape
        self.df = self.n - self.p
        w = np.ones(self.n)
        for _ in range(3):
            coef, *_ = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)
            resid = y - X @ coef
            pooled = float(resid @ resid / self.df)
            s2 = {g: float(np.mean(resid[hist == g] ** 2) * self.n / self.df) if (hist == g).sum() > self.p else pooled for g in (True, False)}
            w = 1 / np.where(hist, s2[True], s2[False])
        self.coef, *_ = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)
        self.sigma2_history, self.sigma2_no_history = s2[True], s2[False]
        self.cov = np.linalg.inv(X.T @ (X * w[:, None]))
        self.names = [f"age_{k:g}" for k in self.knots] + ["age_slope_after_tail", "momentum"]
        return self


def v_metrics(y_change, pred_change):
    e = np.asarray(y_change) - np.asarray(pred_change)
    ape = np.abs(np.exp(-e) - 1)                         # |forecast / actual - 1|
    return {"rmse_log": rmse(e), "typical_error_pct": 100 * float(np.median(ape)), "within_20_pct": 100 * float(np.mean(ape <= 0.2)),
            "bias_log": float(e.mean()), "overpay_risk_pct": 100 * float(np.mean(np.exp(-e) > 1.2))}   # forecast >20% above actual


def cv_value(df, make, reps=REPS):
    """Repeated CV split by player. Returns per-repeat predicted log change and in-80%-range flags (reps x n)."""
    pc, in80, iscore = (np.zeros((reps, len(df))) for _ in range(3))
    for r in range(reps):
        fold = group_folds(df.player_id.values, FOLDS, r)
        for k in range(FOLDS):
            tr, te = df[fold != k], df[fold == k]
            assert not set(tr.player_id) & set(te.player_id)       # leakage guard: no player on both sides
            m = fit_v(tr, make)
            pr = pred_v(m, te)
            lo, hi = interval(pr, 0.8)
            pc[r, fold == k] = pr["log_change"]
            in80[r, fold == k] = (te.target_value_eur.values >= lo) & (te.target_value_eur.values <= hi)
            iscore[r, fold == k] = interval_score(lo, hi, te.target_value_eur.values)
    return {"pred": pc, "in80": in80, "iscore": iscore, "rmse_by_rep": [rmse(df.log_change.values - pc[r]) for r in range(reps)]}


# ---- baselines and model, development CV and backtest
w6_dev = np.log(1 + vdev.age_at_origin.map(week6_adjustment).values)
w6_bt = np.log(1 + vbt.age_at_origin.map(week6_adjustment).values)
cv1 = cv_value(vdev, lambda: ValueForecastModel(1, True, False))
m1_dev = fit_v(vdev)
pr_bt = pred_v(m1_dev, vbt)
rows = []
tile = lambda a: np.tile(a, REPS)              # development metrics use all CV repeats stacked, as in Week 10
cands = {"No change (value stays the same)": (np.zeros(len(vdev)), np.zeros(len(vbt))),
         "Week 6 rule (fixed age adjustment)": (w6_dev, w6_bt),
         "Week 10 model (age curve + momentum)": (cv1["pred"], pr_bt["log_change"])}
boot = {}
for name, (pd_, pb_) in cands.items():
    cvm = v_metrics(tile(vdev.log_change.values), pd_.ravel() if pd_.ndim > 1 else tile(pd_))
    btm = v_metrics(vbt.log_change.values, pb_)
    rows.append({"model": name, **{f"cv_{k}": v for k, v in cvm.items()}, **{f"backtest_{k}": v for k, v in btm.items()}})
    if not name.startswith("Week 10"):
        e_base = vbt.log_change.values - pb_
        e_mod = vbt.log_change.values - pr_bt["log_change"]
        boot[name] = cluster_bootstrap(lambda a, b: rmse(a) - rmse(b), [e_base, e_mod], vbt.player_id.values)
t1 = pd.DataFrame(rows)
t1["backtest_rmse_gain_vs_this_baseline_ci95"] = [f"{boot[n][0]:.3f} ({boot[n][1]:.3f} to {boot[n][2]:.3f})" if n in boot else "" for n in t1.model]
# coverage / decision metrics for the model
bt_in80 = ((vbt.target_value_eur >= interval(pr_bt, .8)[0]) & (vbt.target_value_eur <= interval(pr_bt, .8)[1])).values
t1.loc[t1.model.str.startswith("Week 10"), "cv_coverage_80_pct"] = 100 * cv1["in80"].mean()
t1.loc[t1.model.str.startswith("Week 10"), "backtest_coverage_80_pct"] = 100 * bt_in80.mean()
save(t1, "kpi1_baseline_comparison.csv")
SUMMARY["kpi1_baseline"] = t1.round(4).to_dict("records")
print(t1[["model", "cv_rmse_log", "cv_typical_error_pct", "backtest_rmse_log", "backtest_typical_error_pct"]].round(3).to_string(index=False))
print("[11] paired bootstrap (RMSE of baseline minus RMSE of model, backtest):", {k: tuple(round(x, 3) for x in v) for k, v in boot.items()})

# ---- decision metric: does the 'pay no more than' price hold?
hit = []
for conf in C.CONFIDENCE_PRICE_LEVELS:
    price = price_for_confidence(pr_bt, conf)
    hit.append({"stated_confidence_pct": 100 * conf, "actual_value_at_or_above_price_pct": 100 * float((vbt.target_value_eur.values >= price).mean()),
                "players": len(vbt)})
save(pd.DataFrame(hit), "kpi1_price_hit_rate.csv")
SUMMARY["kpi1_price_hit_rate"] = hit

# ---- error analysis: where does it fail (development out-of-fold, then backtest)
def seg_frame_v(df, change_pred, in80):
    d = df[["player_id", "position", "age_at_origin", "market_value_eur", "has_history"]].copy()
    d["err"] = df.log_change.values - change_pred
    d["abs_pct"] = 100 * np.abs(np.exp(-d.err) - 1)
    d["in80"] = in80
    d["age_band"] = band(d.age_at_origin, [0, 19, 22, 26, 30, 99], ["Under 19", "19 to 21", "22 to 25", "26 to 29", "30 and over"])
    edges = vdev.market_value_eur.quantile([0, 1 / 3, 2 / 3, 1]).values.copy()
    edges[0], edges[-1] = 0, np.inf                                  # value tiers use development terciles
    d["value_tier"] = band(d.market_value_eur, edges, ["Low value", "Mid value", "High value"])
    d["history"] = np.where(d.has_history == 1, "Has valuation one year ago", "No earlier valuation")
    d["age_vs_training"] = np.where(d.age_at_origin > vdev.age_at_origin.max(), "Older than any development player", "Age seen in training")
    return d


seg_rows = {"dev": seg_frame_v(vdev, cv1["pred"].mean(0), cv1["in80"].mean(0)), "bt": seg_frame_v(vbt, pr_bt["log_change"], bt_in80.astype(float))}
for key, d in seg_rows.items():
    parts = []
    for col in ["age_band", "position", "value_tier", "history", "age_vs_training"]:
        if key == "dev" and col == "age_vs_training":
            continue
        s = segment_table(d, col, "err", "in80", "abs_pct")
        s.insert(0, "segment_type", col)
        s = s.rename(columns={col: "segment"})
        parts.append(s)
    save(pd.concat(parts), f"kpi1_segments_{'dev_cv' if key == 'dev' else 'backtest'}.csv")
print("[11] KPI 1 backtest segments:\n", pd.read_csv(OUT / "kpi1_segments_backtest.csv")[["segment_type", "segment", "n", "typical_abs_pct_error", "bias", "coverage_80_pct"]].round(2).to_string(index=False))

# ====================================================================== KPI 1 improvements
imp1 = []

# -- Attempt A: pooled linear tail in the age curve (point-forecast change -> CV rule)
print("\n[11] KPI 1 attempt A: pooled tail of the age curve (decided on development CV)")
base_cv = cv1["rmse_by_rep"]
tail_res = {}
for tf in (28, 29, 30, 31):
    c = cv_value(vdev, lambda tf=tf: TailValueModel(tf))
    tail_res[tf] = c
best_tf = min(tail_res, key=lambda k: np.mean(tail_res[k]["rmse_by_rep"]))
bt_tail = tail_res[best_tf]["rmse_by_rep"]
gain = 1 - np.mean(bt_tail) / np.mean(base_cv)
wins = int(sum(a < b for a, b in zip(bt_tail, base_cv)))
adopt_A = gain >= ADOPT_MIN_REL_GAIN and wins >= 2
for tf, c in tail_res.items():
    print(f"      tail from {tf}: CV RMSE {np.mean(c['rmse_by_rep']):.4f} (Week 10: {np.mean(base_cv):.4f})")
imp1.append({"attempt": "A. Pool the oldest part of the age curve into one straight line", "reason": "Backtest errors were concentrated in players older than the "
             "training ages, where a free point per year is estimated from one or two players",
             "evidence_set": "development cross-validation", "metric": "CV RMSE (log)", "week10": np.mean(base_cv), "revised": np.mean(bt_tail),
             "relative_gain_pct": 100 * gain, "wins_in_repeats": f"{wins} of {REPS}", "rule": f"gain >= {100 * ADOPT_MIN_REL_GAIN:.0f}% and wins >= 2 of 3",
             "decision": "ADOPT" if adopt_A else "REJECT", "note": f"best tail start tested: age {best_tf}"})
# information only (NOT used for any decision): what the one-off backtest says about attempt A
m_tail = fit_v(vdev, lambda: TailValueModel(best_tf))
pr_tail = pred_v(m_tail, vbt)
info_tail = v_metrics(vbt.log_change.values, pr_tail["log_change"])
SUMMARY["kpi1_attempt_A_info_only_backtest"] = info_tail

# -- Attempt B: widen the ranges outside the trained ages (range change -> stress-test rule)
print("[11] KPI 1 attempt B: wider ranges outside the trained ages (stress test on development data)")


def stress_factor(df, cutoffs=(29, 30, 31, 32)):
    """Train on players younger than a cut-off, test on players older than anyone in that training set; the
    root-mean-square of (error / the model's own standard error) says how many times too narrow the range is."""
    z, who = [], []
    for c in cutoffs:
        tr = df[df.age_at_origin < c]
        m = fit_v(tr)
        te = df[df.age_at_origin > m.age_max]
        if len(te) == 0:
            continue
        pr = pred_v(m, te)
        z.append((te.log_change.values - pr["log_change"]) / pr["se_pred"])
        who.append(te.player_id.values)
    z, who = np.concatenate(z), np.concatenate(who)
    est, lo, hi = cluster_bootstrap(lambda a: float(np.sqrt(np.mean(np.square(a)))), [z], who, n_boot=1000)
    return est, lo, hi, len(z), len(set(who))


k_dev, k_lo, k_hi, n_z, n_p = stress_factor(vdev)
known_v = pd.concat([vdev, vbt], ignore_index=True)
k_final, kf_lo, kf_hi, nf_z, nf_p = stress_factor(known_v)
# in-range reference: the same statistic for ordinary cross-validation rows (should be about 1)
inrange_z = []
fold0 = group_folds(vdev.player_id.values, FOLDS, 0)
for k in range(FOLDS):
    m = fit_v(vdev[fold0 != k]); te = vdev[fold0 == k]; pr = pred_v(m, te)
    inrange_z.append((te.log_change.values - pr["log_change"]) / pr["se_pred"])
inrange_rms = rmse(np.concatenate(inrange_z))
adopt_B = k_dev > EXTRAP_RULE_THRESHOLD
K_DEV = round(max(1.0, k_dev), 2) if adopt_B else 1.0
K_FINAL = round(max(1.0, k_final), 2) if adopt_B else 1.0
print(f"      in-range RMS standardised error {inrange_rms:.2f}; out-of-range: dev {k_dev:.2f} (95% CI {k_lo:.2f}-{k_hi:.2f}, {n_z} test rows from "
      f"{n_p} players); dev+backtest {k_final:.2f} (CI {kf_lo:.2f}-{kf_hi:.2f}). Factor used: backtest model x{K_DEV}, live model x{K_FINAL}")
imp1.append({"attempt": "B. Widen the prediction range for players older than anyone in the training data",
             "reason": "In the backtest all 11 players older than the training ages fell outside their 80% range",
             "evidence_set": f"development stress test ({n_z} rows, {n_p} players)", "metric": "RMS of error / standard error (1.0 = honest range)",
             "week10": k_dev, "revised": 1.0, "relative_gain_pct": np.nan, "wins_in_repeats": "",
             "rule": f"factor > {EXTRAP_RULE_THRESHOLD}", "decision": "ADOPT" if adopt_B else "REJECT",
             "note": f"stress test says ranges are {k_dev:.1f}x too narrow outside the trained ages (95% CI {k_lo:.1f} to {k_hi:.1f}); in-range value is {inrange_rms:.2f}. Factor applied: {K_DEV}. After applying it the stress-test value is 1.0 by construction, so the independent check is the one-off backtest"})

# ---- single backtest comparison: Week 10 vs retained revised
print("[11] KPI 1 final backtest: Week 10 model vs retained revised model (run once)")
m1_rev = fit_v(vdev)
m1_rev.extrap_inflation = K_DEV
pr_rev = pred_v(m1_rev, vbt)
flag = pr_rev["extrapolated"] == 1


def range_stats(pr, df, label):
    out = {"version": label}
    for lvl in (0.8, 0.95):
        lo, hi = interval(pr, lvl)
        inn = (df.target_value_eur.values >= lo) & (df.target_value_eur.values <= hi)
        out[f"coverage_{int(100 * lvl)}_all_pct"] = 100 * inn.mean()
        out[f"coverage_{int(100 * lvl)}_older_than_training_pct"] = 100 * inn[flag].mean()
        out[f"coverage_{int(100 * lvl)}_age_seen_pct"] = 100 * inn[~flag].mean()
    lo, hi = interval(pr, 0.8)
    out["median_80_range_width_pct_of_forecast_age_seen"] = 100 * float(np.median((hi - lo)[~flag] / point(pr)[~flag]))
    out["median_80_range_width_pct_of_forecast_older"] = 100 * float(np.median((hi - lo)[flag] / point(pr)[flag]))
    out["mean_80_interval_score_older"] = float(interval_score(lo, hi, df.target_value_eur.values)[flag].mean())
    out["typical_error_pct"] = 100 * float(np.median(np.abs(point(pr) / df.target_value_eur.values - 1)))
    out["rmse_log"] = rmse(np.log(df.target_value_eur.values) - pr["log_pred"])
    out["players_older_than_training"] = int(flag.sum())
    return out


ba1 = pd.DataFrame([range_stats(pr_bt, vbt, "Week 10 model"), range_stats(pr_rev, vbt, "Week 11 retained model (point forecast unchanged, wider range outside trained ages)")])
save(ba1, "kpi1_before_after_backtest.csv")
print(ba1.T.round(2).to_string())
# information-only line for attempt A
SUMMARY["kpi1_before_after"] = ba1.round(3).to_dict("records")
bp1 = vbt[["row_id", "player_id", "position", "age_at_origin", "market_value_eur", "target_value_eur"]].copy()
bp1["forecast_eur"] = point(pr_rev).round(2)
for lvl in (0.8, 0.95):
    lo, hi = interval(pr_rev, lvl)
    bp1[f"pi{int(100 * lvl)}_low_eur"], bp1[f"pi{int(100 * lvl)}_high_eur"] = lo.round(2), hi.round(2)
    bp1[f"inside_pi{int(100 * lvl)}"] = ((vbt.target_value_eur >= lo) & (vbt.target_value_eur <= hi)).astype(int)
bp1["abs_pct_error"] = (100 * np.abs(bp1.forecast_eur / bp1.target_value_eur - 1)).round(2)
bp1["older_than_training"] = flag.astype(int)
bp1.to_csv(C.MODEL / "week11_value_backtest_predictions.csv", index=False)
imp1_df = pd.DataFrame(imp1)
save(imp1_df, "kpi1_improvements.csv")
SUMMARY["kpi1_improvements"] = imp1_df.round(4).to_dict("records")
SUMMARY["kpi1_settings"] = {"extrap_inflation_backtest_model": K_DEV, "extrap_inflation_final_model": K_FINAL,
                            "stress_test": {"dev_factor": k_dev, "dev_ci95": [k_lo, k_hi], "dev_rows": n_z, "dev_players": n_p,
                                            "dev_plus_backtest_factor": k_final, "dev_plus_backtest_ci95": [kf_lo, kf_hi], "in_range_rms": inrange_rms}}

# ====================================================================== KPI 2
print("\n[11] ===== KPI 2: passes per 90 =====")
yd, yb = pdev.target_passes_p90.values, pbt.target_passes_p90.values


def p_metrics(y, p):
    e = np.asarray(p) - np.asarray(y)
    return {"mae": float(np.mean(np.abs(e))), "rmse": rmse(e), "r2": float(1 - np.sum(e ** 2) / np.sum((y - np.mean(y)) ** 2)),
            "within_5_passes_pct": 100 * float(np.mean(np.abs(e) <= 5)), "bias": float(e.mean())}


def cv_perf(df, name, params, feats=FEATURES, target=None, reps=REPS):
    y = df.target_passes_p90.values
    oof = np.zeros((reps, len(df)))
    for r in range(reps):
        fold = group_folds(df.player_id.values, FOLDS, r)
        for k in range(FOLDS):
            tr, te = df[fold != k], df[fold == k]
            assert not set(tr.player_id) & set(te.player_id)
            t_tr = tr.target_passes_p90 - (tr.passes_last if target == "delta" else 0)
            est = make_estimator(name, params).fit(tr[feats], t_tr)
            oof[r, fold == k] = est.predict(te[feats]) + (te.passes_last.values if target == "delta" else 0)
    return oof


# ---- model choice re-run under the version-independent folds (Week 10 grid)
grid = [("Linear regression (ridge)", {})] + [("Random forest", {"min_samples_leaf": l, "max_features": f}) for l in (5, 10, 20) for f in (0.6, 1.0)] + \
       [("Gradient boosting", {"max_depth": d, "n_estimators": n}) for d in (2, 3) for n in (200, 400)]
choice_rows, oofs = [], {}
for name, params in grid:
    o = cv_perf(pdev, name, params)
    oofs[(name, json.dumps(params))] = o
    per = [rmse(o[r] - yd) for r in range(REPS)]
    choice_rows.append({"model": name, "params": json.dumps(params), "cv_rmse": np.mean(per), "cv_rmse_sd_across_repeats": np.std(per, ddof=1),
                        "cv_mae": np.mean([np.mean(np.abs(o[r] - yd)) for r in range(REPS)]),
                        "is_week10_choice": (name == PNAME and params == PPARAMS)})
choice = pd.DataFrame(choice_rows).sort_values("cv_rmse").reset_index(drop=True)
choice["rank"] = np.arange(1, len(choice) + 1)
save(choice, "kpi2_model_choice_cv.csv")
print(choice[["rank", "model", "params", "cv_rmse", "cv_rmse_sd_across_repeats", "is_week10_choice"]].round(3).to_string(index=False))
o_base = oofs[(PNAME, json.dumps(PPARAMS))]
base_per_rep = [rmse(o_base[r] - yd) for r in range(REPS)]
best_row = choice.iloc[0]
SUMMARY["kpi2_model_choice"] = {"week10_choice": f"{PNAME} {PPARAMS}", "week10_rank_under_week11_folds": int(choice[choice.is_week10_choice]["rank"].iloc[0]),
                                "best_under_week11_folds": f"{best_row.model} {best_row.params}", "best_rmse": float(best_row.cv_rmse),
                                "week10_choice_rmse": float(np.mean(base_per_rep)), "fold_noise_sd": float(np.std(base_per_rep, ddof=1))}

# ---- baselines + model: development CV and backtest
pos_mean_dev = pdev.groupby("position").target_passes_p90.mean()
m2_dev = PerformanceForecastModel(PNAME, PPARAMS).fit(pdev, pj["resid_quantiles_june2025"])
f2_bt = m2_dev.predict_frame(pbt)
baselines = {"No change (same as last season)": (pdev.passes_last.values, pbt.passes_last.values),
             "Position average": (pdev.position.map(pos_mean_dev).values, pbt.position.map(pos_mean_dev).values),
             "Week 10 model (gradient boosting)": (o_base, f2_bt.forecast.values)}
rows, boot2 = [], {}
for name, (pd_, pb_) in baselines.items():
    rows.append({"model": name, **{f"cv_{k}": v for k, v in p_metrics(np.tile(yd, REPS), pd_.ravel() if pd_.ndim > 1 else np.tile(pd_, REPS)).items()}, **{f"backtest_{k}": v for k, v in p_metrics(yb, pb_).items()}})
    if not name.startswith("Week 10"):
        boot2[name] = cluster_bootstrap(lambda a, b: float(np.mean(np.abs(a)) - np.mean(np.abs(b))), [pb_ - yb, f2_bt.forecast.values - yb], pbt.player_id.values)
t2 = pd.DataFrame(rows)
t2["backtest_mae_gain_vs_this_baseline_ci95"] = [f"{boot2[n][0]:.2f} ({boot2[n][1]:.2f} to {boot2[n][2]:.2f})" if n in boot2 else "" for n in t2.model]
bt_in80_2 = ((yb >= f2_bt.lo80) & (yb <= f2_bt.hi80)).values
t2.loc[t2.model.str.startswith("Week 10"), "backtest_coverage_80_pct"] = 100 * bt_in80_2.mean()
save(t2, "kpi2_baseline_comparison.csv")
SUMMARY["kpi2_baseline"] = t2.round(4).to_dict("records")
print(t2[["model", "cv_mae", "cv_rmse", "backtest_mae", "backtest_rmse", "backtest_r2"]].round(2).to_string(index=False))

# ---- decision metric: will the player reach the need's target style? (MID need, so MID players)
tgt = float(need.target_passes_p90.iloc[0])
mid_d, mid_b = (pdev.position == "MID").values, (pbt.position == "MID").values
dec = []
for label, pred_d, pred_b in [("Week 10 model", o_base.mean(0), f2_bt.forecast.values), ("No change (last season)", pdev.passes_last.values, pbt.passes_last.values)]:
    for split, mask, p, y in (("development CV", mid_d, pred_d, yd), ("backtest", mid_b, pred_b, yb)):
        pr_hit, act = p[mask] >= tgt, y[mask] >= tgt
        dec.append({"method": label, "set": split, "midfielders": int(mask.sum()), "actually_reach_target": int(act.sum()),
                    "correct_call_pct": 100 * float((pr_hit == act).mean()),
                    "of_players_called_fit_how_many_really_fit_pct": 100 * float(act[pr_hit].mean()) if pr_hit.any() else np.nan,
                    "of_real_fits_how_many_found_pct": 100 * float(pr_hit[act].mean()) if act.any() else np.nan})
save(pd.DataFrame(dec), "kpi2_target_reach.csv")
SUMMARY["kpi2_target_reach"] = pd.DataFrame(dec).round(2).to_dict("records")
SUMMARY["kpi2_target_passes"] = tgt
print(pd.DataFrame(dec).round(1).to_string(index=False))

# ---- error analysis
def seg_frame_p(df, pred, in80):
    d = df[["player_id", "position", "age", "minutes_last", "has_before", "passes_last"]].copy()
    d["err"] = pred - df.target_passes_p90.values
    d["in80"] = in80
    d["age_band"] = band(d.age, [0, 22, 26, 30, 99], ["Under 22", "22 to 25", "26 to 29", "30 and over"])
    d["minutes_band"] = band(d.minutes_last, [0, 1500, 2200, 9999], ["900 to 1,500", "1,500 to 2,200", "Over 2,200"])
    d["history"] = np.where(d.has_before == 1, "Two seasons of data", "One season of data")
    d["passing_level"] = band(d.passes_last, [-1, 30, 50, 999], ["Low passer (<30)", "Mid passer (30-50)", "High passer (>50)"])
    return d


oof_resid_stack = (yd[None, :] - o_base)            # actual - forecast, reps x n
rq10, rq90 = (float({float(k): v for k, v in pj["resid_quantiles_june2025"].items()}[q]) for q in (0.1, 0.9))
cover_dev = ((yd[None, :] >= o_base + rq10) & (yd[None, :] <= o_base + rq90)).mean(0)
dsd, dsb = seg_frame_p(pdev, o_base.mean(0), cover_dev), seg_frame_p(pbt, f2_bt.forecast.values, bt_in80_2.astype(float))
for key, d in (("dev_cv", dsd), ("backtest", dsb)):
    parts = []
    for col in ["position", "age_band", "minutes_band", "history", "passing_level"]:
        s = segment_table(d, col, "err", "in80")
        s.insert(0, "segment_type", col)
        parts.append(s.rename(columns={col: "segment"}))
    save(pd.concat(parts), f"kpi2_segments_{key}.csv")
print("[11] KPI 2 backtest segments:\n", pd.read_csv(OUT / "kpi2_segments_backtest.csv")[["segment_type", "segment", "n", "mae", "bias", "coverage_80_pct"]].round(2).to_string(index=False))

# ====================================================================== KPI 2 improvements
imp2 = []

# -- Attempt A: prediction ranges per group (range rule). Cross-fitted on development so the quantiles are never scored on the rows that made them.
print("\n[11] KPI 2 attempt A: separate prediction ranges for goalkeepers and outfield players")
grp_d = np.where(pdev.position == "GK", "GK", "Outfield")
players_d = pdev.player_id.values
uniq = np.unique(players_d)
res_pool, res_grp, sc_pool, sc_grp = [], [], [], []
cov_by = {("pooled", g): [] for g in ("GK", "Outfield")}
cov_by.update({("group", g): [] for g in ("GK", "Outfield")})
rng = np.random.default_rng(11)
for rep in range(60):
    half = set(rng.permutation(uniq)[: len(uniq) // 2])
    in_a = np.array([p in half for p in players_d])
    for fit_mask in (in_a, ~in_a):
        ev = ~fit_mask
        r_fit = oof_resid_stack[:, fit_mask]
        q_pool = {q: np.quantile(r_fit, q) for q in (0.1, 0.9)}
        for g in ("GK", "Outfield"):
            gm = ev & (grp_d == g)
            if not gm.any():
                continue
            q_g = {q: np.quantile(oof_resid_stack[:, fit_mask & (grp_d == g)], q) for q in (0.1, 0.9)}
            pred_ev = o_base.mean(0)[gm]
            for tag, q in (("pooled", q_pool), ("group", q_g)):
                lo, hi = pred_ev + q[0.1], pred_ev + q[0.9]
                cov_by[(tag, g)].append(float(np.mean((yd[gm] >= lo) & (yd[gm] <= hi))))
                (sc_pool if tag == "pooled" else sc_grp).append(float(np.mean(interval_score(lo, hi, yd[gm]))))
gaps = {t: max(abs(100 * np.mean(cov_by[(t, g)]) - 80) for g in ("GK", "Outfield")) for t in ("pooled", "group")}
mean_score = {"pooled": float(np.mean(sc_pool)), "group": float(np.mean(sc_grp))}
adopt_2A = gaps["group"] < gaps["pooled"] and mean_score["group"] < mean_score["pooled"]
cov_tab = pd.DataFrame([{"ranges": t, "group": g, "coverage_80_pct (cross-fitted, development)": 100 * np.mean(cov_by[(t, g)])} for t in ("pooled", "group") for g in ("GK", "Outfield")])
save(cov_tab, "kpi2_range_crossfit_dev.csv")
print(cov_tab.round(1).to_string(index=False), "\n      worst coverage gap:", {k: round(v, 1) for k, v in gaps.items()}, "mean interval score:", {k: round(v, 2) for k, v in mean_score.items()})
imp2.append({"attempt": "A. Separate 80%/95% prediction ranges for goalkeepers and outfield players",
             "reason": "Development error SD is about half for goalkeepers (6.7) vs outfield players (13.4), but one pooled error distribution sets every range",
             "evidence_set": "development, cross-fitted by player halves (60 repeats)", "metric": "worst gap of 80% coverage from 80 (points); mean 80% interval score",
             "week10": f"gap {gaps['pooled']:.1f}; score {mean_score['pooled']:.1f}", "revised": f"gap {gaps['group']:.1f}; score {mean_score['group']:.1f}",
             "relative_gain_pct": 100 * (1 - mean_score["group"] / mean_score["pooled"]), "wins_in_repeats": "",
             "rule": "gap smaller AND score lower", "decision": "ADOPT" if adopt_2A else "REJECT", "note": "point forecast is unchanged"})

# -- Attempt B: engineered predictors
print("[11] KPI 2 attempt B: extra predictors (change since the season before, passes relative to the position)")
pdev2 = pdev.copy()
pdev2["passes_change"] = pdev2.passes_last - pdev2.passes_before
pdev2["passes_vs_position"] = pdev2.passes_last - pdev2.groupby("position").passes_last.transform("mean")
oB = cv_perf(pdev2, PNAME, PPARAMS, feats=FEATURES + ["passes_change", "passes_vs_position"])
perB = [rmse(oB[r] - yd) for r in range(REPS)]
gainB, winsB = 1 - np.mean(perB) / np.mean(base_per_rep), sum(a < b for a, b in zip(perB, base_per_rep))
imp2.append({"attempt": "B. Add two predictors: change since the season before and passes relative to the position average",
             "reason": "Passing level and its trend are the strongest signals; the model might use them better in engineered form",
             "evidence_set": "development cross-validation", "metric": "CV RMSE (passes per 90)", "week10": np.mean(base_per_rep), "revised": np.mean(perB),
             "relative_gain_pct": 100 * gainB, "wins_in_repeats": f"{winsB} of {REPS}", "rule": "gain >= 2% and wins >= 2 of 3",
             "decision": "ADOPT" if (gainB >= ADOPT_MIN_REL_GAIN and winsB >= 2) else "REJECT", "note": ""})
print(f"      CV RMSE {np.mean(perB):.3f} vs {np.mean(base_per_rep):.3f} ({100 * gainB:+.1f}%), wins {winsB}/3")

# -- Attempt C: a simpler model (depth 1, more trees) and a different target (change vs last season)
print("[11] KPI 2 attempt C: simpler model settings; predicting the change from last season")
oC = cv_perf(pdev, "Gradient boosting", {"max_depth": 1, "n_estimators": 300})
perC = [rmse(oC[r] - yd) for r in range(REPS)]
gainC, winsC = 1 - np.mean(perC) / np.mean(base_per_rep), sum(a < b for a, b in zip(perC, base_per_rep))
imp2.append({"attempt": "C1. Simpler model settings (gradient boosting, depth 1, 300 trees)", "reason": "Small data (334 rows): a simpler model may generalise better",
             "evidence_set": "development cross-validation", "metric": "CV RMSE (passes per 90)", "week10": np.mean(base_per_rep), "revised": np.mean(perC),
             "relative_gain_pct": 100 * gainC, "wins_in_repeats": f"{winsC} of {REPS}", "rule": "gain >= 2% and wins >= 2 of 3",
             "decision": "ADOPT" if (gainC >= ADOPT_MIN_REL_GAIN and winsC >= 2) else "REJECT", "note": ""})
oD = cv_perf(pdev, PNAME, PPARAMS, target="delta")
perD = [rmse(oD[r] - yd) for r in range(REPS)]
gainD, winsD = 1 - np.mean(perD) / np.mean(base_per_rep), sum(a < b for a, b in zip(perD, base_per_rep))
imp2.append({"attempt": "C2. Predict the change from last season instead of the level", "reason": "Mean reversion may be easier to learn as a change",
             "evidence_set": "development cross-validation", "metric": "CV RMSE (passes per 90)", "week10": np.mean(base_per_rep), "revised": np.mean(perD),
             "relative_gain_pct": 100 * gainD, "wins_in_repeats": f"{winsD} of {REPS}", "rule": "gain >= 2% and wins >= 2 of 3",
             "decision": "ADOPT" if (gainD >= ADOPT_MIN_REL_GAIN and winsD >= 2) else "REJECT", "note": ""})
print(f"      depth-1: {np.mean(perC):.3f} ({100 * gainC:+.1f}%, wins {winsC}/3); delta target: {np.mean(perD):.3f} ({100 * gainD:+.1f}%, wins {winsD}/3)")

# -- hygiene check: carries imputation from development medians only
def impute_dev_only(df):
    d = df.copy()
    d["carries_last"] = d.carries_last_raw.fillna(d.position.map(med_dev))
    return d


m_h = PerformanceForecastModel(PNAME, PPARAMS).fit(impute_dev_only(pdev), pj["resid_quantiles_june2025"])
rmse_h = rmse(m_h.predict_frame(impute_dev_only(pbt)).forecast.values - yb)
rmse_o = rmse(f2_bt.forecast.values - yb)
SUMMARY["kpi2_carries_imputation_check"] = {"backtest_rmse_all_row_medians": rmse_o, "backtest_rmse_development_only_medians": rmse_h, "difference": rmse_h - rmse_o}
print(f"[11] carries-imputation check: backtest RMSE {rmse_o:.3f} (Week 10) vs {rmse_h:.3f} (development-only medians)")

# ---- retained KPI 2 model: build group ranges from out-of-fold residuals, then ONE backtest comparison
def group_quantiles(df):
    y = df.target_passes_p90.values
    o = cv_perf(df, PNAME, PPARAMS)
    res = y[None, :] - o
    g = np.where(df.position == "GK", "GK", "Outfield")
    return {gr: {float(q): float(np.quantile(res[:, g == gr], q)) for q in QUANTILE_GRID} for gr in ("GK", "Outfield")}


GROUP_Q_BT = group_quantiles(pdev) if adopt_2A else None
known_p = pd.concat([pdev, pbt], ignore_index=True)
GROUP_Q_FINAL = group_quantiles(known_p) if adopt_2A else None
m2_rev = PerformanceForecastModel(PNAME, PPARAMS).fit(pdev, pj["resid_quantiles_june2025"])
if adopt_2A:
    m2_rev.set_group_ranges(GROUP_Q_BT)
f2_rev = m2_rev.predict_frame(pbt)
assert np.allclose(f2_rev.forecast.values, f2_bt.forecast.values)          # point forecast unchanged


def range_stats_p(f, label):
    out = {"version": label}
    for lvl, (a, b) in (("80", ("lo80", "hi80")), ("95", ("lo95", "hi95"))):
        inn = ((yb >= f[a]) & (yb <= f[b])).values
        out[f"coverage_{lvl}_all_pct"] = 100 * inn.mean()
        for g in ("GK", "DEF", "MID", "FWD"):
            out[f"coverage_{lvl}_{g}_pct"] = 100 * inn[(pbt.position == g).values].mean()
    out["median_80_width_GK"] = float((f.hi80 - f.lo80)[(pbt.position == "GK").values].median())
    out["median_80_width_outfield"] = float((f.hi80 - f.lo80)[(pbt.position != "GK").values].median())
    out["mean_80_interval_score"] = float(np.mean(interval_score(f.lo80, f.hi80, yb)))
    out["mae"] = float(np.mean(np.abs(f.forecast.values - yb)))
    return out


ba2 = pd.DataFrame([range_stats_p(f2_bt, "Week 10 model (one pooled range)"), range_stats_p(f2_rev, "Week 11 retained model (ranges by group, same point forecast)")])
save(ba2, "kpi2_before_after_backtest.csv")
SUMMARY["kpi2_before_after"] = ba2.round(3).to_dict("records")
print(ba2.T.round(1).to_string())
bp2 = pbt[["player_id", "position", "age", "passes_last", "passes_before", "has_before", "carries_last", "minutes_last", "target_passes_p90"]].copy()
bp2 = pd.concat([bp2, f2_rev], axis=1)
bp2["error"] = bp2.forecast - bp2.target_passes_p90
bp2["inside_pi80"] = ((bp2.target_passes_p90 >= bp2.lo80) & (bp2.target_passes_p90 <= bp2.hi80)).astype(int)
bp2.round(3).to_csv(C.MODEL / "week11_perf_backtest_predictions.csv", index=False)
imp2_df = pd.DataFrame(imp2)
save(imp2_df, "kpi2_improvements.csv")
SUMMARY["kpi2_improvements"] = imp2_df.round(4).to_dict("records")

# ====================================================================== settings read by the app + summary
settings = {
    "note": "Written by code/11_evaluate_models.py. The app applies these on top of the Week 10 model files, which are left unchanged.",
    "value": {"extrap_inflation_june2025_model": K_DEV, "extrap_inflation_final_model": K_FINAL},
    "performance": {"group_ranges_adopted": bool(adopt_2A), "group_resid_quantiles_june2025": GROUP_Q_BT, "group_resid_quantiles_final": GROUP_Q_FINAL},
}
(C.MODEL / "week11_settings.json").write_text(json.dumps(settings, indent=1))
SUMMARY["decisions"] = {"kpi1": {"retained_point_forecast": "Week 10 age-curve model" if not adopt_A else "pooled tail model",
                                 "range_change_adopted": bool(adopt_B)},
                        "kpi2": {"retained_point_forecast": f"Week 10 {PNAME}", "range_change_adopted": bool(adopt_2A)}}
(OUT / "week11_summary.json").write_text(json.dumps(SUMMARY, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
print("\n[11] done. Retained: KPI 1 =", SUMMARY["decisions"]["kpi1"], "| KPI 2 =", SUMMARY["decisions"]["kpi2"])
print(f"[11] wrote outputs/week11/*.csv, week11_summary.json, data/model/week11_settings.json")
