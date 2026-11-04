"""
evaluation_utils.py
Small helpers shared by the Week 11 evaluation (step 11).

group_folds       Deterministic cross-validation folds, split by player. Written here instead of using
                  sklearn's GroupKFold because GroupKFold changed how it assigns groups between scikit-learn versions,
                  which moved the Week 10 cross-validation numbers by about 0.2 passes per 90. These folds depend only
                  on numpy's seeded generator, so every group member gets the same numbers.
cluster_bootstrap Paired bootstrap that resamples whole players, for the uncertainty of a difference between two models.
interval_score    Winkler score of a prediction interval (width + penalty for missing), lower is better.
"""
import numpy as np
import pandas as pd


def group_folds(groups, k=5, repeat=0, seed=20251003):
    """Fold number (0..k-1) for every row; all rows of one player share a fold. `repeat` gives a different split."""
    groups = np.asarray(groups)
    ids = np.unique(groups)
    order = np.random.default_rng(seed + repeat).permutation(len(ids))
    fold_of = dict(zip(ids, order % k))
    return np.array([fold_of[g] for g in groups])


def cluster_bootstrap(stat, arrays, groups, n_boot=2000, seed=7):
    """Resample players with replacement. stat(*arrays) -> number. Returns (estimate, 2.5th pct, 97.5th pct).
    `arrays` are equally long numpy arrays (e.g. the errors of model A and model B)."""
    groups = np.asarray(groups)
    arrays = [np.asarray(a) for a in arrays]
    idx_by_player = pd.Series(np.arange(len(groups))).groupby(groups).apply(lambda s: s.values).to_dict()
    players = np.array(list(idx_by_player))
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(players, size=len(players), replace=True)
        idx = np.concatenate([idx_by_player[p] for p in pick])
        out[b] = stat(*[a[idx] for a in arrays])
    return float(stat(*arrays)), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def interval_score(lo, hi, y, level=0.80):
    """Winkler / interval score: (hi - lo) + 2/alpha x (distance outside the interval). Lower is better."""
    alpha = 1 - level
    lo, hi, y = (np.asarray(a, dtype=float) for a in (lo, hi, y))
    return (hi - lo) + (2 / alpha) * np.maximum(lo - y, 0) + (2 / alpha) * np.maximum(y - hi, 0)


def band(series, edges, labels, right=False):
    return pd.cut(series, edges, labels=labels, right=right)


def segment_table(df, by, err_col, covered_col=None, abs_pct_col=None, min_n=20):
    """One row per segment: n, typical error, bias, and coverage. Segments with fewer than min_n rows are flagged."""
    g = df.groupby(by, observed=True)
    out = pd.DataFrame({"n": g.size(), "bias": g[err_col].mean(), "rmse": g[err_col].apply(lambda e: float(np.sqrt(np.mean(np.square(e)))))})
    if abs_pct_col:
        out["typical_abs_pct_error"] = g[abs_pct_col].median()
    else:
        out["mae"] = g[err_col].apply(lambda e: float(np.mean(np.abs(e))))
    if covered_col:
        out["coverage_80_pct"] = 100 * g[covered_col].mean()
    out["small_sample"] = out.n < min_n
    return out.reset_index()
