"""
Tests for the Week 11 additions. Run from the Predictive_Model folder:   python -m unittest discover tests -v
Needs only pandas, numpy, scipy and scikit-learn (the same as step 11). The tests that need the saved Week 10 files read them
from data/model, so run them from a checkout of the repository.
"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import config as C  # noqa: E402
import input_checks as IC  # noqa: E402
from value_model import ValueForecastModel, interval, point  # noqa: E402
import performance_model as PM  # noqa: E402
from evaluation_utils import group_folds, interval_score  # noqa: E402

VT = pd.read_csv(C.MODEL / "value_model_table.csv")
PT = pd.read_csv(C.MODEL / "performance_model_table.csv")
REF = IC.reference_ranges(VT, PT)
AGE_RANGE = (16.5, 33.9)


class TestInputChecks(unittest.TestCase):
    def levels(self, issues):
        return [i.level for i in issues]

    def test_normal_value_scenario_has_no_issue(self):
        self.assertEqual(IC.verdict(IC.check_value_inputs(500_000, 450_000, True, 24, 550_000, REF, AGE_RANGE)), "ok")

    def test_negative_or_missing_values_block(self):
        self.assertIn("block", self.levels(IC.check_value_inputs(-5, 450_000, True, 24, 550_000, REF, AGE_RANGE)))
        self.assertIn("block", self.levels(IC.check_value_inputs(500_000, np.nan, True, 24, 550_000, REF, AGE_RANGE)))

    def test_young_player_cannot_have_history(self):
        self.assertEqual(IC.verdict(IC.check_value_inputs(100_000, 90_000, True, 16.4, 110_000, REF, AGE_RANGE)), "block")
        self.assertEqual(IC.verdict(IC.check_value_inputs(100_000, 90_000, False, 16.6, 110_000, REF, AGE_RANGE)), "ok")
        # 16.4 is below the youngest training age (16.5): not impossible, so no block, but the model has never seen it
        self.assertEqual(IC.verdict(IC.check_value_inputs(100_000, 90_000, False, 16.4, 110_000, REF, AGE_RANGE)), "do_not_rely")

    def test_impossible_jumps_block_but_big_ones_only_warn(self):
        self.assertEqual(IC.verdict(IC.check_value_inputs(5_000_000, 100_000, True, 24, 5_000_000, REF, AGE_RANGE)), "block")
        big = IC.check_value_inputs(900_000, 400_000, True, 22, 900_000, REF, AGE_RANGE)
        self.assertNotIn("block", self.levels(big))

    def test_age_outside_training_is_do_not_rely(self):
        issues = IC.check_value_inputs(500_000, 450_000, True, 35.0, 550_000, REF, AGE_RANGE)
        self.assertEqual(IC.verdict(issues, extrapolated=True), "do_not_rely")
        self.assertEqual(IC.verdict(issues), "do_not_rely")                                   # the age issue alone is severe enough
        self.assertEqual(IC.verdict(IC.check_value_inputs(500_000, 450_000, True, 45.0, 550_000, REF, AGE_RANGE)), "block")

    def test_asking_price_unit_error_blocks(self):
        self.assertEqual(IC.verdict(IC.check_value_inputs(500_000, 450_000, True, 24, 50_000_000, REF, AGE_RANGE)), "block")

    def test_perf_limits(self):
        ok = IC.check_perf_inputs(45, 42, True, 10, 2000, 24, "MID", REF, (17, 34))
        self.assertEqual(IC.verdict(ok), "ok")
        self.assertEqual(IC.verdict(IC.check_perf_inputs(45, 42, True, 10, 4000, 24, "MID", REF, (17, 34))), "block")      # > 3,420 minutes
        self.assertEqual(IC.verdict(IC.check_perf_inputs(200, 42, True, 10, 2000, 24, "MID", REF, (17, 34))), "block")     # impossible passes
        self.assertEqual(IC.verdict(IC.check_perf_inputs(45, 42, True, 10, 2000, 17.2, "MID", REF, (17, 34))), "block")    # two seasons at 17
        self.assertEqual(IC.verdict(IC.check_perf_inputs(45, 42, True, 10, 500, 24, "MID", REF, (17, 34))), "do_not_rely")  # fewer minutes than any training row
        self.assertEqual(IC.verdict(IC.check_perf_inputs(80, 42, True, 10, 2000, 24, "GK", REF, (17, 34))), "caution")     # GK passing like a midfielder

    def test_reference_ranges_come_from_training_data(self):
        self.assertGreater(REF["value_max"], 1_000_000)
        self.assertLess(REF["by_position"]["GK"]["passes"][1], REF["by_position"]["MID"]["passes"][1])


class TestModelBackwardCompatibility(unittest.TestCase):
    def test_week10_value_model_unchanged_when_inflation_is_one(self):
        m = ValueForecastModel.from_dict(json.loads((C.MODEL / "value_model_june2025.json").read_text()))
        bp = pd.read_csv(C.MODEL / "backtest_predictions.csv")
        pr = m.predict(bp.market_value_eur, bp.age_at_origin, bp.prev_value_eur)
        self.assertTrue(np.allclose(point(pr), bp.forecast_eur, atol=0.01))
        lo, _ = interval(pr, 0.8)
        self.assertTrue(np.allclose(lo, bp.pi80_low_eur, atol=0.01))

    def test_inflation_widens_only_players_outside_training_ages(self):
        m = ValueForecastModel.from_dict(json.loads((C.MODEL / "value_model_june2025.json").read_text()))
        ages = np.array([25.0, 40.0])
        base = m.predict([1e6, 1e6], ages, [9e5, 9e5])
        m.extrap_inflation = 2.0
        wide = m.predict([1e6, 1e6], ages, [9e5, 9e5])
        self.assertEqual(wide["se_pred"][0], base["se_pred"][0])
        self.assertAlmostEqual(wide["se_pred"][1] / base["se_pred"][1], 2.0)
        self.assertTrue(np.allclose(wide["log_pred"], base["log_pred"]))             # the central forecast never changes

    def test_group_ranges_change_ranges_not_forecasts(self):
        dev = PT[PT.split == "development"].reset_index(drop=True)
        j = json.loads((C.MODEL / "performance_model.json").read_text())
        m = PM.PerformanceForecastModel(j["name"], j["params"]).fit(dev, j["resid_quantiles_june2025"])
        x = pd.concat([PM.row(40, 38, True, 8, 2000, 25, "GK"), PM.row(40, 38, True, 8, 2000, 25, "MID")], ignore_index=True)
        before = m.predict_frame(x)
        m.set_group_ranges({"GK": {q: 0.5 * v for q, v in m.resid_q.items()}, "Outfield": dict(m.resid_q)})
        after = m.predict_frame(x)
        self.assertTrue(np.allclose(before.forecast, after.forecast))
        self.assertLess(after.hi80[0] - after.lo80[0], before.hi80[0] - before.lo80[0])      # GK narrower
        self.assertAlmostEqual(after.hi80[1], before.hi80[1])                                # outfield identical


class TestEvaluationHelpers(unittest.TestCase):
    def test_folds_never_split_a_player_and_are_repeatable(self):
        g = np.repeat(np.arange(40), 3)
        f1, f2 = group_folds(g, 5, 0), group_folds(g, 5, 0)
        self.assertTrue((f1 == f2).all())
        for pid in range(40):
            self.assertEqual(len(set(f1[g == pid])), 1)
        self.assertFalse((group_folds(g, 5, 1) == f1).all())

    def test_interval_score_penalises_misses(self):
        self.assertEqual(float(interval_score(0, 10, 5)), 10.0)
        self.assertGreater(float(interval_score(0, 10, 20)), 10.0)


class TestSettingsFile(unittest.TestCase):
    def test_week11_settings_present_and_sane(self):
        s = json.loads((C.MODEL / "week11_settings.json").read_text())
        self.assertGreaterEqual(s["value"]["extrap_inflation_june2025_model"], 1.0)
        gq = s["performance"]["group_resid_quantiles_june2025"]
        self.assertLess(gq["GK"]["0.9"] - gq["GK"]["0.1"], gq["Outfield"]["0.9"] - gq["Outfield"]["0.1"])


if __name__ == "__main__":
    unittest.main()
