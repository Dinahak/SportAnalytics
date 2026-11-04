"""
input_checks.py
Week 11. Checks every what-if scenario before the application forecasts it.

Three outcomes, so the user always knows what the tool did with an implausible scenario:
  block    The values are impossible or contradict each other (a negative price, a player aged 16 with a valuation from a
           year before his 16th birthday, a 50x jump in a year). The forecast is NOT computed. The user is told which
           input to fix.
  caution  The values are possible but outside what the model learned from (age or value beyond the training data, an
           unusual jump, very few minutes). The forecast is shown, labelled 'Use with caution', and the reason is repeated
           in the scenario summary.
  note     Information only.
Plausible ranges for the 'caution' level come from the training tables (1st to 99th percentile), not from guesses.
'Block' limits are physical or data-definition limits (config.MAX_VALID_PASSES_P90, 38 matches x 90 minutes, minimum age
for recorded valuations).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config as C

MAX_MINUTES = 38 * 90          # a season of 38 league matches
MIN_AGE, MAX_AGE = 16.0, 40.0
EXTREME_ONE_YEAR_RATIO = 10.0  # value now / value a year ago beyond 10x or below 0.1x: almost certainly a typing or unit error
EXTREME_ASK_RATIO = 20.0       # asking price beyond 20x or below 1/20 of the value: same
CAUTION_ASK_RATIO = (0.5, 2.0)
MIN_MINUTES_IN_TRAINING = 900


@dataclass
class Issue:
    level: str        # "block" | "caution" | "note"
    fields: tuple     # the inputs involved
    message: str      # what is wrong, in plain language
    handling: str     # what the application did about it
    severe: bool = False   # a 'caution' that makes the forecast unreliable ('Do not rely'): the model is outside what it was trained on


def reference_ranges(value_table, perf_table):
    """Plausible ranges taken from the rows the models were fitted on (development + backtest, i.e. rows with an outcome)."""
    v = value_table[value_table.split != "live"]
    p = perf_table[perf_table.split != "live"]
    ref = {
        "value_min": float(v.market_value_eur.min()), "value_max": float(v.market_value_eur.max()),
        "momentum_p1": float(v[v.has_history == 1].momentum_log.quantile(0.01)),
        "momentum_p99": float(v[v.has_history == 1].momentum_log.quantile(0.99)),
        "minutes_min": float(p.minutes_last.min()),
        "change_p99": float((p[p.has_before == 1].passes_last - p[p.has_before == 1].passes_before).abs().quantile(0.99)),
        "by_position": {},
    }
    for pos, g in p.groupby("position"):
        ratio = (g.carries_last / g.passes_last)
        ref["by_position"][pos] = {
            "passes": (float(g.passes_last.quantile(0.01)), float(g.passes_last.quantile(0.99))),
            "carries": (float(g.carries_last.quantile(0.01)), float(g.carries_last.quantile(0.99))),
            "carries_ratio_p99": float(ratio.quantile(0.99)),
        }
    return ref


def _bad(x):
    return x is None or not np.isfinite(x)


# ====================================================================== KPI 1
def check_value_inputs(value, prev, has_history, age, ask, ref, age_range, budget=None):
    """age_range = (youngest, oldest) age the model was fitted on."""
    out = []
    if any(_bad(x) or x <= 0 for x in (value, ask)) or (has_history and (_bad(prev) or prev <= 0)):
        out.append(Issue("block", ("Market value now", "Market value one year ago", "Asking price"),
                         "A value or price is zero, negative or missing.", "No forecast. Enter positive amounts in euros."))
        return out
    if _bad(age) or not MIN_AGE <= age <= MAX_AGE:
        out.append(Issue("block", ("Age",), f"Age must be between {MIN_AGE:.0f} and {MAX_AGE:.0f}. Valuations of players younger than "
                         f"{C.MIN_PLAYER_AGE_FOR_DATA} are not recorded, and the data holds no players older than {MAX_AGE:.0f}.",
                         "No forecast. Change the age."))
        return out
    if has_history and age < C.MIN_PLAYER_AGE_FOR_DATA + 1:
        out.append(Issue("block", ("Age", "Market value one year ago"),
                         f"A player aged {age:.1f} cannot have a valuation from one year ago: valuations are only recorded from age "
                         f"{C.MIN_PLAYER_AGE_FOR_DATA}.", "No forecast. Switch off 'valuation from one year ago' or raise the age to 17 or more."))
    if has_history:
        ratio = value / prev
        if ratio > EXTREME_ONE_YEAR_RATIO or ratio < 1 / EXTREME_ONE_YEAR_RATIO:
            out.append(Issue("block", ("Market value now", "Market value one year ago"),
                             f"The value changed by a factor of {ratio:.1f} in one year. That is almost certainly a typing or unit error.",
                             "No forecast. Check both values."))
    ar = ask / value
    if ar > EXTREME_ASK_RATIO or ar < 1 / EXTREME_ASK_RATIO:
        out.append(Issue("block", ("Asking price", "Market value now"),
                         f"The asking price is {ar:.1f} times the current value. That is almost certainly a typing or unit error.",
                         "No forecast. Check the asking price."))
    if any(i.level == "block" for i in out):
        return out
    # ---- caution level
    if age < age_range[0] or age > age_range[1]:
        out.append(Issue("caution", ("Age",), f"Age {age:.1f} is outside the ages the model was fitted on ({age_range[0]:.1f} to {age_range[1]:.1f}). "
                         "In the backtest every player in this situation fell outside the original 80% range.",
                         "Forecast shown with a range widened by the factor measured in the stress test. Marked 'Do not rely'.", severe=True))
    if value < ref["value_min"] or value > ref["value_max"]:
        out.append(Issue("caution", ("Market value now",), f"A value of {value:,.0f} euros is outside the values the model has seen "
                         f"({ref['value_min']:,.0f} to {ref['value_max']:,.0f}).", "Forecast shown; treat it as an extrapolation."))
    if has_history:
        m = float(np.log(value / prev))
        if m < ref["momentum_p1"] or m > ref["momentum_p99"]:
            out.append(Issue("caution", ("Market value now", "Market value one year ago"),
                             f"A one-year change of {100 * (value / prev - 1):+.0f}% is more extreme than 98 out of 100 changes in the training data.",
                             "Forecast shown; the momentum effect is estimated from few similar cases."))
    if ar < CAUTION_ASK_RATIO[0] or ar > CAUTION_ASK_RATIO[1]:
        out.append(Issue("caution", ("Asking price",), f"The asking price is {ar:.1f} times the current value. Asking prices in the data "
                         "rarely fall outside half to double the value.", "Forecast shown; check the price."))
    if budget is not None and ask > budget:
        out.append(Issue("note", ("Asking price",), "The asking price is above the budget for this squad need.",
                         "The bid reading will say REJECT (asking price above budget)."))
    return out


# ====================================================================== KPI 2
def check_perf_inputs(passes, before, has_before, carries, minutes, age, position, ref, age_range):
    out = []
    if any(_bad(x) for x in (passes, carries, minutes, age)) or (has_before and _bad(before)):
        return [Issue("block", ("Inputs",), "A value is missing.", "No forecast.")]
    if not 0 <= passes <= C.MAX_VALID_PASSES_P90 or (has_before and not 0 <= before <= C.MAX_VALID_PASSES_P90):
        out.append(Issue("block", ("Passes per 90, last season", "Passes per 90, season before"),
                         f"Passes per 90 must be between 0 and {C.MAX_VALID_PASSES_P90} (higher values are treated as impossible in the data cleaning).",
                         "No forecast. Change the passing figure."))
    if not 0 <= carries <= C.MAX_VALID_CARRIES_P90:
        out.append(Issue("block", ("Carries per 90, last season",), f"Carries per 90 must be between 0 and {C.MAX_VALID_CARRIES_P90}.", "No forecast."))
    if not 0 <= minutes <= MAX_MINUTES:
        out.append(Issue("block", ("Minutes played, last season",), f"Minutes must be between 0 and {MAX_MINUTES:,} (38 matches of 90 minutes).", "No forecast."))
    if not MIN_AGE <= age <= MAX_AGE:
        out.append(Issue("block", ("Age",), f"Age must be between {MIN_AGE:.0f} and {MAX_AGE:.0f}.", "No forecast."))
    if has_before and age < C.MIN_PLAYER_AGE_FOR_DATA + 2:
        out.append(Issue("block", ("Age", "Passes per 90, season before"),
                         f"A player aged {age:.1f} next season cannot have two full seasons of data: statistics are only recorded from age "
                         f"{C.MIN_PLAYER_AGE_FOR_DATA}.", "No forecast. Switch off 'earlier season' or raise the age to 18 or more."))
    if any(i.level == "block" for i in out):
        return out
    pos = ref["by_position"][position]
    if age < age_range[0] or age > age_range[1]:
        out.append(Issue("caution", ("Age",), f"Age {age:.1f} is outside the ages the model was fitted on ({age_range[0]:.1f} to {age_range[1]:.1f}).",
                         "Forecast shown, marked 'Do not rely'.", severe=True))
    if minutes < MIN_MINUTES_IN_TRAINING:
        out.append(Issue("caution", ("Minutes played, last season",), f"The model was fitted only on players with at least {MIN_MINUTES_IN_TRAINING} minutes. "
                         "Below that it has no examples, per-90 figures are noisy, and the forecast can swing by many passes for no real reason.",
                         "Forecast shown, marked 'Do not rely'; the range is too narrow here.", severe=True))
    if not pos["passes"][0] <= passes <= pos["passes"][1]:
        out.append(Issue("caution", ("Passes per 90, last season",), f"{passes:.1f} passes per 90 is outside the 1st to 99th percentile for {position} players "
                         f"in the training data ({pos['passes'][0]:.0f} to {pos['passes'][1]:.0f}).", "Forecast shown; few similar players were seen."))
    if not pos["carries"][0] <= carries <= pos["carries"][1]:
        out.append(Issue("caution", ("Carries per 90, last season",), f"{carries:.1f} carries per 90 is unusual for a {position} "
                         f"({pos['carries'][0]:.1f} to {pos['carries'][1]:.1f} in the training data).", "Forecast shown."))
    if passes > 0 and carries / passes > pos["carries_ratio_p99"]:
        out.append(Issue("caution", ("Carries per 90, last season", "Passes per 90, last season"),
                         "This player carries the ball far more often relative to his passing than 99 out of 100 players in his position.",
                         "Forecast shown; check that carries and passes belong to the same player and season."))
    if has_before and abs(passes - before) > ref["change_p99"]:
        out.append(Issue("caution", ("Passes per 90, last season", "Passes per 90, season before"),
                         f"A change of {passes - before:+.0f} passes per 90 between two seasons happened in under 1% of the training rows.",
                         "Forecast shown; the model pulls extreme seasons back towards the middle."))
    return out


# ====================================================================== reading the issues
def verdict(issues, extrapolated=False):
    """'block' | 'do_not_rely' | 'caution' | 'ok' - the one label shown next to the forecast."""
    if any(i.level == "block" for i in issues):
        return "block"
    if extrapolated or any(i.level == "caution" and i.severe for i in issues):
        return "do_not_rely"
    if any(i.level == "caution" for i in issues):
        return "caution"
    return "ok"


VERDICT_TEXT = {"ok": "Normal: inputs are inside the range the model learned from.",
                "caution": "Use with caution: at least one input is unusual compared with the training data.",
                "do_not_rely": "Do not rely: the scenario is outside what the model was trained on.",
                "block": "Not forecast: the inputs are impossible or contradict each other."}

HANDLING_POLICY = (
    "**How the tool handles odd scenarios.** Impossible or contradictory inputs (negative prices, a 16-year-old with a valuation from a "
    "year before, a 20-fold jump in a year, more than 3,420 minutes) **stop the forecast** and say which input to fix. Possible but unusual "
    "inputs **still produce a forecast, but it is labelled**: 'Use with caution' for unusual values (a very large one-year jump, an unusual asking "
    "price, passing figures outside the usual range for the position), and 'Do not rely' where the model has never seen anything similar (an age "
    "outside the training ages, fewer than 900 minutes). The reason is repeated in the scenario summary. Limits for 'unusual' come from the training "
    "data (1st to 99th percentile). Outside the ages the value model was trained on, its prediction range is also widened.")
