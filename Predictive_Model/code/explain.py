"""
explain.py
Predictor importance, predictor contributions and plain-language text for both KPI models. Used by the application.

KPI 1 (value forecast, ValueForecastModel) is additive on the log scale:
    log(forecast) = log(value now) + f(age) + g x momentum
Each term depends on one predictor, so its contribution to one prediction is exact:
    contribution_j = term_j(this player) - average of term_j over the model's training rows
This is the SHAP value of a linear model with independent predictors (the same thing shap.LinearExplainer returns), so
the contributions of the three predictors add up exactly to log(forecast) - average log(forecast) in training.
On the EUR scale the effects multiply: forecast = average forecast x exp(c_value) x exp(c_age) x exp(c_momentum).

KPI 2 (performance forecast, PerformanceForecastModel) uses SHAP values from shap.TreeExplainer (performance_model.py).

Overall importance (all players) = mean absolute contribution over the training rows. It says how much a predictor
moves predictions in general. A contribution (one player) says how much it moved THIS prediction, and in which direction.
Neither is evidence that changing the predictor would change the outcome: they describe how the model uses the data.
"""
import numpy as np
import pandas as pd

VALUE_PREDICTORS = ["Market value now", "Age", "Change in value over the last year"]


# ====================================================================== KPI 1: value forecast
def _value_terms(model, value_now, age, prev_value):
    value_now = np.atleast_1d(np.asarray(value_now, dtype=float))
    age = np.atleast_1d(np.asarray(age, dtype=float))
    prev_value = np.atleast_1d(np.asarray(prev_value, dtype=float))
    X = model._design(value_now, age, prev_value)
    k = len(model.knots)
    age_term = X[:, :k] @ model.coef[:k]
    mom_term = X[:, k] * model.coef[k] if model.use_momentum else np.zeros(len(age))
    return pd.DataFrame({VALUE_PREDICTORS[0]: np.log(value_now), VALUE_PREDICTORS[1]: age_term, VALUE_PREDICTORS[2]: mom_term})


class ValueExplainer:
    def __init__(self, model, background):
        """background: the rows the model was fitted on (market_value_eur, age_at_origin, prev_value_eur)."""
        self.model = model
        self.bg_terms = _value_terms(model, background.market_value_eur, background.age_at_origin, background.prev_value_eur)
        self.mean_terms = self.bg_terms.mean()
        self.base_log = float(self.mean_terms.sum())
        self.base_eur = float(np.exp(self.base_log))           # forecast for the 'average' training player (geometric mean)
        self.base_change_log = float(self.mean_terms[VALUE_PREDICTORS[1:]].sum())

    def contributions(self, value_now, age, prev_value):
        """Log-scale contributions of one prediction (Series) - they add up to log(forecast) - base_log."""
        t = _value_terms(self.model, [value_now], [age], [np.nan if prev_value is None else prev_value]).iloc[0]
        return t - self.mean_terms

    def global_importance(self):
        """Mean |contribution| over the training rows, in log points and as a typical % effect on the forecast."""
        imp = (self.bg_terms - self.mean_terms).abs().mean().sort_values(ascending=False)
        return pd.DataFrame({"predictor": imp.index, "mean_abs_log": imp.values, "typical_pct": 100 * (np.exp(imp.values) - 1)})

    def bg_contributions(self):
        return self.bg_terms - self.mean_terms


def pct_effect(c_log):
    return 100 * (np.exp(c_log) - 1)


def size_word(abs_effect, small, large):
    return "a small" if abs_effect < small else "a large" if abs_effect >= large else "a moderate"


def value_sentences(contrib, inputs, explainer):
    """Plain-language reading of each contribution to one value forecast (direction, size, practical meaning)."""
    out = []
    for name in contrib.sort_values(key=np.abs, ascending=False).index:
        e = pct_effect(contrib[name])
        direction = "raises" if e > 0 else "lowers"
        size = size_word(abs(e), 5, 20)
        if abs(e) < 0.5:
            out.append(f"**{name}** ({inputs[name]}) has almost no effect on this forecast: it is close to the training average.")
            continue
        if name == VALUE_PREDICTORS[0]:
            meaning = ("The forecast starts from today's value, so a player valued above the typical player in the data is "
                       "forecast above it too. This sets the level of the forecast; it says nothing about whether the price is fair.")
        elif name == VALUE_PREDICTORS[1]:
            meaning = ("In the training data, players of this age " + ("tended to gain value" if e > 0 else "tended to lose value or gain less")
                       + " over the next 12 months than the average player. This is the age pattern of the market, not a property of this player.")
        else:
            meaning = ("The model has learned that part of last year's movement tends to reverse: after a rise it expects a smaller next "
                       "change, after a fall a partial recovery." if inputs[name] != "no earlier valuation" else
                       "Without an earlier valuation the model uses no momentum, so this predictor sits at its neutral value.")
        out.append(f"**{name}** ({inputs[name]}) {direction} the forecast by about **{abs(e):.0f}%**, {size} effect. {meaning}")
    return out


# ====================================================================== KPI 2: performance forecast
def perf_sentences(contrib, inputs):
    """Plain-language reading of SHAP contributions (passes per 90) to one performance forecast."""
    out = []
    for name in contrib.sort_values(key=np.abs, ascending=False).index:
        v = contrib[name]
        if abs(v) < 0.3:
            out.append(f"**{name}** ({inputs[name]}) has almost no effect on this forecast.")
            continue
        size = size_word(abs(v), 1.5, 5)
        direction = "adds" if v > 0 else "takes away"
        meaning = {
            "Passes per 90, last season": "Passing volume carries over from one season to the next, though only partly: the model "
                                          "pulls extreme seasons back towards the middle.",
            "Passes per 90, season before": "A second season makes the estimate of the player's usual level steadier; a high or low "
                                            "earlier season moves the forecast in the same direction.",
            "Carries per 90, last season": "Players who carry the ball more also tend to be involved more in possession.",
            "Minutes played, last season": "Minutes say how much the per-90 figures can be trusted and how established the player is.",
            "Age": "Passing volume in the training data differs by age; this reflects that pattern, not the effect of getting older.",
            "Position": "Positions have different typical passing volumes (goalkeepers far fewer).",
        }[name]
        out.append(f"**{name}** ({inputs[name]}) {direction} **{abs(v):.1f} passes per 90**, {size} effect. {meaning}")
    return out


NOT_CAUSAL = ("These are associations the model learned from past (synthetic) data. They show how the model arrives at its "
              "forecast. They are not evidence that changing a predictor in real life would change the outcome.")


# ====================================================================== what the user can and cannot change (Week 11)
ROLE_FIXED, ROLE_EXTERNAL, ROLE_DECISION = "Fixed characteristic", "External condition", "You decide"

# Keys are the input names used in the what-if summaries and comparison tables.
VARIABLE_ROLES = {
    "KPI 1": {
        "Market value now": (ROLE_EXTERNAL, "Set by the public valuation, not by the club. It changes only if the valuation is revised before you bid."),
        "Market value one year ago": (ROLE_FIXED, "A past fact about the player."),
        "Age": (ROLE_FIXED, "Cannot be changed. Changing it here means asking about a different player (or a different time)."),
        "Asking price": (ROLE_DECISION, "The one input you can influence: by negotiating, or by deciding the most you will pay. "
                                        "It is not a predictor of the player's value; it is what the forecast is compared with."),
    },
    "KPI 2": {
        "Passes/90 last season": (ROLE_FIXED, "Observed history. Changing it asks about a different player or a corrected data point."),
        "Passes/90 season before": (ROLE_FIXED, "Observed history."),
        "Carries/90": (ROLE_FIXED, "Observed history."),
        "Minutes": (ROLE_EXTERNAL, "Depends on coaches, injuries and the squad around the player, not on the recruiter."),
        "Age": (ROLE_FIXED, "Cannot be changed."),
        "Position": (ROLE_FIXED, "Fixed for this model. A club could use the player elsewhere, but the model has never seen that happen."),
    },
}
DECISION_LEVERS = ("What the Head of Recruitment can actually decide: **which player to pursue**, **the most to pay** (the asking price and bid), "
                   "**when to act** (which window), and **which squad need** the signing is meant to fill. The predictors themselves describe the "
                   "player and the market. Changing them in a scenario is a way to test how sensitive the forecast is, or to compare players, "
                   "not an action the club can take.")


def changed_roles(kpi, changed):
    """Split the changed input names into (club decides, fixed or external)."""
    roles = VARIABLE_ROLES[kpi]
    return [k for k in changed if roles.get(k, (None,))[0] == ROLE_DECISION], [k for k in changed if roles.get(k, (None,))[0] != ROLE_DECISION]


def roles_table(kpi):
    return pd.DataFrame([{"Input": k, "Who controls it": r, "What it means": t} for k, (r, t) in VARIABLE_ROLES[kpi].items()])


# ====================================================================== side-by-side comparison (Week 11)
def _signed(d, fmt):
    return "no change" if abs(d) < 1e-9 else ("+" if d > 0 else "-") + fmt(abs(d))


def _money_change(a, b, eur):
    return "no change" if abs(b - a) < 0.5 else f"{_signed(b - a, eur)} ({100 * (b / a - 1):+.1f}%)"


def value_comparison(cur, base, now, bf, eur, extras, pinned=None):
    """KPI 1 table: inputs, then predictions. cur/base = raw inputs {value, prev, hist, age, ask}; now/bf = forecast dicts.
    base/bf are None for a new player (no baseline). pinned = {"label", "cur", "now", "extras"} for a second saved scenario."""
    roles = VARIABLE_ROLES["KPI 1"]

    def cells(raw, f, ex):
        return {
            "Market value now": eur(raw["value"]), "Market value one year ago": eur(raw["prev"]) if raw["hist"] else "none",
            "Age": f"{raw['age']:.1f}", "Asking price": eur(raw["ask"]),
            "Forecast value at the next valuation": eur(f["forecast"]), "Expected change from today": f"{f['change_pct']:+.1f}%",
            "80% prediction range": f"{eur(f['lo80'])} to {eur(f['hi80'])}", "Chance the fee holds": f"{100 * f['prob']:.0f}%",
            "Asking price against the forecast": f"{100 * (raw['ask'] / f['forecast'] - 1):+.1f}%",
            "Bid reading (Week 6 bands)": ex["action"], "How far to rely on it": ex["reliability"]}

    cur_c = cells(cur, now, extras)
    base_c = cells(base, bf, extras["base"]) if base else None
    pin_c = cells(pinned["cur"], pinned["now"], pinned["extras"]) if pinned else None
    rows = []
    for item in cur_c:
        is_input = item in roles
        row = {"Part": "Input" if is_input else "Prediction", "Item": item, "Who controls it": roles[item][0] if is_input else ""}
        if base_c:
            row["Baseline (recorded)"] = base_c[item]
        if pin_c:
            row[f"Pinned: {pinned['label']}"] = pin_c[item]
        row["Current scenario"] = cur_c[item]
        if base_c:
            row["Change from baseline"] = _change_cell(item, cur, base, now, bf, eur) or ("no change" if base_c[item] == cur_c[item] else "changed")
        rows.append(row)
    return pd.DataFrame(rows)


def _change_cell(item, cur, base, now, bf, eur):
    if item in ("Market value now", "Asking price"):
        k = "value" if item == "Market value now" else "ask"
        return _money_change(base[k], cur[k], eur)
    if item == "Market value one year ago":
        if cur["hist"] and base["hist"]:
            return _money_change(base["prev"], cur["prev"], eur)
        return "no change" if cur["hist"] == base["hist"] else "history added or removed"
    if item == "Age":
        return _signed(cur["age"] - base["age"], lambda d: f"{d:.1f} years")
    if item == "Forecast value at the next valuation":
        return _money_change(bf["forecast"], now["forecast"], eur)
    if item == "Expected change from today":
        return _signed(now["change_pct"] - bf["change_pct"], lambda d: f"{d:.1f} points")
    if item == "80% prediction range":
        dl, dh = now["lo80"] - bf["lo80"], now["hi80"] - bf["hi80"]
        return "no change" if abs(dl) < 0.5 and abs(dh) < 0.5 else f"low {_signed(dl, eur)}, high {_signed(dh, eur)}"
    if item == "Chance the fee holds":
        return _signed(100 * (now["prob"] - bf["prob"]), lambda d: f"{d:.0f} points")
    if item == "Asking price against the forecast":
        a0, a1 = 100 * (base["ask"] / bf["forecast"] - 1), 100 * (cur["ask"] / now["forecast"] - 1)
        return _signed(a1 - a0, lambda d: f"{d:.1f} points")
    return ""


def perf_comparison(cur, base, now, bf, target, extras, pinned=None):
    roles = VARIABLE_ROLES["KPI 2"]

    def cells(raw, f, ex):
        return {
            "Passes/90 last season": f"{raw['passes']:.1f}", "Passes/90 season before": f"{raw['before']:.1f}" if raw["hist"] else "none",
            "Carries/90": f"{raw['carries']:.1f}", "Minutes": f"{raw['minutes']:,}", "Age": f"{raw['age']:.1f}", "Position": raw["pos"],
            "Forecast passes per 90": f"{f['forecast']:.1f}", "80% prediction range": f"{f['lo80']:.1f} to {f['hi80']:.1f}",
            f"Chance of reaching the target style ({target:.0f})": f"{100 * f['p_target']:.0f}%",
            "Forecast minus target style": f"{f['forecast'] - target:+.1f}", "How far to rely on it": ex["reliability"]}

    cur_c = cells(cur, now, extras)
    base_c = cells(base, bf, extras["base"]) if base else None
    pin_c = cells(pinned["cur"], pinned["now"], pinned["extras"]) if pinned else None
    rows = []
    for item in cur_c:
        is_input = item in roles
        row = {"Part": "Input" if is_input else "Prediction", "Item": item, "Who controls it": roles[item][0] if is_input else ""}
        if base_c:
            row["Baseline (recorded)"] = base_c[item]
        if pin_c:
            row[f"Pinned: {pinned['label']}"] = pin_c[item]
        row["Current scenario"] = cur_c[item]
        if base_c:
            ch = ""
            if item == "Passes/90 last season":
                ch = _signed(cur["passes"] - base["passes"], lambda d: f"{d:.1f}")
            elif item == "Passes/90 season before":
                ch = _signed((cur["before"] if cur["hist"] else 0) - (base["before"] if base["hist"] else 0), lambda d: f"{d:.1f}") if cur["hist"] == base["hist"] else "history added or removed"
            elif item == "Carries/90":
                ch = _signed(cur["carries"] - base["carries"], lambda d: f"{d:.1f}")
            elif item == "Minutes":
                ch = _signed(cur["minutes"] - base["minutes"], lambda d: f"{d:,.0f}")
            elif item == "Age":
                ch = _signed(cur["age"] - base["age"], lambda d: f"{d:.1f} years")
            elif item == "Position":
                ch = "no change" if cur["pos"] == base["pos"] else f"{base['pos']} to {cur['pos']}"
            elif item == "Forecast passes per 90":
                ch = _signed(now["forecast"] - bf["forecast"], lambda d: f"{d:.1f}")
            elif item == "80% prediction range":
                dl, dh = now["lo80"] - bf["lo80"], now["hi80"] - bf["hi80"]
                ch = "no change" if abs(dl) < 0.05 and abs(dh) < 0.05 else f"low {_signed(dl, lambda d: f'{d:.1f}')}, high {_signed(dh, lambda d: f'{d:.1f}')}"
            elif item.startswith("Chance of reaching"):
                ch = _signed(100 * (now["p_target"] - bf["p_target"]), lambda d: f"{d:.0f} points")
            elif item == "Forecast minus target style":
                ch = _signed((now["forecast"] - bf["forecast"]), lambda d: f"{d:.1f}")
            row["Change from baseline"] = ch or ("no change" if base_c[item] == cur_c[item] else "changed")
        rows.append(row)
    return pd.DataFrame(rows)


# ====================================================================== automatic what-if summaries
def _fmt_inputs(inputs, base_inputs):
    parts = []
    for k, v in inputs.items():
        b = base_inputs.get(k) if base_inputs else None
        parts.append(f"{k}: **{v}**" + (f" (baseline {b})" if b is not None and b != v else ""))
    return parts


def _role_sentence(kpi, changed):
    """One sentence on whether the club could really change what the scenario changes."""
    decide, fixed = changed_roles(kpi, changed)
    if not changed:
        return None
    verb = lambda names: "is" if len(names) == 1 and names[0] != "Minutes" else "are"
    if fixed and not decide:
        return (f"{', '.join(fixed)} {verb(fixed)} fixed or outside the club's control, so this comparison shows how the model "
                "treats a different player or a corrected data point. It is not something the club could do.")
    if decide and not fixed:
        return f"{', '.join(decide)} is the one input the club can influence, by negotiating or by setting a limit."
    return (f"{', '.join(decide)} can be influenced by the club; {', '.join(fixed)} {verb(fixed)} fixed or outside its control, "
            "so the combined change mixes a decision with a different player.")


def _reliability_line(extras):
    notes = extras.get("cautions") or []
    if extras.get("verdict") == "do_not_rely":
        return "**Do not rely on this forecast.** " + " ".join(notes)
    if notes:
        return "**Use with caution.** " + " ".join(notes)
    return None


def value_summary(player, inputs, base_inputs, now, base, eur, extras):
    """Markdown summary of one value what-if run. now/base: forecast dicts (base None for a new player).
    extras: action, base_action, action_changed, range_split, verdict, cautions, typical_error_pct, coverage_80, price_hit_75 (Week 11 keys optional)."""
    changed = [k for k in inputs if base_inputs and inputs[k] != base_inputs.get(k)]
    lines = [f"**Scenario for {player}** — KPI 1, market value at the next yearly valuation.",
             "**Inputs:** " + "; ".join(_fmt_inputs(inputs, base_inputs)) + "."]
    role = _role_sentence("KPI 1", changed)
    if role:
        lines.append("**What was changed:** " + role)
    lines.append(f"**Model forecast for this scenario:** {eur(now['forecast'])} (80% range {eur(now['lo80'])} to {eur(now['hi80'])}); "
                 f"chance the player is worth at least the asking price: {100 * now['prob']:.0f}%.")
    tradeoffs = []
    if base is None:
        lines.append("**Comparison with the baseline:** no recorded baseline exists for a new player; the forecast stands alone. "
                     "Pick a recorded case to see a baseline next to a scenario.")
    elif not changed:
        lines.append(f"**Comparison with the baseline:** none. These are the recorded values, so this *is* the baseline ({eur(base['forecast'])}).")
    else:
        d = now["forecast"] - base["forecast"]
        dp = 100 * (now["forecast"] / base["forecast"] - 1)
        dprob = 100 * (now["prob"] - base["prob"])
        if abs(dp) < 0.5:
            lines.append(f"**Comparison with the baseline:** changing {', '.join(changed)} leaves the central forecast essentially unchanged "
                         f"({eur(base['forecast'])}); the chance the fee holds moves from {100 * base['prob']:.0f}% to {100 * now['prob']:.0f}%.")
        else:
            lines.append(f"**Comparison with the baseline:** the model's central forecast moves from {eur(base['forecast'])} to {eur(now['forecast'])}, "
                         f"{'+' if d >= 0 else '-'}{eur(abs(d))} ({dp:+.1f}%), and the chance the fee holds from {100 * base['prob']:.0f}% to "
                         f"{100 * now['prob']:.0f}%. This is a difference between two model predictions, not a measured effect.")
        w0, w1 = (base["hi80"] - base["lo80"]) / base["forecast"], (now["hi80"] - now["lo80"]) / now["forecast"]
        if "Asking price" in changed and base_inputs:
            tradeoffs.append(f"A different asking price does not change what the player is forecast to be worth; it changes how big a premium "
                             f"you would pay over that forecast ({100 * (extras.get('ask_now', 0) / now['forecast'] - 1):+.0f}% now, "
                             f"{100 * (extras.get('ask_base', 0) / base['forecast'] - 1):+.0f}% before) and therefore how likely the fee is to hold "
                             f"({dprob:+.0f} points). Whether the seller would accept is outside the model.")
        if abs(dp) >= 0.5 and w1 > 1.15 * w0:
            tradeoffs.append("The range around the new forecast is wider than before, so the central number looks more certain than it is.")
        if dp > 0.5 and dprob < -1:
            tradeoffs.append("The forecast rose but the chance the fee holds fell, because the asking price moved up by more than the forecast.")
        if dp < -0.5 and dprob > 1:
            tradeoffs.append("The forecast fell but the chance the fee holds rose, because the asking price fell by more than the forecast.")
        if extras.get("action_changed"):
            tradeoffs.append(f"The Week 6 bid reading changes from {extras['base_action']} to {extras['action']}.")
    if tradeoffs:
        lines.append("**Trade-offs:** " + " ".join(tradeoffs))
    imp = [f"Under the Week 6 bid bands the asking price reads **{extras['action']}** against this forecast."]
    if extras.get("range_split"):
        imp.append("The reading changes inside the 80% range, so the bid is a judgement call.")
    imp.append("The forecast can inform how high to bid and how much risk you accept (use the 'pay no more than' prices); it does not decide whether the player fits the squad need.")
    lines.append("**What this could inform:** " + " ".join(imp))
    rel = _reliability_line(extras)
    if rel:
        lines.append("**Reliability of this scenario:** " + rel)
    lim = ["These are model-based differences from synthetic data, not proven effects and not guaranteed outcomes."]
    if extras.get("typical_error_pct") is not None:
        lim.append(f"In the backtest the forecast was typically {extras['typical_error_pct']:.0f}% away from the actual value, and the 80% range "
                   f"held {extras['coverage_80']:.0f}% of actual values. One player in five is still expected to land outside it.")
    else:
        lim.append("One player in five is expected to land outside the 80% range.")
    lim.append("The model uses only value, age and last year's change: injuries, contract length, form and the buying club are not in it.")
    lines.append("**Limitations:** " + " ".join(lim))
    return "\n\n".join(lines)


def perf_summary(player, inputs, base_inputs, now, base, target, extras):
    changed = [k for k in inputs if base_inputs and inputs[k] != base_inputs.get(k)]
    lines = [f"**Scenario for {player}** — KPI 2, passes per 90 minutes next season.",
             "**Inputs:** " + "; ".join(_fmt_inputs(inputs, base_inputs)) + "."]
    role = _role_sentence("KPI 2", changed)
    if role:
        lines.append("**What was changed:** " + role)
    lines.append(f"**Model forecast for this scenario:** {now['forecast']:.1f} passes per 90 (80% range {now['lo80']:.1f} to {now['hi80']:.1f}). "
                 f"Chance of reaching the need's target style of {target:.0f}: {100 * extras['p_target']:.0f}%.")
    tradeoffs = []
    if base is None:
        lines.append("**Comparison with the baseline:** no recorded baseline exists for a new player; the forecast stands alone.")
    elif not changed:
        lines.append(f"**Comparison with the baseline:** none. These are the recorded values, so this *is* the baseline ({base['forecast']:.1f}).")
    else:
        d = now["forecast"] - base["forecast"]
        if abs(d) < 0.05:
            lines.append(f"**Comparison with the baseline:** changing {', '.join(changed)} leaves the forecast unchanged ({base['forecast']:.1f}). "
                         "Tree models predict in steps, so small input changes often move nothing.")
        else:
            lines.append(f"**Comparison with the baseline:** the model's forecast moves from {base['forecast']:.1f} to {now['forecast']:.1f} passes per 90 "
                         f"({d:+.1f}), and the chance of reaching the target from {100 * extras['base_p_target']:.0f}% to {100 * extras['p_target']:.0f}%. "
                         "This is a difference between two model predictions, not a measured effect.")
            if now["forecast"] >= target > base["forecast"]:
                tradeoffs.append("The scenario crosses the target line, but the range is wide: a forecast just above the target is close to a coin flip.")
            if "Minutes" in changed:
                tradeoffs.append("Minutes also change how much the per-90 figures can be trusted, not only the forecast.")
    gap = now["forecast"] - target
    if tradeoffs:
        lines.append("**Trade-offs:** " + " ".join(tradeoffs))
    imp = (f"The forecast is {abs(gap):.1f} passes per 90 {'above' if gap >= 0 else 'below'} the target style for this squad need, "
           + ("so on passing volume the player is expected to fit the role." if gap >= 0 else
              "so on passing volume the player is expected to fall short of the role unless his usage changes."))
    if extras.get("recall_pct") is not None and gap < 0:
        imp += (f" Be careful with a 'no': in the backtest only {extras['recall_pct']:.0f}% of the midfielders who really reached the target were "
                "forecast to, so the model under-calls high passers.")
    lines.append("**What this could inform:** " + imp + " The tactical fit score (TSS) uses the same statistic, so a shortfall here is a risk to the fit that screening measured on past seasons.")
    rel = _reliability_line(extras)
    if rel:
        lines.append("**Reliability of this scenario:** " + rel)
    lim = ["These are model-based differences from synthetic data, not the effect of an intervention: coaching or a new role could change passing in ways the model has never seen, "
           "and no input here is something a recruiter can change."]
    lim.append(f"In the backtest the typical error was about {extras['mae']:.0f} passes per 90 (about {extras.get('mae_group', extras['mae']):.0f} for this kind of player), "
               "and one forecast in five falls outside the 80% range. Passing volume also depends on the new team's style, which the model does not know.")
    lines.append("**Limitations:** " + " ".join(lim))
    return "\n\n".join(lines)
