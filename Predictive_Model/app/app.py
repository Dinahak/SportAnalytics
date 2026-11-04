"""
app.py
Football Recruitment Decision-Support Tool - application prototype.
Run from the project folder (the one containing README.txt):   streamlit run app/app.py

Screens
  1. Candidates           the screening result for one squad need (Week 6 rules) with both KPI forecasts added
  2. KPIs and models      what each KPI measures, why it matters, its predictors and why its algorithm was chosen
  3. Forecast accuracy    how well each model predicted outcomes it had not seen (backtests)
  4. Explain a forecast   overall predictor importance vs. the contributions to one player's forecast (SHAP)
  5. What-if analysis     change a player's predictors; the model re-predicts against a clearly marked baseline and
                          writes a summary of every scenario. Week 11: input checks, which inputs the club can change,
                          a side-by-side baseline / scenario table and a pinned second scenario
  6. Validation and limits  (Week 11) how the models were tested, against what baseline, where they fail, what was improved

KPI 1  Market value at the next yearly valuation   code/value_model.py        (steps 07-09)
KPI 2  Passes per 90 minutes next season           code/performance_model.py  (step 10)
The app only reads data/final/, data/model/ and outputs/. It predicts with the same model code that was tested.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import config as C  # noqa: E402
import input_checks as IC  # noqa: E402
from value_model import ValueForecastModel, interval, point, price_for_confidence, prob_at_least  # noqa: E402
import performance_model as PM  # noqa: E402
from explain import (DECISION_LEVERS, NOT_CAUSAL, VALUE_PREDICTORS, ValueExplainer, pct_effect, perf_comparison,  # noqa: E402
                     perf_sentences, perf_summary, roles_table, value_comparison, value_sentences, value_summary)

# ------------------------------------------------------------------ colours (one validated palette, light surface)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BAND_80, BAND_95, GREY = "#9ec5f4", "#cde2fb", "#c3c2b7"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
STATUS_ORDER = ["SHORTLIST", "MONITOR", "REJECT", "EXCLUDED"]
EUR_TICKS = dict(tickvals=[1e3, 1e4, 1e5, 1e6, 1e7], ticktext=["€1K", "€10K", "€100K", "€1M", "€10M"])

st.set_page_config(page_title="Recruitment decision support", layout="wide")


# ------------------------------------------------------------------ data
@st.cache_data
def load():
    f, m, o = C.FINAL, C.MODEL, C.OUTPUTS
    d = {
        "need": pd.read_csv(f / "squad_need.csv"),
        "run": pd.read_csv(f / "prediction_run.csv"),
        "case": pd.read_csv(f / "scouting_case.csv"),
        "cf": pd.read_csv(m / "case_forecast.csv"),
        "bp": pd.read_csv(m / "backtest_predictions.csv"),
        "age_curve": pd.read_csv(m / "age_curve.csv"),
        "cv": pd.read_csv(o / "model_selection_cv.csv"),
        "coverage": pd.read_csv(o / "model_backtest_coverage.csv"),
        "by_age": pd.read_csv(o / "model_backtest_by_age.csv"),
        "checks": pd.read_csv(o / "model_validation_results.csv"),
        "summary": json.loads((o / "model_summary.json").read_text()),
        "vt": pd.read_csv(m / "value_model_table.csv"),
        "pt": pd.read_csv(m / "performance_model_table.csv"),
        "pcf": pd.read_csv(m / "case_performance_forecast.csv"),
        "pbp": pd.read_csv(m / "performance_backtest_predictions.csv"),
        "pcv": pd.read_csv(o / "performance_model_cv.csv"),
        "psum": json.loads((o / "performance_summary.json").read_text()),
        "pjson": json.loads((m / "performance_model.json").read_text()),
        # ---- Week 11
        "w11set": json.loads((m / "week11_settings.json").read_text()),
        "w11sum": json.loads((o / "week11" / "week11_summary.json").read_text()),
        "w11": {n: pd.read_csv(o / "week11" / f"{n}.csv") for n in (
            "split_design", "leakage_checks", "kpi1_baseline_comparison", "kpi2_baseline_comparison", "kpi1_segments_backtest",
            "kpi2_segments_backtest", "kpi1_improvements", "kpi2_improvements", "kpi1_before_after_backtest", "kpi2_before_after_backtest",
            "kpi1_price_hit_rate", "kpi2_target_reach", "kpi2_model_choice_cv")},
    }
    cases = d["case"].merge(d["run"].drop(columns=["player_id", "asking_price_eur"]), on="case_id") \
        .merge(d["cf"].drop(columns=["player_id", "run_id", "asking_price_eur", "screening_status"]), on="case_id")
    cases = cases.merge(d["pcf"][["case_id", "perf_forecast", "perf_lo80", "perf_hi80", "actual_passes_p90"]], on="case_id", how="left")
    cases["status_rank"] = cases.screening_status.map({s: i for i, s in enumerate(STATUS_ORDER)})
    d["cases"] = cases.sort_values(["status_rank", "tactical_score", "risk_forecast"], ascending=[True, False, False])
    return d


@st.cache_resource
def load_models():
    read = lambda name: ValueForecastModel.from_dict(json.loads((C.MODEL / name).read_text()))
    w11 = json.loads((C.MODEL / "week11_settings.json").read_text())["value"]
    models = {C.BACKTEST_ORIGIN: read("value_model_june2025.json"), C.LIVE_ORIGIN: read("value_model.json")}
    models[C.BACKTEST_ORIGIN].extrap_inflation = w11["extrap_inflation_june2025_model"]      # Week 11: wider range outside trained ages
    models[C.LIVE_ORIGIN].extrap_inflation = w11["extrap_inflation_final_model"]
    return models


@st.cache_resource
def load_perf_models():
    """KPI 2 models, refitted from the saved training table exactly as step 10 fitted them (same rows, order and seed)."""
    t, j = D["pt"], D["pjson"]
    dev, bt = t[t.split == "development"].reset_index(drop=True), t[t.split == "backtest"].reset_index(drop=True)
    models = {C.BACKTEST_ORIGIN: PM.PerformanceForecastModel(j["name"], j["params"]).fit(dev, j["resid_quantiles_june2025"]),
              C.LIVE_ORIGIN: PM.PerformanceForecastModel(j["name"], j["params"]).fit(pd.concat([dev, bt], ignore_index=True),
                                                                                   j["resid_quantiles_final"])}
    w11 = D["w11set"]["performance"]                                    # Week 11: separate ranges for goalkeepers and outfield players
    if w11["group_ranges_adopted"]:
        models[C.BACKTEST_ORIGIN].set_group_ranges(w11["group_resid_quantiles_june2025"])
        models[C.LIVE_ORIGIN].set_group_ranges(w11["group_resid_quantiles_final"])
    return models


@st.cache_resource
def load_value_explainers():
    vt = D["vt"]
    return {C.BACKTEST_ORIGIN: ValueExplainer(MODELS[C.BACKTEST_ORIGIN], vt[vt.split == "development"]),
            C.LIVE_ORIGIN: ValueExplainer(MODELS[C.LIVE_ORIGIN], vt[vt.split != "live"])}


D, MODELS = load(), load_models()
PMODELS, VEX = load_perf_models(), load_value_explainers()
BT = D["summary"]["backtest"]
PBT = D["psum"]["backtest"]
REF = IC.reference_ranges(D["vt"], D["pt"])                              # plausible ranges for the input checks (from training data)
W11S = D["w11sum"]
_pb = D["pbp"].assign(group=lambda d: np.where(d.position == "GK", "GK", "Outfield"))
PBT_MAE_GROUP = _pb.groupby("group").error.apply(lambda e: float(e.abs().mean())).to_dict()      # typical error by group (Week 10 backtest)
_tr = D["w11"]["kpi2_target_reach"]
RECALL_PCT = float(_tr[(_tr.method == "Week 10 model") & (_tr.set == "backtest")].of_real_fits_how_many_found_pct.iloc[0])


# ------------------------------------------------------------------ helpers
def eur(x):
    if x is None or pd.isna(x):
        return "n/a"
    a = abs(x)
    if a >= 1e6:
        return f"€{x / 1e6:.2f}M"
    if a >= 1e5:
        return f"€{x / 1e3:.0f}K"
    if a >= 1e3:
        return f"€{x / 1e3:.1f}K"
    return f"€{x:.0f}"


def pct(x, signed=False):
    return "n/a" if pd.isna(x) else f"{x:+.0f}%" if signed else f"{x:.0f}%"


def style(fig, height=340, xtitle=None, ytitle=None, legend=True):
    fig.update_layout(height=height, margin=dict(l=64, r=24, t=40 if legend else 14, b=56), paper_bgcolor=SURFACE,
                      plot_bgcolor=SURFACE, font=dict(family=FONT, size=13, color=INK2), showlegend=legend,
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, font=dict(color=INK2)),
                      hoverlabel=dict(bgcolor="white", font=dict(color=INK, family=FONT)))
    axis = dict(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED), automargin=True)
    fig.update_xaxes(title=dict(text=xtitle, font=dict(color=INK2, size=13), standoff=10), **axis)
    fig.update_yaxes(title=dict(text=ytitle, font=dict(color=INK2, size=13), standoff=10), **axis)
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})


def forecast(model, value, age, prev, ask):
    """One forecast with everything the screens show. prev = None when the player has no earlier valuation."""
    pr = model.predict([value], [age], [np.nan if prev is None else prev])
    lo80, hi80 = interval(pr, C.MAIN_INTERVAL)
    lo95, hi95 = interval(pr, C.WIDE_INTERVAL)
    clo, chi = interval(pr, C.WIDE_INTERVAL, "confidence")
    out = {"pred": pr, "forecast": float(point(pr)[0]), "change_pct": float(100 * (np.exp(pr["log_change"][0]) - 1)),
           "lo80": float(lo80[0]), "hi80": float(hi80[0]), "lo95": float(lo95[0]), "hi95": float(hi95[0]),
           "ci_lo": float(clo[0]), "ci_hi": float(chi[0]), "extrapolated": bool(pr["extrapolated"][0]),
           "prob": float(prob_at_least(pr, ask)[0]),
           "sure": {c: float(price_for_confidence(pr, c)[0]) for c in C.CONFIDENCE_PRICE_LEVELS}}
    return out


def bid_rule(ask, value, budget):
    """Week 6 bid rules 7-10, applied to any value estimate. Returns (action, max bid or None, premium)."""
    prem = ask / value - 1
    if ask > budget:
        return "REJECT (asking price above budget)", None, prem
    if prem <= C.PREMIUM_BID_MAX:
        return "BID AT ASKING PRICE", ask, prem
    if prem <= C.PREMIUM_COUNTER_MAX:
        return "COUNTER-BID", min(value * C.COUNTER_BID_CAP, budget), prem
    return "REJECT (overpriced)", None, prem


def need_picker():
    need = D["need"]
    labels = {f"{r.need_code} · {r.position_code} · {r.target_window}": r.need_id for r in need.itertuples()}
    choice = st.sidebar.selectbox("Squad need", list(labels), index=len(labels) - 1, key="need_label")
    nd = need[need.need_id == labels[choice]].iloc[0]
    st.sidebar.caption(f"Budget {eur(nd.budget_allocated_eur)} · age {nd.min_age} to {nd.max_age} · "
                       f"target style {nd.target_passes_p90:.0f} passes and {nd.target_carries_p90:.0f} carries per 90")
    st.sidebar.caption("All data is synthetic. No real players or clubs.")
    return nd


# ====================================================================== screen 1: candidates
def screen_candidates():
    nd = NEED
    cases = D["cases"][D["cases"].need_id == nd.need_id]
    target_year = int(cases.target_value_date.iloc[0][:4])
    st.title("Candidates")
    st.caption(f"Screening result for {nd.need_code} ({nd.position_code}, {nd.target_window}) under the Week 6 rules, with the two KPI "
               f"forecasts added: market value in January {target_year} (KPI 1) and passes per 90 next season (KPI 2). "
               f"Prediction run: {cases.run_timestamp.iloc[0][:10]}.")

    counts = cases.screening_status.value_counts()
    cols = st.columns(5)
    cols[0].metric("Cases in pool", len(cases))
    for col, s in zip(cols[1:], STATUS_ORDER):
        col.metric(s.capitalize(), int(counts.get(s, 0)))

    f1, f2 = st.columns([3, 1])
    keep = f1.multiselect("Status", STATUS_ORDER, default=STATUS_ORDER[:3])
    budget_only = f2.toggle("Within budget only", value=True)
    view = cases[cases.screening_status.isin(keep)]
    if budget_only:
        view = view[view.asking_price_eur <= nd.budget_allocated_eur]

    fc = f"Forecast Jan {target_year}"
    table = pd.DataFrame({
        "Case": view.case_id, "Player": "Player " + view.player_id.astype(str), "Age": view.age_at_run.round(1),
        "Status": view.screening_status.str.capitalize(), "TSS": view.tactical_score, "RAPF": view.risk_forecast, "Risk": view.risk_band,
        "Value now": view.market_value_eur, "Asking price": view.asking_price_eur, fc: view.forecast_eur,
        "80% range": [f"{eur(a)} to {eur(b)}" for a, b in zip(view.pi80_low_eur, view.pi80_high_eur)],
        "Chance fee holds": 100 * view.prob_value_at_least_ask,
        "Premium (Week 6)": view.week6_premium_pct,
        "Passes/90 forecast": view.perf_forecast, "vs target style": view.perf_forecast - nd.target_passes_p90,
        "Reason for status": view.status_reason,
    })
    money = st.column_config.NumberColumn(format="euro", step=1, width="small")
    score = st.column_config.NumberColumn(format="%.0f", width="small")
    st.subheader("Ranked candidates")
    st.caption("Sorted by status, then TSS, then RAPF. 'Chance fee holds' is the chance that the player is worth at least the asking "
               f"price at the January {target_year} valuation. Select a row to open that player in the what-if screen.")
    picked = st.dataframe(
        table, hide_index=True, width="stretch", height=min(38 + 35 * len(table), 460), on_select="rerun",
        selection_mode="single-row", key="cand_table",
        column_config={"Case": st.column_config.NumberColumn(width="small"), "Age": st.column_config.NumberColumn(format="%.1f", width="small"),
                       "Value now": money, "Asking price": money, fc: money, "TSS": score, "RAPF": score,
                       "Risk": st.column_config.TextColumn(width="small"),
                       "Passes/90 forecast": st.column_config.NumberColumn(format="%.1f", width="small",
                                                                           help="KPI 2: forecast passes per 90 minutes next season."),
                       "vs target style": st.column_config.NumberColumn(format="%+.1f", width="small",
                                                                        help=f"KPI 2 forecast minus the need's target style ({nd.target_passes_p90:.0f} passes per 90)."),
                       "Premium (Week 6)": st.column_config.NumberColumn(format="%+.0f%%", width="small",
                                                                         help="Asking price against the Week 6 fair market value."),
                       "Chance fee holds": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100, width="small")})
    rows = picked.selection.rows
    if rows:
        cid = int(table.Case.iloc[rows[0]])
        if st.button(f"Open what-if forecast for {table.Player.iloc[rows[0]]}", type="primary"):
            st.session_state.wi_request = cid
            st.switch_page(PAGE_WHATIF)

    st.subheader("Will the fee hold its value?")
    st.caption(f"Chance that each player is worth at least the asking price at the January {target_year} valuation, "
               "according to the value forecast model. 50% means the asking price equals the central forecast.")
    bars = view.sort_values("prob_value_at_least_ask")
    fig = go.Figure(go.Bar(
        y="Player " + bars.player_id.astype(str) + " · " + bars.screening_status.str.capitalize(),
        x=100 * bars.prob_value_at_least_ask, orientation="h", marker=dict(color=BLUE, cornerradius=4), width=0.6,
        text=[pct(v) for v in 100 * bars.prob_value_at_least_ask], textposition="outside", textfont=dict(color=INK2),
        customdata=np.column_stack([bars.asking_price_eur.map(eur), bars.forecast_eur.map(eur),
                                    bars.pi80_low_eur.map(eur), bars.pi80_high_eur.map(eur)]),
        hovertemplate="%{y}<br>Asking price %{customdata[0]}<br>Forecast %{customdata[1]} "
                      "(80% range %{customdata[2]} to %{customdata[3]})<br>Chance worth at least asking price: %{x:.0f}%<extra></extra>"))
    fig.add_vline(x=50, line=dict(color=AXIS, width=1))
    style(fig, height=max(220, 24 * len(bars) + 60), xtitle="Chance worth at least the asking price (%)", legend=False)
    fig.update_xaxes(range=[0, 108])
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK2))
    show(fig)
    st.caption("The forecast does not change a player's screening status. Bid rules apply to shortlisted players only.")


# ====================================================================== screen 2: forecast accuracy
def screen_accuracy():
    st.title("Forecast accuracy")
    t1, t2 = st.tabs(["KPI 1 · Market value next year", "KPI 2 · Passes per 90 next season"])
    with t1:
        accuracy_value()
    with t2:
        accuracy_perf()


def accuracy_value():
    bp, s = D["bp"], D["summary"]
    st.markdown(
        f"The value forecast model predicts a player's **market value at the next yearly valuation**, from his value now, "
        f"his value a year ago and his age. This screen shows how far it can be trusted. The test: the model was fitted on "
        f"data up to January 2025 only, then asked in June 2025 to forecast the January 2026 values of "
        f"**{BT['players']:.0f} players**. Those values are now known, so each forecast can be checked.")

    k = st.columns(4)
    k[0].metric("Typical forecast error", f"{BT['median_abs_pct_error']:.0f}%",
                help="Median absolute error of the forecast, as a share of the actual value. Half of the forecasts were closer than this.")
    k[0].caption(f"Week 6 rule: {BT['week6_rule_median_abs_pct_error']:.0f}% · assuming no change: {BT['no_change_median_abs_pct_error']:.0f}%")
    k[1].metric("Forecasts within 20% of actual", f"{BT['within_20_pct']:.0f}%")
    k[1].caption(f"{BT['r2_change']:.0%} of the yearly change in value explained")
    k[2].metric("80% range contained the actual value", f"{BT['coverage_80']:.0f}%", help="A well-calibrated 80% range should contain about 80% of the actual values.")
    k[2].caption("Target: 80%")
    k[3].metric("95% range contained the actual value", f"{BT['coverage_95']:.0f}%")
    k[3].caption("Target: 95%")

    left, right = st.columns(2)
    with left:
        st.subheader("Forecast against actual value")
        st.caption("Each dot is one player. Dots on the line were forecast exactly. Both axes use a log scale.")
        groups = [("Inside the 80% range", (bp.inside_pi80 == 1) & (bp.extrapolated == 0), BLUE),
                  ("Outside the 80% range", (bp.inside_pi80 == 0) & (bp.extrapolated == 0), ORANGE),
                  ("Flagged: age not seen in training", bp.extrapolated == 1, AQUA)]
        fig = go.Figure()
        lim = [bp[["forecast_eur", "target_value_eur"]].min().min() * 0.7, bp[["forecast_eur", "target_value_eur"]].max().max() * 1.4]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=AXIS, width=1), hoverinfo="skip", showlegend=False))
        for name, mask, colour in groups:
            g = bp[mask]
            fig.add_trace(go.Scatter(
                x=g.forecast_eur, y=g.target_value_eur, mode="markers", name=f"{name} ({len(g)})",
                marker=dict(color=colour, size=9, line=dict(color=SURFACE, width=2)),
                customdata=np.column_stack([g.player_id, g.age_at_origin.round(1), g.forecast_eur.map(eur), g.target_value_eur.map(eur),
                                            g.pi80_low_eur.map(eur), g.pi80_high_eur.map(eur), g.abs_pct_error.round(0)]),
                hovertemplate="Player %{customdata[0]}, age %{customdata[1]}<br>Forecast %{customdata[2]} "
                              "(80% range %{customdata[4]} to %{customdata[5]})<br>Actual %{customdata[3]}<br>Error %{customdata[6]}%<extra></extra>"))
        style(fig, height=400, xtitle="Forecast made in June 2025", ytitle="Actual value, January 2026")
        fig.update_xaxes(type="log", **EUR_TICKS)
        fig.update_yaxes(type="log", **EUR_TICKS)
        show(fig)
    with right:
        st.subheader("Are the ranges honest?")
        st.caption("For each range width, the share of actual values that fell inside it. On the line means the stated uncertainty is right.")
        cov = D["coverage"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[45, 100], y=[45, 100], mode="lines", name="Perfectly calibrated", line=dict(color=AXIS, width=1), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=cov.nominal_pct, y=cov.actual_pct, mode="lines+markers+text", name="Value forecast model",
                                 line=dict(color=BLUE, width=2), marker=dict(color=BLUE, size=9, line=dict(color=SURFACE, width=2)),
                                 text=[f"{v:.0f}%" if n in (80, 95) else "" for n, v in zip(cov.nominal_pct, cov.actual_pct)],
                                 textposition="bottom right", textfont=dict(color=INK2),
                                 hovertemplate="%{x:.0f}% range contained %{y:.1f}% of actual values<extra></extra>"))
        style(fig, height=400, xtitle="Stated range (%)", ytitle="Actual values inside the range (%)")
        fig.update_xaxes(range=[45, 100])
        fig.update_yaxes(range=[45, 100])
        show(fig)

    st.info(
        f"**What this means for a decision.** A forecast is usually within about {BT['median_abs_pct_error']:.0f}% of the outcome, and a "
        f"typical 80% range runs from {BT['median_pi80_low_pct']:.0f}% to +{BT['median_pi80_high_pct']:.0f}% around it. The fixed ±12.5% "
        f"range used by the Week 6 fair-value rule contained the actual value only {BT['week6_range_coverage']:.0f}% of the time, so it "
        f"was far too narrow to bid against. The Week 6 bid bands (10% and 25% premium) are narrower than the real uncertainty, which is "
        f"why the what-if screen reports a chance, not only a single premium.")

    left, right = st.columns(2)
    with left:
        st.subheader("Where the model is weaker")
        st.caption("Typical forecast error by age at the forecast date (backtest).")
        ba = D["by_age"]
        fig = go.Figure(go.Bar(x=ba.age_band, y=ba.median_abs_pct_error, marker=dict(color=BLUE, cornerradius=4), width=0.35,
                               text=[f"{v:.0f}%" for v in ba.median_abs_pct_error], textposition="outside", textfont=dict(color=INK2),
                               customdata=np.column_stack([ba.players, ba.coverage_80, ba.coverage_95]),
                               hovertemplate="%{x}: %{customdata[0]} players<br>Typical error %{y:.1f}%<br>80% range contained "
                                             "%{customdata[1]:.0f}%, 95% range %{customdata[2]:.0f}%<extra></extra>"))
        style(fig, height=300, ytitle="Typical forecast error (%)", legend=False)
        fig.update_yaxes(range=[0, ba.median_abs_pct_error.max() * 1.25])
        fig.update_xaxes(showgrid=False)
        show(fig)
        st.warning(
            f"{BT['extrapolated_players']:.0f} players were older than anyone the June 2025 model had seen, and the tool flagged them. "
            f"They were its worst forecasts: without them the typical error is {BT['median_abs_pct_error_seen_ages']:.0f}% and the 80% "
            f"range contained {BT['coverage_80_seen_ages']:.0f}%. When the what-if screen shows this flag, do not rely on the forecast.",
            icon=":material/warning:")
    with right:
        st.subheader("What the model learned about age")
        st.caption("Expected change in value over the next 12 months by age, for a player whose value was stable last year. "
                   "The band is the 95% confidence interval of the curve.")
        ac = D["age_curve"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pd.concat([ac.age, ac.age[::-1]]), y=pd.concat([ac.ci95_high_pct, ac.ci95_low_pct[::-1]]),
                                 fill="toself", fillcolor=BAND_95, line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=ac.age, y=ac.expected_change_pct, mode="lines", line=dict(color=BLUE, width=2),
                                 customdata=np.column_stack([ac.ci95_low_pct, ac.ci95_high_pct]),
                                 hovertemplate="Age %{x:.1f}: %{y:+.0f}% (95% CI %{customdata[0]:+.0f}% to %{customdata[1]:+.0f}%)<extra></extra>"))
        fig.add_hline(y=0, line=dict(color=AXIS, width=1))
        style(fig, height=300, xtitle="Age at the forecast date", ytitle="Expected change in value (%)", legend=False)
        show(fig)
        st.caption("Young players gain value, players in their mid twenties hold it and older players lose it. The sharp steps at 18 "
                   "and 32 to 33 come from the synthetic data; on real data the curve would be re-estimated.")

    st.subheader("Why this model")
    st.caption("Every candidate was compared by cross-validation on the 2023 and 2024 forecasts, split by player. The chosen model "
               "was then tested once on the 2025 forecasts. Lower error is better.")
    cv = D["cv"].copy()
    short = {"No change (value stays the same)": "No change", "Week 6 rule (fixed age adjustment)": "Week 6 rule (fixed age adjustment)",
             "Linear regression (age, age squared)": "Linear regression (age)",
             "Linear regression (age, age squared, momentum, position, performance)": "Linear regression (all predictors)",
             "Random forest (all predictors)": "Random forest", "Gradient boosting (all predictors)": "Gradient boosting"}
    cv["label"] = [short.get(m, m.replace("Age-curve regression, point every ", "Age curve, every ").replace(" year(s)", " yr")
                             .replace(", with momentum", "")) + (" (chosen)" if c else "") for m, c in zip(cv.model, cv.chosen)]
    cvs = cv.sort_values("backtest_median_abs_pct_error", ascending=False)
    fig = go.Figure(go.Bar(y=cvs.label, x=cvs.backtest_median_abs_pct_error, orientation="h", width=0.6,
                           marker=dict(color=[BLUE if c else GREY for c in cvs.chosen], cornerradius=4),
                           text=[f"{v:.0f}%" for v in cvs.backtest_median_abs_pct_error], textposition="outside", textfont=dict(color=INK2),
                           customdata=np.column_stack([cvs.rmse_log, cvs.backtest_rmse_log]),
                           hovertemplate="%{y}<br>Backtest typical error %{x:.1f}%<br>Cross-validation RMSE (log) %{customdata[0]:.3f}"
                                         "<br>Backtest RMSE (log) %{customdata[1]:.3f}<extra></extra>"))
    style(fig, height=24 * len(cvs) + 70, xtitle="Typical forecast error in the backtest (%)", legend=False)
    fig.update_xaxes(range=[0, cvs.backtest_median_abs_pct_error.max() * 1.12])
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK2))
    show(fig)
    st.caption("Performance statistics (passes, carries, minutes) and position were tested and did not improve the forecast, so they are "
               "not inputs. The two machine-learning models were no more accurate at the ages they had seen and give no direct "
               "prediction interval, so the simpler age-curve regression was kept.")

    with st.expander("Show the numbers behind these charts"):
        st.markdown("**Model comparison** (RMSE on the log scale; 'seen ages' leaves out the flagged players)")
        st.dataframe(cv[["label", "family", "rmse_log", "median_abs_pct_error", "coverage_80", "coverage_95", "backtest_rmse_log",
                         "backtest_median_abs_pct_error", "backtest_rmse_log_seen_ages", "backtest_median_abs_pct_error_seen_ages"]]
                     .rename(columns={"label": "Model", "family": "Family", "rmse_log": "CV RMSE", "median_abs_pct_error": "CV typical error %",
                                      "coverage_80": "CV 80% coverage", "coverage_95": "CV 95% coverage", "backtest_rmse_log": "Backtest RMSE",
                                      "backtest_median_abs_pct_error": "Backtest typical error %",
                                      "backtest_rmse_log_seen_ages": "Backtest RMSE, seen ages",
                                      "backtest_median_abs_pct_error_seen_ages": "Backtest typical error %, seen ages"}).round(3),
                     hide_index=True, width="stretch")
        a, b = st.columns(2)
        a.markdown("**Coverage of the prediction ranges (backtest)**")
        a.dataframe(D["coverage"].rename(columns={"nominal_pct": "Stated range %", "actual_pct": "Actual inside %", "n": "Players"}),
                    hide_index=True, width="stretch")
        b.markdown("**Accuracy by age band (backtest)**")
        b.dataframe(D["by_age"].rename(columns={"age_band": "Age", "players": "Players", "median_abs_pct_error": "Typical error %",
                                                "coverage_80": "80% coverage", "coverage_95": "95% coverage",
                                                "mean_log_error": "Mean log error"}), hide_index=True, width="stretch")
        st.markdown("**Backtest forecasts, one row per player**")
        st.dataframe(bp.drop(columns=["row_id", "se_log"]), hide_index=True, width="stretch", height=260)
    with st.expander(f"Model and data validation checks ({(D['checks'].result == 'PASS').sum()} of {len(D['checks'])} passed)"):
        st.dataframe(D["checks"], hide_index=True, width="stretch")


# ====================================================================== screen 3: what-if
NEW = 0


def screen_whatif():
    st.title("What-if analysis")
    st.caption("Change a player's predictors and the model re-predicts. The **baseline** is always the forecast from the player's "
               "recorded values, so you can compare it with your scenario. Every scenario gets an automatic summary and is added "
               "to the scenario log. These are model-based scenarios, not guaranteed effects of real-world changes.")
    t1, t2 = st.tabs(["KPI 1 · Market value next year", "KPI 2 · Passes per 90 next season"])
    with t1:
        whatif_value()
    with t2:
        whatif_perf()
    scenario_log()


def whatif_value():
    nd = NEED
    cases = D["cases"][D["cases"].need_id == nd.need_id].set_index("case_id")
    year = int(cases.origin_year.iloc[0])
    model = MODELS[year]
    target = f"January {year + 1}"
    st.caption(f"Change the predictors or the asking price and the model re-predicts the value at the {target} valuation, with its "
               f"uncertainty. Model fitted on {model.n} past forecasts, ages {model.age_min:.1f} to {model.age_max:.1f}.")
    hit = D["w11"]["kpi1_price_hit_rate"].set_index("stated_confidence_pct").actual_value_at_or_above_price_pct
    st.info(f"**How much to trust this.** In the 2025 test a forecast was typically {BT['median_abs_pct_error']:.0f}% away from the actual value, "
            f"and the 'pay no more than' price marked 75% sure held for {hit[75.0]:.0f}% of players. It is least reliable for players older than "
            f"{model.age_max:.0f} (marked 'Do not rely'). Details: screen *6. Validation and limits*.", icon=":material/verified:")

    ids = {"New player (enter the values yourself)": NEW}
    ids.update({f"Case {cid} · Player {r.player_id} · {r.screening_status.capitalize()}": int(cid) for cid, r in cases.iterrows()})
    label_of = {v: k for k, v in ids.items()}
    if st.session_state.get("wi_request") in label_of:
        st.session_state.wi_label = label_of[st.session_state.pop("wi_request")]
    if st.session_state.get("wi_label") not in ids:
        short = cases[cases.screening_status == "SHORTLIST"]
        st.session_state.wi_label = label_of[int(short.index[0]) if len(short) else int(cases.index[0])]

    def recorded(cid):
        if cid == NEW:
            return {"value": 500_000.0, "prev": 450_000.0, "hist": True, "age": 24.0, "ask": 550_000.0}
        r = cases.loc[cid]
        hist = bool(r.has_history)
        return {"value": float(r.market_value_eur), "prev": float(r.prev_value_eur) if hist else float(r.market_value_eur),
                "hist": hist, "age": round(float(r.age_at_origin), 1), "ask": float(r.asking_price_eur)}

    def reset():
        for k, v in recorded(ids[st.session_state.wi_label]).items():
            st.session_state[f"wi_{k}"] = v

    if st.session_state.get("wi_loaded") != st.session_state.wi_label:
        reset()
        st.session_state.wi_loaded = st.session_state.wi_label

    with st.expander("Which inputs can I change, and which can't I?"):
        st.markdown(DECISION_LEVERS)
        st.dataframe(roles_table("KPI 1"), hide_index=True, width="stretch")
    inp, out = st.columns([1, 2.7], gap="large")
    with inp:
        st.selectbox("Start from", list(ids), key="wi_label")
        cid = ids[st.session_state.wi_label]
        base_in = recorded(cid)
        st.markdown("**Predictors**")
        st.number_input("Market value now (EUR) · external", min_value=100.0, max_value=50_000_000.0, step=10_000.0, format="%.0f", key="wi_value",
                        help="Latest public valuation before the prediction run.")
        st.toggle("Player has a valuation from one year ago", key="wi_hist")
        st.number_input("Market value one year ago (EUR) · fixed", min_value=100.0, max_value=50_000_000.0, step=10_000.0, format="%.0f",
                        key="wi_prev", disabled=not st.session_state.wi_hist,
                        help="The model expects part of last year's rise or fall to reverse.")
        st.slider("Age at the prediction run · fixed", 16.0, 36.0, step=0.1, format="%.1f", key="wi_age")
        st.markdown("**Decision input**")
        st.number_input("Asking price (EUR) · you decide", min_value=100.0, max_value=50_000_000.0, step=10_000.0, format="%.0f", key="wi_ask",
                        help="The asking price is what the forecast is compared with. It is not used to predict the player's value.")
        # Week 11 user test: the participant did not see why the price moved 'Chance fee holds' but not the forecast
        st.caption("Changing the asking price updates the chance that the player's future value will cover the fee. "
                   "It does not change the predicted market value.")
        st.button("Reset to the recorded values", on_click=reset, disabled=cid == NEW)

    s = st.session_state
    issues = IC.check_value_inputs(s.wi_value, s.wi_prev if s.wi_hist else None, s.wi_hist, s.wi_age, s.wi_ask, REF,
                                   (model.age_min, model.age_max), nd.budget_allocated_eur)
    if IC.verdict(issues) == "block":                      # impossible or contradictory inputs: no forecast
        with out:
            show_blocked(issues)
        return
    now = forecast(model, s.wi_value, s.wi_age, s.wi_prev if s.wi_hist else None, s.wi_ask)
    changed = cid != NEW and (s.wi_hist != base_in["hist"] or any(abs(s[f"wi_{k}"] - base_in[k]) > 1e-6 for k in ("value", "prev", "age", "ask")))
    base = forecast(model, base_in["value"], base_in["age"], base_in["prev"] if base_in["hist"] else None, base_in["ask"]) if cid != NEW else None

    with out:
        if cid != NEW:
            r = cases.loc[cid]
            st.markdown(f"**Player {r.player_id}** · screening status **{r.screening_status}** ({r.status_reason}) · "
                        f"TSS {r.tactical_score:.0f} · RAPF {'n/a' if pd.isna(r.risk_forecast) else format(r.risk_forecast, '.0f')} · risk {r.risk_band}")
        verdict = IC.verdict(issues, bool(now["extrapolated"]))
        show_issues(issues, verdict)

        def delta(key, fmt, smallest):
            """Change against the recorded values; nothing is shown when it rounds to zero."""
            d = now[key] - base[key] if changed and base else 0.0
            return fmt(d) if abs(d) >= smallest else None

        k = st.columns([1, 1, 1.9, 1])
        k[0].metric(f"Forecast, Jan {year + 1}", eur(now["forecast"]),
                    delta("forecast", lambda d: ("+" if d >= 0 else "-") + eur(abs(d)), 0.005 * now["forecast"]),
                    help=f"Central forecast of the market value at the {target} valuation.")
        k[1].metric("Change from today", pct(now["change_pct"], signed=True), delta("change_pct", lambda d: f"{d:+.0f} pts", 0.5))
        k[2].metric("80% prediction range", f"{eur(now['lo80'])} to {eur(now['hi80'])}",
                    help="Four times out of five, the actual value is expected to fall inside this range.")
        k[3].metric("Chance fee holds", f"{100 * now['prob']:.0f}%", delta("prob", lambda d: f"{100 * d:+.0f} pts", 0.005),
                    help=f"Chance that the player is worth at least the asking price at the {target} valuation.")

        if changed:
            st.caption("The small arrows show the change against the recorded values.")

        # ---- Week 11: baseline and scenario side by side (with an optional pinned second scenario)
        cur_raw = {"value": s.wi_value, "prev": s.wi_prev, "hist": s.wi_hist, "age": s.wi_age, "ask": s.wi_ask}
        ex_cur = {"action": bid_rule(s.wi_ask, now["forecast"], nd.budget_allocated_eur)[0], "reliability": VERDICT_WORD[verdict]}
        ex_cur["base"] = None
        if base:
            b_issues = IC.check_value_inputs(base_in["value"], base_in["prev"] if base_in["hist"] else None, base_in["hist"], base_in["age"],
                                             base_in["ask"], REF, (model.age_min, model.age_max))
            ex_cur["base"] = {"action": bid_rule(base_in["ask"], base["forecast"], nd.budget_allocated_eur)[0],
                              "reliability": VERDICT_WORD[IC.verdict(b_issues, bool(base["extrapolated"]))]}
        st.markdown("**Baseline and scenario side by side**")
        pin = pin_controls("pin_v", cid, year, f"saved {pd.Timestamp.now():%H:%M}", cur_raw, now, ex_cur)
        st.dataframe(value_comparison(cur_raw, base_in if base else None, slim(now), slim(base) if base else None, eur, ex_cur, pinned=pin),
                     hide_index=True, width="stretch")
        st.caption("Differences are between two model predictions. They are not proven effects and not guaranteed outcomes.")

        # ---- ranges against the asking price
        moved = changed and (s.wi_hist != base_in["hist"] or any(abs(s[f"wi_{k}"] - base_in[k]) > 1e-6 for k in ("value", "prev", "age")))
        rows = ([("Recorded values", base)] if moved else []) + [("What-if" if moved else "Recorded values", now)]
        fig = go.Figure()
        for i, (name, f) in enumerate(rows):
            first = i == 0
            fig.add_trace(go.Scatter(x=[f["lo95"], f["hi95"]], y=[name, name], mode="lines", line=dict(color=BAND_95, width=18),
                                     name="95% prediction range", showlegend=first, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=[f["lo80"], f["hi80"]], y=[name, name], mode="lines", line=dict(color=BAND_80, width=18),
                                     name="80% prediction range", showlegend=first, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=[f["forecast"]], y=[name], mode="markers", name="Forecast", showlegend=first,
                                     marker=dict(color=BLUE, size=13, line=dict(color=SURFACE, width=2)),
                                     hovertemplate=f"{name}<br>Forecast {eur(f['forecast'])}<br>80% range {eur(f['lo80'])} to {eur(f['hi80'])}"
                                                   f"<br>95% range {eur(f['lo95'])} to {eur(f['hi95'])}<extra></extra>"))
        if changed and abs(s.wi_ask - base_in["ask"]) > 1:
            fig.add_vline(x=base_in["ask"], line=dict(color=GREY, width=2))
            fig.add_trace(go.Scatter(x=[base_in["ask"]] * len(rows), y=[n for n, _ in rows], mode="markers", name="Recorded asking price",
                                     marker=dict(color=GREY, size=9, symbol="square"),
                                     hovertemplate=f"Recorded asking price {eur(base_in['ask'])}<extra></extra>"))
        fig.add_vline(x=s.wi_ask, line=dict(color=ORANGE, width=2))
        fig.add_trace(go.Scatter(x=[s.wi_ask] * len(rows), y=[n for n, _ in rows], mode="markers", name="Asking price",
                                 marker=dict(color=ORANGE, size=9, symbol="square"), hovertemplate=f"Asking price {eur(s.wi_ask)}<extra></extra>"))
        if cid != NEW and pd.notna(cases.loc[cid].actual_next_value_eur):
            actual = float(cases.loc[cid].actual_next_value_eur)
            fig.add_trace(go.Scatter(x=[actual], y=["Recorded values"], mode="markers", name=f"Actual value, {target}",
                                     marker=dict(color=AQUA, size=11, symbol="diamond", line=dict(color=SURFACE, width=2)),
                                     hovertemplate=f"Actual value {eur(actual)}<extra></extra>"))
        style(fig, height=150 + 50 * len(rows), xtitle=f"Value at the {target} valuation (EUR)")
        fig.update_xaxes(tickprefix="€", tickformat="~s")
        fig.update_yaxes(showgrid=False, tickfont=dict(color=INK2), categoryorder="array", categoryarray=[n for n, _ in rows][::-1])
        show(fig)

        # ---- automatic summary, contributions and log (KPI 1)
        vex = VEX[year]
        def v_inputs(i):
            return {"Market value now": eur(i["value"]), "Market value one year ago": eur(i["prev"]) if i["hist"] else "no earlier valuation",
                    "Age": f"{i['age']:.1f}", "Asking price": eur(i["ask"])}
        cur_in = {"value": s.wi_value, "prev": s.wi_prev, "hist": s.wi_hist, "age": s.wi_age, "ask": s.wi_ask}
        act_now, _, _ = bid_rule(s.wi_ask, now["forecast"], nd.budget_allocated_eur)
        extras = {"action": act_now, "verdict": verdict, "cautions": [i.message for i in issues if i.level == "caution"],
                  "typical_error_pct": BT["median_abs_pct_error"], "coverage_80": BT["coverage_80"],
                  "ask_now": s.wi_ask, "ask_base": base_in["ask"],
                  "range_split": bid_rule(s.wi_ask, now["lo80"], nd.budget_allocated_eur)[0] != bid_rule(s.wi_ask, now["hi80"], nd.budget_allocated_eur)[0]}
        if base is not None:
            extras["base_action"] = bid_rule(base_in["ask"], base["forecast"], nd.budget_allocated_eur)[0]
            extras["action_changed"] = changed and extras["base_action"] != act_now
        player = "a new player" if cid == NEW else f"Player {cases.loc[cid].player_id} (case {cid})"
        summary = value_summary(player, v_inputs(cur_in), v_inputs(base_in) if cid != NEW else None, now, base, eur, extras)
        scenario_box("KPI 1", player, summary, v_inputs(cur_in))

        c_now = vex.contributions(s.wi_value, s.wi_age, s.wi_prev if s.wi_hist else None)
        c_base = vex.contributions(base_in["value"], base_in["age"], base_in["prev"] if base_in["hist"] else None) if cid != NEW else None
        st.markdown("**Why the model gives this forecast: predictor contributions**")
        st.caption(f"Each bar is how much one predictor moves this forecast away from the forecast for an average player in the training "
                   f"data ({eur(vex.base_eur)}), in %. The effects multiply: {eur(vex.base_eur)} × all effects = {eur(now['forecast'])}. "
                   "Grey = baseline (recorded values), blue = what-if.")
        contrib_chart(VALUE_PREDICTORS, None if (c_base is None or not moved) else pct_effect(c_base).values, pct_effect(c_now).values,
                      "Effect on the forecast (%)", "%+.0f%%")

        # ---- decision reading
        a, b = st.columns([1.25, 1])
        with a:
            st.markdown("**Reading for the bid decision**")
            act, max_bid, prem = bid_rule(s.wi_ask, now["forecast"], nd.budget_allocated_eur)
            act_lo, _, _ = bid_rule(s.wi_ask, now["lo80"], nd.budget_allocated_eur)
            act_hi, _, _ = bid_rule(s.wi_ask, now["hi80"], nd.budget_allocated_eur)
            st.markdown(
                f"- The asking price is **{pct(100 * prem, signed=True)}** against the forecast value. Under the Week 6 bid bands this "
                f"reads **{act}**" + (f", up to **{eur(max_bid)}**." if act == "COUNTER-BID" else "."))
            if act_lo == act_hi:
                st.markdown(f"- The reading is **the same across the whole 80% range**, so the uncertainty does not change it.")
            else:
                st.markdown(f"- The reading **changes inside the 80% range**: {act_lo.lower()} at the low end, {act_hi.lower()} at the "
                            f"high end. Treat it as a judgement call and use the prices on the right to set a limit.")
            if cid != NEW:
                w6_act, w6_max, w6_prem = bid_rule(s.wi_ask, float(r.week6_fmv_eur), nd.budget_allocated_eur)
                st.markdown(f"- For comparison, the Week 6 fair value ({eur(r.week6_fmv_eur)}, value today) reads **{w6_act}** at this price "
                            f"({pct(100 * w6_prem, signed=True)}).")
                if r.screening_status != "SHORTLIST":
                    st.markdown(f"- This player is on **{r.screening_status}** in screening. The forecast informs the price; it does not "
                                "change the status.")
        with b:
            st.markdown("**How sure do you want to be that the fee holds its value?**")
            st.dataframe(pd.DataFrame({"Chance fee holds": [f"{c:.0%}" for c in now["sure"]],
                                       "Pay no more than": [eur(v) for v in now["sure"].values()]}), hide_index=True, width="stretch")
            st.caption(f"95% confidence interval for the average player with these inputs: {eur(now['ci_lo'])} to {eur(now['ci_hi'])}. "
                       "It is narrower than the prediction range because one player varies more than the average.")

        # ---- sensitivity
        st.markdown("**How the forecast responds to one predictor**")
        vary = st.radio("Vary", ["Age", "Change in value over the last year", "Market value now"], horizontal=True, label_visibility="collapsed")
        n = 81
        prev_now = s.wi_prev if s.wi_hist else np.nan
        if vary == "Age":
            xs = np.linspace(16, 36, n); here = s.wi_age; xt = "Age at the prediction run"
            pr = model.predict(np.full(n, s.wi_value), xs, np.full(n, prev_now))
        elif vary == "Market value now":
            xs = s.wi_value * np.linspace(0.5, 1.5, n); here = s.wi_value; xt = "Market value now (EUR), last year's value held fixed"
            pr = model.predict(xs, np.full(n, s.wi_age), np.full(n, prev_now))
        else:
            ch = np.linspace(-50, 100, n); xs = ch; xt = "Change in value over the last year (%)"
            here = 100 * (s.wi_value / s.wi_prev - 1) if s.wi_hist else 0.0
            pr = model.predict(np.full(n, s.wi_value), np.full(n, s.wi_age), s.wi_value / (1 + ch / 100))
        lo, hi = interval(pr, C.MAIN_INTERVAL)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.concatenate([xs, xs[::-1]]), y=np.concatenate([hi, lo[::-1]]), fill="toself", fillcolor=BAND_80,
                                 line=dict(width=0), name="80% prediction range", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xs, y=point(pr), mode="lines", name="Forecast", line=dict(color=BLUE, width=2),
                                 customdata=np.column_stack([[eur(v) for v in point(pr)], [eur(v) for v in lo], [eur(v) for v in hi]]),
                                 hovertemplate="%{x:,.1f}<br>Forecast %{customdata[0]}<br>80% range %{customdata[1]} to %{customdata[2]}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[xs.min(), xs.max()], y=[s.wi_ask, s.wi_ask], mode="lines", name="Asking price",
                                 line=dict(color=ORANGE, width=2), hovertemplate=f"Asking price {eur(s.wi_ask)}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[here], y=[now["forecast"]], mode="markers", name="Current setting",
                                 marker=dict(color=BLUE, size=11, line=dict(color=SURFACE, width=2)), hoverinfo="skip"))
        if vary == "Age":
            for edge in (model.age_min, model.age_max):
                fig.add_vline(x=edge, line=dict(color=AXIS, width=1))
        style(fig, height=320, xtitle=xt, ytitle=f"Value at the {target} valuation")
        fig.update_yaxes(tickprefix="€", tickformat="~s")
        if vary == "Market value now":
            fig.update_xaxes(tickprefix="€", tickformat="~s")
        show(fig)
        if vary == "Age":
            st.caption(f"The thin vertical lines mark the youngest and oldest ages in the training data ({model.age_min:.1f} and "
                       f"{model.age_max:.1f}). Outside them the forecast is an extrapolation.")
        with st.expander("Show the numbers"):
            cols = {"Forecast value": "forecast", "Expected change %": "change_pct", "80% low": "lo80", "80% high": "hi80",
                    "95% low": "lo95", "95% high": "hi95", "Chance worth at least asking price %": "prob"}
            tab = pd.DataFrame({name: {c: (100 * f[k] if k == "prob" else f[k]) for c, k in cols.items()} for name, f in rows}).round(1)
            st.dataframe(tab, width="stretch")


# ====================================================================== shared: input checks, comparison helpers (Week 11)
VERDICT_WORD = {"ok": "Normal", "caution": "Use with caution", "do_not_rely": "Do not rely", "block": "Not forecast"}


def slim(f):
    """The few numbers of a forecast that the comparison table and the pinned scenario need."""
    return {k: f[k] for k in ("forecast", "change_pct", "lo80", "hi80", "prob", "p_target") if k in f}


def show_blocked(issues):
    st.error("**This scenario cannot be forecast.** The inputs are impossible or contradict each other, so the tool does not guess.",
             icon=":material/block:")
    for i in issues:
        if i.level == "block":
            st.markdown(f"- **{' / '.join(i.fields)}:** {i.message} *{i.handling}*")
    st.caption(IC.HANDLING_POLICY)


def show_issues(issues, verdict):
    """Cautions and notes for a scenario that was forecast, plus the one-line reliability label."""
    cautions, notes = [i for i in issues if i.level == "caution"], [i for i in issues if i.level == "note"]
    if verdict == "do_not_rely":
        st.error("**Do not rely on this forecast.** " + " ".join(i.message for i in cautions), icon=":material/error:")
    elif verdict == "caution":
        st.warning("**Use with caution.** " + " ".join(i.message for i in cautions), icon=":material/warning:")
    for i in notes:
        st.info(i.message, icon=":material/info:")
    if cautions:
        with st.expander("What the tool did about this"):
            for i in cautions:
                st.markdown(f"- **{' / '.join(i.fields)}:** {i.handling}")
            st.caption(IC.HANDLING_POLICY)
    why = IC.VERDICT_TEXT[verdict].split(": ", 1)[1]
    st.caption(f"Reliability of this scenario: **{VERDICT_WORD[verdict]}**. {why[0].upper() + why[1:]}")


def pin_controls(key, cid, year, label, cur_raw, now, extras):
    """'Pin this scenario' keeps one scenario so a second one can be compared with it and with the baseline."""
    pin = st.session_state.get(key)
    pin = pin if pin and pin["case"] == cid and pin["year"] == year else None
    a, b, _ = st.columns([1.3, 1, 3])
    if a.button("Pin this scenario to compare", key=f"{key}_btn", disabled=cid == NEW,
                help="Keeps this scenario as an extra column, so you can change the inputs again and compare two scenarios with the baseline."):
        st.session_state[key] = {"case": cid, "year": year, "label": label, "cur": dict(cur_raw), "now": slim(now), "extras": dict(extras)}
        st.rerun()
    # Week 11 user test: the participant hesitated at the Pin button, so the steps are spelled out
    st.caption("To compare two scenarios: set up scenario A and press **Pin**, then change the inputs for scenario B. "
               "The table then shows the baseline, the pinned scenario A and the current scenario B side by side.")
    if pin and b.button("Clear pin", key=f"{key}_clear"):
        st.session_state.pop(key)
        st.rerun()
    return pin


# ====================================================================== shared: contributions chart, summary box, log
def contrib_chart(names, base_vals, now_vals, xtitle, fmt):
    """Horizontal bars of predictor contributions. base_vals None -> one series."""
    fig = go.Figure()
    order = list(np.array(names)[np.argsort(np.abs(now_vals))])
    if base_vals is not None:
        fig.add_trace(go.Bar(y=names, x=base_vals, orientation="h", name="Baseline", marker=dict(color=GREY, cornerradius=3),
                             text=[fmt % v for v in base_vals], textposition="outside", textfont=dict(color=INK2),
                             hovertemplate="Baseline · %{y}: %{x:+.1f}<extra></extra>"))
    fig.add_trace(go.Bar(y=names, x=now_vals, orientation="h", name="What-if" if base_vals is not None else "This forecast",
                         marker=dict(color=[BLUE if v >= 0 else ORANGE for v in now_vals], cornerradius=3),
                         text=[fmt % v for v in now_vals], textposition="outside", textfont=dict(color=INK2),
                         hovertemplate="%{y}: %{x:+.1f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=AXIS, width=1))
    lim = max(1e-6, np.max(np.abs(np.concatenate([now_vals, base_vals if base_vals is not None else []])))) * 1.6
    style(fig, height=70 + 46 * len(names) * (2 if base_vals is not None else 1), xtitle=xtitle, legend=base_vals is not None)
    fig.update_layout(barmode="group", bargap=0.25)
    fig.update_xaxes(range=[-lim, lim])
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK2), categoryorder="array", categoryarray=order)
    show(fig)


def scenario_box(kpi, player, summary, inputs):
    """Show the automatic summary and add it to the scenario log when the scenario is new."""
    st.markdown("**Automatic scenario summary**")
    with st.container(border=True):
        st.markdown(summary)
    log = st.session_state.setdefault("scenario_log", [])
    key = (kpi, player, tuple(inputs.items()))
    if not any(e["key"] == key for e in log[-20:]):
        log.append({"key": key, "time": pd.Timestamp.now().strftime("%H:%M:%S"), "kpi": kpi, "player": player, "summary": summary})


def scenario_log():
    log = st.session_state.get("scenario_log", [])
    with st.expander(f"Scenario log ({len(log)} run{'s' if len(log) != 1 else ''} this session)"):
        if not log:
            st.caption("No scenarios yet.")
            return
        text = "\n\n---\n\n".join(f"### {e['time']} · {e['kpi']} · {e['player']}\n\n{e['summary']}" for e in log)
        st.download_button("Download the log (Markdown)", text, file_name="what_if_scenarios.md")
        for e in reversed(log[-10:]):
            st.markdown(f"**{e['time']} · {e['kpi']} · {e['player']}**")
            st.markdown(e["summary"])
            st.divider()


# ====================================================================== screen 2: KPIs and models
def screen_kpis():
    st.title("KPIs and models")
    st.caption("The decision-maker is the club's Head of Recruitment. The tool supports one question: should we bid for this "
               "player, and up to what price?")
    a, b = st.columns(2, gap="large")
    with a:
        with st.container(border=True):
            st.subheader("KPI 1 · Market value at the next yearly valuation")
            st.markdown(
                "**Measures:** the player's market value (EUR) about 12 months after the transfer window.\n\n"
                "**Why it matters:** a fee is paid today but judged later. If the player's value stays above the fee, the club "
                "keeps an asset it can sell; if it falls, the fee is a loss. This protects financial sustainability and sets the "
                "maximum bid.\n\n"
                "**Predictors:** market value now (sets the level), age (values rise for young players and fall after the late "
                "twenties), change in value over the last year (part of a rise or fall tends to reverse).\n\n"
                f"**Algorithm:** linear regression on the log scale with a piecewise age curve. Chosen from {len(D['cv'])} candidates "
                "by cross-validation. Values are skewed and errors proportional, which the log scale handles; there are only "
                "547 player-years, so a small model is safer; and the regression gives honest prediction ranges. Random forest and "
                "gradient boosting were no more accurate at the ages they had seen.")
    with b:
        with st.container(border=True):
            st.subheader("KPI 2 · Passes per 90 minutes next season")
            st.markdown(
                f"**Measures:** how many passes per 90 minutes the player will make next season.\n\n"
                f"**Why it matters:** the squad need defines the target playing style in passes per 90 ({NEED.target_passes_p90:.0f} "
                "for this need). Screening measures fit on past seasons; this KPI asks whether the player will still fit when he "
                "plays for us. A signing who falls short of the style is an operational failure even at a fair price.\n\n"
                "**Predictors:** passes per 90 last season and the season before (form carries over, partly), carries per 90 "
                "(involvement on the ball), minutes played (how reliable the per-90 figures are), age and position (typical "
                "volumes differ).\n\n"
                f"**Algorithm:** {D['psum']['chosen_model'].lower()} (depth {D['psum']['params']['max_depth']}, "
                f"{D['psum']['params']['n_estimators']} trees). It had the lowest cross-validated error of "
                f"{(D['pcv'].family == 'learned').sum()} settings across ridge regression, random forest and gradient boosting. "
                "Passing depends on age and position in non-linear ways (it peaks in the mid-twenties and goalkeepers are far "
                "lower), which shallow trees capture without hand-built terms, and SHAP explains each prediction exactly.")
    st.info("A third candidate KPI, minutes played next season (availability), was tested and dropped: no predictor in the data "
            "was related to it (all correlations about 0), so a model could not beat the average.", icon=":material/info:")


# ====================================================================== screen 3b: accuracy, KPI 2
def accuracy_perf():
    pb, cv = D["pbp"], D["pcv"]
    st.markdown(f"Fitted on the 2023 and 2024 seasons, then asked to forecast the **2025 season of {PBT['players']} players** "
                "before it was played.")
    k = st.columns(4)
    k[0].metric("Typical error", f"{PBT['mae']:.1f} passes/90", help="Mean absolute error in the backtest.")
    k[0].caption(f"Same as last season: {PBT['no_change_mae']:.1f}")
    k[1].metric("Variation explained (R²)", f"{PBT['r2']:.2f}")
    k[2].metric("80% range contained actual", f"{PBT['coverage_80']:.0f}%")
    k[2].caption("Target: 80%")
    k[3].metric("95% range contained actual", f"{PBT['coverage_95']:.0f}%")
    k[3].caption("Target: 95%")
    left, right = st.columns(2)
    with left:
        st.subheader("Forecast against actual")
        fig = go.Figure()
        lim = [0, max(pb.forecast.max(), pb.target_passes_p90.max()) * 1.05]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=AXIS, width=1), hoverinfo="skip", showlegend=False))
        for name, mask, colour in [("Inside the 80% range", pb.inside_pi80 == 1, BLUE), ("Outside", pb.inside_pi80 == 0, ORANGE)]:
            g = pb[mask]
            fig.add_trace(go.Scatter(x=g.forecast, y=g.target_passes_p90, mode="markers", name=f"{name} ({len(g)})",
                                     marker=dict(color=colour, size=8, line=dict(color=SURFACE, width=1.5)),
                                     customdata=np.column_stack([g.player_id, g.position, g.age.round(1)]),
                                     hovertemplate="Player %{customdata[0]} (%{customdata[1]}, %{customdata[2]})<br>Forecast %{x:.1f}"
                                                   "<br>Actual %{y:.1f}<extra></extra>"))
        style(fig, height=380, xtitle="Forecast passes per 90 (2025)", ytitle="Actual passes per 90 (2025)")
        show(fig)
    with right:
        st.subheader("Why this model")
        c = cv.copy()
        c["label"] = [m + ("" if not p or p != p else " " + ", ".join(f"{k}={v}" for k, v in json.loads(p).items())) +
                      (" (chosen)" if ch else "") for m, p, ch in zip(c.model, c.params, c.chosen)]
        c = c.sort_values("cv_rmse", ascending=False)
        fig = go.Figure(go.Bar(y=c.label, x=c.cv_rmse, orientation="h", width=0.6,
                               marker=dict(color=[BLUE if ch else GREY for ch in c.chosen], cornerradius=3),
                               text=[f"{v:.1f}" for v in c.cv_rmse], textposition="outside", textfont=dict(color=INK2),
                               hovertemplate="%{y}<br>CV RMSE %{x:.2f}<extra></extra>"))
        style(fig, height=22 * len(c) + 80, xtitle="Cross-validation RMSE (passes per 90, lower is better)", legend=False)
        fig.update_xaxes(range=[0, c.cv_rmse.max() * 1.15])
        fig.update_yaxes(showgrid=False, tickfont=dict(color=INK2, size=11))
        show(fig)
    st.caption("Passing volume is hard to forecast: even the best model leaves a typical error of about 9 passes per 90, so read "
               "the 80% range, not only the central forecast.")


# ====================================================================== screen 4: explain a forecast
def screen_explain():
    nd = NEED
    cases = D["cases"][D["cases"].need_id == nd.need_id].set_index("case_id")
    year = int(cases.origin_year.iloc[0])
    st.title("Explain a forecast")
    st.caption("**Overall importance** (left) shows how much each predictor moves forecasts across all players. **Contributions** "
               "(right) show how each predictor moved one player's forecast, and in which direction.")
    labels = {f"Case {cid} · Player {r.player_id} · {r.screening_status.capitalize()}": cid for cid, r in cases.iterrows()}
    pick = st.selectbox("Player", list(labels), key="ex_player")
    cid = labels[pick]
    r = cases.loc[cid]
    t1, t2 = st.tabs(["KPI 1 · Market value next year", "KPI 2 · Passes per 90 next season"])

    with t1:
        vex = VEX[year]
        prev = float(r.prev_value_eur) if r.has_history else None
        c = vex.contributions(float(r.market_value_eur), float(r.age_at_origin), prev)
        fc = float(np.exp(vex.base_log + c.sum()))
        imp = vex.global_importance()
        left, right = st.columns(2, gap="large")
        with left:
            st.subheader("Overall importance · all players")
            fig = go.Figure(go.Bar(y=imp.predictor, x=imp.typical_pct, orientation="h", width=0.55, marker=dict(color=GREY, cornerradius=3),
                                   text=[f"±{v:.0f}%" for v in imp.typical_pct], textposition="outside", textfont=dict(color=INK2),
                                   hovertemplate="%{y}: typical effect ±%{x:.0f}%<extra></extra>"))
            style(fig, height=220, xtitle="Typical effect on a forecast (%)", legend=False)
            fig.update_xaxes(range=[0, imp.typical_pct.max() * 1.3])
            fig.update_yaxes(showgrid=False, autorange="reversed", tickfont=dict(color=INK2))
            show(fig)
            st.caption("Value now dominates because players' values differ up to a hundredfold; it sets the level. Age and last "
                       "year's change decide how the value moves from there.")
        with right:
            st.subheader(f"Contributions · Player {r.player_id}")
            contrib_chart(VALUE_PREDICTORS, None, pct_effect(c).values, "Effect on this forecast (%)", "%+.0f%%")
            st.caption(f"Average training player {eur(vex.base_eur)} × these effects = **{eur(fc)}** forecast for Player {r.player_id}.")
        chg = f"{100 * (r.market_value_eur / r.prev_value_eur - 1):+.0f}%" if r.has_history else "no earlier valuation"
        st.markdown("**What the contributions mean**")
        for line in value_sentences(c, {"Market value now": eur(r.market_value_eur), "Age": f"{r.age_at_origin:.1f}",
                                        "Change in value over the last year": chg}, vex):
            st.markdown("- " + line)
        st.info(NOT_CAUSAL, icon=":material/info:")

    with t2:
        pm = PMODELS[year]
        pc = D["pcf"].set_index("case_id")
        if cid not in pc.index or pd.isna(pc.loc[cid].perf_forecast):
            st.warning("This player has no completed season before the run, so there is no performance forecast to explain.")
        else:
            p = pc.loc[cid]
            x = PM.row(p.passes_last, p.passes_before, bool(p.has_before), p.carries_last, p.minutes_last, p.age, p.position)
            sv, base_v = pm.shap_values(x)
            sv = sv.iloc[0]
            imp = pm.global_importance()
            left, right = st.columns(2, gap="large")
            with left:
                st.subheader("Overall importance · all players")
                fig = go.Figure(go.Bar(y=imp.predictor, x=imp.mean_abs_shap, orientation="h", width=0.55,
                                       marker=dict(color=GREY, cornerradius=3), text=[f"±{v:.1f}" for v in imp.mean_abs_shap],
                                       textposition="outside", textfont=dict(color=INK2),
                                       hovertemplate="%{y}: mean |SHAP| %{x:.2f} passes/90<extra></extra>"))
                style(fig, height=300, xtitle="Mean |SHAP value| (passes per 90)", legend=False)
                fig.update_xaxes(range=[0, imp.mean_abs_shap.max() * 1.2])
                fig.update_yaxes(showgrid=False, autorange="reversed", tickfont=dict(color=INK2))
                show(fig)
            with right:
                st.subheader(f"Contributions · Player {r.player_id}")
                contrib_chart(list(sv.index), None, sv.values, "SHAP value (passes per 90)", "%+.1f")
                st.caption(f"Average forecast {base_v:.1f} + these contributions = **{base_v + sv.sum():.1f}** passes per 90.")
            st.subheader("How the model uses one predictor across all players")
            grp = st.selectbox("Predictor", list(PM.GROUPS)[:5] + ["Position"], key="ex_dep")
            bg = pm.background
            bsv, _ = pm.shap_values(bg)
            xval = bg[PM.GROUPS[grp][0]] if grp != "Position" else pd.Series(
                np.array(PM.POSITIONS)[bg[[f"pos_{q}" for q in PM.POSITIONS]].values.argmax(1)], index=bg.index)
            fig = go.Figure(go.Scatter(x=xval, y=bsv[grp], mode="markers", marker=dict(color=BLUE, size=6, opacity=0.55),
                                       hovertemplate="%{x}<br>SHAP %{y:+.1f}<extra></extra>", name="Training players"))
            this_x = {"Passes per 90, last season": p.passes_last, "Passes per 90, season before": p.passes_before,
                      "Carries per 90, last season": p.carries_last, "Minutes played, last season": p.minutes_last,
                      "Age": p.age, "Position": p.position}[grp]
            fig.add_trace(go.Scatter(x=[this_x], y=[sv[grp]], mode="markers", name=f"Player {r.player_id}",
                                     marker=dict(color=ORANGE, size=13, symbol="diamond", line=dict(color=SURFACE, width=2))))
            fig.add_hline(y=0, line=dict(color=AXIS, width=1))
            style(fig, height=300, xtitle=grp, ytitle="SHAP value (passes per 90)")
            show(fig)
            st.caption("Each dot is one training player. Above zero, the predictor pushed that forecast up.")
            st.markdown("**What the contributions mean**")
            ins = {"Passes per 90, last season": f"{p.passes_last:.1f}",
                   "Passes per 90, season before": f"{p.passes_before:.1f}" if p.has_before else "no earlier season",
                   "Carries per 90, last season": f"{p.carries_last:.1f}" + (" (estimated, missing)" if p.carries_missing else ""),
                   "Minutes played, last season": f"{p.minutes_last:,.0f}", "Age": f"{p.age:.1f}", "Position": p.position}
            for line in perf_sentences(sv, ins):
                st.markdown("- " + line)
            st.info(NOT_CAUSAL, icon=":material/info:")


# ====================================================================== screen 5b: what-if, KPI 2
def whatif_perf():
    nd = NEED
    cases = D["cases"][D["cases"].need_id == nd.need_id].set_index("case_id")
    year = int(cases.origin_year.iloc[0])
    pm = PMODELS[year]
    pc = D["pcf"].set_index("case_id")
    tgt = float(nd.target_passes_p90)
    st.caption(f"Forecast of passes per 90 in the {year}/{str(year + 1)[2:]} season. Model fitted on {pm.n} player-seasons, "
               f"ages {pm.age_min:.1f} to {pm.age_max:.1f}. Target style for {nd.need_code}: {tgt:.0f} passes per 90.")
    st.info(f"**How much to trust this.** In the 2025 test a forecast was typically {PBT['mae']:.1f} passes per 90 away (goalkeepers {PBT_MAE_GROUP['GK']:.1f}, "
            f"outfield players {PBT_MAE_GROUP['Outfield']:.1f}). The model finds only {RECALL_PCT:.0f}% of the midfielders who really reach the target style, "
            "so a forecast below the target is a warning, not a rejection. Details: screen *6. Validation and limits*.", icon=":material/verified:")
    ids = {"New player (enter the values yourself)": NEW}
    ids.update({f"Case {cid} · Player {r.player_id} · {r.screening_status.capitalize()}": int(cid) for cid, r in cases.iterrows()
                if cid in pc.index and pd.notna(pc.loc[cid].perf_forecast)})
    if st.session_state.get("pw_label") not in ids:
        st.session_state.pw_label = list(ids)[1] if len(ids) > 1 else list(ids)[0]

    def recorded(cid):
        if cid == NEW:
            return {"passes": 45.0, "before": 42.0, "hist": True, "carries": 10.0, "minutes": 2000, "age": 24.0, "pos": "MID"}
        p = pc.loc[cid]
        return {"passes": round(float(p.passes_last), 1), "before": round(float(p.passes_before), 1), "hist": bool(p.has_before),
                "carries": round(float(p.carries_last), 1), "minutes": int(p.minutes_last), "age": round(float(p.age), 1), "pos": p.position}

    def reset():
        for k, v in recorded(ids[st.session_state.pw_label]).items():
            st.session_state[f"pw_{k}"] = v

    if st.session_state.get("pw_loaded") != st.session_state.pw_label:
        reset()
        st.session_state.pw_loaded = st.session_state.pw_label

    with st.expander("Which inputs can I change, and which can't I?"):
        st.markdown(DECISION_LEVERS)
        st.dataframe(roles_table("KPI 2"), hide_index=True, width="stretch")
    inp, out = st.columns([1, 2.7], gap="large")
    with inp:
        st.selectbox("Start from", list(ids), key="pw_label")
        cid = ids[st.session_state.pw_label]
        base_in = recorded(cid)
        st.number_input("Passes per 90, last season · fixed", 0.0, 150.0, step=1.0, format="%.1f", key="pw_passes")
        st.toggle("Player has an earlier season", key="pw_hist")
        st.number_input("Passes per 90, season before · fixed", 0.0, 150.0, step=1.0, format="%.1f", key="pw_before",
                        disabled=not st.session_state.pw_hist)
        st.number_input("Carries per 90, last season · fixed", 0.0, 60.0, step=0.5, format="%.1f", key="pw_carries")
        st.slider("Minutes played, last season · external", 0, 3420, step=30, key="pw_minutes")
        st.slider("Age next season · fixed", 16.0, 36.0, step=0.1, format="%.1f", key="pw_age")
        st.selectbox("Position · fixed", PM.POSITIONS, key="pw_pos")
        st.button("Reset to the recorded values", on_click=reset, disabled=cid == NEW, key="pw_reset")

    s = st.session_state
    cur = {k: s[f"pw_{k}"] for k in base_in}
    issues = IC.check_perf_inputs(cur["passes"], cur["before"], cur["hist"], cur["carries"], cur["minutes"], cur["age"], cur["pos"], REF,
                                  (pm.age_min, pm.age_max))
    if IC.verdict(issues) == "block":                      # impossible or contradictory inputs: no forecast
        with out:
            show_blocked(issues)
        return

    def run(i):
        x = PM.row(i["passes"], i["before"], i["hist"], i["carries"], i["minutes"], i["age"], i["pos"])
        f = pm.predict_frame(x).iloc[0].to_dict()
        f["p_target"] = pm.prob_at_least(f["forecast"], tgt, group=pm.group_of(i["pos"]))
        f["shap"], f["base"] = pm.shap_values(x)
        return f

    now = run(cur)
    base = run(base_in) if cid != NEW else None
    changed = cid != NEW and cur != base_in

    with out:
        verdict = IC.verdict(issues, bool(now["extrapolated"]))
        show_issues(issues, verdict)
        k = st.columns(3)
        dlt = lambda key, fmt, small: (fmt(now[key] - base[key]) if changed and abs(now[key] - base[key]) >= small else None)
        k[0].metric("Forecast passes per 90", f"{now['forecast']:.1f}", dlt("forecast", lambda d: f"{d:+.1f}", 0.05))
        k[1].metric("80% prediction range", f"{now['lo80']:.1f} to {now['hi80']:.1f}")
        k[2].metric(f"Chance of reaching {tgt:.0f}", f"{100 * now['p_target']:.0f}%", dlt("p_target", lambda d: f"{100 * d:+.0f} pts", 0.005))

        # ---- Week 11: baseline and scenario side by side (with an optional pinned second scenario)
        ex_cur = {"reliability": VERDICT_WORD[verdict], "base": None}
        if base:
            b_issues = IC.check_perf_inputs(base_in["passes"], base_in["before"], base_in["hist"], base_in["carries"], base_in["minutes"],
                                            base_in["age"], base_in["pos"], REF, (pm.age_min, pm.age_max))
            ex_cur["base"] = {"reliability": VERDICT_WORD[IC.verdict(b_issues, bool(base["extrapolated"]))]}
        st.markdown("**Baseline and scenario side by side**")
        pin = pin_controls("pin_p", cid, year, f"saved {pd.Timestamp.now():%H:%M}", cur, now, ex_cur)
        st.dataframe(perf_comparison(cur, base_in if base else None, slim(now), slim(base) if base else None, tgt, ex_cur, pinned=pin),
                     hide_index=True, width="stretch")
        st.caption("Differences are between two model predictions. They are not proven effects and not guaranteed outcomes.")

        rows = ([("Baseline (recorded)", base)] if changed else []) + [("What-if" if changed else "Baseline (recorded)" if cid != NEW else "Forecast", now)]
        fig = go.Figure()
        for i, (name, f) in enumerate(rows):
            fig.add_trace(go.Scatter(x=[f["lo95"], f["hi95"]], y=[name, name], mode="lines", line=dict(color=BAND_95, width=18),
                                     name="95% range", showlegend=i == 0, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=[f["lo80"], f["hi80"]], y=[name, name], mode="lines", line=dict(color=BAND_80, width=18),
                                     name="80% range", showlegend=i == 0, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=[f["forecast"]], y=[name], mode="markers", name="Forecast", showlegend=i == 0,
                                     marker=dict(color=BLUE if name != "Baseline (recorded)" or not changed else GREY, size=13,
                                                 line=dict(color=SURFACE, width=2)),
                                     hovertemplate=f"{name}: {f['forecast']:.1f} (80% {f['lo80']:.1f} to {f['hi80']:.1f})<extra></extra>"))
        fig.add_vline(x=tgt, line=dict(color=ORANGE, width=2))
        fig.add_trace(go.Scatter(x=[tgt] * len(rows), y=[n for n, _ in rows], mode="markers", name="Target style",
                                 marker=dict(color=ORANGE, size=9, symbol="square"), hovertemplate=f"Target style {tgt:.0f}<extra></extra>"))
        style(fig, height=150 + 50 * len(rows), xtitle="Passes per 90 next season")
        fig.update_yaxes(showgrid=False, tickfont=dict(color=INK2), categoryorder="array", categoryarray=[n for n, _ in rows][::-1])
        show(fig)

        def p_inputs(i):
            return {"Passes/90 last season": f"{i['passes']:.1f}", "Passes/90 season before": f"{i['before']:.1f}" if i["hist"] else "none",
                    "Carries/90": f"{i['carries']:.1f}", "Minutes": f"{i['minutes']:,}", "Age": f"{i['age']:.1f}", "Position": i["pos"]}
        player = "a new player" if cid == NEW else f"Player {cases.loc[cid].player_id} (case {cid})"
        extras = {"p_target": now["p_target"], "base_p_target": base["p_target"] if base else None, "mae": PBT["mae"],
                  "mae_group": PBT_MAE_GROUP[pm.group_of(cur["pos"])], "recall_pct": RECALL_PCT, "verdict": verdict,
                  "cautions": [i.message for i in issues if i.level == "caution"]}
        summary = perf_summary(player, p_inputs(cur), p_inputs(base_in) if cid != NEW else None, now, base, tgt, extras)
        scenario_box("KPI 2", player, summary, p_inputs(cur))

        st.markdown("**Why the model gives this forecast: SHAP contributions**")
        st.caption(f"Each bar adds to or takes from the average forecast ({now['base']:.1f}). They sum to the forecast. "
                   "Grey = baseline, blue/orange = what-if.")
        sv_now = now["shap"].iloc[0]
        contrib_chart(list(sv_now.index), base["shap"].iloc[0].values if changed else None, sv_now.values,
                      "SHAP value (passes per 90)", "%+.1f")

        st.markdown("**How the forecast responds to one predictor**")
        vary = st.radio("Vary", ["Passes per 90, last season", "Age", "Minutes played, last season"], horizontal=True,
                        label_visibility="collapsed", key="pw_vary")
        key, rng = {"Passes per 90, last season": ("passes", np.linspace(10, 100, 61)), "Age": ("age", np.linspace(16, 36, 61)),
                    "Minutes played, last season": ("minutes", np.linspace(300, 3420, 61))}[vary]
        xs = pd.concat([PM.row(**{**{"passes_last": cur["passes"], "passes_before": cur["before"], "has_before": cur["hist"],
                                     "carries_last": cur["carries"], "minutes_last": cur["minutes"], "age": cur["age"],
                                     "position": cur["pos"]}, {"passes": "passes_last", "age": "age", "minutes": "minutes_last"}[key]: v})
                        for v in rng], ignore_index=True)
        pf = pm.predict_frame(xs)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.concatenate([rng, rng[::-1]]), y=np.concatenate([pf.hi80, pf.lo80[::-1]]), fill="toself",
                                 fillcolor=BAND_80, line=dict(width=0), name="80% range", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=rng, y=pf.forecast, mode="lines", line=dict(color=BLUE, width=2), name="Forecast",
                                 hovertemplate="%{x:.1f}: %{y:.1f}<extra></extra>"))
        fig.add_hline(y=tgt, line=dict(color=ORANGE, width=2))
        fig.add_trace(go.Scatter(x=[cur[key]], y=[now["forecast"]], mode="markers", name="Current setting",
                                 marker=dict(color=BLUE, size=11, line=dict(color=SURFACE, width=2)), hoverinfo="skip"))
        style(fig, height=300, xtitle=vary, ytitle="Passes per 90 next season")
        show(fig)
        st.caption("The orange line is the target style. Flat stretches are normal: a tree model predicts in steps.")


def seg_val(name, segment, col):
    t = D["w11"][name]
    return float(t[t.segment == segment][col].iloc[0])


S_TARGET = float(W11S["kpi2_target_passes"])


# ====================================================================== screen 6: validation and limits (Week 11)
def screen_validation():
    w = D["w11"]
    st.title("Validation and limits")
    st.caption("Written for the Head of Recruitment: how far each forecast can be trusted, where it goes wrong, and what changed in Week 11. "
               "Every number comes from forecasts checked against outcomes the model had not seen. The data is synthetic, so the figures show how the "
               "method behaves, not how it would perform on real players.")

    k1 = w["kpi1_baseline_comparison"].set_index("model")
    m1, n1, w6 = k1.loc["Week 10 model (age curve + momentum)"], k1.loc["No change (value stays the same)"], k1.loc["Week 6 rule (fixed age adjustment)"]
    k2 = w["kpi2_baseline_comparison"].set_index("model")
    m2, n2, pa = k2.loc["Week 10 model (gradient boosting)"], k2.loc["No change (same as last season)"], k2.loc["Position average"]
    hit = w["kpi1_price_hit_rate"].set_index("stated_confidence_pct").actual_value_at_or_above_price_pct
    tr = w["kpi2_target_reach"]
    trb, trn = (tr[(tr.method == m) & (tr.set == "backtest")].iloc[0] for m in ("Week 10 model", "No change (last season)"))
    ba1, ba2 = w["kpi1_before_after_backtest"], w["kpi2_before_after_backtest"]
    ext = ba1.iloc[0]

    a, b = st.columns(2, gap="large")
    with a:
        with st.container(border=True):
            st.subheader("KPI 1 · Market value next year")
            st.markdown(
                f"**Can I rely on it?** Yes for most players, with one clear exception.\n\n"
                f"- A forecast is typically **{m1.backtest_typical_error_pct:.0f}%** away from the real value. Assuming no change is "
                f"{n1.backtest_typical_error_pct:.0f}% away and the Week 6 rule {w6.backtest_typical_error_pct:.0f}%.\n"
                f"- The 'pay no more than' prices held: the price marked 75% sure held for **{hit[75.0]:.0f}%** of players, the 90% price for **{hit[90.0]:.0f}%**.\n"
                f"- Forecasts were more than 20% above the real value for {m1.backtest_overpay_risk_pct:.0f}% of players (the risk of overpaying).\n"
                f"- **Exception:** players older than the oldest in the training data ({int(ext.players_older_than_training)} of {BT['players']:.0f}). Their typical error was "
                f"{seg_val('kpi1_segments_backtest', 'Older than any development player', 'typical_abs_pct_error'):.0f}%. Do not rely on a forecast with the "
                "'Do not rely' label.")
    with b:
        with st.container(border=True):
            st.subheader("KPI 2 · Passes per 90 next season")
            st.markdown(
                f"**Can I rely on it?** As a rough guide, not as a verdict on one player.\n\n"
                f"- A forecast is typically **{m2.backtest_mae:.1f} passes per 90** away. Assuming no change is {n2.backtest_mae:.1f} away and the position average {pa.backtest_mae:.1f}.\n"
                f"- It explains **{100 * m2.backtest_r2:.0f}%** of the differences between players (no change: {100 * n2.backtest_r2:.0f}%).\n"
                f"- Calling 'will reach the target style' was right for **{trb.correct_call_pct:.0f}%** of midfielders ({trn.correct_call_pct:.0f}% by assuming no change), "
                f"but the model found only **{trb.of_real_fits_how_many_found_pct:.0f}%** of the midfielders who really reached the target. "
                "**A forecast below the target does not rule a player out.**\n"
                "- Goalkeepers are forecast most tightly; defenders and players aged 26 to 29 were under-forecast in 2025.")

    t_design, t_base, t_fail, t_improve, t_missing = st.tabs(["How it was tested", "Against a simple baseline", "Where it fails", "What we improved", "What is still missing"])

    with t_design:
        st.markdown(
            "**Time-ordered split.** Models were built on forecasts made in June 2023 and June 2024. They were then tested on forecasts made in June 2025, "
            "whose outcomes (January 2026, season 2025) are now known. This matches how the tool is used: a forecast is always made before the outcome exists. "
            "Inside the building period, models were compared by cross-validation split by player, so a player never helps predict himself.\n\n"
            "**Building and testing are kept apart.** Choices (which model, which improvement) were made only on the building period, using rules written down in advance. "
            "The 2025 test was run once per model version. One caution: the 2025 results were already visible in Week 10, so they are not a pristine test. "
            "The first untouched test is the January 2027 outcome of the June 2026 forecasts.")
        st.dataframe(w["split_design"].rename(columns={"kpi": "KPI", "part": "Part", "forecast_origins": "Forecasts made", "rows": "Rows", "players": "Players",
                                                       "outcome_period": "Outcome", "age_range": "Ages"}), hide_index=True, width="stretch")
        lk = w["leakage_checks"]
        st.markdown(f"**Data-leakage checks: {(lk.result == 'PASS').sum()} passed, {(lk.result == 'WARN').sum()} finding, {(lk.result == 'FAIL').sum()} failed.** "
                    "Each asks: would this predictor really have been known when the forecast is needed?")
        st.dataframe(lk.rename(columns={"kpi": "KPI", "check": "Check", "result": "Result", "detail": "Detail"}), hide_index=True, width="stretch")

    with t_base:
        st.markdown("A model is only useful if it beats a rule you could apply in your head. Lower error is better; the last column is the model's gain over that "
                    "baseline with a 95% interval (resampling whole players). An interval that stays above zero means the gain is not luck.")
        st.subheader("KPI 1 · Market value")
        t = w["kpi1_baseline_comparison"][["model", "cv_typical_error_pct", "backtest_typical_error_pct", "backtest_rmse_log", "backtest_within_20_pct",
                                           "backtest_overpay_risk_pct", "backtest_rmse_gain_vs_this_baseline_ci95"]]
        st.dataframe(t.rename(columns={"model": "Model or baseline", "cv_typical_error_pct": "Typical error %, building period",
                                       "backtest_typical_error_pct": "Typical error %, 2025 test", "backtest_rmse_log": "RMSE (log), 2025 test",
                                       "backtest_within_20_pct": "Within 20% of actual %", "backtest_overpay_risk_pct": "Forecast >20% too high %",
                                       "backtest_rmse_gain_vs_this_baseline_ci95": "Model's RMSE gain, 95% interval"}).round(2), hide_index=True, width="stretch")
        st.caption("RMSE (log) measures the typical size of the error in percentage terms; 0.28 is roughly a 28% typical swing, with large misses counted more.")
        st.dataframe(w["kpi1_price_hit_rate"].rename(columns={"stated_confidence_pct": "Price marked this % sure", "actual_value_at_or_above_price_pct":
                                                              "Share of players whose real value was at or above that price %", "players": "Players"}).round(1),
                     hide_index=True, width="stretch")
        st.subheader("KPI 2 · Passes per 90")
        t = w["kpi2_baseline_comparison"][["model", "cv_mae", "backtest_mae", "backtest_rmse", "backtest_r2", "backtest_within_5_passes_pct",
                                           "backtest_mae_gain_vs_this_baseline_ci95"]]
        st.dataframe(t.rename(columns={"model": "Model or baseline", "cv_mae": "Typical error (passes/90), building period", "backtest_mae": "Typical error, 2025 test",
                                       "backtest_rmse": "RMSE, 2025 test", "backtest_r2": "R²", "backtest_within_5_passes_pct": "Within 5 passes %",
                                       "backtest_mae_gain_vs_this_baseline_ci95": "Model's error reduction, 95% interval"}).round(2), hide_index=True, width="stretch")
        st.markdown(f"**Decision view: will a midfielder reach the need's target style ({S_TARGET:.0f} passes per 90)?**")
        st.dataframe(tr.rename(columns={"method": "Method", "set": "Evaluated on", "midfielders": "Midfielders", "actually_reach_target": "Really reached it",
                                        "correct_call_pct": "Correct call %", "of_players_called_fit_how_many_really_fit_pct": "Of those called a fit, really fit %",
                                        "of_real_fits_how_many_found_pct": "Of real fits, found %"}).round(1), hide_index=True, width="stretch")
        st.caption("Small groups (53 midfielders in the 2025 test, 13 who reached the target): read these percentages as rough.")
        mc = w["kpi2_model_choice_cv"]
        mine, top = mc[mc.is_week10_choice].iloc[0], mc.iloc[0]
        st.markdown(f"**Is gradient boosting really the best KPI 2 model?** Under the version-independent folds used in Week 11 it ranks {int(mine['rank'])} of {len(mc)}; "
                    f"the best is a {top.model.lower()}. But the gap ({mine.cv_rmse - top.cv_rmse:.2f} passes per 90, {100 * (mine.cv_rmse / top.cv_rmse - 1):.1f}%) is about the size of the run-to-run noise "
                    f"({top.cv_rmse_sd_across_repeats:.2f}), so the Week 10 choice was kept under the pre-set rule (a change needs a 2% gain).")

    with t_fail:
        st.markdown("Where a model does worse than its average, a forecast for that kind of player deserves less weight. Rows with fewer than 20 players are marked small.")
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.subheader("KPI 1 · Market value (2025 test)")
            sg = w["kpi1_segments_backtest"][["segment_type", "segment", "n", "typical_abs_pct_error", "bias", "coverage_80_pct", "small_sample"]]
            st.dataframe(sg.rename(columns={"segment_type": "Group by", "segment": "Segment", "n": "Players", "typical_abs_pct_error": "Typical error %",
                                            "bias": "Bias (log; + = forecast too low)", "coverage_80_pct": "80% range held %", "small_sample": "Small"}).round(2),
                         hide_index=True, width="stretch", height=430)
            st.markdown("**What this means for you**")
            st.markdown("- **Players older than the training ages:** the model has never seen them and forecast every one of them too low. The range is now widened: "
                        f"it held for {ba1.iloc[1].coverage_80_older_than_training_pct:.0f}% of them, against {ext.coverage_80_older_than_training_pct:.0f}% before. That is better but still poor, "
                        "so treat any forecast with the 'Do not rely' label as a guess.\n"
                        "- **Low-value players** were forecast slightly too low and their range held less often. Inexpensive signings carry more surprise.\n"
                        "- **Players without a valuation from a year ago** (only 9) have wider uncertainty; the tool already uses a wider range for them.")
        with c2:
            st.subheader("KPI 2 · Passes per 90 (2025 test)")
            sg = w["kpi2_segments_backtest"][["segment_type", "segment", "n", "mae", "bias", "coverage_80_pct", "small_sample"]]
            st.dataframe(sg.rename(columns={"segment_type": "Group by", "segment": "Segment", "n": "Players", "mae": "Typical error (passes/90)",
                                            "bias": "Bias (forecast minus actual)", "coverage_80_pct": "80% range held %", "small_sample": "Small"}).round(2),
                         hide_index=True, width="stretch", height=430)
            st.markdown("**What this means for you**")
            st.markdown("- **Defenders and players aged 26 to 29** were under-forecast in 2025 (about 4 to 6 passes per 90 too low). This did not appear in the building "
                        "period. Defenders' average passing was about 4 passes per 90 higher in 2025 than in 2023 and 2024, a shift for one position that the model cannot see.\n"
                        "- **High passers (over 50 per 90)** have the widest errors and their range held least often: the tool is least certain exactly where the target style sits.\n"
                        "- **Goalkeepers** are forecast within about 5 passes per 90, so a goalkeeper forecast is more informative than an outfield one.")

    with t_improve:
        st.markdown("Each improvement was judged on the building period with a rule set in advance. An improvement that fails is reported, not hidden.")
        for title, name in (("KPI 1 · Market value", "kpi1_improvements"), ("KPI 2 · Passes per 90", "kpi2_improvements")):
            st.subheader(title)
            imp = w[name][["attempt", "evidence_set", "metric", "week10", "revised", "wins_in_repeats", "rule", "decision", "note"]]
            st.dataframe(imp.rename(columns={"attempt": "Attempt", "evidence_set": "Judged on", "metric": "Measure", "week10": "Week 10", "revised": "Revised",
                                             "wins_in_repeats": "Better in repeats", "rule": "Rule set in advance", "decision": "Decision", "note": "Note"}),
                         hide_index=True, width="stretch")
        st.markdown("**Final comparison on the 2025 test, run once per version.**")
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("*KPI 1: ranges for players older than the training ages*")
            st.dataframe(ba1[["version", "coverage_80_all_pct", "coverage_80_older_than_training_pct", "coverage_95_older_than_training_pct",
                              "median_80_range_width_pct_of_forecast_older"]].rename(columns={
                "version": "Version", "coverage_80_all_pct": "80% range held, all %", "coverage_80_older_than_training_pct": "80% range held, older players %",
                "coverage_95_older_than_training_pct": "95% range held, older players %", "median_80_range_width_pct_of_forecast_older": "Typical 80% range width, older players (% of forecast)"
            }).round(1), hide_index=True, width="stretch")
            st.caption("The central forecast and the typical error are unchanged. Widening helped but did not fix these cases: they were mostly forecast far too low, "
                       "so the real fix is more data on older players.")
        with c2:
            st.markdown("*KPI 2: 80% range held, by position*")
            fig = go.Figure()
            pos = ["GK", "DEF", "MID", "FWD"]
            for i, (name, colour) in enumerate(zip(ba2.version, (GREY, BLUE))):
                fig.add_trace(go.Bar(x=pos, y=[ba2.iloc[i][f"coverage_80_{p}_pct"] for p in pos], name=["Week 10 (one pooled range)", "Week 11 (ranges by group)"][i],
                                     marker=dict(color=colour, cornerradius=3), text=[f"{ba2.iloc[i][f'coverage_80_{p}_pct']:.0f}%" for p in pos],
                                     textposition="outside", textfont=dict(color=INK2)))
            fig.add_hline(y=80, line=dict(color=ORANGE, width=2))
            style(fig, height=300, ytitle="Actual values inside the 80% range (%)")
            fig.update_layout(barmode="group")
            fig.update_yaxes(range=[0, 112])
            fig.update_xaxes(showgrid=False)
            show(fig)
            st.caption("The orange line is the target (80%). Goalkeepers' ranges were far too wide and defenders' too narrow; the point forecasts are unchanged. "
                       f"A goalkeeper's 80% range is now about {ba2.iloc[1].median_80_width_GK:.0f} passes wide, not {ba2.iloc[0].median_80_width_GK:.0f}.")

    with t_missing:
        st.markdown(
            "- **Real data.** Everything here is synthetic. The effects are illustrative, and both models must be refitted on real valuations and event data.\n"
            "- **Few old players.** Only 11 players in the 2025 test were older than the training ages, and only 53 players aged 30 or more in the building period. The age curve at the "
            "oldest ages is the least certain part of KPI 1.\n"
            "- **Short history.** Two forecast years for building and one for testing. Shifts that hit a whole position (such as the 2025 rise in defenders' passing) cannot be learned from so few years.\n"
            "- **What each model leaves out.** KPI 1 ignores injuries, contract length, form and the buying club's reputation. KPI 2 ignores the new team's playing style, which is the biggest "
            "driver of passing volume after a move.\n"
            "- **The 2025 test is not pristine.** Its results were seen in Week 10. The June 2026 forecasts, scored in January 2027, are the first clean test.\n"
            "- **No cost of being wrong.** Errors are counted in percentage or passes, not in the money a wrong bid would cost; a decision-based score would need the club's budget and alternatives.\n"
            "- **Association, not cause.** Predictor effects describe patterns in past data. They do not show that changing an input would change the outcome.")


# ====================================================================== navigation
PAGE_CANDIDATES = st.Page(screen_candidates, title="1. Candidates", url_path="candidates", default=True)
PAGE_KPIS = st.Page(screen_kpis, title="2. KPIs and models", url_path="kpis")
PAGE_ACCURACY = st.Page(screen_accuracy, title="3. Forecast accuracy", url_path="forecast-accuracy")
PAGE_EXPLAIN = st.Page(screen_explain, title="4. Explain a forecast", url_path="explain")
PAGE_WHATIF = st.Page(screen_whatif, title="5. What-if analysis", url_path="what-if")
PAGE_VALIDATION = st.Page(screen_validation, title="6. Validation and limits", url_path="validation")
page = st.navigation([PAGE_CANDIDATES, PAGE_KPIS, PAGE_ACCURACY, PAGE_EXPLAIN, PAGE_WHATIF, PAGE_VALIDATION])
st.sidebar.markdown("### Football recruitment decision support")
NEED = need_picker()
page.run()
