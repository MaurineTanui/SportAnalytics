"""
app.py
Football Recruitment Decision-Support Tool - application prototype.
Run from the project folder (the one containing README.txt):   streamlit run app/app.py

Screens
  1. Candidates         the screening result for one squad need (Week 6 rules) with the value forecast added
  2. Forecast accuracy  how well the value forecast model predicted values it had not seen (backtest)
  3. What-if forecast   change a player's predictors and the asking price; the model re-predicts with its uncertainty

The app only reads data/final/, data/model/ and outputs/. It predicts with code/value_model.py, the same code
that was tested in steps 08 and 09.
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
from value_model import ValueForecastModel, interval, point, price_for_confidence, prob_at_least  # noqa: E402

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
    }
    cases = d["case"].merge(d["run"].drop(columns=["player_id", "asking_price_eur"]), on="case_id") \
        .merge(d["cf"].drop(columns=["player_id", "run_id", "asking_price_eur", "screening_status"]), on="case_id")
    cases["status_rank"] = cases.screening_status.map({s: i for i, s in enumerate(STATUS_ORDER)})
    d["cases"] = cases.sort_values(["status_rank", "tactical_score", "risk_forecast"], ascending=[True, False, False])
    return d


@st.cache_resource
def load_models():
    read = lambda name: ValueForecastModel.from_dict(json.loads((C.MODEL / name).read_text()))
    return {C.BACKTEST_ORIGIN: read("value_model_june2025.json"), C.LIVE_ORIGIN: read("value_model.json")}


D, MODELS = load(), load_models()
BT = D["summary"]["backtest"]


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
    st.caption(f"Screening result for {nd.need_code} ({nd.position_code}, {nd.target_window}) under the Week 6 rules, with the "
               f"value forecast for January {target_year} added. Prediction run: {cases.run_timestamp.iloc[0][:10]}.")

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
        "Premium (Week 6)": view.week6_premium_pct, "Reason for status": view.status_reason,
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
    bp, s = D["bp"], D["summary"]
    st.title("Forecast accuracy")
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
    nd = NEED
    cases = D["cases"][D["cases"].need_id == nd.need_id].set_index("case_id")
    year = int(cases.origin_year.iloc[0])
    model = MODELS[year]
    target = f"January {year + 1}"
    st.title("What-if forecast")
    st.caption(f"Change the predictors or the asking price and the model re-predicts the value at the {target} valuation, with its "
               f"uncertainty. Model fitted on {model.n} past forecasts, ages {model.age_min:.1f} to {model.age_max:.1f}.")

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

    inp, out = st.columns([1, 2.7], gap="large")
    with inp:
        st.selectbox("Start from", list(ids), key="wi_label")
        cid = ids[st.session_state.wi_label]
        base_in = recorded(cid)
        st.markdown("**Predictors**")
        st.number_input("Market value now (EUR)", min_value=100.0, max_value=50_000_000.0, step=10_000.0, format="%.0f", key="wi_value",
                        help="Latest public valuation before the prediction run.")
        st.toggle("Player has a valuation from one year ago", key="wi_hist")
        st.number_input("Market value one year ago (EUR)", min_value=100.0, max_value=50_000_000.0, step=10_000.0, format="%.0f",
                        key="wi_prev", disabled=not st.session_state.wi_hist,
                        help="The model expects part of last year's rise or fall to reverse.")
        st.slider("Age at the prediction run", 16.0, 36.0, step=0.1, format="%.1f", key="wi_age")
        st.markdown("**Decision input**")
        st.number_input("Asking price (EUR)", min_value=100.0, max_value=50_000_000.0, step=10_000.0, format="%.0f", key="wi_ask")
        st.button("Reset to the recorded values", on_click=reset, disabled=cid == NEW)

    s = st.session_state
    now = forecast(model, s.wi_value, s.wi_age, s.wi_prev if s.wi_hist else None, s.wi_ask)
    changed = cid != NEW and (s.wi_hist != base_in["hist"] or any(abs(s[f"wi_{k}"] - base_in[k]) > 1e-6 for k in ("value", "prev", "age", "ask")))
    base = forecast(model, base_in["value"], base_in["age"], base_in["prev"] if base_in["hist"] else None, base_in["ask"]) if cid != NEW else None

    with out:
        if cid != NEW:
            r = cases.loc[cid]
            st.markdown(f"**Player {r.player_id}** · screening status **{r.screening_status}** ({r.status_reason}) · "
                        f"TSS {r.tactical_score:.0f} · RAPF {'n/a' if pd.isna(r.risk_forecast) else format(r.risk_forecast, '.0f')} · risk {r.risk_band}")
        if now["extrapolated"]:
            st.error(f"Age {s.wi_age:.1f} is outside the ages the model has seen ({model.age_min:.1f} to {model.age_max:.1f}). The forecast "
                     "holds the age effect at the nearest age it knows and the range is too narrow. Do not rely on it.", icon=":material/error:")

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


# ====================================================================== navigation
PAGE_CANDIDATES = st.Page(screen_candidates, title="1. Candidates", url_path="candidates", default=True)
PAGE_ACCURACY = st.Page(screen_accuracy, title="2. Forecast accuracy", url_path="forecast-accuracy")
PAGE_WHATIF = st.Page(screen_whatif, title="3. What-if forecast", url_path="what-if")
page = st.navigation([PAGE_CANDIDATES, PAGE_ACCURACY, PAGE_WHATIF])
st.sidebar.markdown("### Football recruitment decision support")
NEED = need_picker()
page.run()
