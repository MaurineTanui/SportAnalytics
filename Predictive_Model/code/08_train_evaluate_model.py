"""
08_train_evaluate_model.py
Step 8. Chooses, tests and fits the value forecast model, then writes everything the application shows.

  A. Model choice     Repeated cross-validation inside the development origins (2023, 2024), split by player so
                      the same player is never in both the training and the test part. Compares two simple
                      baselines, three regressions and two machine-learning models. The backtest year is not used.
  B. Backtest         The chosen model is fitted on the development origins only and forecasts the January 2026
                      values from June 2025 data (195 players it has never seen the outcome of). Accuracy and
                      interval coverage are measured here. Nothing is tuned on this step.
  C. Final model      Refitted on all origins with a known outcome (2023-2025) and used for the 2026 window.
  D. Case forecasts   One forecast per scouting case, made with the model that existed at that prediction run
                      (Summer 2025 cases: backtest model; Summer 2026 cases: final model).

Outputs: outputs/model_selection_cv.csv, model_backtest_metrics.csv, model_backtest_coverage.csv,
         model_backtest_by_age.csv, model_summary.json
         data/model/backtest_predictions.csv, value_model.json, value_model_june2025.json, model_coefficients.csv, age_curve.csv,
         player_forecast.csv, case_forecast.csv
This script never reads data/withheld/.
"""
import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
import config as C
from value_model import ValueForecastModel, interval, point, price_for_confidence, prob_at_least

C.OUTPUTS.mkdir(parents=True, exist_ok=True)
mt = pd.read_csv(C.MODEL / "value_model_table.csv")
mt["log_now"] = np.log(mt.market_value_eur)
mt["perf_z0"] = mt.perf_z.fillna(0.0)
mt["minutes_k"] = mt.minutes_played.fillna(mt.minutes_played.median()).astype(float) / 1000
mt["pos_code"] = mt.position.astype("category").cat.codes
dev = mt[mt.split == "development"].reset_index(drop=True)
bt = mt[mt.split == "backtest"].reset_index(drop=True)
live = mt[mt.split == "live"].reset_index(drop=True)
ML_FEATURES = ["log_now", "age_at_origin", "momentum_log", "has_history", "perf_z0", "has_perf", "minutes_k", "pos_code"]


def week6_adjustment(age):
    for limit, adj in C.FMV_AGE_ADJUSTMENT:
        if age <= limit:
            return adj
    return C.FMV_AGE_ADJUSTMENT[-1][1]


def fit_curve(tr, step, momentum=True, level=False):
    return ValueForecastModel(step, momentum, level).fit(tr.market_value_eur, tr.age_at_origin, tr.prev_value_eur,
                                                         tr.target_value_eur)


def curve(step, momentum=True, level=False):
    def f(tr, te):
        pr = fit_curve(tr, step, momentum, level).predict(te.market_value_eur, te.age_at_origin, te.prev_value_eur)
        return pr["log_change"], pr
    return f


def ols(formula):
    return lambda tr, te: (smf.ols(formula, data=tr).fit().predict(te).values, None)


def forest(tr, te):
    m = RandomForestRegressor(n_estimators=400, min_samples_leaf=5, random_state=C.SEED)
    return m.fit(tr[ML_FEATURES], tr.log_change).predict(te[ML_FEATURES]), None


def boosting(tr, te):
    m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, random_state=C.SEED)
    return m.fit(tr[ML_FEATURES], tr.log_change).predict(te[ML_FEATURES]), None


# every candidate predicts the log change in value; name -> (family, function)
CANDIDATES = {
    "No change (value stays the same)": ("Baseline", lambda tr, te: (np.zeros(len(te)), None)),
    "Week 6 rule (fixed age adjustment)": ("Baseline", lambda tr, te: (np.log(1 + te.age_at_origin.map(week6_adjustment)).values, None)),
    "Linear regression (age, age squared)": ("Regression", ols("log_change ~ age_at_origin + I(age_at_origin ** 2)")),
    "Linear regression (age, age squared, momentum, position, performance)":
        ("Regression", ols("log_change ~ age_at_origin + I(age_at_origin ** 2) + momentum_log + position + perf_z0 + has_perf + minutes_k")),
}
for step in C.KNOT_STEP_OPTIONS:
    CANDIDATES[f"Age-curve regression, point every {step} year(s), with momentum"] = ("Age curve", curve(step))
CANDIDATES["Age-curve regression, point every 1 year(s), no momentum"] = ("Age curve (test)", curve(1, momentum=False))
CANDIDATES["Age-curve regression, point every 1 year(s), momentum + value level"] = ("Age curve (test)", curve(1, level=True))
CANDIDATES["Random forest (all predictors)"] = ("Machine learning", forest)
CANDIDATES["Gradient boosting (all predictors)"] = ("Machine learning", boosting)


def metrics(actual_change, pred_change):
    e = np.asarray(actual_change) - np.asarray(pred_change)
    ape = np.abs(np.exp(-e) - 1)                       # absolute % error of the EUR forecast
    return {"rmse_log": np.sqrt(np.mean(e ** 2)), "mae_log": np.mean(np.abs(e)), "median_abs_pct_error": 100 * np.median(ape),
            "within_20_pct": 100 * np.mean(ape <= 0.20), "bias_log": e.mean()}


def covered(pr, actual_eur, level):
    lo, hi = interval(pr, level)
    return (np.asarray(actual_eur) >= lo) & (np.asarray(actual_eur) <= hi)


# ------------------------------------------------------------------ A. model choice by cross-validation (development only)
ids = dev.player_id.unique()
cv_rows = []
for name, (family, fn) in CANDIDATES.items():
    fold_rmse, all_err, c80, c95 = [], [], [], []
    for rep in range(C.CV_REPEATS):
        rng = np.random.default_rng(C.SEED + rep)
        fold = dev.player_id.map(dict(zip(ids, rng.permutation(len(ids)) % C.CV_FOLDS))).values
        for k in range(C.CV_FOLDS):
            tr, te = dev[fold != k], dev[fold == k]
            pred, pr = fn(tr, te)
            fold_rmse.append(np.sqrt(np.mean((te.log_change.values - pred) ** 2)))
            all_err.append(np.column_stack([te.log_change.values, pred]))
            if pr is not None:
                c80.append(covered(pr, te.target_value_eur, C.MAIN_INTERVAL))
                c95.append(covered(pr, te.target_value_eur, C.WIDE_INTERVAL))
    a = np.vstack(all_err)
    row = {"model": name, "family": family, **metrics(a[:, 0], a[:, 1]),
           "rmse_log_fold_sd": np.std(fold_rmse, ddof=1),
           "coverage_80": 100 * np.concatenate(c80).mean() if c80 else np.nan,
           "coverage_95": 100 * np.concatenate(c95).mean() if c95 else np.nan}
    cv_rows.append(row)
cv = pd.DataFrame(cv_rows)
pick = cv[cv.family == "Age curve"].sort_values("rmse_log").iloc[0]
STEP = int(pick.model.split("every ")[1].split(" ")[0])
cv["chosen"] = (cv.model == pick.model).astype(int)
print(f"[08] chosen by cross-validation: {pick.model}  (CV RMSE {pick.rmse_log:.4f})")

# ------------------------------------------------------------------ B. backtest: fit on development, forecast origin 2025
bt_rows = []
seen_age = bt.age_at_origin.between(dev.age_at_origin.min(), dev.age_at_origin.max()).values   # ages present in training
for name, (family, fn) in CANDIDATES.items():
    pred, _ = fn(dev, bt)
    in_range = metrics(bt.log_change.values[seen_age], pred[seen_age])
    bt_rows.append({"model": name, "family": family, **metrics(bt.log_change.values, pred),
                    "rmse_log_seen_ages": in_range["rmse_log"], "median_abs_pct_error_seen_ages": in_range["median_abs_pct_error"]})
btm = pd.DataFrame(bt_rows)
cv = cv.merge(btm.drop(columns="family").add_prefix("backtest_").rename(columns={"backtest_model": "model"}), on="model")
cv.round(4).to_csv(C.OUTPUTS / "model_selection_cv.csv", index=False)

m_bt = fit_curve(dev, STEP)
(C.MODEL / "value_model_june2025.json").write_text(json.dumps(m_bt.to_dict(), indent=1))   # the model as it stood in June 2025
pr = m_bt.predict(bt.market_value_eur, bt.age_at_origin, bt.prev_value_eur)
bp = bt[["row_id", "player_id", "position", "age_at_origin", "market_value_eur", "prev_value_eur", "has_history",
         "target_value_date", "target_value_eur"]].copy()
bp["forecast_eur"] = point(pr).round(2)
for lvl in (C.MAIN_INTERVAL, C.WIDE_INTERVAL):
    lo, hi = interval(pr, lvl)
    bp[f"pi{int(lvl * 100)}_low_eur"], bp[f"pi{int(lvl * 100)}_high_eur"] = lo.round(2), hi.round(2)
    bp[f"inside_pi{int(lvl * 100)}"] = covered(pr, bt.target_value_eur, lvl).astype(int)
bp["abs_pct_error"] = (100 * np.abs(bp.forecast_eur / bp.target_value_eur - 1)).round(2)
bp["log_error"] = (np.log(bp.target_value_eur) - pr["log_pred"]).round(5)
bp["se_log"] = pr["se_pred"].round(5)                               # standard error of the forecast (log scale)
bp["extrapolated"] = pr["extrapolated"]
w6 = bt.market_value_eur * (1 + bt.age_at_origin.map(week6_adjustment))
bp["week6_rule_forecast_eur"] = w6.round(2)
bp["inside_week6_range"] = ((bt.target_value_eur >= w6 * (1 - C.FMV_RANGE)) & (bt.target_value_eur <= w6 * (1 + C.FMV_RANGE))).astype(int)
bp.to_csv(C.MODEL / "backtest_predictions.csv", index=False)

cov_rows = [{"nominal_pct": 100 * lvl, "actual_pct": round(100 * covered(pr, bt.target_value_eur, lvl).mean(), 1),
             "n": len(bt)} for lvl in C.INTERVAL_LEVELS]
pd.DataFrame(cov_rows).to_csv(C.OUTPUTS / "model_backtest_coverage.csv", index=False)

bands = pd.cut(bp.age_at_origin, [0, 19, 22, 26, 30, 99], labels=["Under 19", "19 to 21", "22 to 25", "26 to 29", "30 and over"], right=False)
by_age = bp.groupby(bands, observed=True).agg(players=("row_id", "size"), median_abs_pct_error=("abs_pct_error", "median"),
                                              coverage_80=("inside_pi80", "mean"), coverage_95=("inside_pi95", "mean"),
                                              mean_log_error=("log_error", "mean")).reset_index(names="age_band")
by_age[["coverage_80", "coverage_95"]] = (100 * by_age[["coverage_80", "coverage_95"]]).round(1)
by_age.round(3).to_csv(C.OUTPUTS / "model_backtest_by_age.csv", index=False)

chosen = btm[btm.model == pick.model].iloc[0]
base = btm[btm.model.str.startswith("No change")].iloc[0]
rule = btm[btm.model.str.startswith("Week 6")].iloc[0]
seen = bp[bp.extrapolated == 0]
ss_res, ss_tot = (bp.log_error ** 2).sum(), ((bt.log_change - dev.log_change.mean()) ** 2).sum()
summary_bt = {
    "players": len(bt), "median_abs_pct_error": chosen.median_abs_pct_error, "within_20_pct": chosen.within_20_pct,
    "rmse_log": chosen.rmse_log, "bias_log": chosen.bias_log,
    "r2_change": 1 - ss_res / ss_tot,                                  # share of the change in value that is explained
    "coverage_80": 100 * bp.inside_pi80.mean(), "coverage_95": 100 * bp.inside_pi95.mean(),
    "median_pi80_low_pct": 100 * float(np.median(bp.pi80_low_eur / bp.forecast_eur - 1)),
    "median_pi80_high_pct": 100 * float(np.median(bp.pi80_high_eur / bp.forecast_eur - 1)),
    "no_change_median_abs_pct_error": base.median_abs_pct_error, "no_change_rmse_log": base.rmse_log,
    "week6_rule_median_abs_pct_error": rule.median_abs_pct_error, "week6_rule_rmse_log": rule.rmse_log,
    "week6_range_coverage": 100 * bp.inside_week6_range.mean(),
    "extrapolated_players": int(bp.extrapolated.sum()),
    "median_abs_pct_error_seen_ages": float(seen.abs_pct_error.median()),
    "coverage_80_seen_ages": 100 * seen.inside_pi80.mean(),
}
pd.DataFrame([{"measure": k, "value": round(float(v), 4)} for k, v in summary_bt.items()]) \
    .to_csv(C.OUTPUTS / "model_backtest_metrics.csv", index=False)

# ------------------------------------------------------------------ C. final model: refit on every origin with a known outcome
known = pd.concat([dev, bt], ignore_index=True)
m_final = fit_curve(known, STEP)
(C.MODEL / "value_model.json").write_text(json.dumps(m_final.to_dict(), indent=1))
se = np.sqrt(np.diag(m_final.cov))
pd.DataFrame({"term": m_final.names, "coefficient": m_final.coef.round(5), "std_error": se.round(5),
              "t_value": (m_final.coef / se).round(2)}).to_csv(C.MODEL / "model_coefficients.csv", index=False)
ages = np.round(np.arange(m_final.age_min, m_final.age_max + 1e-9, 0.25), 2)
ch, se_c = m_final.age_curve(ages)
t95 = 1.96
pd.DataFrame({"age": ages, "expected_change_pct": (100 * (np.exp(ch) - 1)).round(2),
              "ci95_low_pct": (100 * (np.exp(ch - t95 * se_c) - 1)).round(2),
              "ci95_high_pct": (100 * (np.exp(ch + t95 * se_c) - 1)).round(2)}).to_csv(C.MODEL / "age_curve.csv", index=False)


def forecast_table(model, rows, model_label):
    p = model.predict(rows.market_value_eur, rows.age_at_origin, rows.prev_value_eur)
    out = rows[["player_id", "origin_year", "forecast_origin", "position", "age_at_origin", "market_value_eur",
                "prev_value_eur", "has_history", "target_value_date"]].copy()
    out["model_fitted_on"] = model_label
    out["forecast_eur"] = point(p).round(2)
    out["expected_change_pct"] = (100 * (np.exp(p["log_change"]) - 1)).round(2)
    for lvl in (C.MAIN_INTERVAL, C.WIDE_INTERVAL):
        lo, hi = interval(p, lvl)
        out[f"pi{int(lvl * 100)}_low_eur"], out[f"pi{int(lvl * 100)}_high_eur"] = lo.round(2), hi.round(2)
    lo, hi = interval(p, C.WIDE_INTERVAL, "confidence")
    out["ci95_low_eur"], out["ci95_high_eur"] = lo.round(2), hi.round(2)
    out["extrapolated"] = p["extrapolated"]
    return out, p


pf, _ = forecast_table(m_final, live, "origins 2023-2025")
pf.to_csv(C.MODEL / "player_forecast.csv", index=False)

# ------------------------------------------------------------------ D. one forecast per scouting case, with the model of its window
case = pd.read_csv(C.FINAL / "scouting_case.csv")
run = pd.read_csv(C.FINAL / "prediction_run.csv", parse_dates=["run_timestamp"])
cases = case.merge(run[["case_id", "run_id", "run_timestamp", "predicted_fmv_eur", "premium_pct", "screening_status"]], on="case_id")
cases["origin_year"] = cases.run_timestamp.dt.year
parts = []
for year, (model, label) in {C.BACKTEST_ORIGIN: (m_bt, "origins 2023-2024"), C.LIVE_ORIGIN: (m_final, "origins 2023-2025")}.items():
    rows = cases[cases.origin_year == year].merge(mt[mt.origin_year == year], on=["player_id", "origin_year"], how="left")
    out, p = forecast_table(model, rows, label)
    out.insert(0, "case_id", rows.case_id.values)
    out.insert(1, "run_id", rows.run_id.values)
    ask = rows.asking_price_eur.values
    out["asking_price_eur"] = ask
    out["week6_fmv_eur"] = rows.predicted_fmv_eur.values
    out["week6_premium_pct"] = rows.premium_pct.values
    out["forecast_premium_pct"] = (100 * (ask / out.forecast_eur - 1)).round(2)
    out["prob_value_at_least_ask"] = prob_at_least(p, ask).round(4)
    for conf in C.CONFIDENCE_PRICE_LEVELS:
        out[f"price_{int(conf * 100)}pct_sure_eur"] = price_for_confidence(p, conf).round(-2)
    out["screening_status"] = rows.screening_status.values
    out["actual_next_value_eur"] = rows.target_value_eur.values      # known for Summer 2025 cases only
    parts.append(out)
cf = pd.concat(parts, ignore_index=True).sort_values("case_id")
assert len(cf) == len(case) and cf.forecast_eur.notna().all()
cf.to_csv(C.MODEL / "case_forecast.csv", index=False)

summary = {
    "chosen_model": pick.model, "knot_step_years": STEP, "target": "market value at the next yearly valuation",
    "development_rows": len(dev), "development_players": int(dev.player_id.nunique()),
    "cv_rmse_log": round(float(pick.rmse_log), 4), "cv_coverage_80": round(float(pick.coverage_80), 1),
    "cv_coverage_95": round(float(pick.coverage_95), 1),
    "backtest": {k: round(float(v), 3) for k, v in summary_bt.items()},
    "final_model": {"rows": m_final.n, "parameters": m_final.p, "age_min": round(m_final.age_min, 2),
                    "age_max": round(m_final.age_max, 2),
                    "residual_sd_log_with_history": round(float(np.sqrt(m_final.sigma2_history)), 4),
                    "residual_sd_log_no_history": round(float(np.sqrt(m_final.sigma2_no_history)), 4),
                    "momentum_coefficient": round(float(m_final.coef[m_final.names.index("momentum")]), 4)},
    "live_forecasts": len(pf), "case_forecasts": len(cf),
}
(C.OUTPUTS / "model_summary.json").write_text(json.dumps(summary, indent=1))

pd.set_option("display.width", 220)
print(cv[["model", "rmse_log", "median_abs_pct_error", "coverage_80", "coverage_95", "backtest_rmse_log",
          "backtest_median_abs_pct_error"]].round(3).to_string(index=False))
print("[08] backtest (June 2025 -> January 2026):", json.dumps(summary["backtest"], indent=1))
print(by_age.round(2).to_string(index=False))
print(pd.DataFrame(cov_rows).to_string(index=False))
print(f"[08] final model: {m_final.n} rows, {m_final.p} coefficients -> data/model/value_model.json; "
      f"{len(pf)} live forecasts, {len(cf)} case forecasts")
