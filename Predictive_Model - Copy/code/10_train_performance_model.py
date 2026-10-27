"""
10_train_performance_model.py
KPI 2 - Performance forecast: passes per 90 minutes in the next season.

  A. Model choice     Repeated cross-validation (5 folds x 3 repeats, split by player) on the development seasons
                      (forecasts of the 2023 and 2024 seasons). Two simple rules and three learning algorithms, each
                      with a small grid of settings, are compared on RMSE. The lowest RMSE wins.
  B. Backtest         The chosen model is refitted on the development seasons and forecasts the 2025 season once
                      (stats up to 2024 known in June 2025). Nothing is tuned on this test.
  C. Final model      Refitted on all seasons with a known outcome (2023-2025) and used for the June 2026 forecasts.

Inputs:  data/final/player_season_stats.csv, player.csv, scouting_case.csv, prediction_run.csv
Outputs: data/model/performance_model_table.csv, performance_model.json, performance_backtest_predictions.csv,
         case_performance_forecast.csv; outputs/performance_model_cv.csv, performance_importance.csv,
         performance_summary.json
"""
import json

import numpy as np
import pandas as pd

import config as C
from performance_model import FEATURES, PerformanceForecastModel, build_table, oof_predictions

stats = pd.read_csv(C.FINAL / "player_season_stats.csv")
players = pd.read_csv(C.FINAL / "player.csv")
table, carries_median = build_table(stats, players)
table["split"] = np.select([table.forecast_season <= 2024, table.forecast_season == C.BACKTEST_ORIGIN,
                            table.forecast_season == C.LIVE_ORIGIN], ["development", "backtest", "live"], "unused")
table = table[(table.split == "live") | ((table.split != "unused") & table.target_passes_p90.notna())].reset_index(drop=True)
table.to_csv(C.MODEL / "performance_model_table.csv", index=False)
dev, bt, live = (table[table.split == s].reset_index(drop=True) for s in ("development", "backtest", "live"))
print(f"[10] rows: development {len(dev)}, backtest {len(bt)}, live {len(live)}")


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


# ------------------------------------------------------------------ A. model choice
CANDIDATES = [("Linear regression (ridge)", {})] + \
    [("Random forest", {"min_samples_leaf": l, "max_features": f}) for l in (5, 10, 20) for f in (0.6, 1.0)] + \
    [("Gradient boosting", {"max_depth": d, "n_estimators": n}) for d in (2, 3) for n in (200, 400)]

y = dev.target_passes_p90
rows = []
# two rules that need no fitting
pos_mean = lambda tr, te: te.position.map(tr.groupby("position").target_passes_p90.mean()).values
for label, pred in [("No change (same as last season)", dev.passes_last.values)]:
    rows.append({"model": label, "family": "rule", "params": "", "cv_rmse": rmse(pred - y), "cv_mae": float(np.mean(np.abs(pred - y)))})
rows.append({"model": "Position average", "family": "rule", "params": "", "cv_rmse": rmse(pos_mean(dev, dev) - y),
             "cv_mae": float(np.mean(np.abs(pos_mean(dev, dev) - y)))})
for name, params in CANDIDATES:
    oof = oof_predictions(name, params, dev[FEATURES], y, dev.player_id.values)
    rows.append({"model": name, "family": "learned", "params": json.dumps(params), "cv_rmse": float(np.mean([rmse(o - y) for o in oof])),
                 "cv_mae": float(np.mean([np.mean(np.abs(o - y)) for o in oof]))})
    print(f"[10] CV {name} {params}: RMSE {rows[-1]['cv_rmse']:.2f}")
cv = pd.DataFrame(rows)
learned = cv[cv.family == "learned"]
best = learned.loc[learned.cv_rmse.idxmin()]
NAME, PARAMS = best.model, json.loads(best.params)
cv["chosen"] = (cv.model == NAME) & (cv.params == best.params)
print(f"[10] chosen: {NAME} {PARAMS}")

# backtest of every family at its best CV setting (for the comparison chart), chosen one flagged
bt_err = {}
for fam in learned.model.unique():
    b = learned[learned.model == fam].sort_values("cv_rmse").iloc[0]
    m = PerformanceForecastModel(fam, json.loads(b.params)).fit(dev)
    bt_err[(fam, b.params)] = m.predict_frame(bt).forecast.values - bt.target_passes_p90.values
cv["backtest_rmse"] = [rmse(bt_err[(m, p)]) if (m, p) in bt_err else np.nan for m, p in zip(cv.model, cv.params)]
cv.loc[cv.model == "No change (same as last season)", "backtest_rmse"] = rmse(bt.passes_last - bt.target_passes_p90)
cv.loc[cv.model == "Position average", "backtest_rmse"] = rmse(pos_mean(dev, bt) - bt.target_passes_p90)
cv.round(3).to_csv(C.OUTPUTS / "performance_model_cv.csv", index=False)

# ------------------------------------------------------------------ B. backtest
m_bt = PerformanceForecastModel(NAME, PARAMS).fit(dev)
pb = pd.concat([bt[["player_id", "position", "forecast_season", "age", "passes_last", "passes_before", "has_before",
                    "carries_last", "minutes_last", "target_passes_p90"]], m_bt.predict_frame(bt)], axis=1)
pb["error"] = pb.forecast - pb.target_passes_p90
pb["inside_pi80"] = ((pb.target_passes_p90 >= pb.lo80) & (pb.target_passes_p90 <= pb.hi80)).astype(int)
pb["inside_pi95"] = ((pb.target_passes_p90 >= pb.lo95) & (pb.target_passes_p90 <= pb.hi95)).astype(int)
pb.round(3).to_csv(C.MODEL / "performance_backtest_predictions.csv", index=False)
y_bt = pb.target_passes_p90
backtest = {
    "players": int(len(pb)),
    "rmse": rmse(pb.error), "mae": float(pb.error.abs().mean()),
    "r2": float(1 - np.sum(pb.error ** 2) / np.sum((y_bt - y_bt.mean()) ** 2)),
    "no_change_rmse": rmse(bt.passes_last - y_bt), "no_change_mae": float(np.mean(np.abs(bt.passes_last - y_bt))),
    "coverage_80": float(100 * pb.inside_pi80.mean()), "coverage_95": float(100 * pb.inside_pi95.mean()),
    "median_pi80_width": float((pb.hi80 - pb.lo80).median()),
}
print("[10] backtest:", json.dumps({k: round(v, 2) for k, v in backtest.items()}))

# ------------------------------------------------------------------ C. final model and forecasts
known = pd.concat([dev, bt], ignore_index=True)
m_final = PerformanceForecastModel(NAME, PARAMS).fit(known)
(C.MODEL / "performance_model.json").write_text(json.dumps(
    {"name": NAME, "params": PARAMS, "features": FEATURES, "carries_median_by_position": carries_median,
     "resid_quantiles_final": m_final.resid_q, "resid_quantiles_june2025": m_bt.resid_q,
     "fit_rows_final": int(m_final.n), "fit_rows_june2025": int(m_bt.n)}, indent=1))
imp = m_final.global_importance()
imp["share_pct"] = 100 * imp.mean_abs_shap / imp.mean_abs_shap.sum()
imp.round(4).to_csv(C.OUTPUTS / "performance_importance.csv", index=False)
print("[10] overall importance (mean |SHAP|, passes per 90):\n", imp.round(2).to_string(index=False))

# forecasts for the scouting cases: June 2025 cases use the June 2025 model, June 2026 cases the final model
cases = pd.read_csv(C.FINAL / "scouting_case.csv").merge(pd.read_csv(C.FINAL / "prediction_run.csv")[["case_id", "run_timestamp"]], on="case_id")
cases["origin_year"] = pd.to_datetime(cases.run_timestamp).dt.year
out = []
for year, model in ((C.BACKTEST_ORIGIN, m_bt), (C.LIVE_ORIGIN, m_final)):
    c = cases[cases.origin_year == year][["case_id", "player_id", "origin_year"]].merge(
        table[table.forecast_season == year], on=["player_id", "origin_year"], how="left")
    ok = c.passes_last.notna()
    pred = model.predict_frame(c[ok])
    pred.index = c[ok].index
    out.append(pd.concat([c, pred], axis=1))
cf = pd.concat(out, ignore_index=True)
cf = cf[["case_id", "player_id", "origin_year", "forecast_season", "position", "age", "passes_last", "passes_before", "has_before",
         "carries_last", "carries_missing", "minutes_last", "forecast", "lo80", "hi80", "lo95", "hi95", "extrapolated", "target_passes_p90"]]
cf = cf.rename(columns={"forecast": "perf_forecast", "lo80": "perf_lo80", "hi80": "perf_hi80", "lo95": "perf_lo95", "hi95": "perf_hi95",
                        "extrapolated": "perf_extrapolated", "target_passes_p90": "actual_passes_p90"})
cf.round(3).to_csv(C.MODEL / "case_performance_forecast.csv", index=False)
print(f"[10] case forecasts: {cf.perf_forecast.notna().sum()} of {len(cf)} cases have stats to forecast from")

summary = {"kpi": "Passes per 90 minutes in the next season", "chosen_model": NAME, "params": PARAMS,
           "development_rows": int(len(dev)), "cv_rmse": float(best.cv_rmse), "backtest": backtest,
           "final_model": {"rows": int(m_final.n), "age_min": m_final.age_min, "age_max": m_final.age_max},
           "live_forecasts": int(len(live))}
(C.OUTPUTS / "performance_summary.json").write_text(json.dumps(summary, indent=1))
