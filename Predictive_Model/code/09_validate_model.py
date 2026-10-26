"""
09_validate_model.py
Step 9. Validates the modelling data (data/model/) and the model outputs, and writes the results to
outputs/model_validation_results.csv. Exits with an error if any check fails.
"""
import json
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
import config as C
from value_model import ValueForecastModel, interval, point, price_for_confidence, prob_at_least

M = C.MODEL
mt = pd.read_csv(M / "value_model_table.csv", parse_dates=["forecast_origin", "value_available_ts", "stats_available_ts",
                                                           "value_date", "target_value_date"])
bp = pd.read_csv(M / "backtest_predictions.csv")
pf = pd.read_csv(M / "player_forecast.csv")
cf = pd.read_csv(M / "case_forecast.csv")
saved = json.loads((M / "value_model.json").read_text())
cvt = pd.read_csv(C.OUTPUTS / "model_selection_cv.csv")
va = pd.read_csv(C.FINAL / "valuation_history.csv", parse_dates=["valuation_date"])
player = pd.read_csv(C.FINAL / "player.csv", parse_dates=["date_of_birth"])
run = pd.read_csv(C.FINAL / "prediction_run.csv", parse_dates=["run_timestamp"])
case = pd.read_csv(C.FINAL / "scouting_case.csv")
res = []


def check(cat, name, ok, details):
    res.append({"check_id": len(res) + 1, "category": cat, "check": name, "result": "PASS" if ok else "FAIL",
                "details": details})


dev, bt, live = (mt[mt.split == s] for s in ("development", "backtest", "live"))
known = mt[mt.split != "live"]

# 1. grain and row counts
dup = int(mt.duplicated(["player_id", "origin_year"]).sum())
check("Grain", "One row per player per forecast origin", dup == 0, f"{len(mt)} rows, {dup} duplicates")
per_year = va.groupby(va.valuation_date.dt.year).size()
ok = all(int((mt.origin_year == y).sum()) == int(per_year[y]) for y in mt.origin_year.unique())
check("Grain", "Every player with a valuation in the origin year has a row", ok,
      ", ".join(f"{y}: {int((mt.origin_year == y).sum())}" for y in sorted(mt.origin_year.unique())))
check("Referential integrity", "Every player_id exists in the player table", mt.player_id.isin(player.player_id).all(),
      f"{mt.player_id.nunique()} players")

# 2. time rules
late_val = int((mt.value_available_ts >= mt.forecast_origin).sum())
late_stat = int((mt.stats_available_ts.notna() & (mt.stats_available_ts >= mt.forecast_origin)).sum())
check("Temporal integrity", "Every predictor was available before the forecast origin", late_val + late_stat == 0,
      f"valuations too late: {late_val}; stats too late: {late_stat}")
check("Temporal integrity", "Every target is dated after the forecast origin", (mt.target_value_date > mt.forecast_origin).all(),
      f"earliest gap: {(mt.target_value_date - mt.forecast_origin).dt.days.min()} days")
runs = set(run.run_timestamp.dt.strftime("%Y-%m-%d"))
origins = set(mt[mt.origin_year.isin([C.BACKTEST_ORIGIN, C.LIVE_ORIGIN])].forecast_origin.dt.strftime("%Y-%m-%d"))
check("Temporal integrity", "Origins 2025 and 2026 equal the dates of the two prediction runs", runs == origins, f"{sorted(origins)}")
check("Temporal integrity", "The backtest model only learned from outcomes known before June 2025",
      dev.target_value_date.max() < bt.forecast_origin.min(), f"latest development target: {dev.target_value_date.max().date()}")
check("Temporal integrity", "The final model only learned from outcomes known before June 2026",
      known.target_value_date.max() < live.forecast_origin.min(), f"latest target used: {known.target_value_date.max().date()}")
uses = [f for f in ("07_build_model_table.py", "08_train_evaluate_model.py", "value_model.py")
        if "WITHHELD" in "\n".join(l for l in (C.ROOT / "code" / f).read_text().splitlines() if "never reads" not in l)]
uses += ["app.py"] if "withheld" in (C.ROOT / "app" / "app.py").read_text(encoding="utf-8").lower() else []
check("Temporal integrity", "Model code and application never read the withheld validation outcomes", not uses,
      "no reference to data/withheld in steps 07-08, value_model.py or app.py" if not uses else f"found in {uses}")

# 3. predictors and target recomputed independently
v = va.assign(year=va.valuation_date.dt.year).set_index(["player_id", "year"]).market_value_eur
now = np.array([v[(p, y)] for p, y in zip(mt.player_id, mt.origin_year)])
prev = np.array([v.get((p, y - 1), np.nan) for p, y in zip(mt.player_id, mt.origin_year)])
nxt = np.array([v.get((p, y + 1), np.nan) for p, y in zip(mt.player_id, mt.origin_year)])
same = np.allclose(now, mt.market_value_eur) and np.allclose(prev, mt.prev_value_eur, equal_nan=True) \
    and np.allclose(nxt, mt.target_value_eur, equal_nan=True)
check("Predictors", "Current, previous and next value match valuation_history", same, f"{len(mt)} rows compared")
age = (mt.forecast_origin - mt.player_id.map(player.set_index("player_id").date_of_birth)).dt.days / 365.25
check("Predictors", "age_at_origin matches date_of_birth", np.allclose(age, mt.age_at_origin, atol=1e-3),
      f"ages {mt.age_at_origin.min():.1f} to {mt.age_at_origin.max():.1f}")
mom = np.where(mt.has_history == 1, np.log(mt.market_value_eur / mt.prev_value_eur), 0.0)
check("Predictors", "momentum_log is log(value / previous value), and 0 without history",
      np.allclose(mom, mt.momentum_log, atol=1e-5) and (mt.has_history == mt.prev_value_eur.notna()).all(),
      f"{int(mt.has_history.sum())} rows with history, {int((mt.has_history == 0).sum())} without")
check("Target", "log_change equals log(target value / current value)",
      np.allclose(np.log(known.target_value_eur / known.market_value_eur), known.log_change, atol=1e-5), f"{len(known)} rows")
check("Missingness", "Target present for all development and backtest rows, absent for live rows",
      known.target_value_eur.notna().all() and live.target_value_eur.isna().all(),
      f"development {len(dev)}, backtest {len(bt)}, live {len(live)}")
check("Missingness", "No missing value in a predictor the model uses", mt[["market_value_eur", "age_at_origin", "momentum_log"]].notna().all().all(),
      "market_value_eur, age_at_origin, momentum_log complete")
rng_ok = (mt.market_value_eur > 0).all() and mt.age_at_origin.between(16, 36).all() and known.log_change.abs().max() < 3
check("Range", "Values positive, ages 16 to 36, yearly change within a plausible range", rng_ok,
      f"largest yearly change: x{np.exp(known.log_change.abs().max()):.1f}")

# 4. the fitted model
refit = ValueForecastModel(saved["knot_step"], saved["use_momentum"], saved["use_level"]).fit(
    known.market_value_eur, known.age_at_origin, known.prev_value_eur, known.target_value_eur)
model = ValueForecastModel.from_dict(saved)
check("Model", "Refitting from the model table reproduces the saved coefficients",
      np.allclose(refit.coef, model.coef, atol=1e-9) and np.allclose(refit.cov, model.cov, atol=1e-9),
      f"{model.p} coefficients, {model.n} rows")
X = refit._design(known.market_value_eur.values, known.age_at_origin.values, known.prev_value_eur.values)
w = 1 / np.where(known.has_history == 1, model.sigma2_history, model.sigma2_no_history)
ind = sm.WLS(known.log_change.values, X, weights=w).fit()
check("Model", "Coefficients match an independent statsmodels weighted least squares fit",
      np.allclose(ind.params, model.coef, atol=1e-4), f"largest difference: {np.abs(ind.params - model.coef).max():.2e}")
chosen = cvt[cvt.chosen == 1].iloc[0]
curve_fam = cvt[cvt.family == "Age curve"]
check("Model", "The chosen model has the lowest cross-validation error of its family", chosen.rmse_log == curve_fam.rmse_log.min(),
      f"{chosen.model}: CV RMSE {chosen.rmse_log:.4f}")
base = cvt[cvt.family == "Baseline"]
check("Model", "The chosen model beats both baselines in cross-validation and in the backtest",
      (chosen.rmse_log < base.rmse_log.min()) and (chosen.backtest_rmse_log < base.backtest_rmse_log.min())
      and (chosen.backtest_median_abs_pct_error < base.backtest_median_abs_pct_error.min()),
      f"backtest median error {chosen.backtest_median_abs_pct_error:.1f}% vs {base.backtest_median_abs_pct_error.min():.1f}% (best baseline)")

# 5. uncertainty
for name, df in (("backtest", bp), ("live", pf), ("case", cf)):
    ok = ((df.pi95_low_eur < df.pi80_low_eur) & (df.pi80_low_eur < df.forecast_eur) & (df.forecast_eur < df.pi80_high_eur)
          & (df.pi80_high_eur < df.pi95_high_eur)).all()
    check("Uncertainty", f"Intervals are ordered (95% low < 80% low < forecast < 80% high < 95% high): {name} forecasts", ok, f"{len(df)} rows")
check("Uncertainty", "The confidence interval is always narrower than the prediction interval",
      ((pf.ci95_low_eur > pf.pi95_low_eur) & (pf.ci95_high_eur < pf.pi95_high_eur)).all(), f"{len(pf)} live forecasts")
c80, c95 = 100 * bp.inside_pi80.mean(), 100 * bp.inside_pi95.mean()
tol = 100 * C.COVERAGE_TOLERANCE
check("Uncertainty", f"Backtest coverage of the 80% and 95% intervals is within {tol:.0f} points of the level",
      abs(c80 - 80) <= tol and abs(c95 - 95) <= tol, f"80% interval covered {c80:.1f}%; 95% interval covered {c95:.1f}%")
seen = bp[bp.extrapolated == 0]
z = seen.log_error / seen.se_log
check("Uncertainty", "Standardised backtest errors have a spread close to 1 (ages seen in training)", 0.7 <= z.std() <= 1.3,
      f"standard deviation {z.std():.2f}, {len(seen)} players")
pr = model.predict(pf.market_value_eur, pf.age_at_origin, pf.prev_value_eur)
ok = all(np.allclose(prob_at_least(pr, price_for_confidence(pr, c)), c, atol=1e-9) for c in C.CONFIDENCE_PRICE_LEVELS)
p_lo, p_hi = prob_at_least(pr, point(pr) * 0.8), prob_at_least(pr, point(pr) * 1.2)
check("Uncertainty", "Probabilities are between 0 and 1, fall as the price rises, and match the price-for-confidence values",
      ok and (p_lo > p_hi).all() and (p_hi > 0).all() and (p_lo < 1).all() and np.allclose(prob_at_least(pr, point(pr)), 0.5),
      "probability at the central forecast is 50%")

# 6. outputs used by the application
lo, hi = interval(pr, C.MAIN_INTERVAL)
check("Application data", "The saved model reproduces the stored live forecasts (same code as the what-if screen)",
      np.allclose(point(pr), pf.forecast_eur, atol=0.01) and np.allclose(lo, pf.pi80_low_eur, atol=0.01)
      and np.allclose(hi, pf.pi80_high_eur, atol=0.01), f"{len(pf)} forecasts compared")
check("Application data", "One live forecast per player with a 2026 valuation", len(pf) == len(live) and pf.player_id.is_unique, f"{len(pf)} players")
check("Application data", "One forecast per scouting case", len(cf) == len(case) and set(cf.case_id) == set(case.case_id), f"{len(cf)} cases")
c25 = cf[cf.origin_year == C.BACKTEST_ORIGIN].merge(bp[["player_id", "forecast_eur"]], on="player_id", suffixes=("", "_bt"))
check("Application data", "Summer 2025 cases use the model that existed in June 2025 (no hindsight)",
      len(c25) == 30 and np.allclose(c25.forecast_eur, c25.forecast_eur_bt), f"{len(c25)} cases match the backtest forecasts")
check("Application data", "Forecasts outside the ages seen in training are flagged",
      (pf.extrapolated == ((pf.age_at_origin < model.age_min) | (pf.age_at_origin > model.age_max)).astype(int)).all(),
      f"live: {int(pf.extrapolated.sum())} flagged; backtest: {int(bp.extrapolated.sum())} flagged "
      f"(training ages {model.age_min:.1f} to {model.age_max:.1f})")

out = pd.DataFrame(res)
out.to_csv(C.OUTPUTS / "model_validation_results.csv", index=False)
print(out[["check_id", "category", "result", "check"]].to_string(index=False))
print(f"[09] {int((out.result == 'PASS').sum())} of {len(out)} checks passed -> outputs/model_validation_results.csv")
if (out.result == "FAIL").any():
    sys.exit(1)
