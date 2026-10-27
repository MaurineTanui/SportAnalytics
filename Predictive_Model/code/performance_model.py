"""
performance_model.py
KPI 2: the performance forecast. It predicts a player's PASSES PER 90 MINUTES in the NEXT season, the statistic the
squad need's target style is defined on. It is used by step 10 (training and testing) and by the application, so the
tool always predicts with exactly the same code that was tested.

Model: a random forest regressor (scikit-learn) on six predictors that are all known at the summer prediction run:
  passes_last     passes per 90 in the latest completed season
  passes_before   passes per 90 in the season before that (= passes_last when there is no earlier season;
                  has_before = 0 tells the model so)
  carries_last    carries per 90 in the latest season (missing values -> median for the position)
  minutes_last    minutes played in the latest season
  age             age on 1 August of the forecast season
  position        GK / DEF / MID / FWD (one-hot)

Uncertainty: the 80% and 95% prediction ranges come from the out-of-fold residuals of repeated cross-validation on the
model's own training rows (an empirical, split-conformal style interval), so they are checked on data the model had
not seen. The backtest (step 10) tests whether they hold.

Explanations: SHAP values (TreeExplainer). For any one prediction the SHAP values of the predictors add up exactly to
(prediction - average prediction over the training rows). Overall importance = mean |SHAP| over the training rows.

The model is refitted from its training table when the app starts (a few hundred rows, under a second) with a fixed
seed, so no binary model file is stored and the result does not depend on the scikit-learn version used to save it.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

POSITIONS = ["DEF", "FWD", "GK", "MID"]
FEATURES = ["passes_last", "passes_before", "has_before", "carries_last", "minutes_last", "age"] + [f"pos_{p}" for p in POSITIONS]
# predictors as the user sees them: one-hot position and the history flag are grouped with the column they describe,
# so SHAP values are added within a group (allowed, because SHAP values are additive)
GROUPS = {
    "Passes per 90, last season": ["passes_last"],
    "Passes per 90, season before": ["passes_before", "has_before"],
    "Carries per 90, last season": ["carries_last"],
    "Minutes played, last season": ["minutes_last"],
    "Age": ["age"],
    "Position": [f"pos_{p}" for p in POSITIONS],
}
SEED = 20251003
QUANTILE_GRID = sorted({round(k / 100, 2) for k in range(1, 100)} | {0.025, 0.975})


def build_table(stats, players):
    """One row per player and forecast season: predictors from the seasons before, target = passes per 90 that season.
    The forecast season S is predicted at the June run of year S, when stats up to season S-1 are known."""
    s = stats.merge(players[["player_id", "position"]], on="player_id").sort_values(["player_id", "season_year"])
    s = s[s.passes_p90.notna()]
    rows = []
    for pid, g in s.groupby("player_id"):
        g = g.set_index("season_year")
        for last in g.index:
            target_season = last + 1
            before = g.loc[last - 1] if (last - 1) in g.index else None
            r = g.loc[last]
            rows.append({
                "player_id": pid, "position": r.position, "forecast_season": target_season, "origin_year": target_season,
                "passes_last": r.passes_p90, "passes_before": before.passes_p90 if before is not None else r.passes_p90,
                "has_before": int(before is not None), "carries_last_raw": r.carries_p90, "minutes_last": r.minutes_played,
                "age": r.age_at_season_start + 1,
                "target_passes_p90": g.loc[target_season].passes_p90 if target_season in g.index else np.nan,
            })
    t = pd.DataFrame(rows)
    med = t.groupby("position").carries_last_raw.median()
    t["carries_missing"] = t.carries_last_raw.isna().astype(int)
    t["carries_last"] = t.carries_last_raw.fillna(t.position.map(med))
    for p in POSITIONS:
        t[f"pos_{p}"] = (t.position == p).astype(int)
    return t, med.to_dict()


def make_estimator(name, params=None):
    params = params or {}
    if name == "Random forest":
        return RandomForestRegressor(n_estimators=params.get("n_estimators", 500), min_samples_leaf=params.get("min_samples_leaf", 10),
                                     max_features=params.get("max_features", 0.6), random_state=SEED, n_jobs=-1)
    if name == "Gradient boosting":
        return GradientBoostingRegressor(n_estimators=params.get("n_estimators", 300), learning_rate=params.get("learning_rate", 0.03),
                                         max_depth=params.get("max_depth", 2), subsample=0.8, random_state=SEED)
    if name == "Linear regression (ridge)":
        return RidgeCV(alphas=np.logspace(-2, 3, 30))
    raise ValueError(name)


def oof_predictions(name, params, X, y, groups, folds=5, repeats=3):
    """Out-of-fold predictions from repeated cross-validation split by player. Returns an array (repeats x n)."""
    out = np.zeros((repeats, len(y)))
    for r in range(repeats):
        rng = np.random.RandomState(SEED + r)
        u = np.unique(groups)
        shuffled = pd.Series(groups).map(dict(zip(u, rng.permutation(len(u))))).values
        for tr, te in GroupKFold(folds).split(X, y, shuffled):
            out[r, te] = make_estimator(name, params).fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
    return out


class PerformanceForecastModel:
    def __init__(self, name="Random forest", params=None):
        self.name, self.params = name, params or {}

    def fit(self, table, resid_q=None):
        """resid_q: residual quantiles saved by step 10. Given -> the slow cross-validation is skipped (used by the app)."""
        X, y = table[FEATURES], table.target_passes_p90
        self.est = make_estimator(self.name, self.params).fit(X, y)
        if resid_q is None:
            oof = oof_predictions(self.name, self.params, X, y, table.player_id.values)
            resid = (y.values[None, :] - oof).ravel()
            resid_q = {q: float(np.quantile(resid, q)) for q in QUANTILE_GRID}
        self.resid_q = {float(k): v for k, v in resid_q.items()}
        self.n = len(table)
        self.age_min, self.age_max = float(table.age.min()), float(table.age.max())
        self.background = X.copy()
        self._explainer = None
        return self

    def predict_frame(self, X):
        """X: DataFrame with FEATURES. Returns a DataFrame with the forecast and its ranges."""
        p = self.est.predict(X[FEATURES])
        out = pd.DataFrame({"forecast": p,
                            "lo80": np.maximum(0, p + self.resid_q[0.10]), "hi80": p + self.resid_q[0.90],
                            "lo95": np.maximum(0, p + self.resid_q[0.025]), "hi95": p + self.resid_q[0.975]})
        out["extrapolated"] = ((X.age < self.age_min) | (X.age > self.age_max)).astype(int).values
        return out

    def prob_at_least(self, forecast, level):
        """Chance that the actual value is at least `level`, from the cross-validated residual distribution."""
        q = np.array(sorted(self.resid_q))
        r = np.array([self.resid_q[k] for k in q])
        # P(forecast + residual >= level) = 1 - F(level - forecast); F interpolated on the residual quantile grid
        return float(1 - np.interp(level - forecast, r, q, left=0.0, right=1.0))

    # ------------------------------------------------------------------ explanations
    @property
    def explainer(self):
        if self._explainer is None:
            import shap
            if self.name == "Linear regression (ridge)":
                self._explainer = shap.LinearExplainer(self.est, self.background)
            else:
                self._explainer = shap.TreeExplainer(self.est)
        return self._explainer

    def shap_values(self, X):
        """SHAP values grouped to the user-facing predictors. Returns (DataFrame n x groups, base value)."""
        sv = self.explainer.shap_values(X[FEATURES])
        raw = pd.DataFrame(sv, columns=FEATURES, index=X.index)
        grouped = pd.DataFrame({g: raw[cols].sum(axis=1) for g, cols in GROUPS.items()})
        base = float(np.atleast_1d(self.explainer.expected_value)[0])
        return grouped, base

    def global_importance(self):
        sv, _ = self.shap_values(self.background)
        imp = sv.abs().mean().sort_values(ascending=False)
        return pd.DataFrame({"predictor": imp.index, "mean_abs_shap": imp.values})


def row(passes_last, passes_before, has_before, carries_last, minutes_last, age, position):
    """One input row for the model from user-facing values."""
    d = {"passes_last": passes_last, "passes_before": passes_before if has_before else passes_last, "has_before": int(has_before),
         "carries_last": carries_last, "minutes_last": minutes_last, "age": age}
    d.update({f"pos_{p}": int(position == p) for p in POSITIONS})
    return pd.DataFrame([d])
