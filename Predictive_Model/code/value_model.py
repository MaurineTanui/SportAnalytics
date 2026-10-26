"""
value_model.py
The value forecast model. It is used by step 08 (training and testing), step 09 (validation) and by the
application (app/app.py), so the tool always predicts with exactly the same code that was tested.

Model (a linear regression on the log scale with an age curve):
    log(value at next valuation / value now) = f(age) + g x momentum + error
  f(age)    a piecewise-linear age curve: one coefficient per knot (every `knot_step` years), joined by straight
            lines. Outside the ages seen in training the curve is held at its last fitted value and the
            prediction is flagged as an extrapolation.
  momentum  log(value now / value one year ago); 0 when the player has no earlier valuation.
  error     normal on the log scale, so values are log-normal (right-skewed), as market values are. The residual
            spread is estimated separately for players with and without an earlier valuation, because the forecast
            is less precise when there is no history (weighted least squares, three re-weighting rounds).

Uncertainty:
  confidence interval  = uncertainty about the AVERAGE player with these inputs (coefficient uncertainty only)
  prediction interval  = uncertainty about THIS player's actual value (coefficient uncertainty + residual spread)
Both use the Student t distribution with n - p degrees of freedom.
"""
import numpy as np
from scipy import stats


class ValueForecastModel:
    def __init__(self, knot_step=1, use_momentum=True, use_level=False):
        self.knot_step, self.use_momentum, self.use_level = knot_step, use_momentum, use_level

    # ------------------------------------------------------------------ design matrix
    def _design(self, value_now, age, prev_value):
        a = np.clip(age, self.age_min, self.age_max)
        X = [np.clip(1 - np.abs(a[:, None] - self.knots[None, :]) / self.knot_step, 0, None)]   # age curve
        if self.use_momentum:
            mom = np.where(np.isnan(prev_value), 0.0, np.log(value_now / np.where(np.isnan(prev_value), value_now, prev_value)))
            X.append(mom[:, None])
        if self.use_level:
            X.append((np.log(value_now) - self.level_centre)[:, None])
        return np.hstack(X)

    @staticmethod
    def _arr(x, n=None):
        if x is None:
            return np.full(n, np.nan)
        return np.atleast_1d(np.asarray(x, dtype=float))

    # ------------------------------------------------------------------ fit
    def fit(self, value_now, age, prev_value, value_next):
        value_now, age, value_next = self._arr(value_now), self._arr(age), self._arr(value_next)
        prev_value = self._arr(prev_value, len(age))
        self.age_min, self.age_max = float(age.min()), float(age.max())
        self.knots = np.arange(np.floor(self.age_min), np.ceil(self.age_max) + self.knot_step, self.knot_step, dtype=float)
        self.level_centre = float(np.log(value_now).mean())
        X, y = self._design(value_now, age, prev_value), np.log(value_next / value_now)
        hist = ~np.isnan(prev_value)
        self.n, self.p = X.shape
        self.df = self.n - self.p
        w = np.ones(self.n)
        for _ in range(3):                                           # round 1 is ordinary least squares
            coef, *_ = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)
            resid = y - X @ coef
            pooled = float(resid @ resid / self.df)
            s2 = {g: float(np.mean(resid[hist == g] ** 2) * self.n / self.df) if (hist == g).sum() > self.p else pooled
                  for g in (True, False)}                            # residual variance (log scale) per group
            w = 1 / np.where(hist, s2[True], s2[False])
        self.coef, *_ = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)   # final weighted fit
        self.sigma2_history, self.sigma2_no_history = s2[True], s2[False]
        self.cov = np.linalg.inv(X.T @ (X * w[:, None]))             # covariance of the coefficients
        self.names = [f"age_{k:g}" for k in self.knots] + ["momentum"] * self.use_momentum + ["value_level"] * self.use_level
        return self

    # ------------------------------------------------------------------ predict
    def predict(self, value_now, age, prev_value=None):
        """Returns a dict of arrays on the log scale; use the helper functions below to read it in EUR."""
        value_now, age = self._arr(value_now), self._arr(age)
        prev_value = self._arr(prev_value, len(age))
        X = self._design(value_now, age, prev_value)
        change = X @ self.coef
        var_mean = np.einsum("ij,jk,ik->i", X, self.cov, X)
        sigma2 = np.where(np.isnan(prev_value), self.sigma2_no_history, self.sigma2_history)
        return {"log_now": np.log(value_now), "log_change": change, "log_pred": np.log(value_now) + change,
                "se_mean": np.sqrt(var_mean), "se_pred": np.sqrt(var_mean + sigma2), "df": self.df,
                "extrapolated": ((age < self.age_min) | (age > self.age_max)).astype(int)}

    def age_curve(self, ages):
        """Expected 12-month change (log scale) by age for a player with flat momentum, with its standard error."""
        ages = self._arr(ages)
        level = np.full(len(ages), np.exp(self.level_centre))
        pr = self.predict(level, ages, level)                        # previous value = current value -> momentum 0
        return pr["log_change"], pr["se_mean"]

    # ------------------------------------------------------------------ save / load (plain numbers, no pickle)
    def to_dict(self):
        return {"knot_step": self.knot_step, "use_momentum": self.use_momentum, "use_level": self.use_level,
                "age_min": self.age_min, "age_max": self.age_max, "knots": self.knots.tolist(),
                "level_centre": self.level_centre, "names": self.names, "coef": self.coef.tolist(),
                "cov": self.cov.tolist(), "sigma2_history": self.sigma2_history,
                "sigma2_no_history": self.sigma2_no_history, "n": self.n, "p": self.p, "df": self.df}

    @classmethod
    def from_dict(cls, d):
        m = cls(d["knot_step"], d["use_momentum"], d["use_level"])
        for k in ("age_min", "age_max", "level_centre", "names", "sigma2_history", "sigma2_no_history", "n", "p", "df"):
            setattr(m, k, d[k])
        m.knots, m.coef, m.cov = np.array(d["knots"]), np.array(d["coef"]), np.array(d["cov"])
        return m


# ---------------------------------------------------------------------- reading a prediction in EUR
def point(pred):
    """Central forecast in EUR (the median of the predictive distribution)."""
    return np.exp(pred["log_pred"])


def interval(pred, level, kind="prediction"):
    """(low, high) in EUR. kind = 'prediction' for this player's value, 'confidence' for the average player."""
    se = pred["se_pred"] if kind == "prediction" else pred["se_mean"]
    t = stats.t.ppf(0.5 + level / 2, pred["df"])
    return np.exp(pred["log_pred"] - t * se), np.exp(pred["log_pred"] + t * se)


def prob_at_least(pred, price):
    """Probability that the value at the next valuation is at least `price` (EUR)."""
    return stats.t.sf((np.log(price) - pred["log_pred"]) / pred["se_pred"], pred["df"])


def price_for_confidence(pred, confidence):
    """The price the future value will reach or exceed with the given probability (a lower quantile)."""
    return np.exp(pred["log_pred"] + stats.t.ppf(1 - confidence, pred["df"]) * pred["se_pred"])
