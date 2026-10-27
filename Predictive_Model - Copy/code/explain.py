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


# ====================================================================== automatic what-if summaries
def _fmt_inputs(inputs, base_inputs):
    parts = []
    for k, v in inputs.items():
        b = base_inputs.get(k) if base_inputs else None
        parts.append(f"{k}: **{v}**" + (f" (baseline {b})" if b is not None and b != v else ""))
    return parts


def value_summary(player, inputs, base_inputs, now, base, eur, extras):
    """Markdown summary of one value what-if run. now/base: forecast dicts (base None for a new player)."""
    changed = [k for k in inputs if base_inputs and inputs[k] != base_inputs.get(k)]
    lines = [f"**Scenario for {player}** — KPI 1, market value at the next yearly valuation.",
             "**Inputs:** " + "; ".join(_fmt_inputs(inputs, base_inputs)) + "."]
    lines.append(f"**Updated prediction:** {eur(now['forecast'])} (80% range {eur(now['lo80'])} to {eur(now['hi80'])}); "
                 f"chance the player is worth at least the asking price: {100 * now['prob']:.0f}%.")
    if base is None:
        lines.append("**Change from baseline:** no recorded baseline exists for a new player; the forecast stands alone.")
    elif not changed:
        lines.append(f"**Change from baseline:** none. These are the recorded values, so this *is* the baseline "
                     f"({eur(base['forecast'])}).")
    else:
        d = now["forecast"] - base["forecast"]
        dp = 100 * (now["forecast"] / base["forecast"] - 1)
        if abs(dp) < 0.5:
            lines.append(f"**Change from baseline:** changing {', '.join(changed)} leaves the forecast essentially unchanged "
                         f"(baseline {eur(base['forecast'])}).")
        else:
            lines.append(f"**Change from baseline:** {eur(base['forecast'])} → {eur(now['forecast'])}, "
                         f"{'+' if d >= 0 else '-'}{eur(abs(d))} ({dp:+.1f}%), after changing {', '.join(changed)}. "
                         f"Chance the fee holds moves from {100 * base['prob']:.0f}% to {100 * now['prob']:.0f}%.")
    imp = [f"Under the Week 6 bid bands the asking price reads **{extras['action']}** against this forecast."]
    if extras.get("action_changed"):
        imp.append(f"That differs from the baseline reading ({extras['base_action']}).")
    if extras.get("range_split"):
        imp.append("The reading changes inside the 80% range, so the bid is a judgement call.")
    lines.append("**Implications:** " + " ".join(imp))
    lim = ["This is a model-based scenario: it shows what the model predicts for these inputs, not what would happen if "
           "the club or the player changed them."]
    if now["extrapolated"]:
        lim.append("The age is outside the ages the model was trained on, so the forecast is an extrapolation and should not be relied on.")
    lim.append(f"Even with correct inputs, one player in five is expected to fall outside the 80% range. "
               "The model uses only value, age and last year's change; injuries, contract and form are not in it. The data is synthetic.")
    lines.append("**Limitations:** " + " ".join(lim))
    return "\n\n".join(lines)


def perf_summary(player, inputs, base_inputs, now, base, target, extras):
    changed = [k for k in inputs if base_inputs and inputs[k] != base_inputs.get(k)]
    lines = [f"**Scenario for {player}** — KPI 2, passes per 90 minutes next season.",
             "**Inputs:** " + "; ".join(_fmt_inputs(inputs, base_inputs)) + "."]
    lines.append(f"**Updated prediction:** {now['forecast']:.1f} passes per 90 (80% range {now['lo80']:.1f} to {now['hi80']:.1f}). "
                 f"Chance of reaching the need's target style of {target:.0f}: {100 * extras['p_target']:.0f}%.")
    if base is None:
        lines.append("**Change from baseline:** no recorded baseline exists for a new player; the forecast stands alone.")
    elif not changed:
        lines.append(f"**Change from baseline:** none. These are the recorded values, so this *is* the baseline ({base['forecast']:.1f}).")
    else:
        d = now["forecast"] - base["forecast"]
        if abs(d) < 0.05:
            lines.append(f"**Change from baseline:** changing {', '.join(changed)} leaves the forecast unchanged ({base['forecast']:.1f}). "
                         "Tree models predict in steps, so small input changes often move nothing.")
        else:
            lines.append(f"**Change from baseline:** {base['forecast']:.1f} → {now['forecast']:.1f} passes per 90 ({d:+.1f}), "
                         f"after changing {', '.join(changed)}. Chance of reaching the target moves from "
                         f"{100 * extras['base_p_target']:.0f}% to {100 * extras['p_target']:.0f}%.")
    gap = now["forecast"] - target
    imp = (f"The forecast is {abs(gap):.1f} passes per 90 {'above' if gap >= 0 else 'below'} the target style for this squad need, "
           + ("so on passing volume the player is expected to fit the role." if gap >= 0 else
              "so on passing volume the player is expected to fall short of the role unless his usage changes."))
    lines.append("**Implications:** " + imp + " The tactical fit score (TSS) uses the same statistic, so a shortfall here is a risk "
                 "to the fit that screening measured on past seasons.")
    lim = ["This is a model-based scenario, not the effect of an intervention: coaching or a new role could change a player's "
           "passing in ways the model has never seen."]
    if now["extrapolated"]:
        lim.append("The age is outside the training ages, so the forecast is an extrapolation.")
    lim.append(f"The typical error in the backtest was about {extras['mae']:.0f} passes per 90, and one forecast in five falls outside "
               "the 80% range. Passing volume also depends on the new team's style, which the model does not know. The data is synthetic.")
    lines.append("**Limitations:** " + " ".join(lim))
    return "\n\n".join(lines)
