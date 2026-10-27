# Progress Report — Week 10

**Project:** Football Recruitment Decision-Support Tool
**Decision-maker:** Head of Recruitment
**Demo video:** _[add link here]_

## Done so far

| Week | Work |
|---|---|
| 5 | Designed the synthetic recruitment database (12 tables), calibrated to StatsBomb and Transfermarkt patterns |
| 6 | Screening and bid rules: tactical fit score (TSS), performance score (RAPF), fair market value |
| 7–8 | Cleaning and validation pipeline (steps 01–06), withheld outcomes for testing |
| 9 | KPI 1 value forecast model, backtest, first version of the app with a what-if screen |
| 10 | KPI 2 model, SHAP explanations, interpretation text, automatic what-if summaries, setup docs |

## KPIs and models

**KPI 1 — Market value at the next yearly valuation.** Tells the club whether a fee will hold its value (financial sustainability, maximum bid). Predictors: value now, age, last year's change in value. Model: log-scale regression with an age curve, picked from 11 candidates by cross-validation. Backtest on 195 players: typical error 12% (vs. 23% for the Week 6 rule); 80% range held 80% of actual values.

**KPI 2 — Passes per 90 minutes next season.** Tells the club whether the player will still fit the target playing style after signing (operational performance). Predictors: passes per 90 in the last two seasons, carries per 90, minutes, age, position. Model: gradient boosting, picked from 11 settings of ridge regression, random forest and gradient boosting by cross-validation. Backtest on 187 players: typical error 9.3 passes/90 (vs. 12.3 assuming no change), R² 0.44; 80% range held 79%.

A third KPI (minutes played next season) was tested and dropped: no predictor was related to it.

## App features added this week

- **KPIs and models** screen: definition, relevance, predictors and algorithm rationale for each KPI
- **Forecast accuracy** for both models
- **Explain a forecast**: overall importance (all players) shown separately from one player's contributions; SHAP dependence plot; plain-language reading of direction and size; non-causal warning
- **What-if analysis** for both KPIs: baseline vs. scenario, contributions for both, automatic summary (inputs, prediction, change, implications, limitations), downloadable scenario log

## Limitations / next steps

- Data is synthetic, so effects are illustrative; models should be refitted on real data.
- Passing volume depends on the new team's style, which the model doesn't see.
- Next: fold KPI 2 into the screening rules (replace the rule-based RAPF), user testing with the decision-maker.
