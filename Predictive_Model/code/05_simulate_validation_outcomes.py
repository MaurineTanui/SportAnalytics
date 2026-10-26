"""
05_simulate_validation_outcomes.py
Step 5. The Week 5 plan keeps 30 validation cases (the Summer 2026 window) aside, with their outcomes withheld.
The Week 5 files contain no outcomes for these cases, so this step simulates them, as a counterfactual:
"what would have happened 12 months later if the player had been signed at the asking price".
The result is written to data/withheld/validation_outcome.csv, separate from the 12 final tables, and is
never read by the metric step (04).

Simulation rules (explicit assumptions, see config.py and report Section 4):
  value after 12 months = latest pre-decision value x exp(age growth + 0.10 x performance z + noise(0, 0.15))
      age growth: under 23 = +0.10, 23-26 = +0.03, 27-29 = -0.05, 30 and over = -0.15
  performance rating   = 50 + 12 x performance z - 0.3 x (age - 27)^2 + noise(0, 6), limited to 0-100
  total cost           = asking price x 1.08 (agent fee) + half of the yearly wage (signing bonus)
"""
import numpy as np
import pandas as pd
import config as C

rng = np.random.default_rng(C.SEED + 5)
C.WITHHELD.mkdir(parents=True, exist_ok=True)
F = C.FINAL
case = pd.read_csv(F / "scouting_case.csv")
dc = pd.read_csv(F / "decision_context.csv", parse_dates=["decision_timestamp"])
td = pd.read_csv(F / "transfer_decision.csv")
player = pd.read_csv(F / "player.csv", parse_dates=["date_of_birth"])
st = pd.read_csv(F / "player_season_stats.csv", parse_dates=["available_from_timestamp"])
va = pd.read_csv(F / "valuation_history.csv", parse_dates=["available_from_timestamp"])
ct = pd.read_csv(F / "contract.csv", parse_dates=["start_date", "end_date"])
pos = player.set_index("player_id").position
dob = player.set_index("player_id").date_of_birth
st["position"] = st.player_id.map(pos)

val = case[case.data_split == "validation"].merge(dc, on="case_id").merge(td[["decision_id", "context_id"]], on="context_id")
rows = []
for i, r in enumerate(val.sort_values("case_id").itertuples(), start=1):
    t = r.decision_timestamp
    avail = st[st.available_from_timestamp < t]
    mine = avail[avail.player_id == r.player_id].sort_values("season_year")
    z = 0.0
    if len(mine):
        last = mine.iloc[-1]
        ref = avail[(avail.position == last.position) & (avail.season_year == last.season_year)]
        parts = [(last[c] - ref[c].mean()) / ref[c].std() for c in ("passes_p90", "carries_p90") if pd.notna(last[c])]
        z = float(np.mean(parts)) if parts else 0.0
    age = (t - dob[r.player_id]).days / 365.25
    growth = 0.10 if age < 23 else 0.03 if age < 27 else -0.05 if age < 30 else -0.15
    last_val = va[(va.player_id == r.player_id) & (va.available_from_timestamp < t)].sort_values("valuation_date") \
        .market_value_eur.iloc[-1]
    post = round(last_val * np.exp(growth + 0.10 * z + rng.normal(0, C.OUTCOME_VALUE_NOISE_SD)), 2)
    perf = float(np.clip(50 + 12 * z - 0.3 * (age - 27) ** 2 + rng.normal(0, 6), 0, 100))
    c = ct[(ct.player_id == r.player_id) & (ct.start_date <= t) & (ct.end_date >= t)]
    wage = float(c.annual_wage_eur.iloc[0]) if len(c) else float(ct.annual_wage_eur.median())
    acq = float(r.asking_price_eur)
    rows.append({"withheld_outcome_id": i, "decision_id": r.decision_id, "case_id": r.case_id,
                 "outcome_basis": "COUNTERFACTUAL_VALIDATION",
                 "outcome_timestamp": (t + pd.Timedelta(days=C.OUTCOME_HORIZON_DAYS)).strftime("%Y-%m-%d %H:%M:%S"),
                 "actual_acquisition_cost_eur": acq,
                 "total_recruitment_cost_eur": round(acq * (1 + C.AGENT_FEE_SHARE) + C.SIGNING_BONUS_SHARE_OF_WAGE * wage, 2),
                 "updated_fmv_post_eur": post, "future_performance_rating": round(perf, 2),
                 "realized_value_change_eur": round(post - acq, 2)})
out = pd.DataFrame(rows)
out.to_csv(C.WITHHELD / "validation_outcome.csv", index=False)
print(f"[05] {len(out)} withheld validation outcomes written to data/withheld/validation_outcome.csv")
