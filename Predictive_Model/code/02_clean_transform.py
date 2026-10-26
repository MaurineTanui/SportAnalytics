"""
02_clean_transform.py
Step 2. Cleaning and calibration of the Week 5 data. Reads data/raw/, fixes the problems found in the
submitted files, calibrates the data to the two Week 5 targets, and writes data/intermediate/.
Every rule writes the number of rows it changed to outputs/cleaning_log.csv.

Calibration (Week 5 targets):
  * money: every EUR field is multiplied by ONE factor so the median market value becomes EUR 275,000
    (one factor keeps all money fields consistent with each other; the Week 5 value is kept in *_week5)
  * passes_p90: rank-preserving mapping onto a log-normal shape with mean 45 for outfield players
    (and 27 for goalkeepers); every player keeps the same rank, only the scale changes
"""
from statistics import NormalDist
import numpy as np
import pandas as pd
import config as C

C.INTERMEDIATE.mkdir(parents=True, exist_ok=True)
C.OUTPUTS.mkdir(parents=True, exist_ok=True)
R = {n: pd.read_csv(C.RAW / f"{n}.csv") for n in [f[:-4] for f in C.RAW_FILES]}
log = []


def note(table, step, rule, n):
    log.append({"table": table, "step": step, "rule": rule, "rows_affected": int(n)})


# ------------------------------------------------------------------ player
player = R["player"].copy()
dob = pd.to_datetime(player.set_index("player_id").date_of_birth)

# ------------------------------------------------------------------ player_season_stats
st = R["player_season_stats"].copy()
frac = (st.minutes_played % 1 != 0)
st["minutes_played"] = st.minutes_played.round().astype(int)
note("player_season_stats", "invalid values", "minutes_played had decimals (e.g. 2935.29); rounded to whole minutes",
     frac.sum())
age = (pd.to_datetime(st.season_year.astype(str) + "-" + C.SEASON_START_MONTH_DAY) - st.player_id.map(dob)).dt.days / 365.25
young = age < C.MIN_PLAYER_AGE_FOR_DATA
st = st[~young].copy()
note("player_season_stats", "invalid values", f"rows for players younger than {C.MIN_PLAYER_AGE_FOR_DATA} at season "
     "start removed (youngest was 13)", young.sum())
st["age_at_season_start"] = age[~young].round(2)
bad_p = st.passes_p90 > C.MAX_VALID_PASSES_P90
st.loc[bad_p, "passes_p90"] = np.nan
note("player_season_stats", "invalid values", f"passes_p90 above {C.MAX_VALID_PASSES_P90} (impossible) set to missing",
     bad_p.sum())
bad_c = st.carries_p90 > C.MAX_VALID_CARRIES_P90
st.loc[bad_c, "carries_p90"] = np.nan
note("player_season_stats", "invalid values", f"carries_p90 above {C.MAX_VALID_CARRIES_P90} (impossible) set to missing",
     bad_c.sum())

# rank-preserving calibration of passes_p90
st["passes_p90_week5"] = R["player_season_stats"].set_index("stat_id").passes_p90.reindex(st.stat_id).values
is_gk = st.player_id.map(player.set_index("player_id").position).eq("GK")
nd = NormalDist()
for mask, mean, cv in [(~is_gk, C.TARGET_OUTFIELD_PASSES_MEAN, C.OUTFIELD_PASSES_CV),
                       (is_gk, C.TARGET_GK_PASSES_MEAN, C.GK_PASSES_CV)]:
    m = mask & st.passes_p90.notna()
    q = (st.loc[m, "passes_p90"].rank(method="average") - 0.5) / m.sum()
    sigma = np.sqrt(np.log(1 + cv ** 2))
    mu = np.log(mean) - sigma ** 2 / 2
    st.loc[m, "passes_p90"] = np.round(np.exp(mu + sigma * q.map(nd.inv_cdf)), 2)
note("player_season_stats", "calibration", "passes_p90 mapped by rank to a log-normal shape (outfield mean 45, "
     "goalkeeper mean 27); Week 5 value kept in passes_p90_week5", st.passes_p90.notna().sum())
st["carries_missing_flag"] = st.carries_p90.isna().astype(int)
st["passes_missing_flag"] = st.passes_p90.isna().astype(int)
note("player_season_stats", "derived variable", "age_at_season_start, carries_missing_flag, passes_missing_flag added",
     len(st))

# ------------------------------------------------------------------ valuation_history
va = R["valuation_history"].copy()
vage = (pd.to_datetime(va.valuation_date) - va.player_id.map(dob)).dt.days / 365.25
vy = vage < C.MIN_PLAYER_AGE_FOR_DATA
va = va[~vy].copy()
note("valuation_history", "invalid values", f"valuations for players younger than {C.MIN_PLAYER_AGE_FOR_DATA} removed",
     vy.sum())
MONEY_FACTOR = C.TARGET_MEDIAN_MARKET_VALUE_EUR / va.market_value_eur.median()
va["market_value_eur_week5"] = va.market_value_eur
va["market_value_eur"] = (va.market_value_eur * MONEY_FACTOR).round(2)
note("valuation_history", "calibration", f"market_value_eur multiplied by {MONEY_FACTOR:.4f} so the median is EUR 275,000; "
     "Week 5 value kept in market_value_eur_week5", len(va))

# ------------------------------------------------------------------ contract
ct = R["contract"].copy()
ct["annual_wage_eur"] = (ct.annual_wage_eur * MONEY_FACTOR).round(2)
note("contract", "calibration", f"annual_wage_eur multiplied by the same money factor ({MONEY_FACTOR:.4f})", len(ct))

# ------------------------------------------------------------------ Week 5 tool outputs and outcomes (money scaled)
pr = R["prediction_run"].copy()
zero = pr.predicted_fmv_eur <= 0
pr["predicted_fmv_eur"] = (pr.predicted_fmv_eur * MONEY_FACTOR).round(2)
note("prediction_run", "calibration", "Week 5 predicted_fmv_eur multiplied by the money factor", len(pr))
note("prediction_run", "invalid values", "Week 5 predicted_fmv_eur equal to 0 flagged; replaced by the recalculated "
     "FMV in step 4", zero.sum())
dc = R["decision_context"].copy()
dc["snapshot_predicted_fmv_eur"] = (dc.snapshot_predicted_fmv_eur * MONEY_FACTOR).round(2)
note("decision_context", "calibration", "Week 5 snapshot_predicted_fmv_eur multiplied by the money factor", len(dc))
to = R["transfer_outcome"].copy()
for col in ["actual_acquisition_cost_eur", "total_recruitment_cost_eur", "updated_fmv_post_eur"]:
    to[col] = (to[col] * MONEY_FACTOR).round(2)
to["realized_value_change_eur"] = (to.updated_fmv_post_eur - to.actual_acquisition_cost_eur).round(2)
note("transfer_outcome", "calibration", "money fields multiplied by the money factor; realized_value_change_eur "
     "recalculated so the identity holds exactly", len(to))

# ------------------------------------------------------------------ write
out = {"club": R["club"], "player": player, "squad_need": R["squad_need"], "scouting_case": R["scouting_case"],
       "player_season_stats": st, "valuation_history": va, "contract": ct, "prediction_run": pr,
       "decision_context": dc, "transfer_decision": R["transfer_decision"], "case_event": R["case_event"],
       "transfer_outcome": to}
for n, df in out.items():
    df.to_csv(C.INTERMEDIATE / f"{n}.csv", index=False)
pd.DataFrame([{"money_factor": MONEY_FACTOR}]).to_csv(C.OUTPUTS / "calibration_factor.csv", index=False)
pd.DataFrame(log).to_csv(C.OUTPUTS / "cleaning_log.csv", index=False)
print(f"[02] money factor = {MONEY_FACTOR:.4f}")
print(pd.DataFrame(log)[["table", "step", "rows_affected"]].to_string(index=False))
