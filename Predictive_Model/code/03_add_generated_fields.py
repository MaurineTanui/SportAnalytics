"""
03_add_generated_fields.py
Step 3. Adds the fields that the Week 6 decision logic needs but the Week 5 data did not contain.
Each added field is either derived by a stated rule or generated with a fixed seed:
  squad_need    : need_code, min_age, max_age                     (manually constructed, config.NEED_SETTINGS)
                  target_passes_p90, target_carries_p90            (derived: 75th percentile of MID players
                                                                     in seasons available before the run)
  scouting_case : asking_price_eur                                 (generated: latest public value x random premium)
                  data_split                                        (derived: Summer 2026 window = validation)
Input/output: data/intermediate/
"""
import numpy as np
import pandas as pd
import config as C

rng = np.random.default_rng(C.SEED)
I = C.INTERMEDIATE
# start from the Week 5 columns so the step gives the same result if it is run again
need = pd.read_csv(C.RAW / "squad_need.csv")
case = pd.read_csv(C.RAW / "scouting_case.csv")
pr = pd.read_csv(I / "prediction_run.csv", parse_dates=["run_timestamp"])
st = pd.read_csv(I / "player_season_stats.csv", parse_dates=["available_from_timestamp"])
va = pd.read_csv(I / "valuation_history.csv", parse_dates=["available_from_timestamp"])
ct = pd.read_csv(I / "contract.csv", parse_dates=["start_date", "end_date"])
pos = pd.read_csv(I / "player.csv").set_index("player_id").position

run_of_need = case.merge(pr[["case_id", "run_timestamp"]], on="case_id").groupby("need_id").run_timestamp.min()

# ------------------------------------------------------------------ squad_need
rows = []
for _, nd in need.iterrows():
    code, amin, amax = C.NEED_SETTINGS[nd.need_id]
    run = run_of_need[nd.need_id]
    ref = st[(st.available_from_timestamp < run) & (st.player_id.map(pos) == nd.position_code)]
    rows.append({"need_code": code, "min_age": amin, "max_age": amax,
                 "target_passes_p90": round(ref.passes_p90.quantile(C.TARGET_PROFILE_PERCENTILE), 1),
                 "target_carries_p90": round(ref.carries_p90.quantile(C.TARGET_PROFILE_PERCENTILE), 1)})
need = pd.concat([need, pd.DataFrame(rows)], axis=1)

# ------------------------------------------------------------------ scouting_case
asks, splits = [], []
win = need.set_index("need_id").target_window
for _, cs in case.iterrows():
    run = run_of_need[cs.need_id]
    pv = va[(va.player_id == cs.player_id) & (va.available_from_timestamp < run)].sort_values("valuation_date")
    last = pv.market_value_eur.iloc[-1]
    c = ct[(ct.player_id == cs.player_id) & (ct.start_date <= run) & (ct.end_date >= run)]
    short = len(c) > 0 and (c.end_date.iloc[0] - run).days / 30.44 < 12
    ask = last * np.exp(rng.normal(C.ASK_PREMIUM_LOG_MEAN, C.ASK_PREMIUM_LOG_SD)) * (C.ASK_SHORT_CONTRACT_FACTOR if short else 1)
    asks.append(float(max(5_000, round(ask, -3))))
    splits.append("validation" if win[cs.need_id] == C.VALIDATION_WINDOW else "development")
case["asking_price_eur"] = asks
case["data_split"] = splits

need.to_csv(I / "squad_need.csv", index=False)
case.to_csv(I / "scouting_case.csv", index=False)
print("[03] squad_need:\n", need.to_string(index=False))
print("[03] asking prices added for", len(case), "cases; split:", case.data_split.value_counts().to_dict())
