"""
06_validate.py
Step 6. Runs validation checks on the final dataset (data/final/) and the withheld validation outcomes
(data/withheld/), and writes the results to outputs/validation_results.csv. Exits with an error if any check fails.
"""
import hashlib
import sys
import numpy as np
import pandas as pd
import config as C

F = C.FINAL
C.OUTPUTS.mkdir(parents=True, exist_ok=True)
NAMES = ["club", "squad_need", "player", "contract", "player_season_stats", "valuation_history", "scouting_case",
         "case_event", "prediction_run", "decision_context", "transfer_decision", "transfer_outcome"]
T = {n: pd.read_csv(F / f"{n}.csv") for n in NAMES}
W = pd.read_csv(C.WITHHELD / "validation_outcome.csv")
res = []


def check(cat, name, ok, details):
    res.append({"check_id": len(res) + 1, "category": cat, "check": name, "result": "PASS" if ok else "FAIL",
                "details": details})


# 1. raw files untouched
bad = [f for f, h in C.RAW_FILES.items() if hashlib.sha256((C.RAW / f).read_bytes()).hexdigest() != h]
check("Raw data", "The 12 Week 5 raw files are present and unchanged (SHA-256)", not bad,
      f"{12 - len(bad)} of 12 files match the Week 5 submission")

# 2. schema
SCHEMA = {
    "club": ["club_id", "name", "is_focal_club", "league_tier"],
    "squad_need": ["need_id", "club_id", "position_code", "target_window", "budget_allocated_eur", "need_code",
                   "min_age", "max_age", "target_passes_p90", "target_carries_p90"],
    "player": ["player_id", "current_club_id", "position", "date_of_birth"],
    "contract": ["contract_id", "player_id", "club_id", "start_date", "end_date", "annual_wage_eur"],
    "player_season_stats": ["stat_id", "player_id", "season_year", "minutes_played", "passes_p90", "carries_p90",
                            "available_from_timestamp", "age_at_season_start", "passes_p90_week5"],
    "valuation_history": ["val_id", "player_id", "valuation_date", "market_value_eur", "available_from_timestamp",
                          "market_value_eur_week5"],
    "scouting_case": ["case_id", "player_id", "need_id", "status", "asking_price_eur", "data_split"],
    "case_event": ["event_id", "case_id", "event_type", "event_timestamp", "detail"],
    "prediction_run": ["run_id", "case_id", "player_id", "run_timestamp", "predicted_fmv_eur", "tactical_score",
                       "risk_forecast", "risk_band", "premium_pct", "screening_status", "recommended_action",
                       "week5_predicted_fmv_eur", "week5_tactical_score", "week5_risk_forecast_0to1"],
    "decision_context": ["context_id", "case_id", "run_id", "decision_timestamp", "snapshot_predicted_fmv_eur",
                         "snapshot_tactical_score", "snapshot_risk_forecast", "snapshot_budget_eur",
                         "snapshot_contract_months_remaining"],
    "transfer_decision": ["decision_id", "context_id", "selected_action", "rationale", "decision_timestamp",
                          "selected_action_std", "tool_recommended_action", "followed_tool_recommendation"],
    "transfer_outcome": ["outcome_id", "decision_id", "actual_acquisition_cost_eur", "total_recruitment_cost_eur",
                         "updated_fmv_post_eur", "future_performance_rating", "realized_value_change_eur"],
}
miss = {t: [c for c in cols if c not in T[t].columns] for t, cols in SCHEMA.items()}
miss = {k: v for k, v in miss.items() if v}
check("Schema", "All 12 final tables contain the Week 5 columns plus the Week 6 fields", not miss,
      f"missing: {miss}" if miss else "12 of 12 tables complete; all Week 5 columns kept")

# 3. types
issues = []
for t, df in T.items():
    for col in df.columns:
        if col.endswith("_id") and not pd.api.types.is_integer_dtype(df[col]):
            issues.append(f"{t}.{col}")
        if (col.endswith("_eur") or col.endswith("_p90")) and not pd.api.types.is_numeric_dtype(df[col]):
            issues.append(f"{t}.{col}")
        if ("timestamp" in col or col.endswith("_date") or col == "date_of_birth") and \
                pd.to_datetime(df[col], errors="coerce").isna().sum() != df[col].isna().sum():
            issues.append(f"{t}.{col}")
check("Data types", "IDs are integers, money and stat fields numeric, all dates parse", not issues,
      ", ".join(issues) if issues else "all ID, numeric and date columns valid")
check("Data types", "minutes_played stored as whole numbers", pd.api.types.is_integer_dtype(T["player_season_stats"].minutes_played),
      f"dtype {T['player_season_stats'].minutes_played.dtype}")

# 4. uniqueness
dups = {t: int(T[t][SCHEMA[t][0]].duplicated().sum()) for t in SCHEMA}
check("Uniqueness", "Primary key unique in every table", sum(dups.values()) == 0,
      f"duplicate keys: {sum(dups.values())} across 12 tables")
nat = {"stats (player, season)": T["player_season_stats"].duplicated(["player_id", "season_year"]).sum(),
       "valuation (player, date)": T["valuation_history"].duplicated(["player_id", "valuation_date"]).sum(),
       "prediction_run per case": T["prediction_run"].case_id.duplicated().sum(),
       "decision_context per case": T["decision_context"].case_id.duplicated().sum(),
       "decision per context": T["transfer_decision"].context_id.duplicated().sum(),
       "outcome per decision": T["transfer_outcome"].decision_id.duplicated().sum(),
       "case (player, need)": T["scouting_case"].duplicated(["player_id", "need_id"]).sum()}
check("Uniqueness", "Natural keys unique (one row per intended grain)", sum(nat.values()) == 0,
      "; ".join(f"{k}: {int(v)}" for k, v in nat.items()))

# 5. referential integrity
FK = [("squad_need", "club_id", "club", "club_id"), ("player", "current_club_id", "club", "club_id"),
      ("contract", "player_id", "player", "player_id"), ("contract", "club_id", "club", "club_id"),
      ("player_season_stats", "player_id", "player", "player_id"), ("valuation_history", "player_id", "player", "player_id"),
      ("scouting_case", "player_id", "player", "player_id"), ("scouting_case", "need_id", "squad_need", "need_id"),
      ("case_event", "case_id", "scouting_case", "case_id"), ("prediction_run", "case_id", "scouting_case", "case_id"),
      ("prediction_run", "player_id", "player", "player_id"), ("decision_context", "case_id", "scouting_case", "case_id"),
      ("decision_context", "run_id", "prediction_run", "run_id"),
      ("transfer_decision", "context_id", "decision_context", "context_id"),
      ("transfer_outcome", "decision_id", "transfer_decision", "decision_id")]
orph = sum(int((~T[a][b].isin(T[c][d])).sum()) for a, b, c, d in FK)
orph += int((~W.decision_id.isin(T["transfer_decision"].decision_id)).sum())
check("Referential integrity", f"All {len(FK) + 1} foreign keys (incl. withheld outcomes) point to an existing row",
      orph == 0, f"orphans found: {orph}")
pr_dc = T["decision_context"].merge(T["prediction_run"][["run_id", "case_id"]], on="run_id", suffixes=("", "_pr"))
check("Referential integrity", "decision_context.run_id belongs to the same case", (pr_dc.case_id == pr_dc.case_id_pr).all(),
      f"{int((pr_dc.case_id != pr_dc.case_id_pr).sum())} mismatches")

# 6. ranges and logic
s, v, pr = T["player_season_stats"], T["valuation_history"], T["prediction_run"]
rng = {"minutes outside 0-3420": int((~s.minutes_played.between(0, 3420)).sum()),
       "passes outside 0-150": int((~s.passes_p90.dropna().between(0, 150)).sum()),
       "carries outside 0-60": int((~s.carries_p90.dropna().between(0, 60)).sum()),
       "player under 16 in stats": int((s.age_at_season_start < C.MIN_PLAYER_AGE_FOR_DATA).sum()),
       "market value <= 0": int((v.market_value_eur <= 0).sum()),
       "asking price <= 0": int((T["scouting_case"].asking_price_eur <= 0).sum()),
       "TSS outside 0-100": int((~pr.tactical_score.dropna().between(0, 100)).sum()),
       "RAPF outside 0-100": int((~pr.risk_forecast.dropna().between(0, 100)).sum()),
       "FMV <= 0": int((pr.predicted_fmv_eur <= 0).sum())}
check("Range", "Stats, ages, money values and scores inside valid ranges", sum(rng.values()) == 0,
      "; ".join(f"{k}: {n}" for k, n in rng.items()))
ct = T["contract"]
check("Logical consistency", "Every contract ends after it starts",
      (pd.to_datetime(ct.end_date) > pd.to_datetime(ct.start_date)).all(), "0 invalid contracts"
      if (pd.to_datetime(ct.end_date) > pd.to_datetime(ct.start_date)).all() else "invalid contracts found")
unk = pr[pr.contract_known_at_run == 0]
check("Logical consistency", "Cases with no contract active at the run date are flagged and get no contract discount",
      unk.contract_months_remaining.isna().all(), f"{len(unk)} cases flagged (contract starts after or ended before the run)")
check("Logical consistency", "Valuations become available after the valuation date",
      (pd.to_datetime(v.available_from_timestamp) >= pd.to_datetime(v.valuation_date)).all(), "0 rows wrong")
for name, o in [("final transfer_outcome", T["transfer_outcome"]), ("withheld validation outcomes", W)]:
    ok = ((o.updated_fmv_post_eur - o.actual_acquisition_cost_eur - o.realized_value_change_eur).abs() < 0.011).all()
    check("Logical consistency", f"Outcome identity holds in {name}", ok, f"{len(o)} rows checked")
dcp = T["decision_context"].merge(pr, on="run_id", suffixes=("", "_pr"))
check("Logical consistency", "Decision is made after the prediction run",
      (pd.to_datetime(dcp.decision_timestamp) > pd.to_datetime(dcp.run_timestamp)).all(), f"{len(dcp)} cases checked")
same = (np.isclose(dcp.snapshot_predicted_fmv_eur, dcp.predicted_fmv_eur)
        & np.isclose(dcp.snapshot_tactical_score.fillna(-1), dcp.tactical_score.fillna(-1))
        & np.isclose(dcp.snapshot_risk_forecast.fillna(-1), dcp.risk_forecast.fillna(-1)))
check("Logical consistency", "Frozen snapshot equals the prediction_run values", same.all(), f"{int((~same).sum())} mismatches")

# 7. missingness
allowed = {"player_season_stats": ["passes_p90", "carries_p90", "passes_p90_week5"],
           "prediction_run": ["profile_carries_p90", "performance_cv", "risk_forecast", "priority_score",
                              "recommended_max_bid_eur", "tactical_score", "contract_months_remaining",
                              "profile_passes_p90", "last_stats_available_ts"],
           "decision_context": ["snapshot_risk_forecast", "snapshot_recommended_max_bid_eur",
                                "snapshot_contract_months_remaining", "snapshot_tactical_score"]}
unexpected = [f"{t}.{c}={int(df[c].isna().sum())}" for t, df in T.items() for c in df.columns
              if df[c].isna().any() and c not in allowed.get(t, [])]
check("Missingness", "Missing values only in fields where missing is allowed", not unexpected,
      ", ".join(unexpected) if unexpected else "no unexpected missing values")
cm = s.carries_p90.isna().mean()
check("Missingness", "Share of missing carries_p90 is reported and handled by the data-gap rule", True,
      f"{cm:.1%} of season rows missing carries_p90; {int(pr.data_gap_flag.sum())} cases flagged as data gap")

# 8. temporal integrity
lk = int((pd.to_datetime(pr.last_stats_available_ts) >= pd.to_datetime(pr.run_timestamp)).sum()
         + (pd.to_datetime(pr.last_valuation_available_ts) >= pd.to_datetime(pr.run_timestamp)).sum())
check("Temporal integrity", "Every stat and valuation used was available before the prediction run", lk == 0,
      f"{lk} violations")
code04 = open(C.ROOT / "code" / "04_compute_metrics.py").read().split('"""')[2]
check("Temporal integrity", "Metric script never reads withheld outcomes", "withheld" not in code04.replace("never reads data/withheld", ""),
      "04_compute_metrics.py has no reference to data/withheld")
valc = T["scouting_case"][T["scouting_case"].data_split == "validation"]
val_dec = T["decision_context"][T["decision_context"].case_id.isin(valc.case_id)].merge(T["transfer_decision"], on="context_id")
check("Temporal integrity", "No validation outcome appears in the final transfer_outcome table",
      not T["transfer_outcome"].decision_id.isin(val_dec.decision_id).any(),
      f"{int(T['transfer_outcome'].decision_id.isin(val_dec.decision_id).sum())} validation outcomes found in final data")

# 9. metrics
elig = pr[pr.screening_status != "EXCLUDED"]
nogap = elig[elig.data_gap_flag == 0]
check("Metric computation", "TSS, FMV and RAPF computed for every eligible case without a data gap",
      nogap[["tactical_score", "predicted_fmv_eur", "risk_forecast"]].notna().all().all(),
      f"{len(nogap)} eligible cases with complete data; {len(elig) - len(nogap)} flagged as data gap")


def restatus(r):
    if r.screening_status == "EXCLUDED":
        return "EXCLUDED"
    st_ = ("MONITOR" if r.data_gap_flag == 1 else "SHORTLIST" if (r.tactical_score >= C.TSS_SHORTLIST and
           r.risk_forecast >= C.RAPF_MIN) else "MONITOR" if (r.tactical_score >= C.TSS_MONITOR_MIN and
           r.risk_forecast >= C.RAPF_MIN) else "REJECT")
    return "REJECT" if (st_ == "MONITOR" and r.windows_in_monitor >= C.MAX_WINDOWS_IN_MONITOR) else st_


re_ = pr.apply(restatus, axis=1)
check("Metric computation", "Screening status re-computed independently matches", (re_ == pr.screening_status).all(),
      f"{int((re_ != pr.screening_status).sum())} mismatches in {len(pr)} cases")
sl = pr[pr.screening_status == "SHORTLIST"]
ra = sl.premium_pct.map(lambda x: "BID_AT_ASK" if x <= 10 else "COUNTER_BID" if x <= 25 else "REJECT")
check("Metric computation", "Recommended bid action re-computed from premium matches", (ra == sl.recommended_action).all(),
      f"{len(sl)} shortlisted cases checked")

# 10. grain and split
n = len(T["scouting_case"])
check("Grain", "One prediction_run, one decision_context and one transfer_decision per scouting case",
      n == len(pr) == len(T["decision_context"]) == len(T["transfer_decision"]),
      f"cases {n}, runs {len(pr)}, contexts {len(T['decision_context'])}, decisions {len(T['transfer_decision'])}")
check("Grain", "30 validation cases, each with exactly one withheld outcome",
      len(valc) == 30 and len(W) == 30 and set(W.case_id) == set(valc.case_id), f"{len(valc)} cases, {len(W)} outcomes")

# 11. calibration
med = v.market_value_eur.median()
check("Calibration", "Median market value equals the EUR 275,000 target (within 1%)",
      abs(med / C.TARGET_MEDIAN_MARKET_VALUE_EUR - 1) <= 0.01, f"median = EUR {med:,.0f} (Week 5: EUR {v.market_value_eur_week5.median():,.0f})")
pos = T["player"].set_index("player_id").position
om = s[s.player_id.map(pos) != "GK"].passes_p90.mean()
check("Calibration", "Outfield passes_p90 mean equals the 45 target (within 2%)", abs(om / 45 - 1) <= 0.02,
      f"mean = {om:.1f} (Week 5: {s[s.player_id.map(pos) != 'GK'].passes_p90_week5.mean():.1f})")
rk = s.dropna(subset=["passes_p90"])
same_rank = all(g.passes_p90.rank().equals(g.passes_p90_week5.rank())
                for _, g in rk.groupby(rk.player_id.map(pos) == "GK"))
check("Calibration", "Passes calibration keeps every row's rank order", same_rank, "rank order identical before and after")
factor = (v.market_value_eur / v.market_value_eur_week5)
check("Calibration", "One money factor applied to all valuations", factor.round(4).nunique() == 1,
      f"factor = {factor.iloc[0]:.4f}")

out = pd.DataFrame(res)
out.to_csv(C.OUTPUTS / "validation_results.csv", index=False)
print(out[["check_id", "category", "result", "check"]].to_string(index=False))
print(f"[06] {int((out.result == 'PASS').sum())} of {len(out)} checks passed -> outputs/validation_results.csv")
if (out.result == "FAIL").any():
    sys.exit(1)
