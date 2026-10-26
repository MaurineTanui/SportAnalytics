"""
04_compute_metrics.py
Step 4. Computes the Week 6 metrics and recommendations for every scouting case and writes the 12 final
tables to data/final/. Only data available before the prediction run is used
(available_from_timestamp < run_timestamp). The Week 5 values are kept next to the new ones (week5_* columns)
so the change from Week 5 to Week 6 logic stays visible.

Recomputed (Week 6 definitions):
  TSS  = 100 x exp(-mean squared z-distance to the need's target profile)          (0-100)
  FMV  = latest public value x (1 + age adjustment) x contract factor               (EUR, +/- 12.5% range)
  RAPF = (50 + 25 x average season z-score) x age factor - 40 x max(0, CV - 0.10)   (0-100)
This script never reads data/withheld/, so validation outcomes cannot leak into the metrics.
"""
import numpy as np
import pandas as pd
import config as C

C.FINAL.mkdir(parents=True, exist_ok=True)
I = C.INTERMEDIATE
club = pd.read_csv(I / "club.csv")
player = pd.read_csv(I / "player.csv", parse_dates=["date_of_birth"])
st = pd.read_csv(I / "player_season_stats.csv", parse_dates=["available_from_timestamp"])
va = pd.read_csv(I / "valuation_history.csv", parse_dates=["available_from_timestamp"])
ct = pd.read_csv(I / "contract.csv", parse_dates=["start_date", "end_date"])
need = pd.read_csv(I / "squad_need.csv")
case = pd.read_csv(I / "scouting_case.csv")
pr5 = pd.read_csv(I / "prediction_run.csv", parse_dates=["run_timestamp"])
dc5 = pd.read_csv(I / "decision_context.csv")
td5 = pd.read_csv(I / "transfer_decision.csv")
ev5 = pd.read_csv(I / "case_event.csv")
to = pd.read_csv(I / "transfer_outcome.csv")

pos_of = player.set_index("player_id").position
dob_of = player.set_index("player_id").date_of_birth
st["position"] = st.player_id.map(pos_of)


def lookup(table, x):
    for limit, v in table:
        if x <= limit:
            return v
    return table[-1][1]


def contract_factor(months):
    for limit, f in C.FMV_CONTRACT_FACTOR:
        if months < limit:
            return f
    return 1.0


def age_factor(age):
    return 0.95 if age < 21 else 1.00 if age < 29 else 0.97 if age < 30 else 0.93 if age < 31 else 0.88


cases = case.merge(pr5[["run_id", "case_id", "run_timestamp"]], on="case_id").merge(need, on="need_id")
monitor_history, rows = {}, []
for run_ts, grp in cases.sort_values(["run_timestamp", "case_id"]).groupby("run_timestamp", sort=True):
    avail = st[st.available_from_timestamp < run_ts]
    ref = avail.groupby(["position", "season_year"]).agg(mp=("passes_p90", "mean"), sp=("passes_p90", "std"),
                                                         mc=("carries_p90", "mean"), sc=("carries_p90", "std"))
    v_av = va[va.available_from_timestamp < run_ts]
    on_monitor = []
    for _, cs in grp.iterrows():
        p = cs.player_id
        pos_sd = avail[avail.position == cs.position_code][["passes_p90", "carries_p90"]].std()
        age = (run_ts - dob_of[p]).days / 365.25
        ps = avail[avail.player_id == p].sort_values("season_year").tail(3)
        w = ps.minutes_played
        mp_ = np.average(ps.passes_p90.dropna(), weights=w[ps.passes_p90.notna()]) if ps.passes_p90.notna().any() else np.nan
        mc_ = np.average(ps.carries_p90.dropna(), weights=w[ps.carries_p90.notna()]) if ps.carries_p90.notna().any() else np.nan
        parts = []
        if not np.isnan(mp_):
            parts.append(((mp_ - cs.target_passes_p90) / pos_sd.passes_p90) ** 2)
        if not np.isnan(mc_):
            parts.append(((mc_ - cs.target_carries_p90) / pos_sd.carries_p90) ** 2)
        tss = 100 * np.exp(-np.mean(parts)) if parts else np.nan
        full = ps.dropna(subset=["passes_p90", "carries_p90"])
        latest_ok = len(ps) > 0 and pd.notna(ps.iloc[-1].passes_p90) and pd.notna(ps.iloc[-1].carries_p90)
        data_gap = (not latest_ok) or (len(full) < C.MIN_SEASONS_FOR_RAPF)
        cv = rapf = np.nan
        if not data_gap:
            z = [((r.passes_p90 - ref.loc[(r.position, r.season_year)].mp) / ref.loc[(r.position, r.season_year)].sp
                  + (r.carries_p90 - ref.loc[(r.position, r.season_year)].mc) / ref.loc[(r.position, r.season_year)].sc) / 2
                 for _, r in full.iterrows()]
            perf = 50 + 25 * np.average(z, weights=full.minutes_played)
            cv = full.passes_p90.std(ddof=1) / full.passes_p90.mean()
            rapf = float(np.clip(perf * age_factor(age) - 40 * max(0.0, cv - 0.10), 0, 100))
        risk = ("Unknown" if np.isnan(cv) else "High" if (cv > 0.30 or age >= 31)
                else "Low" if (cv < 0.15 and age <= 28) else "Medium")
        pv = v_av[v_av.player_id == p].sort_values("valuation_date")
        last_val, last_val_ts = pv.market_value_eur.iloc[-1], pv.available_from_timestamp.iloc[-1]
        c = ct[(ct.player_id == p) & (ct.start_date <= run_ts) & (ct.end_date >= run_ts)]
        contract_known = len(c) > 0
        months = int((c.end_date.iloc[0] - run_ts).days // 30.44) if contract_known else None
        cfac = contract_factor(months) if contract_known else 1.0
        fmv = last_val * (1 + lookup(C.FMV_AGE_ADJUSTMENT, age)) * cfac
        ask = cs.asking_price_eur
        prem = (ask - fmv) / fmv
        wim = monitor_history.get(p, 0)
        in_age = cs.min_age <= age < cs.max_age + 1
        if not in_age or ask > cs.budget_allocated_eur or pos_of[p] != cs.position_code:
            status, reason = "EXCLUDED", "age outside need range" if not in_age else "asking price above budget"
        elif data_gap:
            status, reason = "MONITOR", "data gap: carries or passes missing, or fewer than 2 full seasons"
        elif tss >= C.TSS_SHORTLIST and rapf >= C.RAPF_MIN:
            status, reason = "SHORTLIST", f"TSS {tss:.1f} >= {C.TSS_SHORTLIST} and RAPF {rapf:.1f} >= {C.RAPF_MIN}"
        elif tss >= C.TSS_MONITOR_MIN and rapf >= C.RAPF_MIN:
            status, reason = "MONITOR", f"borderline fit: TSS {tss:.1f} between {C.TSS_MONITOR_MIN} and 69"
        else:
            status = "REJECT"
            reason = f"RAPF {rapf:.1f} < {C.RAPF_MIN}" if rapf < C.RAPF_MIN else f"TSS {tss:.1f} < {C.TSS_MONITOR_MIN}"
        if status == "MONITOR" and wim >= C.MAX_WINDOWS_IN_MONITOR:
            status, reason = "REJECT", "on MONITOR for 2 windows without improvement"
        action, max_bid, prio, ovr = None, np.nan, np.nan, 0
        if status == "SHORTLIST":
            wt, wr, wv = C.PRIORITY_WEIGHTS
            prio = wt * tss + wr * rapf + wv * min(100.0, max(0.0, 100 - 4 * prem * 100))
            if prem <= C.PREMIUM_BID_MAX:
                action, max_bid = "BID_AT_ASK", ask
            elif prem <= C.PREMIUM_COUNTER_MAX:
                action, max_bid = "COUNTER_BID", min(fmv * C.COUNTER_BID_CAP, cs.budget_allocated_eur)
            else:
                action = "REJECT"
            ovr = int(risk == "High" and action != "REJECT")
        else:
            action = "MONITOR" if status == "MONITOR" else "REJECT"
        if status == "MONITOR":
            on_monitor.append(p)
        rows.append({
            "run_id": cs.run_id, "case_id": cs.case_id, "player_id": p,
            "run_timestamp": run_ts.strftime("%Y-%m-%d %H:%M:%S"), "model_version": "dss-rules-v1.0 (Week 6)",
            "age_at_run": round(age, 2), "seasons_used": len(ps),
            "last_stats_available_ts": ps.available_from_timestamp.max().strftime("%Y-%m-%d %H:%M:%S") if len(ps) else None,
            "last_valuation_available_ts": last_val_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "profile_passes_p90": round(mp_, 2), "profile_carries_p90": None if np.isnan(mc_) else round(mc_, 2),
            "tactical_score": None if np.isnan(tss) else round(tss, 2), "latest_market_value_eur": last_val,
            "contract_known_at_run": int(contract_known), "contract_months_remaining": months,
            "predicted_fmv_eur": round(fmv, 2), "fmv_low_eur": round(fmv * (1 - C.FMV_RANGE), 2),
            "fmv_high_eur": round(fmv * (1 + C.FMV_RANGE), 2),
            "performance_cv": None if np.isnan(cv) else round(cv, 4),
            "risk_forecast": None if np.isnan(rapf) else round(rapf, 2), "risk_band": risk,
            "data_gap_flag": int(data_gap), "asking_price_eur": ask, "premium_pct": round(prem * 100, 2),
            "windows_in_monitor": wim, "screening_status": status, "status_reason": reason,
            "priority_score": None if np.isnan(prio) else round(prio, 2), "recommended_action": action,
            "recommended_max_bid_eur": None if np.isnan(max_bid) else round(max_bid, -3),
            "override_required_flag": ovr,
        })
    for p in on_monitor:
        monitor_history[p] = monitor_history.get(p, 0) + 1

pr = pd.DataFrame(rows).sort_values("run_id")
w5 = pr5.set_index("run_id")
pr["week5_predicted_fmv_eur"] = pr.run_id.map(w5.predicted_fmv_eur)
pr["week5_tactical_score"] = pr.run_id.map(w5.tactical_score)
pr["week5_risk_forecast_0to1"] = pr.run_id.map(w5.risk_forecast)

# ------------------------------------------------------------------ decision_context (frozen snapshot at decision time)
p_idx = pr.set_index("run_id")
dc = dc5[["context_id", "case_id", "run_id", "decision_timestamp"]].copy()
snap = {"snapshot_predicted_fmv_eur": "predicted_fmv_eur", "snapshot_tactical_score": "tactical_score",
        "snapshot_risk_forecast": "risk_forecast", "snapshot_risk_band": "risk_band",
        "snapshot_asking_price_eur": "asking_price_eur", "snapshot_premium_pct": "premium_pct",
        "snapshot_contract_months_remaining": "contract_months_remaining",
        "snapshot_screening_status": "screening_status", "snapshot_recommended_action": "recommended_action",
        "snapshot_recommended_max_bid_eur": "recommended_max_bid_eur"}
for k, v in snap.items():
    dc[k] = dc.run_id.map(p_idx[v])
dc["snapshot_budget_eur"] = dc5.snapshot_budget_eur
dc["snapshot_budget_left_eur"] = dc5.snapshot_budget_eur   # all decisions in a window share one timestamp
dc["week5_snapshot_predicted_fmv_eur"] = dc5.snapshot_predicted_fmv_eur
dc["week5_snapshot_tactical_score"] = dc5.snapshot_tactical_score
dc["week5_snapshot_risk_forecast_0to1"] = dc5.snapshot_risk_forecast
dc["week5_snapshot_contract_months_remaining"] = dc5.snapshot_contract_months_remaining
for k, v in [("rule_tss_shortlist", C.TSS_SHORTLIST), ("rule_tss_monitor_min", C.TSS_MONITOR_MIN),
             ("rule_rapf_min", C.RAPF_MIN), ("rule_premium_bid_max", C.PREMIUM_BID_MAX),
             ("rule_premium_counter_max", C.PREMIUM_COUNTER_MAX)]:
    dc[k] = v

# ------------------------------------------------------------------ transfer_decision (Week 5 decisions kept)
td = td5.copy()
td["selected_action_std"] = td.selected_action.str.upper()
rec = td.context_id.map(dc.set_index("context_id").snapshot_recommended_action)
td["tool_recommended_action"] = rec
td["followed_tool_recommendation"] = (((td.selected_action_std == "BID") & rec.isin(["BID_AT_ASK", "COUNTER_BID"]))
                                      | (td.selected_action_std == rec)).astype(int)

# ------------------------------------------------------------------ case_event (Week 5 events + derived events)
ev = ev5.copy()
ev["detail"] = "Week 5 record"
add = []
for _, r in pr.iterrows():
    add.append({"case_id": r.case_id, "event_type": "PREDICTION_RUN", "event_timestamp": r.run_timestamp,
                "detail": f"derived from prediction_run {r.run_id}: {r.screening_status}"})
for _, r in td.merge(dc[["context_id", "case_id"]], on="context_id").iterrows():
    add.append({"case_id": r.case_id, "event_type": "DECISION_RECORDED", "event_timestamp": r.decision_timestamp,
                "detail": f"derived from transfer_decision {r.decision_id}: {r.selected_action}"})
add = pd.DataFrame(add)
add.insert(0, "event_id", range(ev.event_id.max() + 1, ev.event_id.max() + 1 + len(add)))
ev = pd.concat([ev, add], ignore_index=True)

# ------------------------------------------------------------------ write the 12 final tables
player["date_of_birth"] = player.date_of_birth.dt.strftime("%Y-%m-%d")
final = {"club": club, "squad_need": need, "player": player, "contract": pd.read_csv(I / "contract.csv"),
         "player_season_stats": pd.read_csv(I / "player_season_stats.csv"),
         "valuation_history": pd.read_csv(I / "valuation_history.csv"), "scouting_case": case,
         "case_event": ev, "prediction_run": pr, "decision_context": dc, "transfer_decision": td,
         "transfer_outcome": to}
for n, df in final.items():
    df.to_csv(C.FINAL / f"{n}.csv", index=False)
print("[04] final tables written. Screening results:")
print(pd.crosstab(pr.case_id.map(case.set_index("case_id").need_id), pr.screening_status))
print(pr[["tactical_score", "risk_forecast", "premium_pct"]].describe().round(1))
print("agreement Week 5 decision vs Week 6 tool:", td.followed_tool_recommendation.mean().round(2))
