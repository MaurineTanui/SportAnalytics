"""
07_build_model_table.py
Step 7. Builds the table the value forecast model is trained and tested on: data/model/value_model_table.csv.
One row = one player at one forecast origin (21 June of 2023, 2024, 2025 and 2026).

Predictors (only data with available_from_timestamp < forecast origin):
  market_value_eur   latest public valuation before the origin (1 January of the origin year)
  prev_value_eur     the valuation one year earlier, if the player has one (has_history = 1)
  momentum_log       log(market_value_eur / prev_value_eur); 0 when there is no earlier valuation
  age_at_origin      age in years on the origin date
  position, passes_z, carries_z, perf_z, minutes_played
                     latest season before the origin; z-scores are within position and season. These are only
                     used to test whether performance data improves the forecast (step 08)
Target (not known at the origin):
  target_value_eur   the next yearly valuation (1 January of the following year)
  log_change         log(target_value_eur / market_value_eur), the quantity the model predicts
Split:
  development = origins 2023 and 2024, backtest = origin 2025, live = origin 2026 (target not known yet).
This script never reads data/withheld/.
"""
import numpy as np
import pandas as pd
import config as C

C.MODEL.mkdir(parents=True, exist_ok=True)
F = C.FINAL
player = pd.read_csv(F / "player.csv", parse_dates=["date_of_birth"])
st = pd.read_csv(F / "player_season_stats.csv", parse_dates=["available_from_timestamp"])
va = pd.read_csv(F / "valuation_history.csv", parse_dates=["valuation_date", "available_from_timestamp"])
pos_of = player.set_index("player_id").position
dob_of = player.set_index("player_id").date_of_birth
st["position"] = st.player_id.map(pos_of)

TS = "%Y-%m-%d %H:%M:%S"
rows = []
for year in C.DEVELOPMENT_ORIGINS + [C.BACKTEST_ORIGIN, C.LIVE_ORIGIN]:
    origin = pd.Timestamp(f"{year}-{C.FORECAST_ORIGIN_MONTH_DAY}")
    split = "development" if year in C.DEVELOPMENT_ORIGINS else "backtest" if year == C.BACKTEST_ORIGIN else "live"
    v_av = va[va.available_from_timestamp < origin].sort_values("valuation_date")
    s_av = st[st.available_from_timestamp < origin]
    ref = s_av.groupby(["position", "season_year"]).agg(mp=("passes_p90", "mean"), sp=("passes_p90", "std"),
                                                        mc=("carries_p90", "mean"), sc=("carries_p90", "std"))
    for p, pv in v_av.groupby("player_id"):
        last = pv.iloc[-1]
        if last.valuation_date.year != year:      # no valuation in the origin year -> nothing to forecast from
            continue
        prev = pv[pv.valuation_date.dt.year == year - 1]
        nxt = va[(va.player_id == p) & (va.valuation_date.dt.year == year + 1)]
        ps = s_av[s_av.player_id == p].sort_values("season_year")
        passes_z = carries_z = np.nan
        season = minutes = stats_ts = None
        if len(ps):
            ls = ps.iloc[-1]
            r = ref.loc[(ls.position, ls.season_year)]
            passes_z, carries_z = (ls.passes_p90 - r.mp) / r.sp, (ls.carries_p90 - r.mc) / r.sc
            season, minutes, stats_ts = int(ls.season_year), int(ls.minutes_played), ls.available_from_timestamp.strftime(TS)
        zs = [z for z in (passes_z, carries_z) if pd.notna(z)]
        v0 = float(last.market_value_eur)
        vprev = float(prev.market_value_eur.iloc[0]) if len(prev) else np.nan
        v1 = float(nxt.market_value_eur.iloc[0]) if len(nxt) else np.nan
        rows.append({
            "player_id": p, "origin_year": year, "forecast_origin": origin.strftime(TS), "split": split,
            "position": pos_of[p], "age_at_origin": round((origin - dob_of[p]).days / 365.25, 4),
            "value_date": last.valuation_date.strftime("%Y-%m-%d"),
            "value_available_ts": last.available_from_timestamp.strftime(TS),
            "market_value_eur": v0, "prev_value_eur": vprev, "has_history": int(len(prev) > 0),
            "momentum_log": round(float(np.log(v0 / vprev)), 6) if len(prev) else 0.0,
            "stats_season": season, "stats_available_ts": stats_ts, "minutes_played": minutes,
            "passes_z": None if pd.isna(passes_z) else round(float(passes_z), 4),
            "carries_z": None if pd.isna(carries_z) else round(float(carries_z), 4),
            "perf_z": round(float(np.mean(zs)), 4) if zs else None, "has_perf": int(len(zs) > 0),
            "target_value_date": f"{year + 1}-01-01",
            "target_value_eur": None if pd.isna(v1) else v1,
            "log_change": None if pd.isna(v1) else round(float(np.log(v1 / v0)), 6),
        })

mt = pd.DataFrame(rows)
mt.insert(0, "row_id", range(1, len(mt) + 1))
for c in ("stats_season", "minutes_played"):
    mt[c] = mt[c].astype("Int64")
mt.to_csv(C.MODEL / "value_model_table.csv", index=False)
print(mt.groupby(["origin_year", "split"]).agg(rows=("row_id", "size"), with_target=("log_change", "count"),
                                               with_history=("has_history", "sum"), with_stats=("has_perf", "sum"),
                                               min_age=("age_at_origin", "min"), max_age=("age_at_origin", "max")).round(1))
print(f"[07] {len(mt)} rows written to data/model/value_model_table.csv")
