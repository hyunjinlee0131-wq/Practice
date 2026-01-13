import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"C:\Users\2512-02\Desktop\유가보조금\R\mock_dataset")

# -----------------------------
# 0) Load
# -----------------------------
dtg  = pd.read_csv(BASE_DIR / "dtg_daily_expanded.csv")
fuel = pd.read_csv(BASE_DIR / "fuel_transaction_expanded.csv", encoding="cp949")
veh  = pd.read_csv(BASE_DIR / "vehicle_profile_expanded.csv")

# column cleanup
for df in (dtg, fuel, veh):
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)

# -----------------------------
# 1) DTG aggregate (vehicle-level)
# -----------------------------
dtg["date"] = pd.to_datetime(dtg["date"], errors="coerce").dt.date

dtg_agg = (
    dtg.groupby("vehicle_id", as_index=False)
       .agg(total_distance_km=("total_distance_km", "sum"),
            total_drive_time_hr=("drive_time_hr", "sum"),
            total_idle_time_min=("idle_time_min", "sum"))
)

# -----------------------------
# 2) Fuel preprocess + aggregate (vehicle-level)
# -----------------------------
fuel["transaction_dt"]   = pd.to_datetime(fuel["time"], errors="coerce")
fuel["transaction_date"] = fuel["transaction_dt"].dt.date
fuel["transaction_hour"] = fuel["transaction_dt"].dt.hour

fuel["fuel_liter"] = pd.to_numeric(fuel["fuel_liter"], errors="coerce").fillna(0)

# vectorized night flag
fuel["is_night"] = ((fuel["transaction_hour"] >= 23) | (fuel["transaction_hour"] < 6)).astype(np.int8)

fuel_agg = (
    fuel.groupby("vehicle_id", as_index=False)
        .agg(actual_fuel_l=("fuel_liter", "sum"),
             refuel_cnt=("transaction_id", "count"),
             night_refuel_cnt=("is_night", "sum"))
)

# -----------------------------
# 3) Merge summary
# -----------------------------
summary = (
    veh.merge(dtg_agg, on="vehicle_id", how="left")
       .merge(fuel_agg, on="vehicle_id", how="left")
)

# numeric safety
for col in ["total_distance_km", "actual_fuel_l"]:
    summary[col] = pd.to_numeric(summary[col], errors="coerce").fillna(0)

for col in ["refuel_cnt", "night_refuel_cnt"]:
    summary[col] = pd.to_numeric(summary[col], errors="coerce").fillna(0).astype(int)

summary["avg_eff_km_per_l"] = pd.to_numeric(summary.get("avg_eff_km_per_l"), errors="coerce")

# expected fuel (avoid div by 0)
eff = summary["avg_eff_km_per_l"].replace(0, np.nan)
summary["expected_fuel_l"] = (summary["total_distance_km"] / eff).fillna(0)

tolerance = 0.10
summary["expected_low"]  = summary["expected_fuel_l"] * (1 - tolerance)
summary["expected_high"] = summary["expected_fuel_l"] * (1 + tolerance)

# -----------------------------
# 4) Metric 1: Over-tank (vehicle-level, any transaction > tank capacity)
#    -> as an indicator, plus binary
# -----------------------------
if "tank_capacity_l" in veh.columns:
    veh["tank_capacity_l"] = pd.to_numeric(veh["tank_capacity_l"], errors="coerce")

    cap_map = veh.set_index("vehicle_id")["tank_capacity_l"]
    fuel_cap = fuel.copy()
    fuel_cap["tank_capacity_l"] = fuel_cap["vehicle_id"].map(cap_map)

    fuel_cap["over_tank_tx"] = (fuel_cap["fuel_liter"] > fuel_cap["tank_capacity_l"]).fillna(False)

    over_tank_any = fuel_cap.groupby("vehicle_id")["over_tank_tx"].any()
    over_tank_cnt = fuel_cap.groupby("vehicle_id")["over_tank_tx"].sum()

    summary["ind_over_tank"]     = summary["vehicle_id"].map(over_tank_any).fillna(False)
    summary["ind_over_tank_cnt"] = summary["vehicle_id"].map(over_tank_cnt).fillna(0).astype(int)
else:
    summary["ind_over_tank"] = False
    summary["ind_over_tank_cnt"] = 0

# -----------------------------
# 5) Metric 2: Max daily refuel count (vehicle-level)
# -----------------------------
daily_cnt = (
    fuel.groupby(["vehicle_id", "transaction_date"])
        .size()
        .rename("daily_refuel_cnt")
        .reset_index()
)
max_daily = daily_cnt.groupby("vehicle_id")["daily_refuel_cnt"].max()

summary["ind_max_daily_refuel"] = summary["vehicle_id"].map(max_daily).fillna(0).astype(int)

# -----------------------------
# 6) Metric 3: Fuel deviation metrics (continuous)
#    - ratio: actual / expected (when expected > 0)
#    - over: max(0, actual - expected_high)
#    - under: max(0, expected_low - actual) with actual>0
# -----------------------------
exp = summary["expected_fuel_l"].replace(0, np.nan)
summary["ind_fuel_ratio"] = (summary["actual_fuel_l"] / exp).replace([np.inf, -np.inf], np.nan).fillna(0)

summary["ind_fuel_over_l"] = (summary["actual_fuel_l"] - summary["expected_high"]).clip(lower=0)

summary["ind_fuel_under_l"] = (summary["expected_low"] - summary["actual_fuel_l"]).clip(lower=0)
summary.loc[summary["actual_fuel_l"] <= 0, "ind_fuel_under_l"] = 0

# -----------------------------
# 7) Metric 4: Station concentration as a SCORE (not a hard flag)
#    - compute max station share per vehicle
#    - sample-size correction: shrink score for small refuel_cnt
# -----------------------------
station_tx = (
    fuel.groupby(["vehicle_id", "station_id"])
        .size()
        .rename("cnt")
        .reset_index()
)

# total tx per vehicle
veh_total = station_tx.groupby("vehicle_id")["cnt"].sum()
veh_max   = station_tx.groupby("vehicle_id")["cnt"].max()

max_share = (veh_max / veh_total).replace([np.inf, -np.inf], np.nan)

summary["ind_station_max_share"] = summary["vehicle_id"].map(max_share).fillna(0)

# sample size correction
# n_effective = min(refuel_cnt, capN) / capN (0~1), or smoother: n/(n+k)
k = 6  # smoothing factor (bigger k => stricter penalty for small n)
n = summary["refuel_cnt"].astype(float)
n_weight = (n / (n + k)).fillna(0)  # 0~1

# concentration score: only penalize when share exceeds a baseline (e.g. 0.6), then scale
baseline = 0.60
raw_conc = ((summary["ind_station_max_share"] - baseline) / (1 - baseline)).clip(lower=0, upper=1)

summary["score_station_conc"] = raw_conc * n_weight

# -----------------------------
# 8) Build Risk Scores (0~100) — fully vectorized
#    You can tune weights easily here.
# -----------------------------
# A) Over-tank: binary + count
score_over_tank = np.where(summary["ind_over_tank"], 25, 0) + np.clip(summary["ind_over_tank_cnt"], 0, 3) * 5

# B) Max daily refuel: scale (>=4 suspicious)
#   0 at <=2, 10 at 3, 20 at 4, 30 at >=5
m = summary["ind_max_daily_refuel"]
score_daily_refuel = np.select(
    [m <= 2, m == 3, m == 4, m >= 5],
    [0,      10,     20,     30],
    default=0
)

# C) Fuel over expected: continuous severity
#   if ratio <=1.1 -> 0, ratio 1.1~1.5 -> up to 30, >1.5 -> 40
r = summary["ind_fuel_ratio"]
score_fuel_over = np.select(
    [r <= 1.10, (r > 1.10) & (r <= 1.50), r > 1.50],
    [0,
     ((r - 1.10) / (1.50 - 1.10) * 30),
     40],
    default=0
)
score_fuel_over = np.nan_to_num(score_fuel_over, nan=0)

# D) Fuel under expected: weak signal (data issue 가능성도 큼)
#   Use small weight; cap at 10
under_l = summary["ind_fuel_under_l"]
score_fuel_under = np.clip(under_l / (summary["expected_fuel_l"].replace(0, np.nan)) * 10, 0, 10)
score_fuel_under = np.nan_to_num(score_fuel_under, nan=0)

# E) Station concentration score: 0~1 -> 0~15
score_station = summary["score_station_conc"] * 15

# total risk score (0~100)
summary["risk_score"] = (
    score_over_tank +
    score_daily_refuel +
    score_fuel_over +
    score_fuel_under +
    score_station
).clip(lower=0, upper=100).round(2)

# Keep component scores for explainability
summary["score_over_tank"]    = np.round(score_over_tank, 2)
summary["score_daily_refuel"] = np.round(score_daily_refuel, 2)
summary["score_fuel_over"]    = np.round(score_fuel_over, 2)
summary["score_fuel_under"]   = np.round(score_fuel_under, 2)
summary["score_station"]      = np.round(score_station, 2)

# -----------------------------
# 9) Risk tier (vectorized)
# -----------------------------
summary["risk_tier"] = np.select(
    [summary["risk_score"] >= 70,
     summary["risk_score"] >= 40,
     summary["risk_score"] >= 20],
    ["HIGH", "MEDIUM", "LOW"],
    default="NONE"
)

# -----------------------------
# 10) Reason string (vectorized-ish; no row apply)
#     Build reason columns, then join.
# -----------------------------
reason_over_tank = np.where(summary["ind_over_tank"], "OVER_TANK", "")
reason_many_refuel = np.where(summary["ind_max_daily_refuel"] >= 4, "MANY_REFUELS_PER_DAY", "")
reason_fuel_over = np.where(summary["ind_fuel_ratio"] > 1.10, "FUEL_OVER_EXPECTED", "")
reason_station = np.where(summary["score_station_conc"] >= 0.6, "STATION_CONCENTRATION", "")  # high conc only
reason_fuel_under = np.where(summary["ind_fuel_under_l"] > 0, "FUEL_UNDER_EXPECTED", "")

reasons = np.vstack([reason_over_tank, reason_many_refuel, reason_fuel_over, reason_station, reason_fuel_under]).T
summary["risk_reason"] = pd.Series(["|".join([x for x in row if x]) for row in reasons], index=summary.index)

# -----------------------------
# 11) Output
# -----------------------------
output_cols = [
    "vehicle_id", "vehicle_no", "ton_class", "fuel_type",
    "total_distance_km", "expected_fuel_l", "expected_low", "expected_high",
    "actual_fuel_l", "refuel_cnt", "night_refuel_cnt",
    "ind_over_tank", "ind_over_tank_cnt", "ind_max_daily_refuel",
    "ind_fuel_ratio", "ind_station_max_share",
    "score_over_tank", "score_daily_refuel", "score_fuel_over", "score_fuel_under", "score_station",
    "risk_score", "risk_tier", "risk_reason"
]
output_cols = [c for c in output_cols if c in summary.columns]

summary = summary.sort_values(["risk_score", "vehicle_id"], ascending=[False, True])

print("\n=== 차량별 Risk Score 기반 이상징후 결과 ===")
print(summary[output_cols])

output_file = BASE_DIR / "vehicle_risk_scored.csv"
summary[output_cols].to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"\n저장 완료: {output_file}")
