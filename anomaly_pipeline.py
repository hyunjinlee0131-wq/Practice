import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"C:\Users\2512-02\Desktop\유가보조금\R\mock_dataset")

# CSV 로드
dtg = pd.read_csv(BASE_DIR / "dtg_daily_expanded.csv")
fuel = pd.read_csv(BASE_DIR / "fuel_transaction_expanded.csv", encoding="cp949")
veh = pd.read_csv(BASE_DIR / "vehicle_profile_expanded.csv")

# 컬럼명 정리
for df in [dtg, fuel, veh]:
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)

# 1) DTG 집계
dtg["date"] = pd.to_datetime(dtg["date"], errors="coerce").dt.date
dtg_agg = dtg.groupby("vehicle_id", as_index=False).agg(
    total_distance_km=("total_distance_km", "sum"),
    total_drive_time_hr=("drive_time_hr", "sum"),
    total_idle_time_min=("idle_time_min", "sum")
)

# 2) 주유 데이터 전처리
fuel["transaction_dt"] = pd.to_datetime(fuel["time"], errors="coerce")
fuel["transaction_date"] = fuel["transaction_dt"].dt.date
fuel["transaction_hour"] = fuel["transaction_dt"].dt.hour
fuel["fuel_liter"] = pd.to_numeric(fuel["fuel_liter"], errors="coerce").fillna(0)

# 3) 주유 집계
fuel_agg = fuel.groupby("vehicle_id", as_index=False).agg(
    actual_fuel_l=("fuel_liter", "sum"),
    refuel_cnt=("transaction_id", "count"),
    night_refuel_cnt=("transaction_hour", lambda s: ((s >= 23) | (s < 6)).sum())
)

# 4) 데이터 병합 및 기대 연료량 계산
summary = veh.merge(dtg_agg, on="vehicle_id", how="left").merge(fuel_agg, on="vehicle_id", how="left")

# 결측치 처리
for col in ["total_distance_km", "actual_fuel_l"]:
    summary[col] = pd.to_numeric(summary[col], errors="coerce").fillna(0)
for col in ["refuel_cnt", "night_refuel_cnt"]:
    summary[col] = pd.to_numeric(summary[col], errors="coerce").fillna(0).astype(int)

# 기대 연료량 계산
summary["avg_eff_km_per_l"] = pd.to_numeric(summary["avg_eff_km_per_l"], errors="coerce")
summary["expected_fuel_l"] = (summary["total_distance_km"] / summary["avg_eff_km_per_l"]).replace([float("inf"), -float("inf")], 0).fillna(0)

tolerance = 0.10
summary["expected_low"] = summary["expected_fuel_l"] * (1 - tolerance)
summary["expected_high"] = summary["expected_fuel_l"] * (1 + tolerance)

# 5) 이상 플래그 설정
# A. 탱크 용량 초과
if "tank_capacity_l" in veh.columns:
    veh["tank_capacity_l"] = pd.to_numeric(veh["tank_capacity_l"], errors="coerce")
    fuel_with_cap = fuel.merge(veh[["vehicle_id", "tank_capacity_l"]], on="vehicle_id", how="left")
    over_tank = (fuel_with_cap["fuel_liter"] > fuel_with_cap["tank_capacity_l"]).groupby(fuel_with_cap["vehicle_id"]).any()
    summary["flag_over_tank"] = summary["vehicle_id"].map(over_tank).fillna(False)
else:
    summary["flag_over_tank"] = False

# B. 하루 4회 이상 주유
daily_refuel = fuel.groupby(["vehicle_id", "transaction_date"]).size().reset_index(name="count")
max_daily = daily_refuel.groupby("vehicle_id")["count"].max()
summary["flag_day_over_4"] = summary["vehicle_id"].map(max_daily >= 4).fillna(False)

# C. 특정 주유소 집중 (80% 이상)
station_counts = fuel.groupby(["vehicle_id", "station_id"]).size().reset_index(name="cnt")
total_counts = station_counts.groupby("vehicle_id")["cnt"].sum()
station_counts["share"] = station_counts["cnt"] / station_counts["vehicle_id"].map(total_counts)
max_share = station_counts.groupby("vehicle_id")["share"].max()
summary["flag_station_conc_80p"] = summary["vehicle_id"].map(max_share >= 0.80).fillna(False)

# D. 연료량 초과/미달
summary["flag_fuel_over_expected"] = summary["actual_fuel_l"] > summary["expected_high"]
summary["flag_fuel_under_expected"] = (summary["actual_fuel_l"] > 0) & (summary["actual_fuel_l"] < summary["expected_low"])

# 6) 이상 확정 및 등급 부여
STRONG_FLAGS = ["flag_over_tank", "flag_day_over_4", "flag_fuel_over_expected"]
summary["anomaly_confirmed"] = summary[STRONG_FLAGS].any(axis=1)

def get_anomaly_level(row):
    has_strong = any(row[f] for f in STRONG_FLAGS)
    has_conc = row["flag_station_conc_80p"]
    if has_strong and has_conc:
        return "HIGH"
    elif has_strong:
        return "MEDIUM"
    elif has_conc:
        return "LOW"
    return "NONE"

summary["anomaly_level"] = summary.apply(get_anomaly_level, axis=1)

def get_anomaly_reason(row):
    reasons = []
    if row["flag_over_tank"]: reasons.append("OVER_TANK")
    if row["flag_day_over_4"]: reasons.append("MANY_REFUELS_PER_DAY")
    if row["flag_fuel_over_expected"]: reasons.append("FUEL_OVER_EXPECTED")
    if row["flag_station_conc_80p"]: reasons.append("STATION_CONCENTRATION")
    if row["flag_fuel_under_expected"]: reasons.append("FUEL_UNDER_EXPECTED")
    return "|".join(reasons)

summary["anomaly_reason"] = summary.apply(get_anomaly_reason, axis=1)

# 7) 정렬 및 출력
level_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "NONE": 3}
summary["_sort"] = summary["anomaly_level"].map(level_order)
summary = summary.sort_values(["_sort", "vehicle_id"]).drop(columns=["_sort"])

output_cols = [
    "vehicle_id", "vehicle_no", "ton_class", "fuel_type",
    "total_distance_km", "expected_fuel_l", "expected_low", "expected_high",
    "actual_fuel_l", "refuel_cnt", "night_refuel_cnt",
    "flag_over_tank", "flag_day_over_4", "flag_fuel_over_expected", "flag_station_conc_80p",
    "anomaly_confirmed", "anomaly_level", "anomaly_reason"
]
output_cols = [c for c in output_cols if c in summary.columns]

print("\n=== 차량별 이상탐지 결과 ===")
print(summary[output_cols])

# 저장
output_file = BASE_DIR / "vehicle_anomaly_final.csv"
summary[output_cols].to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"\n저장 완료: {output_file}")