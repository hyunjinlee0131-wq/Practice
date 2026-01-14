import numpy as np
import pandas as pd

def apply_baseline_rules(summary: pd.DataFrame,
                         fuel: pd.DataFrame,
                         veh: pd.DataFrame,
                         tolerance: float = 0.10,
                         station_baseline: float = 0.60,
                         station_k: int = 6) -> pd.DataFrame:

    summary = summary.copy()

    # expected fuel
    eff = summary["avg_eff_km_per_l"].replace(0, np.nan)
    summary["expected_fuel_l"] = (summary["total_distance_km"] / eff).fillna(0)
    summary["expected_low"]  = summary["expected_fuel_l"] * (1 - tolerance)
    summary["expected_high"] = summary["expected_fuel_l"] * (1 + tolerance)

    # 4) Over-tank
    if "tank_capacity_l" in veh.columns:
        veh = veh.copy()
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

    # 5) Max daily refuel
    daily_cnt = (
        fuel.groupby(["vehicle_id", "transaction_date"])
            .size()
            .rename("daily_refuel_cnt")
            .reset_index()
    )
    max_daily = daily_cnt.groupby("vehicle_id")["daily_refuel_cnt"].max()
    summary["ind_max_daily_refuel"] = summary["vehicle_id"].map(max_daily).fillna(0).astype(int)

    # 6) Fuel deviation
    exp = summary["expected_fuel_l"].replace(0, np.nan)
    summary["ind_fuel_ratio"] = (summary["actual_fuel_l"] / exp).replace([np.inf, -np.inf], np.nan).fillna(0)

    summary["ind_fuel_over_l"] = (summary["actual_fuel_l"] - summary["expected_high"]).clip(lower=0)
    summary["ind_fuel_under_l"] = (summary["expected_low"] - summary["actual_fuel_l"]).clip(lower=0)
    summary.loc[summary["actual_fuel_l"] <= 0, "ind_fuel_under_l"] = 0

    # 7) Station concentration score
    station_tx = (
        fuel.groupby(["vehicle_id", "station_id"])
            .size()
            .rename("cnt")
            .reset_index()
    )
    veh_total = station_tx.groupby("vehicle_id")["cnt"].sum()
    veh_max   = station_tx.groupby("vehicle_id")["cnt"].max()
    max_share = (veh_max / veh_total).replace([np.inf, -np.inf], np.nan)

    summary["ind_station_max_share"] = summary["vehicle_id"].map(max_share).fillna(0)

    n = summary["refuel_cnt"].astype(float)
    n_weight = (n / (n + station_k)).fillna(0)

    raw_conc = ((summary["ind_station_max_share"] - station_baseline) / (1 - station_baseline)).clip(0, 1)
    summary["score_station_conc"] = raw_conc * n_weight

    # 8) Scores
    score_over_tank = np.where(summary["ind_over_tank"], 25, 0) + np.clip(summary["ind_over_tank_cnt"], 0, 3) * 5

    m = summary["ind_max_daily_refuel"]
    score_daily_refuel = np.select(
        [m <= 2, m == 3, m == 4, m >= 5],
        [0,      10,     20,     30],
        default=0
    )

    r = summary["ind_fuel_ratio"]
    score_fuel_over = np.select(
        [r <= 1.10, (r > 1.10) & (r <= 1.50), r > 1.50],
        [0,
         ((r - 1.10) / (1.50 - 1.10) * 30),
         40],
        default=0
    )
    score_fuel_over = np.nan_to_num(score_fuel_over, nan=0)

    under_l = summary["ind_fuel_under_l"]
    score_fuel_under = np.clip(under_l / (summary["expected_fuel_l"].replace(0, np.nan)) * 10, 0, 10)
    score_fuel_under = np.nan_to_num(score_fuel_under, nan=0)

    score_station = summary["score_station_conc"] * 15

    summary["risk_score"] = (
        score_over_tank +
        score_daily_refuel +
        score_fuel_over +
        score_fuel_under +
        score_station
    ).clip(0, 100).round(2)

    summary["score_over_tank"]    = np.round(score_over_tank, 2)
    summary["score_daily_refuel"] = np.round(score_daily_refuel, 2)
    summary["score_fuel_over"]    = np.round(score_fuel_over, 2)
    summary["score_fuel_under"]   = np.round(score_fuel_under, 2)
    summary["score_station"]      = np.round(score_station, 2)

    # 9) Tier
    summary["risk_tier"] = np.select(
        [summary["risk_score"] >= 70,
         summary["risk_score"] >= 40,
         summary["risk_score"] >= 20],
        ["HIGH", "MEDIUM", "LOW"],
        default="NONE"
    )

    # 10) Reason
    reason_over_tank = np.where(summary["ind_over_tank"], "OVER_TANK", "")
    reason_many_refuel = np.where(summary["ind_max_daily_refuel"] >= 4, "MANY_REFUELS_PER_DAY", "")
    reason_fuel_over = np.where(summary["ind_fuel_ratio"] > 1.10, "FUEL_OVER_EXPECTED", "")
    reason_station = np.where(summary["score_station_conc"] >= 0.6, "STATION_CONCENTRATION", "")
    reason_fuel_under = np.where(summary["ind_fuel_under_l"] > 0, "FUEL_UNDER_EXPECTED", "")

    reasons = np.vstack([reason_over_tank, reason_many_refuel, reason_fuel_over, reason_station, reason_fuel_under]).T
    summary["risk_reason"] = pd.Series(["|".join([x for x in row if x]) for row in reasons], index=summary.index)

    return summary
