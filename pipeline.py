import numpy as np
import pandas as pd
from pathlib import Path

def load_and_build_summary(base_dir: Path):
    dtg  = pd.read_csv(base_dir / "dtg_daily_expanded.csv")
    fuel = pd.read_csv(base_dir / "fuel_transaction_expanded.csv", encoding="cp949")
    veh  = pd.read_csv(base_dir / "vehicle_profile_expanded.csv")

    for df in (dtg, fuel, veh):
        df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)

    # DTG aggregate
    dtg["date"] = pd.to_datetime(dtg["date"], errors="coerce").dt.date
    dtg_agg = (
        dtg.groupby("vehicle_id", as_index=False)
           .agg(total_distance_km=("total_distance_km", "sum"),
                total_drive_time_hr=("drive_time_hr", "sum"),
                total_idle_time_min=("idle_time_min", "sum"))
    )

    # Fuel preprocess
    fuel["transaction_dt"]   = pd.to_datetime(fuel["time"], errors="coerce")
    fuel["transaction_date"] = fuel["transaction_dt"].dt.date
    fuel["transaction_hour"] = fuel["transaction_dt"].dt.hour

    fuel["fuel_liter"] = pd.to_numeric(fuel["fuel_liter"], errors="coerce").fillna(0)
    fuel["is_night"] = ((fuel["transaction_hour"] >= 23) | (fuel["transaction_hour"] < 6)).astype(np.int8)

    fuel_agg = (
        fuel.groupby("vehicle_id", as_index=False)
            .agg(actual_fuel_l=("fuel_liter", "sum"),
                 refuel_cnt=("transaction_id", "count"),
                 night_refuel_cnt=("is_night", "sum"))
    )

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

    return summary, fuel, veh
