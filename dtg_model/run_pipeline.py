import pandas as pd
from pathlib import Path
from datetime import datetime

from pipeline import load_and_build_summary
from rules_baseline import apply_baseline_rules
from refund_engine import RefundParams, run_refund_engine

BASE_DIR = Path(r"C:\Users\2512-02\Desktop\유가보조금\R\mock_dataset")

def safe_to_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"[OK] Saved: {path}")
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt = path.with_name(f"{path.stem}_{ts}{path.suffix}")
        df.to_csv(alt, index=False, encoding="utf-8-sig")
        print(f"[WARN] Permission denied. Saved to: {alt}")

def main():
    summary, fuel, veh = load_and_build_summary(BASE_DIR)
    summary = apply_baseline_rules(summary, fuel, veh)

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

    params = RefundParams(
    tolerance=0.10,
    unit_price_krw_per_l=500,
    cap_mode="by_ton_class",
    cap_by_ton_class={3: 800, 5: 1000, 8: 1200, 10: 1300, 12: 1500}
)

    summary_refund = run_refund_engine(summary, params)

    summary = summary.sort_values(["risk_score", "vehicle_id"], ascending=[False, True])

    print("\n=== 차량별 Risk Score 기반 이상징후 결과 ===")
    print(summary[output_cols])
    
    summary_refund.to_csv(
    BASE_DIR / "refund_decision.csv",
    index=False,
    encoding="utf-8-sig"
)

    safe_to_csv(summary[output_cols], BASE_DIR / "vehicle_risk_scored.csv")

if __name__ == "__main__":
    main()
