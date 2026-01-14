# refund_engine.py
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class RefundParams:
    # Gate 허용오차 (예: 0.10 = 10%)
    tolerance: float = 0.10

    # Gate 기준 (ratio 기반으로 쓰고 싶으면 True)
    use_ratio_gate: bool = False
    ratio_threshold: float = 1.10  # actual/expected <= 1.10 pass

    # 환급 단가(원/L) - 임시값. 실제 사업/제도 값으로 대체
    unit_price_krw_per_l: float = 500.0

    # 한도 산정 방식
    # 1) 고정 한도(L)
    cap_mode: str = "fixed"  # "fixed" | "by_ton_class" | "by_vehicle_col"
    fixed_cap_l: float = 1000.0

    # 2) 톤급별 한도(L) (예시)
    cap_by_ton_class: Optional[Dict[int, float]] = None

    # 3) vehicle profile 컬럼에 한도량이 있을 때
    cap_vehicle_col: str = "subsidy_cap_l"


def _ensure_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df


def compute_expected_fuel(summary: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    """
    summary에 expected_fuel_l, expected_high/low를 계산해서 붙임
    필요 컬럼: total_distance_km, avg_eff_km_per_l
    """
    out = summary.copy()

    out = _ensure_numeric(out, ["total_distance_km", "avg_eff_km_per_l", "actual_fuel_l"])

    eff = out["avg_eff_km_per_l"].replace(0, np.nan)
    out["expected_fuel_l"] = (out["total_distance_km"] / eff).fillna(0)

    out["expected_low"] = out["expected_fuel_l"] * (1 - tolerance)
    out["expected_high"] = out["expected_fuel_l"] * (1 + tolerance)
    return out


def compute_cap_l(summary: pd.DataFrame, params: RefundParams) -> pd.Series:
    """
    차량별 지급한도량(L) 계산
    """
    if params.cap_mode == "fixed":
        return pd.Series(params.fixed_cap_l, index=summary.index)

    if params.cap_mode == "by_ton_class":
        if not params.cap_by_ton_class:
            raise ValueError("cap_mode='by_ton_class'인데 cap_by_ton_class가 비어 있습니다.")
        if "ton_class" not in summary.columns:
            raise ValueError("summary에 ton_class 컬럼이 없습니다.")
        ton = pd.to_numeric(summary["ton_class"], errors="coerce").fillna(-1).astype(int)
        return ton.map(params.cap_by_ton_class).fillna(params.fixed_cap_l)

    if params.cap_mode == "by_vehicle_col":
        col = params.cap_vehicle_col
        if col not in summary.columns:
            raise ValueError(f"cap_mode='by_vehicle_col'인데 summary에 '{col}' 컬럼이 없습니다.")
        return pd.to_numeric(summary[col], errors="coerce").fillna(0)

    raise ValueError("cap_mode는 'fixed', 'by_ton_class', 'by_vehicle_col' 중 하나여야 합니다.")


def apply_gate(summary: pd.DataFrame, params: RefundParams) -> pd.DataFrame:
    """
    Gate(pass/fail) 판정
    기본: actual_fuel_l <= expected_high
    옵션: ratio gate(actual/expected <= ratio_threshold)
    """
    out = summary.copy()
    out = _ensure_numeric(out, ["actual_fuel_l", "expected_fuel_l", "expected_high"])

    if params.use_ratio_gate:
        exp = out["expected_fuel_l"].replace(0, np.nan)
        ratio = (out["actual_fuel_l"] / exp).replace([np.inf, -np.inf], np.nan).fillna(0)
        out["gate_metric"] = ratio
        out["gate_pass"] = ratio <= params.ratio_threshold
        out["gate_reason"] = np.where(out["gate_pass"], "PASS", f"FAIL_RATIO_GT_{params.ratio_threshold}")
    else:
        out["gate_metric"] = out["actual_fuel_l"] - out["expected_high"]
        out["gate_pass"] = out["actual_fuel_l"] <= out["expected_high"]
        out["gate_reason"] = np.where(out["gate_pass"], "PASS", "FAIL_FUEL_GT_EXPECTED_HIGH")

    out["gate_status"] = np.where(out["gate_pass"], "PASS", "FAIL")
    return out


def calculate_refund(summary: pd.DataFrame, params: RefundParams) -> pd.DataFrame:
    """
    Gate 통과 시에만 환급 산식 실행.
    환급액 = min(주유량, 한도량) * 단가
    """
    out = summary.copy()

    # cap liters
    cap_l = compute_cap_l(out, params)
    out["subsidy_cap_l"] = cap_l

    out = _ensure_numeric(out, ["actual_fuel_l", "subsidy_cap_l"])
    out["unit_price"] = params.unit_price_krw_per_l

    # Calculator runs only when gate_pass
    out["refund_liter"] = np.where(
        out["gate_pass"],
        np.minimum(out["actual_fuel_l"], out["subsidy_cap_l"]),
        0.0
    )

    out["refund_amount"] = (out["refund_liter"] * out["unit_price"]).round(0)

    out["refund_status"] = np.where(out["gate_pass"], "APPROVE", "HOLD")

    return out


def run_refund_engine(summary: pd.DataFrame,
                      params: Optional[RefundParams] = None) -> pd.DataFrame:
    """
    One-shot: expected 계산 -> gate -> calculator
    """
    if params is None:
        params = RefundParams()

    out = compute_expected_fuel(summary, tolerance=params.tolerance)
    out = apply_gate(out, params=params)
    out = calculate_refund(out, params=params)
    return out


# -----------------------------
# Example usage (run manually)
# -----------------------------
if __name__ == "__main__":
    # 예시: 다른 파일에서 summary 만든 후 아래처럼 호출하세요.
    # from pipeline import load_and_build_summary
    # summary, fuel, veh = load_and_build_summary(BASE_DIR)
    #
    # params = RefundParams(
    #     tolerance=0.10,
    #     unit_price_krw_per_l=500,
    #     cap_mode="by_ton_class",
    #     cap_by_ton_class={3: 800, 5: 1000, 8: 1200, 10: 1300, 12: 1500}
    # )
    # result = run_refund_engine(summary, params=params)
    # print(result[["vehicle_id","gate_status","refund_status","refund_liter","refund_amount"]].head())
    pass
