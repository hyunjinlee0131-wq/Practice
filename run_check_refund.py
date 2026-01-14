from pathlib import Path

from pipeline import load_and_build_summary
from refund_engine import RefundParams, run_refund_engine

BASE_DIR = Path(r"C:\Users\2512-02\Desktop\유가보조금\R\mock_dataset")

def main():
    # 1️⃣ 데이터 로드 + summary 생성
    summary, fuel, veh = load_and_build_summary(BASE_DIR)

    # 2️⃣ 환급 파라미터 설정 (임시값)
    params = RefundParams(
        tolerance=0.10,                 # 허용오차 10%
        unit_price_krw_per_l=500,        # 단가 (의미 없음, 출력용)
        cap_mode="fixed",
        fixed_cap_l=1000                 # 한도 (의미 없음, 출력용)
    )

    # 3️⃣ DTG Gate + 환급 판정
    result = run_refund_engine(summary, params)

    # 4️⃣ 터미널 출력 (핵심)
    print("\n=== DTG 기반 환급 판정 결과 ===")
    for _, row in result.iterrows():
        vid = row["vehicle_id"]
        status = "✅ 환급 가능" if row["refund_status"] == "APPROVE" else "❌ 환급 불가(보류)"
        reason = row["gate_reason"]

        print(f"- 차량 {vid}: {status} | 사유: {reason}")

if __name__ == "__main__":
    main()
