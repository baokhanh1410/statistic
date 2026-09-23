"""
src/credit_scoring.py
=====================
Module hệ thống chấm điểm tín dụng theo thang FICO (300–850).

Bao gồm chuyển đổi PD (Probability of Default) sang điểm số,
phân hạng rủi ro tín dụng, và demo chấm điểm khách hàng đơn lẻ.

Functions:
    - prob_to_fico_score  : Chuyển xác suất vỡ nợ PD → Điểm FICO
    - get_credit_rating   : Phân hạng Excellent / Good / Fair / Poor
    - score_new_customer  : Chấm điểm & khuyến nghị cho 1 khách hàng mới
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, Tuple


# ==============================================================================
# CONSTANTS & RATING TIERS
# ==============================================================================

FICO_MIN = 300
FICO_MAX = 850
FICO_RANGE = FICO_MAX - FICO_MIN  # 550

# Thang phân hạng FICO
RATING_TIERS = [
    {
        "min": 750, "max": 850,
        "label": "Xuất sắc (Excellent) ⭐⭐⭐⭐⭐",
        "risk": "Rất thấp",
        "color": "\033[92m",  # Green
        "recommendation": "✅ PHÊ DUYỆT NGAY - Ưu đãi lãi suất cao nhất, hạn mức tối đa.",
    },
    {
        "min": 670, "max": 749,
        "label": "Tốt (Good) ⭐⭐⭐⭐",
        "risk": "Thấp",
        "color": "\033[94m",  # Blue
        "recommendation": "✅ PHÊ DUYỆT TIÊU CHUẨN - Hạn mức và lãi suất chuẩn.",
    },
    {
        "min": 580, "max": 669,
        "label": "Trung bình (Fair) ⭐⭐⭐",
        "risk": "Trung bình",
        "color": "\033[93m",  # Yellow
        "recommendation": "⚠️  PHÊ DUYỆT CÓ ĐIỀU KIỆN - Hạn mức thấp, giám sát trả nợ chặt chẽ.",
    },
    {
        "min": 300, "max": 579,
        "label": "Kém (Poor) ⭐⭐",
        "risk": "Cao",
        "color": "\033[91m",  # Red
        "recommendation": "❌ TỪ CHỐI - Rủi ro vỡ nợ cao, không cấp tín dụng.",
    },
]
RESET_COLOR = "\033[0m"


# ==============================================================================
# CORE FUNCTIONS
# ==============================================================================

def prob_to_fico_score(
    prob_default: float,
    min_score: int = FICO_MIN,
    max_score: int = FICO_MAX,
) -> float:
    """
    Chuyển xác suất vỡ nợ (PD) thành điểm tín dụng theo thang FICO.

    Công thức:
        Score = max_score - (PD × (max_score - min_score))
        Score = 850 - (PD × 550)

    Điểm cao → PD thấp → ít rủi ro vỡ nợ.
    Điểm thấp → PD cao → nhiều rủi ro vỡ nợ.

    Args:
        prob_default (float) : Xác suất vỡ nợ PD ∈ [0, 1].
        min_score    (int)   : Điểm tối thiểu (mặc định 300).
        max_score    (int)   : Điểm tối đa (mặc định 850).

    Returns:
        float: Điểm tín dụng FICO ∈ [min_score, max_score].

    Examples:
        >>> prob_to_fico_score(0.0)   # PD = 0% → điểm tối đa
        850.0
        >>> prob_to_fico_score(1.0)   # PD = 100% → điểm tối thiểu
        300.0
        >>> prob_to_fico_score(0.5)   # PD = 50% → điểm trung bình
        575.0
    """
    prob_default = float(np.clip(prob_default, 0.0, 1.0))
    score = max_score - (prob_default * (max_score - min_score))
    return round(score, 2)


def get_credit_rating(score: float) -> Dict[str, str]:
    """
    Phân loại điểm FICO vào 4 hạng tín dụng với khuyến nghị kinh doanh.

    Thang phân hạng:
        750 – 850: Xuất sắc (Excellent) → Duyệt ngay, ưu đãi lãi suất
        670 – 749: Tốt (Good)           → Duyệt tiêu chuẩn
        580 – 669: Trung bình (Fair)    → Hạn mức thấp, giám sát nợ
        300 – 579: Kém (Poor)           → Từ chối cấp tín dụng

    Args:
        score (float): Điểm tín dụng FICO ∈ [300, 850].

    Returns:
        Dict[str, str]: Dictionary gồm:
            - 'rating'         : Tên hạng tín dụng
            - 'risk_level'     : Mức độ rủi ro
            - 'recommendation' : Khuyến nghị kinh doanh
    """
    score = float(score)
    for tier in RATING_TIERS:
        if tier["min"] <= score <= tier["max"]:
            return {
                "rating": tier["label"],
                "risk_level": tier["risk"],
                "recommendation": tier["recommendation"],
            }

    # Fallback an toàn
    return {
        "rating": "Không xác định",
        "risk_level": "Không xác định",
        "recommendation": "⚠️  Không thể xác định hạng tín dụng.",
    }


def score_new_customer(
    pipeline_model: Any,
    customer_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Chấm điểm tín dụng cho một khách hàng mới từ đầu đến cuối.

    Pipeline:
        1. Chuyển customer_dict → DataFrame 1 dòng
        2. Dự đoán PD qua pipeline_model (phải có .predict_proba())
        3. Tính điểm FICO từ PD
        4. Phân hạng tín dụng
        5. Xuất bản kết quả đầy đủ

    Args:
        pipeline_model (Any)          : Mô hình scikit-learn đã fit (Pipeline hoặc classifier).
                                        Bắt buộc có predict_proba().
        customer_dict  (Dict[str, Any]): Dữ liệu 1 khách hàng (gồm tất cả feature gốc).

    Returns:
        Dict[str, Any]: Dictionary kết quả gồm:
            - 'prob_default' : Xác suất vỡ nợ (float)
            - 'fico_score'   : Điểm tín dụng FICO (float)
            - 'rating'       : Hạng tín dụng (str)
            - 'risk_level'   : Mức độ rủi ro (str)
            - 'recommendation': Khuyến nghị kinh doanh (str)

    Example:
        >>> customer = {
        ...     'LIMIT_BAL': 50000, 'SEX': 2, 'EDUCATION': 2, 'MARRIAGE': 1, 'AGE': 28,
        ...     'PAY_0': 0, 'PAY_2': 0, 'PAY_3': 0, 'PAY_4': 0, 'PAY_5': 0, 'PAY_6': 0,
        ...     'BILL_AMT1': 10000, 'BILL_AMT2': 9000, 'BILL_AMT3': 8500,
        ...     'BILL_AMT4': 8000, 'BILL_AMT5': 7500, 'BILL_AMT6': 7000,
        ...     'PAY_AMT1': 2000, 'PAY_AMT2': 2000, 'PAY_AMT3': 2000,
        ...     'PAY_AMT4': 2000, 'PAY_AMT5': 2000, 'PAY_AMT6': 2000,
        ... }
        >>> result = score_new_customer(best_pipeline, customer)
    """
    # 1. Chuyển dict → DataFrame
    df_customer = pd.DataFrame([customer_dict])

    # 2. Dự đoán PD
    try:
        prob_default = pipeline_model.predict_proba(df_customer)[0, 1]
    except Exception as e:
        raise RuntimeError(f"Lỗi khi dự đoán: {e}. Đảm bảo pipeline đã được fit và customer_dict đúng cấu trúc.") from e

    # 3. Tính điểm FICO
    fico_score = prob_to_fico_score(prob_default)

    # 4. Phân hạng
    rating_info = get_credit_rating(fico_score)

    # 5. Tổng hợp kết quả
    result = {
        "prob_default": round(float(prob_default), 4),
        "fico_score": fico_score,
        **rating_info,
    }

    # 6. In bản kết quả đẹp
    _print_scorecard(result)

    return result


def _print_scorecard(result: Dict[str, Any]) -> None:
    """
    In kết quả chấm điểm tín dụng theo định dạng bản thẩm định đẹp.

    Args:
        result (Dict[str, Any]): Kết quả từ score_new_customer().
    """
    # Tìm màu tương ứng
    color = "\033[93m"  # Default Yellow
    for tier in RATING_TIERS:
        if tier["label"] == result["rating"]:
            color = tier["color"]
            break

    border = "═" * 55
    print(f"\n╔{border}╗")
    print(f"║{'BẢN THẨM ĐỊNH TÍN DỤNG - CREDIT SCORECARD':^55}║")
    print(f"╠{border}╣")
    print(f"║  💰  Xác suất vỡ nợ (PD):  {result['prob_default'] * 100:>6.2f}%{' ':>19}║")
    print(f"║  📊  Điểm FICO:              {result['fico_score']:>6.1f} / 850{' ':>16}║")
    print(f"║  🏷️   Hạng tín dụng:         {color}{result['rating']}{RESET_COLOR:}{' ':>3}║")
    print(f"║  ⚡  Mức độ rủi ro:          {result['risk_level']:<27}║")
    print(f"╠{border}╣")
    print(f"║  🏦  Khuyến nghị:                                     ║")
    # Wrap recommendation text
    rec = result["recommendation"]
    rec_lines = [rec[i:i+50] for i in range(0, len(rec), 50)]
    for line in rec_lines:
        print(f"║  {line:<53}║")
    print(f"╚{border}╝\n")


def batch_score_customers(
    pipeline_model: Any,
    df_customers: pd.DataFrame,
    id_col: str = None,
) -> pd.DataFrame:
    """
    Chấm điểm tín dụng cho nhiều khách hàng (batch mode).

    Args:
        pipeline_model (Any)          : Mô hình scikit-learn đã fit.
        df_customers   (pd.DataFrame) : DataFrame nhiều khách hàng.
        id_col         (str, opt)     : Tên cột ID khách hàng (nếu có).

    Returns:
        pd.DataFrame: DataFrame bổ sung cột 'prob_default', 'fico_score', 'rating'.
    """
    df_result = df_customers.copy()

    prob_defaults = pipeline_model.predict_proba(df_customers)[:, 1]
    df_result["prob_default"] = prob_defaults
    df_result["fico_score"] = [prob_to_fico_score(p) for p in prob_defaults]
    df_result["rating"] = [get_credit_rating(s)["rating"] for s in df_result["fico_score"]]
    df_result["risk_level"] = [get_credit_rating(s)["risk_level"] for s in df_result["fico_score"]]
    df_result["recommendation"] = [get_credit_rating(s)["recommendation"] for s in df_result["fico_score"]]

    print(f"✅ Đã chấm điểm {len(df_result):,} khách hàng.")
    print("\nPhân bố hạng tín dụng:")
    print(df_result["rating"].value_counts().to_string())

    return df_result
