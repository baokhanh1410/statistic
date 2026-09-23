"""
src/feature_engineering.py
===========================
Module tạo đặc trưng tín dụng theo khung RFM & chỉ số tài chính nghiệp vụ.

Recency  (R): Tính gần đây của trễ hạn trả nợ.
Frequency(F): Tần suất và mức độ trễ hạn trong 6 tháng.
Monetary (M): Tổng dư nợ, số tiền trả, tỉ lệ sử dụng hạn mức & chỉ số tài chính.

Functions:
    - _add_recency_features   : Tạo các biến Recency (r_*)
    - _add_frequency_features : Tạo các biến Frequency (f_*)
    - _add_monetary_features  : Tạo các biến Monetary & Financial (m_*)
    - build_features          : Hàm tổng hợp gọi cả 3 nhóm trên
"""

import pandas as pd
import numpy as np

# Cột trạng thái trả nợ (PAY_0 = Tháng 9, PAY_6 = Tháng 4)
PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
# Cột dư nợ sao kê (BILL_AMT1 = Tháng 9, BILL_AMT6 = Tháng 4)
BILL_COLS = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
# Cột số tiền đã trả (PAY_AMT1 = Tháng 9, PAY_AMT6 = Tháng 4)
PAY_AMT_COLS = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]

EPSILON = 1e-5  # Tránh chia cho 0


# ==============================================================================
# RECENCY FEATURES (R)
# ==============================================================================

def _add_recency_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Thêm các biến Recency phản ánh mức độ trễ hạn gần đây:

    - r_weighted_delay : Trọng số thời gian (T9×6, T8×5, T7×4, T6×3, T5×2, T4×1)
    - r_delay_trend    : Xu hướng (PAY_0 - PAY_6) > 0 = ngày càng tệ hơn
    - r_recent_delay   : Mức trễ tháng gần nhất (PAY_0)

    Args:
        df (pd.DataFrame): DataFrame đầu vào.

    Returns:
        pd.DataFrame: DataFrame đã bổ sung cột r_*.
    """
    # Trọng số: tháng 9 (PAY_0) trọng số cao nhất = 6, tháng 4 (PAY_6) = 1
    weights = [6, 5, 4, 3, 2, 1]
    pay_values = df[PAY_COLS].values  # shape (n, 6)

    # r_weighted_delay: chỉ tính trọng số khi PAY > 0 (thực sự trễ hạn)
    delay_matrix = np.maximum(pay_values, 0)  # clamp âm về 0
    df["r_weighted_delay"] = (delay_matrix * weights).sum(axis=1)

    # r_delay_trend: xu hướng trễ hạn từ tháng 4 đến tháng 9
    df["r_delay_trend"] = df["PAY_0"] - df["PAY_6"]

    # r_recent_delay: mức trễ tháng gần nhất
    df["r_recent_delay"] = df["PAY_0"]

    return df


# ==============================================================================
# FREQUENCY FEATURES (F)
# ==============================================================================

def _add_frequency_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Thêm các biến Frequency phản ánh tần suất trễ hạn trong 6 tháng:

    - f_total_delay_months  : Tổng số tháng có trễ hạn (PAY > 0)
    - f_max_delay           : Mức trễ cao nhất từng ghi nhận
    - f_full_pay_months     : Số tháng trả đầy đủ đúng hạn (PAY <= -1)
    - f_consecutive_delay   : Cờ trễ 2 tháng liên tiếp gần nhất (PAY_0>0 & PAY_2>0)

    Args:
        df (pd.DataFrame): DataFrame đầu vào.

    Returns:
        pd.DataFrame: DataFrame đã bổ sung cột f_*.
    """
    pay_values = df[PAY_COLS].values

    # f_total_delay_months: số tháng trễ (PAY > 0)
    df["f_total_delay_months"] = (pay_values > 0).sum(axis=1)

    # f_max_delay: mức trễ lớn nhất
    df["f_max_delay"] = pay_values.max(axis=1)

    # f_full_pay_months: số tháng trả đủ đúng hạn (PAY <= -1 nghĩa là trả đủ)
    df["f_full_pay_months"] = (pay_values <= -1).sum(axis=1)

    # f_consecutive_delay: trễ hạn 2 tháng liên tiếp gần nhất (T9 & T8)
    # PAY_0 = tháng 9, PAY_2 = tháng 8 (trong UCI dataset)
    df["f_consecutive_delay"] = (
        (df["PAY_0"] > 0) & (df["PAY_2"] > 0)
    ).astype(int)

    return df


# ==============================================================================
# MONETARY & FINANCIAL RATIO FEATURES (M)
# ==============================================================================

def _add_monetary_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Thêm các biến Monetary & chỉ số tài chính:

    - m_total_bill_amt   : Tổng dư nợ sao kê 6 tháng
    - m_total_pay_amt    : Tổng số tiền đã thanh toán 6 tháng
    - m_net_debt         : Dư nợ ròng còn lại (total_bill - total_pay)
    - m_avg_utilization  : Tỉ lệ sử dụng hạn mức trung bình (avg_bill / LIMIT_BAL)
    - m_repayment_ratio  : Tỉ lệ trả nợ / dư nợ (total_pay / (total_bill + 1))
    - m_bill_growth      : Tốc độ tăng dư nợ so với hạn mức [(BILL1 - BILL6) / LIMIT_BAL]

    ⚠️  Mọi phép chia đều được bảo vệ bởi EPSILON hoặc + 1 để tránh ZeroDivisionError.

    Args:
        df (pd.DataFrame): DataFrame đầu vào.

    Returns:
        pd.DataFrame: DataFrame đã bổ sung cột m_*.
    """
    bill_values = df[BILL_COLS].values
    pay_amt_values = df[PAY_AMT_COLS].values
    limit_bal = df["LIMIT_BAL"].values

    # m_total_bill_amt & m_total_pay_amt
    df["m_total_bill_amt"] = bill_values.sum(axis=1)
    df["m_total_pay_amt"] = pay_amt_values.sum(axis=1)

    # m_net_debt
    df["m_net_debt"] = df["m_total_bill_amt"] - df["m_total_pay_amt"]

    # m_avg_utilization: mean(BILL_AMT) / LIMIT_BAL  [clip trong [0, 2] vì có thể > 100%]
    avg_bill = bill_values.mean(axis=1)
    safe_limit = np.where(limit_bal <= 0, EPSILON, limit_bal)
    df["m_avg_utilization"] = np.clip(avg_bill / safe_limit, 0, 2)

    # m_repayment_ratio: total_pay / (total_bill + 1)
    df["m_repayment_ratio"] = df["m_total_pay_amt"] / (df["m_total_bill_amt"] + 1)

    # m_bill_growth: (BILL_AMT1 - BILL_AMT6) / LIMIT_BAL
    bill_growth_raw = df["BILL_AMT1"] - df["BILL_AMT6"]
    df["m_bill_growth"] = bill_growth_raw / (safe_limit)

    return df


# ==============================================================================
# MASTER FEATURE ENGINEERING PIPELINE
# ==============================================================================

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hàm tổng hợp: áp dụng toàn bộ quy trình tạo biến RFM & tài chính.

    Gọi lần lượt:
        1. _add_recency_features   (r_*)
        2. _add_frequency_features (f_*)
        3. _add_monetary_features  (m_*)

    Args:
        df (pd.DataFrame): DataFrame đầu vào (đã làm sạch).

    Returns:
        pd.DataFrame: DataFrame với các cột đặc trưng mới được bổ sung.

    Example:
        >>> df_raw = load_data('data/UCI_Credit_Card.csv')
        >>> df_clean = clean_raw_data(df_raw)
        >>> df_features = build_features(df_clean)
    """
    df = df.copy()

    # Áp dụng từng nhóm đặc trưng
    df = _add_recency_features(df)
    df = _add_frequency_features(df)
    df = _add_monetary_features(df)

    # Đặc trưng mới được tạo
    new_features = [
        # Recency
        "r_weighted_delay", "r_delay_trend", "r_recent_delay",
        # Frequency
        "f_total_delay_months", "f_max_delay", "f_full_pay_months", "f_consecutive_delay",
        # Monetary
        "m_total_bill_amt", "m_total_pay_amt", "m_net_debt",
        "m_avg_utilization", "m_repayment_ratio", "m_bill_growth",
    ]

    print(f"✅ Feature Engineering hoàn tất: +{len(new_features)} biến RFM & tài chính mới.")
    print(f"   Tổng số cột sau engineering: {df.shape[1]}")
    return df


def get_engineered_feature_names() -> list:
    """
    Trả về danh sách tên các biến được tạo bởi feature engineering.

    Returns:
        list: Danh sách tên cột đặc trưng RFM.
    """
    return [
        "r_weighted_delay", "r_delay_trend", "r_recent_delay",
        "f_total_delay_months", "f_max_delay", "f_full_pay_months", "f_consecutive_delay",
        "m_total_bill_amt", "m_total_pay_amt", "m_net_debt",
        "m_avg_utilization", "m_repayment_ratio", "m_bill_growth",
    ]
