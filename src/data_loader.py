"""
src/data_loader.py
==================
Module nạp và kiểm tra chất lượng dữ liệu tín dụng.

Functions:
    - load_data              : Đọc file CSV và in thông tin cơ bản
    - check_data_quality     : Kiểm tra missing values, duplicates, dtypes
    - get_target_distribution: Phân phối biến mục tiêu (0 / 1)
"""

import pandas as pd
import numpy as np
from typing import Tuple


def load_data(file_path: str) -> pd.DataFrame:
    """
    Đọc tập dữ liệu UCI Credit Card từ file CSV.

    Args:
        file_path (str): Đường dẫn đến file CSV.

    Returns:
        pd.DataFrame: DataFrame chứa toàn bộ dữ liệu thô.

    Raises:
        FileNotFoundError: Nếu file không tồn tại.
    """
    print(f"📂 Đang nạp dữ liệu từ: {file_path}")
    df = pd.read_csv(file_path)
    print(f"✅ Nạp thành công! Kích thước: {df.shape[0]:,} dòng × {df.shape[1]} cột")
    return df


def check_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kiểm tra toàn diện chất lượng tập dữ liệu:
    missing values, duplicates, kiểu dữ liệu.

    Args:
        df (pd.DataFrame): DataFrame cần kiểm tra.

    Returns:
        pd.DataFrame: Bảng tóm tắt missing values theo cột.
    """
    print("\n" + "=" * 60)
    print("📊 KIỂM TRA CHẤT LƯỢNG DỮ LIỆU")
    print("=" * 60)

    # --- Shape ---
    print(f"\n🔢 Kích thước dữ liệu: {df.shape[0]:,} dòng × {df.shape[1]} cột")

    # --- Missing values ---
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_summary = pd.DataFrame({
        "Missing Count": missing,
        "Missing (%)": missing_pct
    })
    missing_cols = missing_summary[missing_summary["Missing Count"] > 0]

    if len(missing_cols) == 0:
        print("\n✅ Không có missing values!")
    else:
        print(f"\n⚠️  Có {len(missing_cols)} cột có missing values:")
        print(missing_cols.to_string())

    # --- Duplicates ---
    n_dup = df.duplicated().sum()
    if n_dup == 0:
        print("\n✅ Không có hàng trùng lặp (duplicates)!")
    else:
        print(f"\n⚠️  Có {n_dup:,} hàng trùng lặp!")

    # --- Data types ---
    print("\n📋 Kiểu dữ liệu từng cột:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"   {dtype}: {count} cột")

    # --- Basic stats ---
    print(f"\n📈 Thống kê cơ bản (numeric):")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(df[numeric_cols].describe().round(2).to_string())

    print("\n" + "=" * 60)
    return missing_summary


def get_target_distribution(
    df: pd.DataFrame,
    target_col: str = "default.payment.next.month"
) -> pd.DataFrame:
    """
    Xuất phân phối biến mục tiêu (0 / 1) với số lượng và tỉ lệ phần trăm.

    Args:
        df          (pd.DataFrame): DataFrame chứa biến mục tiêu.
        target_col  (str)         : Tên cột mục tiêu. Mặc định: 'default.payment.next.month'.

    Returns:
        pd.DataFrame: Bảng phân phối lớp mục tiêu.
    """
    print(f"\n📊 Phân phối biến mục tiêu: '{target_col}'")
    counts = df[target_col].value_counts().sort_index()
    pct = (counts / len(df) * 100).round(2)

    dist = pd.DataFrame({
        "Nhãn": ["0 - Không vỡ nợ", "1 - Vỡ nợ (Default)"],
        "Số lượng": counts.values,
        "Tỉ lệ (%)": pct.values
    })

    print(dist.to_string(index=False))
    print(f"\n📌 Tỉ lệ mất cân bằng lớp: {pct.iloc[1]:.2f}% positive class (vỡ nợ)")
    return dist
