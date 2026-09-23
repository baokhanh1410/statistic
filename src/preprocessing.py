"""
src/preprocessing.py
====================
Module tiền xử lý dữ liệu tín dụng.

Bao gồm: làm sạch dị biệt phân loại, tách train/test (stratified),
xây dựng ColumnTransformer an toàn (không data leakage).

Functions:
    - clean_raw_data   : Loại ID, gom nhóm EDUCATION & MARRIAGE
    - split_data       : Chia X/y train/test với stratify
    - get_preprocessor : Tạo ColumnTransformer (RobustScaler + OHE)
"""

import pandas as pd
import numpy as np
from typing import Tuple, List

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Làm sạch dữ liệu thô:
    1. Loại bỏ cột 'ID' (định danh, không có giá trị dự đoán).
    2. Gom nhóm EDUCATION: {0, 4, 5, 6} -> 4 (Khác/Other).
    3. Gom nhóm MARRIAGE:  {0, 3}       -> 3 (Khác/Other).

    Args:
        df (pd.DataFrame): DataFrame gốc sau khi nạp.

    Returns:
        pd.DataFrame: DataFrame đã làm sạch (copy).
    """
    df = df.copy()

    # 1. Loại bỏ cột ID
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
        print("✅ Đã loại bỏ cột 'ID'.")

    # 2. Làm sạch EDUCATION: các giá trị 0, 4, 5, 6 -> 4 (Other)
    if "EDUCATION" in df.columns:
        invalid_edu = [0, 4, 5, 6]
        n_edu = df["EDUCATION"].isin(invalid_edu).sum()
        df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
        print(f"✅ EDUCATION: gom {n_edu:,} giá trị dị biệt {{0,4,5,6}} → 4 (Khác).")

    # 3. Làm sạch MARRIAGE: các giá trị 0, 3 -> 3 (Other)
    if "MARRIAGE" in df.columns:
        invalid_mar = [0, 3]
        n_mar = df["MARRIAGE"].isin(invalid_mar).sum()
        df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
        print(f"✅ MARRIAGE: gom {n_mar:,} giá trị dị biệt {{0,3}} → 3 (Khác).")

    print(f"✅ Dữ liệu sau làm sạch: {df.shape[0]:,} dòng × {df.shape[1]} cột.")
    return df


def split_data(
    df: pd.DataFrame,
    target_col: str = "default.payment.next.month",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Tách dữ liệu thành tập train và test với stratified sampling.

    Args:
        df           (pd.DataFrame): DataFrame đầu vào đã làm sạch.
        target_col   (str)         : Tên cột mục tiêu.
        test_size    (float)       : Tỉ lệ tập test (mặc định 0.2).
        random_state (int)         : Seed tái lặp (mặc định 42).

    Returns:
        Tuple[X_train, X_test, y_train, y_test]:
            X_train, X_test: features DataFrames
            y_train, y_test: target Series
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"\n✅ Chia tập dữ liệu (stratified, test_size={test_size}):")
    print(f"   Train: {X_train.shape[0]:,} mẫu  |  Test: {X_test.shape[0]:,} mẫu")
    print(f"   Tỉ lệ vỡ nợ Train: {y_train.mean():.4f}  |  Test: {y_test.mean():.4f}")
    return X_train, X_test, y_train, y_test


def get_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> ColumnTransformer:
    """
    Xây dựng scikit-learn ColumnTransformer tiền xử lý:
    - RobustScaler cho biến số (ít nhạy cảm với outliers).
    - OneHotEncoder cho biến phân loại.

    ⚠️  CHỈ fit trên tập TRAIN để tránh Data Leakage!

    Args:
        numeric_cols     (List[str]): Danh sách cột số.
        categorical_cols (List[str]): Danh sách cột phân loại.

    Returns:
        ColumnTransformer: Bộ tiền xử lý chưa được fit.
    """
    numeric_transformer = Pipeline(steps=[
        ("scaler", RobustScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    print(f"✅ Preprocessor đã tạo:")
    print(f"   Numeric  ({len(numeric_cols)} cột): RobustScaler")
    print(f"   Category ({len(categorical_cols)} cột): OneHotEncoder")
    return preprocessor


def get_feature_columns(df: pd.DataFrame, target_col: str = "default.payment.next.month") -> Tuple[List[str], List[str]]:
    """
    Tự động phân loại cột số và cột phân loại (sau khi đã loại target & ID).

    Args:
        df         (pd.DataFrame): DataFrame đã làm sạch.
        target_col (str)         : Tên cột mục tiêu (sẽ loại khỏi kết quả).

    Returns:
        Tuple[numeric_cols, categorical_cols]: Hai danh sách tên cột.
    """
    exclude = [target_col, "ID"] if "ID" in df.columns else [target_col]
    feature_df = df.drop(columns=[c for c in exclude if c in df.columns])

    # Phân loại theo kiểu dữ liệu gốc TRƯỚC khi đặt tên feature engineering
    # Biến phân loại: SEX, EDUCATION, MARRIAGE (tuy là số nguyên nhưng là categorical)
    categorical_cols = ["SEX", "EDUCATION", "MARRIAGE"]
    categorical_cols = [c for c in categorical_cols if c in feature_df.columns]
    numeric_cols = [c for c in feature_df.columns if c not in categorical_cols]

    return numeric_cols, categorical_cols
