"""
src/models.py
=============
Module khởi tạo mô hình, Cross-Validation và Hyperparameter Tuning.

Functions:
    - get_baseline_models       : Dictionary các mô hình baseline
    - train_cross_validation    : 5-Fold Stratified CV với nhiều metrics
    - tune_xgboost_lightgbm     : Tinh chỉnh siêu tham số XGBoost & LightGBM
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import (
    StratifiedKFold, cross_validate, RandomizedSearchCV, GridSearchCV
)
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def get_baseline_models(scale_pos_weight: float = 3.5) -> Dict[str, Any]:
    """
    Trả về dictionary các mô hình baseline đã được cấu hình xử lý mất cân bằng lớp.

    Tỉ lệ scale_pos_weight cho XGBoost = n_negative / n_positive
    Với UCI dataset: 23364 / 6636 ≈ 3.52

    Args:
        scale_pos_weight (float): Tỉ lệ trọng số âm/dương cho XGBoost.

    Returns:
        Dict[str, Any]: Dictionary {tên model: đối tượng model}.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=0.1,
            solver="lbfgs",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            subsample=0.8,
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
            verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            class_weight="balanced",
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        ),
    }
    print(f"✅ Đã khởi tạo {len(models)} mô hình baseline.")
    return models


def train_cross_validation(
    models_dict: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 5,
) -> pd.DataFrame:
    """
    Chạy Stratified K-Fold Cross-Validation cho tất cả mô hình.

    Args:
        models_dict (Dict[str, Any]) : Dictionary {tên: model}.
        X_train     (np.ndarray)     : Ma trận đặc trưng training.
        y_train     (np.ndarray)     : Nhãn training.
        cv          (int)            : Số fold (mặc định 5).

    Returns:
        pd.DataFrame: Bảng tổng hợp CV scores trung bình và độ lệch chuẩn.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    scoring = {
        "roc_auc": "roc_auc",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
    }

    results = []
    print(f"\n{'='*65}")
    print(f"🔄 STRATIFIED {cv}-FOLD CROSS-VALIDATION")
    print(f"{'='*65}")

    for name, model in models_dict.items():
        print(f"\n⏳ Đang CV: {name}...")
        cv_results = cross_validate(
            model, X_train, y_train,
            cv=skf, scoring=scoring,
            n_jobs=-1, return_train_score=False
        )

        row = {"Model": name}
        for metric in scoring.keys():
            scores = cv_results[f"test_{metric}"]
            row[f"{metric.upper()} Mean"] = scores.mean().round(4)
            row[f"{metric.upper()} Std"] = scores.std().round(4)

        results.append(row)
        print(f"   ✅ {name}: ROC-AUC = {row['ROC_AUC Mean']:.4f} ± {row['ROC_AUC Std']:.4f}  |  "
              f"F1 = {row['F1 Mean']:.4f} ± {row['F1 Std']:.4f}")

    cv_df = pd.DataFrame(results).set_index("Model")
    print(f"\n{'='*65}")
    print("📊 BẢNG TỔNG HỢP CV SCORES:")
    print(f"{'='*65}")
    print(cv_df[["ROC_AUC Mean", "ROC_AUC Std", "F1 Mean", "F1 Std",
                 "PRECISION Mean", "RECALL Mean"]].to_string())
    return cv_df


def tune_xgboost_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str = "XGBoost",
    n_iter: int = 30,
    cv: int = 5,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Tinh chỉnh siêu tham số XGBoost hoặc LightGBM bằng RandomizedSearchCV.

    Args:
        X_train    (np.ndarray): Ma trận đặc trưng training.
        y_train    (np.ndarray): Nhãn training.
        model_name (str)       : 'XGBoost' hoặc 'LightGBM'.
        n_iter     (int)       : Số lần thử tham số (mặc định 30).
        cv         (int)       : Số fold cho CV nội bộ (mặc định 5).

    Returns:
        Tuple[best_estimator, best_params]:
            - best_estimator: mô hình tốt nhất sau tuning.
            - best_params   : dictionary các tham số tốt nhất.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    if model_name == "XGBoost":
        # Tính scale_pos_weight từ y_train
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        spw = neg / pos

        base_model = XGBClassifier(
            scale_pos_weight=spw,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
            verbosity=0,
        )
        param_dist = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 4, 5, 6, 7],
            "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "reg_alpha": [0, 0.01, 0.1, 0.5, 1.0],
            "reg_lambda": [1.0, 1.5, 2.0, 3.0],
        }

    elif model_name == "LightGBM":
        base_model = LGBMClassifier(
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )
        param_dist = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [4, 6, 8, 10, -1],
            "num_leaves": [15, 31, 63, 127],
            "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "reg_alpha": [0, 0.01, 0.1, 0.5, 1.0],
            "reg_lambda": [0, 0.01, 0.1, 1.0],
        }
    else:
        raise ValueError(f"model_name phải là 'XGBoost' hoặc 'LightGBM', nhận được: '{model_name}'")

    print(f"\n🔧 Hyperparameter Tuning: {model_name} (n_iter={n_iter}, cv={cv})")
    print("   ⏳ Đang tìm kiếm tham số tốt nhất... (có thể mất vài phút)")

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=skf,
        n_jobs=-1,
        random_state=42,
        verbose=0,
        refit=True,
    )
    search.fit(X_train, y_train)

    best_params = search.best_params_
    best_score = search.best_score_

    print(f"\n✅ Kết quả Tuning {model_name}:")
    print(f"   Best CV ROC-AUC: {best_score:.4f}")
    print(f"   Best Params:")
    for k, v in sorted(best_params.items()):
        print(f"     {k}: {v}")

    return search.best_estimator_, best_params
