"""
Credit Scoring & Default Prediction - src package
==================================================
Trường Đại học Kinh tế Quốc dân (NEU)
Môn học: Machine Learning - Bài tập giữa kỳ
Đề tài: Dự đoán Điểm tín dụng & Xác suất vỡ nợ thẻ tín dụng
Nhóm 4:
    - Nguyễn Bảo Khánh  (MSV: 11247178)
    - Nguyễn Phạm Tuân  (MSV: 11247242)
    - Lớp: DS66B

Package structure:
    - data_loader        : Nạp và kiểm tra chất lượng dữ liệu
    - preprocessing      : Làm sạch, chia tập, tiền xử lý
    - feature_engineering: Tạo biến RFM & chỉ số tài chính
    - visualization      : Trực quan hóa EDA & kết quả mô hình
    - models             : Khởi tạo mô hình & Cross-Validation
    - evaluation         : Đánh giá & so sánh mô hình
    - credit_scoring     : Hệ thống chấm điểm tín dụng FICO
"""

from .data_loader import load_data, check_data_quality, get_target_distribution
from .preprocessing import clean_raw_data, split_data, get_preprocessor
from .feature_engineering import build_features
from .visualization import (
    plot_target_distribution,
    plot_correlation_heatmap,
    plot_delinquency_vs_default,
    plot_feature_importance,
)
from .models import get_baseline_models, train_cross_validation, tune_xgboost_lightgbm
from .evaluation import evaluate_all_models, plot_roc_and_pr_curves, plot_confusion_matrices
from .credit_scoring import prob_to_fico_score, get_credit_rating, score_new_customer

__all__ = [
    # data_loader
    "load_data",
    "check_data_quality",
    "get_target_distribution",
    # preprocessing
    "clean_raw_data",
    "split_data",
    "get_preprocessor",
    # feature_engineering
    "build_features",
    # visualization
    "plot_target_distribution",
    "plot_correlation_heatmap",
    "plot_delinquency_vs_default",
    "plot_feature_importance",
    # models
    "get_baseline_models",
    "train_cross_validation",
    "tune_xgboost_lightgbm",
    # evaluation
    "evaluate_all_models",
    "plot_roc_and_pr_curves",
    "plot_confusion_matrices",
    # credit_scoring
    "prob_to_fico_score",
    "get_credit_rating",
    "score_new_customer",
]
