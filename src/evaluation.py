"""
src/evaluation.py
=================
Module đánh giá và so sánh các mô hình trên tập test.

Functions:
    - evaluate_all_models      : Bảng tổng hợp metrics trên test set
    - plot_roc_and_pr_curves   : ROC Curve & Precision-Recall Curve
    - plot_confusion_matrices  : Ma trận nhầm lẫn cho các mô hình
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import Dict, Any

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
    confusion_matrix,
    classification_report,
)

# Bảng màu nhất quán với visualization.py
MODEL_COLORS = [
    "#4361EE",  # Logistic Regression
    "#F72585",  # Random Forest
    "#7209B7",  # Gradient Boosting
    "#3A86FF",  # XGBoost
    "#06D6A0",  # LightGBM
]


def evaluate_all_models(
    models_dict: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Đánh giá tất cả mô hình trên tập test, tính toán đầy đủ các metrics.

    Args:
        models_dict (Dict[str, Any]) : Dictionary {tên model: đối tượng model đã fit}.
        X_test      (np.ndarray)     : Ma trận đặc trưng test.
        y_test      (np.ndarray)     : Nhãn test.
        threshold   (float)          : Ngưỡng phân loại (mặc định 0.5).

    Returns:
        pd.DataFrame: Bảng tổng hợp metrics với các cột:
                      Accuracy, Precision, Recall, F1-Score, ROC-AUC, Gini, PR-AUC.
    """
    print(f"\n{'='*70}")
    print(f"📊 ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST (threshold={threshold})")
    print(f"{'='*70}")

    results = []
    for name, model in models_dict.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Tính toán metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        gini = 2 * roc_auc - 1

        # PR-AUC
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall_vals, precision_vals)

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(roc_auc, 4),
            "Gini Index": round(gini, 4),
            "PR-AUC": round(pr_auc, 4),
        })
        print(f"\n✅ {name}:")
        print(f"   Accuracy={acc:.4f} | Precision={prec:.4f} | Recall={rec:.4f} | F1={f1:.4f}")
        print(f"   ROC-AUC={roc_auc:.4f} | Gini={gini:.4f} | PR-AUC={pr_auc:.4f}")

    eval_df = pd.DataFrame(results).set_index("Model")

    print(f"\n{'='*70}")
    print("📋 BẢNG TỔNG HỢP KẾT QUẢ:")
    print(f"{'='*70}")
    print(eval_df.to_string())

    # Tô đậm model tốt nhất theo ROC-AUC
    best_model = eval_df["ROC-AUC"].idxmax()
    print(f"\n🏆 Mô hình tốt nhất (ROC-AUC): {best_model} ({eval_df.loc[best_model, 'ROC-AUC']:.4f})")

    return eval_df


def plot_roc_and_pr_curves(
    models_dict: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    figsize: tuple = (15, 6),
) -> None:
    """
    Vẽ đường cong ROC Curve và Precision-Recall Curve cho tất cả mô hình.

    Args:
        models_dict (Dict[str, Any]) : Dictionary {tên model: đối tượng model đã fit}.
        X_test      (np.ndarray)     : Ma trận đặc trưng test.
        y_test      (np.ndarray)     : Nhãn test.
        figsize     (tuple)          : Kích thước figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle("Đường cong Đánh giá Mô hình: ROC Curve & Precision-Recall Curve",
                 fontsize=14, fontweight="bold")

    colors = MODEL_COLORS
    lw = 2.2

    for (name, model), color in zip(models_dict.items(), colors):
        y_proba = model.predict_proba(X_test)[:, 1]

        # --- ROC Curve ---
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        axes[0].plot(fpr, tpr, color=color, lw=lw,
                     label=f"{name} (AUC = {roc_auc:.4f})")

        # --- PR Curve ---
        prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(rec_vals, prec_vals)
        axes[1].plot(rec_vals, prec_vals, color=color, lw=lw,
                     label=f"{name} (PR-AUC = {pr_auc:.4f})")

    # --- ROC panel ---
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier (AUC = 0.5)")
    axes[0].set_xlabel("False Positive Rate (FPR)", fontsize=11)
    axes[0].set_ylabel("True Positive Rate (TPR / Recall)", fontsize=11)
    axes[0].set_title("ROC Curve", fontsize=12, fontweight="bold")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1.02])
    axes[0].grid(True, alpha=0.4)

    # --- PR panel ---
    baseline_pr = y_test.mean()
    axes[1].axhline(y=baseline_pr, color="gray", linestyle="--", lw=1,
                    label=f"Random Classifier (PR = {baseline_pr:.3f})")
    axes[1].set_xlabel("Recall", fontsize=11)
    axes[1].set_ylabel("Precision", fontsize=11)
    axes[1].set_title("Precision-Recall Curve", fontsize=12, fontweight="bold")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1.05])
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(
    models_dict: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
    figsize: tuple = (18, 10),
) -> None:
    """
    Vẽ ma trận nhầm lẫn (Confusion Matrix) cho tất cả mô hình.

    Args:
        models_dict (Dict[str, Any]) : Dictionary {tên model: đối tượng model đã fit}.
        X_test      (np.ndarray)     : Ma trận đặc trưng test.
        y_test      (np.ndarray)     : Nhãn test.
        threshold   (float)          : Ngưỡng phân loại (mặc định 0.5).
        figsize     (tuple)          : Kích thước figure.
    """
    n_models = len(models_dict)
    cols = min(n_models, 3)
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if n_models > 1 else [axes]

    fig.suptitle("Ma trận Nhầm lẫn (Confusion Matrix) - Tập Test",
                 fontsize=14, fontweight="bold")

    class_labels = ["Không vỡ nợ (0)", "Vỡ nợ (1)"]
    cmap = sns.color_palette("Blues", as_cmap=True)

    for idx, (name, model) in enumerate(models_dict.items()):
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        cm = confusion_matrix(y_test, y_pred)
        cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        # Tạo annotations: count + %
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm[i, j]:,}\n({cm_normalized[i, j]:.1f}%)"

        ax = axes[idx]
        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap=cmap,
            ax=ax,
            xticklabels=class_labels,
            yticklabels=class_labels,
            linewidths=1,
            linecolor="white",
            annot_kws={"size": 10, "weight": "bold"},
            cbar=False,
        )
        ax.set_title(name, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Nhãn Dự đoán", fontsize=9)
        ax.set_ylabel("Nhãn Thực tế", fontsize=9)
        ax.tick_params(labelsize=8)

    # Ẩn các subplot thừa
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.show()


def print_classification_report(
    models_dict: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> None:
    """
    In báo cáo phân loại chi tiết cho từng mô hình.

    Args:
        models_dict (Dict[str, Any]) : Dictionary {tên model: đối tượng model đã fit}.
        X_test      (np.ndarray)     : Ma trận đặc trưng test.
        y_test      (np.ndarray)     : Nhãn test.
        threshold   (float)          : Ngưỡng phân loại.
    """
    for name, model in models_dict.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        print(f"\n{'='*60}")
        print(f"📋 Classification Report: {name}")
        print(f"{'='*60}")
        print(classification_report(y_test, y_pred,
                                    target_names=["Không vỡ nợ (0)", "Vỡ nợ (1)"]))
