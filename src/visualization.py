"""
src/visualization.py
====================
Module trực quan hóa EDA & kết quả mô hình.

Functions:
    - plot_target_distribution    : Biểu đồ phân phối biến mục tiêu
    - plot_correlation_heatmap    : Heatmap tương quan
    - plot_delinquency_vs_default : Tỉ lệ nợ xấu theo số tháng trễ hạn
    - plot_feature_importance     : Biểu đồ cột ngang feature importance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import List, Optional, Any

# Bảng màu nhất quán toàn dự án
PALETTE = {
    "primary": "#4361EE",
    "secondary": "#F72585",
    "accent": "#3A86FF",
    "positive": "#06D6A0",
    "negative": "#EF233C",
    "neutral": "#8D99AE",
    "bg": "#1E2030",
    "bg_light": "#2D3250",
    "text": "#FFFFFF",
}

# Style theme chung
FIGSIZE_WIDE = (14, 6)
FIGSIZE_SQUARE = (10, 8)


def _setup_style() -> None:
    """Thiết lập style toàn cục cho matplotlib."""
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#F8F9FA",
        "axes.grid": True,
        "grid.alpha": 0.4,
        "font.family": "DejaVu Sans",
    })


def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str = "default.payment.next.month",
    figsize: tuple = (12, 5),
) -> None:
    """
    Vẽ biểu đồ thanh và biểu đồ tròn phân phối biến mục tiêu (0 vs 1).

    Args:
        df         (pd.DataFrame): DataFrame chứa biến mục tiêu.
        target_col (str)         : Tên cột mục tiêu.
        figsize    (tuple)       : Kích thước figure.
    """
    _setup_style()
    counts = df[target_col].value_counts().sort_index()
    labels = ["0 - Không vỡ nợ", "1 - Vỡ nợ (Default)"]
    colors = [PALETTE["positive"], PALETTE["negative"]]
    pcts = counts / counts.sum() * 100

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle("Phân phối Biến Mục tiêu: Tình trạng Vỡ nợ",
                 fontsize=15, fontweight="bold", y=1.02)

    # --- Bar chart ---
    bars = axes[0].bar(labels, counts.values, color=colors, edgecolor="white",
                       linewidth=1.5, width=0.5, alpha=0.9)
    axes[0].set_title("Số lượng mẫu theo nhãn", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Số lượng mẫu", fontsize=11)
    axes[0].set_xlabel("")
    for bar, count, pct in zip(bars, counts.values, pcts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                     f"{count:,}\n({pct:.1f}%)", ha="center", va="bottom",
                     fontsize=10, fontweight="bold")
    axes[0].set_ylim(0, max(counts.values) * 1.2)
    axes[0].tick_params(axis="x", labelsize=10)

    # --- Pie chart ---
    wedges, texts, autotexts = axes[1].pie(
        counts.values,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        explode=(0, 0.06),
        shadow=True,
        textprops={"fontsize": 10},
    )
    for at in autotexts:
        at.set_fontweight("bold")
        at.set_fontsize(11)
    axes[1].set_title("Tỉ lệ phần trăm", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.show()
    print(f"📊 Tỉ lệ mất cân bằng: {pcts.iloc[0]:.1f}% (0) vs {pcts.iloc[1]:.1f}% (1)")


def plot_correlation_heatmap(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    figsize: tuple = (14, 11),
    title: str = "Ma trận Tương quan",
) -> None:
    """
    Vẽ heatmap tương quan Pearson giữa các biến.

    Args:
        df     (pd.DataFrame)   : DataFrame đầu vào.
        cols   (List[str], opt) : Danh sách cột cần vẽ (None = tất cả numeric).
        figsize (tuple)         : Kích thước figure.
        title  (str)            : Tiêu đề biểu đồ.
    """
    _setup_style()
    if cols is None:
        cols = df.select_dtypes(include=np.number).columns.tolist()

    corr_matrix = df[cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    fig, ax = plt.subplots(figsize=figsize)
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        linewidths=0.5,
        linecolor="white",
        square=True,
        annot_kws={"size": 7},
        ax=ax,
        vmin=-1, vmax=1,
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    plt.tight_layout()
    plt.show()


def plot_delinquency_vs_default(
    df: pd.DataFrame,
    target_col: str = "default.payment.next.month",
    figsize: tuple = (12, 5),
) -> None:
    """
    Vẽ biểu đồ tỉ lệ vỡ nợ theo số tháng trễ hạn (f_total_delay_months).

    Args:
        df         (pd.DataFrame): DataFrame sau feature engineering.
        target_col (str)         : Tên cột mục tiêu.
        figsize    (tuple)       : Kích thước figure.
    """
    _setup_style()

    if "f_total_delay_months" not in df.columns:
        raise ValueError("Cần chạy build_features() trước để có cột 'f_total_delay_months'.")

    grouped = df.groupby("f_total_delay_months")[target_col].agg(["mean", "count"])
    grouped.columns = ["default_rate", "count"]
    grouped["default_rate_pct"] = grouped["default_rate"] * 100

    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    # Bar chart: số lượng mẫu
    bars = ax1.bar(
        grouped.index, grouped["count"],
        color=PALETTE["neutral"], alpha=0.5, label="Số lượng mẫu", width=0.6
    )
    ax1.set_xlabel("Số tháng trễ hạn (f_total_delay_months)", fontsize=11)
    ax1.set_ylabel("Số lượng mẫu", fontsize=11, color=PALETTE["neutral"])

    # Line chart: tỉ lệ vỡ nợ
    ax2.plot(
        grouped.index, grouped["default_rate_pct"],
        color=PALETTE["negative"], marker="o", linewidth=2.5,
        markersize=7, label="Tỉ lệ vỡ nợ (%)", zorder=5
    )
    ax2.set_ylabel("Tỉ lệ vỡ nợ (%)", fontsize=11, color=PALETTE["negative"])

    # Annotate points
    for x, y in zip(grouped.index, grouped["default_rate_pct"]):
        ax2.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=8,
                     color=PALETTE["negative"], fontweight="bold")

    ax1.set_title("Tỉ lệ Vỡ nợ theo Số tháng Trễ hạn", fontsize=14, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 15,
    figsize: tuple = (10, 7),
    title: str = "Top Feature Importance",
) -> None:
    """
    Vẽ biểu đồ cột ngang top N đặc trưng quan trọng nhất từ mô hình cây.

    Args:
        model        (Any)      : Mô hình đã được huấn luyện (cần thuộc tính feature_importances_).
        feature_names (List[str]): Tên các đặc trưng.
        top_n        (int)      : Số lượng đặc trưng cần hiển thị (mặc định 15).
        figsize      (tuple)    : Kích thước figure.
        title        (str)      : Tiêu đề biểu đồ.
    """
    _setup_style()

    # Lấy importances từ model
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        print("⚠️  Mô hình này không có thuộc tính feature_importances_ hoặc coef_.")
        return

    # Tạo Series và sắp xếp
    feat_series = pd.Series(importances, index=feature_names).sort_values(ascending=True)
    top_feats = feat_series.tail(top_n)

    # Tạo màu gradient
    n = len(top_feats)
    colors = [plt.cm.RdYlGn(i / n) for i in range(n)]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(range(n), top_feats.values, color=colors,
                   edgecolor="white", linewidth=0.5, height=0.7)

    ax.set_yticks(range(n))
    ax.set_yticklabels(top_feats.index, fontsize=9)
    ax.set_xlabel("Mức độ quan trọng (Feature Importance)", fontsize=11)
    ax.set_title(f"{title} (Top {top_n})", fontsize=13, fontweight="bold")

    # Annotate values
    for bar, val in zip(bars, top_feats.values):
        ax.text(bar.get_width() + max(top_feats.values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)

    ax.set_xlim(0, max(top_feats.values) * 1.15)
    plt.tight_layout()
    plt.show()
