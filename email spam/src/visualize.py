"""Visualization module for SMS/Email Spam Classifier.

Generates polished, publication-grade figures saved into /outputs:
1. Class distribution (Ham vs Spam counts and percentages)
2. Message length and word count distributions
3. Model comparison metrics bar chart
4. Confusion matrices for all models
5. ROC curves with AUC comparison
6. Top 20 spam-indicative features/words
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set visual styling
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10


def plot_class_distribution(df: pd.DataFrame, output_path: str) -> None:
    """Plot class balance between Ham (0) and Spam (1)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))

    counts = df["label"].value_counts().rename({0: "Ham (Legitimate)", 1: "Spam"})
    total = len(df)
    palette = ["#2b6cb0", "#e53e3e"]

    bars = ax.bar(counts.index, counts.values, color=palette, width=0.5, edgecolor="#1a202c", linewidth=1.2)
    ax.set_title("SMS/Email Dataset: Class Distribution (Imbalance)", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("Message Count", fontsize=11)
    ax.set_ylim(0, max(counts.values) * 1.15)

    for bar, count in zip(bars, counts.values):
        pct = (count / total) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            count + (max(counts.values) * 0.02),
            f"{count:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    sns.despine(top=True, right=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Visualizer] Saved class distribution plot to: {output_path}")


def plot_message_length_distribution(df: pd.DataFrame, output_path: str) -> None:
    """Plot character and word count distributions for Ham vs Spam."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ham_df = df[df["label"] == 0]
    spam_df = df[df["label"] == 1]

    # Character length plot (capped at 400 for clean visualization)
    sns.histplot(
        ham_df["char_length"].clip(upper=350),
        color="#2b6cb0",
        label=f"Ham (Mean: {ham_df['char_length'].mean():.0f})",
        kde=True,
        ax=ax1,
        bins=35,
        alpha=0.6,
        stat="density",
    )
    sns.histplot(
        spam_df["char_length"].clip(upper=350),
        color="#e53e3e",
        label=f"Spam (Mean: {spam_df['char_length'].mean():.0f})",
        kde=True,
        ax=ax1,
        bins=35,
        alpha=0.6,
        stat="density",
    )
    ax1.set_title("Character Length Distribution", fontweight="bold", fontsize=12)
    ax1.set_xlabel("Characters per Message")
    ax1.set_ylabel("Density")
    ax1.legend(loc="upper right")

    # Word count plot (capped at 70 for clean visualization)
    sns.histplot(
        ham_df["word_count"].clip(upper=60),
        color="#2b6cb0",
        label=f"Ham (Mean: {ham_df['word_count'].mean():.1f})",
        kde=True,
        ax=ax2,
        bins=30,
        alpha=0.6,
        stat="density",
    )
    sns.histplot(
        spam_df["word_count"].clip(upper=60),
        color="#e53e3e",
        label=f"Spam (Mean: {spam_df['word_count'].mean():.1f})",
        kde=True,
        ax=ax2,
        bins=30,
        alpha=0.6,
        stat="density",
    )
    ax2.set_title("Word Count Distribution", fontweight="bold", fontsize=12)
    ax2.set_xlabel("Words per Message")
    ax2.set_ylabel("Density")
    ax2.legend(loc="upper right")

    plt.suptitle("Exploratory Data Analysis: Message Length Comparison", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Visualizer] Saved message length plot to: {output_path}")


def plot_confusion_matrices(cm_dict: dict, output_path: str) -> None:
    """Plot confusion matrices for each trained model side by side.

    cm_dict format: {model_name: confusion_matrix_2x2}
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    n_models = len(cm_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 4.5))

    if n_models == 1:
        axes = [axes]

    class_names = ["Ham (0)", "Spam (1)"]

    for ax, (name, cm) in zip(axes, cm_dict.items()):
        total = np.sum(cm)
        annot = np.empty_like(cm, dtype=object)
        for i in range(2):
            for j in range(2):
                pct = (cm[i, j] / total) * 100
                annot[i, j] = f"{cm[i, j]:,}\n({pct:.1f}%)"

        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap="Blues",
            cbar=False,
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            linewidths=1.5,
            linecolor="white",
            annot_kws={"size": 11, "weight": "bold"},
        )
        ax.set_title(f"{name}\nConfusion Matrix", fontweight="bold", fontsize=12)
        ax.set_xlabel("Predicted Label", fontweight="semibold")
        ax.set_ylabel("True Label", fontweight="semibold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Visualizer] Saved confusion matrices to: {output_path}")


def plot_model_comparison(metrics_df: pd.DataFrame, output_path: str) -> None:
    """Plot grouped bar chart comparing performance metrics across models.

    metrics_df contains columns: ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    melted = metrics_df.melt(
        id_vars=["Model"],
        value_vars=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
        var_name="Metric",
        value_name="Score",
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=melted,
        x="Metric",
        y="Score",
        hue="Model",
        palette="viridis",
        edgecolor="#1a202c",
        linewidth=0.8,
        ax=ax,
    )

    ax.set_title("Model Performance Comparison (Evaluation on Test Set)", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11)
    ax.set_ylim(0.80, 1.02)  # Focus in on top discrimination range
    ax.legend(title="Model", frameon=True, loc="lower right")

    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(
                f"{h:.3f}",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=45,
                xytext=(0, 3),
                textcoords="offset points",
            )

    sns.despine(top=True, right=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Visualizer] Saved model comparison chart to: {output_path}")


def plot_top_spam_words(
    features: list,
    weights: list,
    output_path: str,
    title: str = "Top 20 Spam-Indicating Words & N-Grams",
    top_n: int = 20
) -> None:
    """Plot horizontal bar chart of top predictive features for spam."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))

    # Take top N
    top_features = features[:top_n][::-1]
    top_weights = weights[:top_n][::-1]

    colors = sns.color_palette("Reds_r", n_colors=top_n)

    bars = ax.barh(top_features, top_weights, color=colors, edgecolor="#4a0e17", linewidth=0.8)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Relative Feature Importance / Coefficient", fontsize=11)
    ax.set_ylabel("Feature / Token", fontsize=11)

    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + (max(top_weights) * 0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{w:.2f}",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="semibold",
        )

    ax.set_xlim(0, max(top_weights) * 1.15)
    sns.despine(top=True, right=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Visualizer] Saved top spam words plot to: {output_path}")


def plot_roc_curves(roc_data: dict, output_path: str) -> None:
    """Plot ROC curves for models.

    roc_data: {model_name: (fpr, tpr, roc_auc)}
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["#2b6cb0", "#38a169", "#d69e2e", "#805ad5"]

    for (name, (fpr, tpr, auc_val)), c in zip(roc_data.items(), colors):
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.4f})", linewidth=2.2, color=c)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance (AUC = 0.5000)", linewidth=1.5)
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True)

    sns.despine(top=True, right=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Visualizer] Saved ROC curve to: {output_path}")
