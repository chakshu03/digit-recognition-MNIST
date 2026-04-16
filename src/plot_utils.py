from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def save_history_plot(history_df: pd.DataFrame, output_path: str | Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history_df["epoch"], history_df["accuracy"], label="Train Accuracy", linewidth=2)
    axes[0].plot(history_df["epoch"], history_df["val_accuracy"], label="Validation Accuracy", linewidth=2, linestyle="--")
    axes[0].set_title("CNN Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history_df["epoch"], history_df["loss"], label="Train Loss", linewidth=2, color="orange")
    axes[1].plot(history_df["epoch"], history_df["val_loss"], label="Validation Loss", linewidth=2, linestyle="--", color="red")
    axes[1].set_title("CNN Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix_plot(y_true, y_pred, output_path: str | Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_classification_report_df(y_true, y_pred) -> pd.DataFrame:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return pd.DataFrame(report).transpose().round(4)


def save_misclassified_plot(images, y_true, y_pred, output_path: str | Path, max_items: int = 16) -> int:
    wrong_indices = [index for index, (truth, pred) in enumerate(zip(y_true, y_pred)) if truth != pred]
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))

    for axis_index, ax in enumerate(axes.flatten()):
        if axis_index >= min(max_items, len(wrong_indices)):
            ax.axis("off")
            continue
        image_index = wrong_indices[axis_index]
        ax.imshow(images[image_index].reshape(28, 28), cmap="gray")
        ax.set_title(f"T:{y_true[image_index]} P:{y_pred[image_index]}", fontsize=8, color="red")
        ax.axis("off")

    fig.suptitle("Misclassified Samples", fontsize=12, fontweight="bold")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return len(wrong_indices)


def save_feature_maps_plot(feature_maps, output_path: str | Path) -> None:
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(feature_maps[0, :, :, i], cmap="viridis")
        ax.axis("off")
    fig.suptitle("Conv1 Feature Maps", fontsize=13, fontweight="bold")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
