from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score

from src.data_utils import normalize_cnn_images
from src.model import build_cnn_model
from src.plot_utils import (
    build_classification_report_df,
    save_confusion_matrix_plot,
    save_feature_maps_plot,
    save_history_plot,
    save_misclassified_plot,
)


PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "mnist_cnn_model.keras"
METADATA_PATH = ARTIFACTS_DIR / "training_metadata.json"
HISTORY_CSV_PATH = ARTIFACTS_DIR / "training_history.csv"
CLASSIFICATION_REPORT_PATH = ARTIFACTS_DIR / "classification_report.csv"
HISTORY_PLOT_PATH = ARTIFACTS_DIR / "history_plot.png"
CONFUSION_MATRIX_PATH = ARTIFACTS_DIR / "confusion_matrix.png"
MODEL_SUMMARY_PATH = ARTIFACTS_DIR / "model_summary.txt"
MISCLASSIFIED_PLOT_PATH = ARTIFACTS_DIR / "misclassified_samples.png"
FEATURE_MAPS_PATH = ARTIFACTS_DIR / "feature_maps.png"
SAMPLE_PREDICTION_PATH = ARTIFACTS_DIR / "sample_prediction.png"


PARAMETERS = {
    "dataset": "MNIST handwritten digits",
    "input_size": "28 x 28 grayscale image",
    "input_shape_for_cnn": [28, 28, 1],
    "train_split": "90% of training set for training",
    "validation_split": "10% of training set for validation",
    "test_split": "Original MNIST test set",
    "architecture": [
        "Conv2D(32, 3x3, relu, same)",
        "MaxPool(2x2)",
        "Conv2D(64, 3x3, relu, same)",
        "MaxPool(2x2)",
        "Flatten",
        "Dense(128, relu)",
        "Dropout(0.5)",
        "Dense(10, softmax)",
    ],
    "conv_blocks": 2,
    "output_classes": 10,
    "optimizer": "Adam",
    "batch_size": 128,
    "epochs": 10,
    "early_stopping_patience": 3,
    "loss_function": "categorical_crossentropy",
    "metrics": ["accuracy", "precision", "recall", "f1-score", "confusion matrix"],
}


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    tf.random.set_seed(42)
    np.random.seed(42)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = normalize_cnn_images(x_train)
    x_test = normalize_cnn_images(x_test)
    y_train_ohe = keras.utils.to_categorical(y_train, 10)
    y_test_ohe = keras.utils.to_categorical(y_test, 10)

    model = build_cnn_model(input_shape=x_train.shape[1:])
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PARAMETERS["early_stopping_patience"],
        restore_best_weights=True,
    )

    history = model.fit(
        x_train,
        y_train_ohe,
        validation_split=0.1,
        epochs=PARAMETERS["epochs"],
        batch_size=PARAMETERS["batch_size"],
        callbacks=[early_stop],
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test_ohe, verbose=0)
    test_probabilities = model.predict(x_test, verbose=0)
    predicted_labels = test_probabilities.argmax(axis=1)
    accuracy = float(accuracy_score(y_test, predicted_labels))

    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", range(1, len(history_df) + 1))
    history_df.to_csv(HISTORY_CSV_PATH, index=False)
    save_history_plot(history_df, HISTORY_PLOT_PATH)

    report_df = build_classification_report_df(y_test, predicted_labels)
    report_df.to_csv(CLASSIFICATION_REPORT_PATH, index_label="label")
    save_confusion_matrix_plot(y_test, predicted_labels, CONFUSION_MATRIX_PATH)
    misclassified_count = save_misclassified_plot(x_test, y_test, predicted_labels, MISCLASSIFIED_PLOT_PATH)

    feature_model = keras.Model(inputs=model.inputs, outputs=model.get_layer("conv1").output)
    feature_maps = feature_model.predict(x_test[0:1], verbose=0)
    save_feature_maps_plot(feature_maps, FEATURE_MAPS_PATH)

    sample_index = 0
    sample_prediction = int(predicted_labels[sample_index])
    sample_truth = int(y_test[sample_index])
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(x_test[sample_index].reshape(28, 28), cmap="gray")
    title_color = "green" if sample_prediction == sample_truth else "red"
    ax.set_title(f"True: {sample_truth} | Predicted: {sample_prediction}", color=title_color, fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(SAMPLE_PREDICTION_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary_lines: list[str] = []
    model.summary(print_fn=summary_lines.append)
    MODEL_SUMMARY_PATH.write_text("\n".join(summary_lines), encoding="utf-8")

    metadata = {
        **PARAMETERS,
        "model_type": "Convolutional Neural Network (CNN)",
        "test_loss": round(float(test_loss), 4),
        "test_accuracy": round(accuracy, 4),
        "keras_test_accuracy": round(float(test_accuracy), 4),
        "total_parameters": int(model.count_params()),
        "misclassified_samples": int(misclassified_count),
        "model_path": str(MODEL_PATH.name),
        "history_plot": str(HISTORY_PLOT_PATH.name),
        "confusion_matrix_plot": str(CONFUSION_MATRIX_PATH.name),
        "classification_report": str(CLASSIFICATION_REPORT_PATH.name),
        "model_summary": str(MODEL_SUMMARY_PATH.name),
        "misclassified_plot": str(MISCLASSIFIED_PLOT_PATH.name),
        "feature_maps_plot": str(FEATURE_MAPS_PATH.name),
        "sample_prediction_plot": str(SAMPLE_PREDICTION_PATH.name),
    }

    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    model.save(MODEL_PATH)
    print("Training completed.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Test accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
