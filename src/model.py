from __future__ import annotations

import tensorflow as tf


def build_cnn_model(input_shape: tuple[int, int, int] = (28, 28, 1)) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape, name="input_layer"),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv1"),
            tf.keras.layers.MaxPooling2D((2, 2), name="pool1"),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2"),
            tf.keras.layers.MaxPooling2D((2, 2), name="pool2"),
            tf.keras.layers.Flatten(name="flatten"),
            tf.keras.layers.Dense(128, activation="relu", name="dense1"),
            tf.keras.layers.Dropout(0.5, name="dropout"),
            tf.keras.layers.Dense(10, activation="softmax", name="output"),
        ],
        name="digit_recognition_cnn",
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
