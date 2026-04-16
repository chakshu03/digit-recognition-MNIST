from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps


IMAGE_SIZE = 28
IMAGE_CHANNELS = 1


def normalize_cnn_images(images: np.ndarray) -> np.ndarray:
    images = images.astype("float32") / 255.0
    return images.reshape(images.shape[0], IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)


def prepare_uploaded_image(image: Image.Image) -> tuple[np.ndarray, Image.Image]:
    grayscale = ImageOps.grayscale(image)
    inverted = ImageOps.invert(grayscale)

    canvas = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=0)
    fitted = ImageOps.contain(inverted, (20, 20))
    offset = ((IMAGE_SIZE - fitted.width) // 2, (IMAGE_SIZE - fitted.height) // 2)
    canvas.paste(fitted, offset)

    processed_array = np.asarray(canvas).astype("float32") / 255.0
    cnn_tensor = processed_array.reshape(1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)
    return cnn_tensor, canvas
