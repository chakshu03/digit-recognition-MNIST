from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps


IMAGE_SIZE = 28
IMAGE_CHANNELS = 1


def normalize_cnn_images(images: np.ndarray) -> np.ndarray:
    images = images.astype("float32") / 255.0
    return images.reshape(images.shape[0], IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)


def _ensure_mnist_style_foreground(image: Image.Image) -> Image.Image:
    grayscale = ImageOps.grayscale(image)
    image_array = np.asarray(grayscale, dtype=np.uint8)

    # MNIST digits are light strokes on a dark background. Invert only when the
    # upload looks like dark ink on a light canvas.
    border_pixels = np.concatenate(
        [
            image_array[0, :],
            image_array[-1, :],
            image_array[:, 0],
            image_array[:, -1],
        ]
    )
    if float(border_pixels.mean()) > 127:
        grayscale = ImageOps.invert(grayscale)

    return ImageOps.autocontrast(grayscale)


def prepare_uploaded_image(image: Image.Image) -> tuple[np.ndarray, Image.Image]:
    normalized = _ensure_mnist_style_foreground(image)
    pixel_array = np.asarray(normalized, dtype=np.uint8)

    # Remove empty margins so the uploaded digit fills the frame similarly to
    # MNIST samples, while keeping a little breathing room around the glyph.
    foreground_mask = pixel_array > 30
    if foreground_mask.any():
        rows, cols = np.where(foreground_mask)
        cropped_array = pixel_array[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
        normalized = Image.fromarray(cropped_array, mode="L")

    canvas = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=0)
    fitted = ImageOps.contain(normalized, (20, 20))
    offset = ((IMAGE_SIZE - fitted.width) // 2, (IMAGE_SIZE - fitted.height) // 2)
    canvas.paste(fitted, offset)

    processed_array = np.asarray(canvas).astype("float32") / 255.0
    cnn_tensor = processed_array.reshape(1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)
    return cnn_tensor, canvas
