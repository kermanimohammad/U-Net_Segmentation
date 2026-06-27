"""Inference utilities for single-image and batch prediction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras

from WWR_Segmentation.config import Config

logger = logging.getLogger("wwr_segmentation")


def load_model(model_path: Union[str, Path]) -> keras.Model:
    """Load a trained segmentation model from disk."""
    logger.info("Loading model from %s", model_path)
    return keras.models.load_model(str(model_path), compile=False)


def preprocess_image(
    image: Union[str, Path, np.ndarray, tf.Tensor],
    target_size: tuple[int, int] = (512, 512),
) -> tf.Tensor:
    """
    Load and preprocess an image for inference.

    Args:
        image: File path or array (H, W, 3).
        target_size: Target (height, width).

    Returns:
        Preprocessed tensor (1, H, W, 3) in float32 [0, 1].
    """
    if isinstance(image, (str, Path)):
        image_bytes = tf.io.read_file(str(image))
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)

    image = tf.cast(image, tf.float32)
    if tf.reduce_max(image) > 1.0:
        image = image / 255.0

    image = tf.image.resize(image, target_size, method="bilinear")
    return tf.expand_dims(image, 0)


def predict(
    model: keras.Model,
    image: Union[str, Path, np.ndarray],
    config: Optional[Config] = None,
    return_probabilities: bool = False,
) -> np.ndarray:
    """
    Run segmentation inference on a single image.

    Args:
        model: Trained model.
        image: Input image path or array.
        config: Configuration (uses default 512×512 if None).
        return_probabilities: If True, return softmax probabilities instead of labels.

    Returns:
        Predicted mask (H, W) or probability map (H, W, C).
    """
    target_size = config.image_size if config else (512, 512)
    batch = preprocess_image(image, target_size)
    probs = model.predict(batch, verbose=0)[0]

    if return_probabilities:
        return probs
    return np.argmax(probs, axis=-1)


def predict_batch(
    model: keras.Model,
    image_paths: List[Union[str, Path]],
    config: Config,
    batch_size: int = 8,
) -> List[np.ndarray]:
    """Run inference on a list of image paths."""
    masks: List[np.ndarray] = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_tensors = [preprocess_image(p, config.image_size)[0] for p in batch_paths]
        batch = tf.stack(batch_tensors)
        probs = model.predict(batch, verbose=0)
        masks.extend([np.argmax(p, axis=-1) for p in probs])
    return masks


def save_prediction(
    mask: np.ndarray,
    save_path: Union[str, Path],
    config: Config,
    colorized: bool = True,
) -> None:
    """
    Save a predicted segmentation mask.

    Args:
        mask: Integer label mask (H, W).
        save_path: Output file path.
        config: Configuration with class colors.
        colorized: If True, save RGB visualization; otherwise save raw labels.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if colorized:
        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for class_idx, color in enumerate(config.class_colors):
            colored[mask == class_idx] = color
        tf.keras.utils.save_img(str(save_path), colored)
    else:
        # Map class indices back to grayscale mask values
        reverse_lut = {v: k for k, v in config.mask_value_to_class.items()}
        gray = np.zeros(mask.shape, dtype=np.uint8)
        for class_idx, gray_val in reverse_lut.items():
            gray[mask == class_idx] = gray_val
        tf.keras.utils.save_img(str(save_path), np.expand_dims(gray, -1))


def run_inference(
    config: Config,
    input_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    model_path: Optional[str] = None,
) -> None:
    """Run batch inference on all images in *input_dir* and save predictions."""
    from WWR_Segmentation.utils import list_image_files

    model = load_model(model_path or config.best_model_path)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else config.predictions_dir / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_image_files(input_dir, config.image_extensions)
    logger.info("Running inference on %d images...", len(image_paths))

    for img_path in image_paths:
        mask = predict(model, img_path, config)
        save_prediction(mask, output_dir / f"{img_path.stem}_pred.png", config)
        logger.info("Saved prediction for %s", img_path.name)

    logger.info("Predictions saved to %s", output_dir)
