"""Training-time segmentation visualization (input / ground truth / prediction)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from WWR_Segmentation.config import Config
from WWR_Segmentation.dataset import build_file_pairs
from WWR_Segmentation.inference import preprocess_image

logger = logging.getLogger("wwr_segmentation")


def colorize_mask(mask: np.ndarray, config: Config) -> np.ndarray:
    """Convert integer label mask (H, W) to RGB."""
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx, color in enumerate(config.class_colors):
        colored[mask == class_idx] = color
    return colored


def _decode_mask_path(mask_path: str, config: Config) -> np.ndarray:
    """Load and resize a grayscale mask to class indices."""
    mask_bytes = tf.io.read_file(mask_path)
    mask = tf.io.decode_image(mask_bytes, channels=1, expand_animations=False)
    mask = tf.cast(mask[..., 0], tf.int32)
    lut = tf.constant(config.mask_lut, dtype=tf.int32)
    mask = tf.gather(lut, tf.clip_by_value(mask, 0, 255))
    mask = tf.image.resize(
        tf.expand_dims(tf.cast(mask, tf.float32), -1),
        config.image_size,
        method="nearest",
    )
    return tf.cast(tf.squeeze(mask, -1), tf.int32).numpy()


def load_fixed_samples(config: Config, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a fixed set of validation samples for consistent epoch comparisons.

    Falls back to train split if validation is empty.
    """
    pairs = build_file_pairs(config, "val")
    if not pairs:
        pairs = build_file_pairs(config, "train")

    pairs = pairs[:num_samples]
    if not pairs:
        raise ValueError("No image/mask pairs found for visualization.")

    images = []
    masks = []
    for image_path, mask_path in pairs:
        batch = preprocess_image(image_path, config.image_size)
        images.append(batch[0].numpy())
        masks.append(_decode_mask_path(mask_path, config))

    return np.stack(images, axis=0), np.stack(masks, axis=0)


def save_epoch_comparison(
    images: np.ndarray,
    true_masks: np.ndarray,
    pred_masks: np.ndarray,
    config: Config,
    save_path: Path,
    epoch: int,
    logs: Optional[Dict[str, float]] = None,
) -> None:
    """Save a grid: each row is Input | Ground Truth | Prediction."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = images.shape[0]

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    title = f"Epoch {epoch + 1}"
    if logs:
        iou = logs.get("val_mean_iou")
        loss = logs.get("val_loss")
        if iou is not None:
            title += f"  |  val IoU: {iou:.4f}"
        if loss is not None:
            title += f"  |  val loss: {loss:.4f}"
    fig.suptitle(title, fontsize=14, y=1.01)

    column_titles = ("Input", "Ground Truth", "Prediction")
    for col, col_title in enumerate(column_titles):
        axes[0, col].set_title(col_title, fontsize=12)

    for row in range(num_samples):
        img = np.clip(images[row], 0.0, 1.0)
        axes[row, 0].imshow(img)
        axes[row, 1].imshow(colorize_mask(true_masks[row], config))
        axes[row, 2].imshow(colorize_mask(pred_masks[row], config))
        for col in range(3):
            axes[row, col].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def show_image_in_notebook(image_path: Path) -> None:
    """Display a saved PNG inline (Google Colab / Jupyter)."""
    try:
        from IPython.display import Image, display

        display(Image(filename=str(image_path)))
    except Exception:
        logger.debug("Notebook display unavailable for %s", image_path)


class EpochVisualizationCallback(keras.callbacks.Callback):
    """
    At the end of each epoch, predict on fixed validation samples and save
    Input | Ground Truth | Prediction figures.
    """

    def __init__(self, config: Config, run_name: str = "training") -> None:
        super().__init__()
        self.config = config
        self.run_name = run_name
        self.save_dir = config.results_dir / "epoch_visualizations" / run_name
        self.images: Optional[np.ndarray] = None
        self.masks: Optional[np.ndarray] = None

    def on_train_begin(self, logs: dict | None = None) -> None:
        self.images, self.masks = load_fixed_samples(
            self.config, self.config.viz_num_samples
        )
        self.save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Epoch visualization enabled — %d fixed samples → %s",
            self.config.viz_num_samples,
            self.save_dir,
        )

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        if self.images is None or self.masks is None:
            return
        if (epoch + 1) % self.config.viz_every_n_epochs != 0:
            return

        preds = self.model.predict(self.images, verbose=0)
        pred_masks = np.argmax(preds, axis=-1)

        save_path = self.save_dir / f"epoch_{epoch + 1:03d}.png"
        save_epoch_comparison(
            self.images,
            self.masks,
            pred_masks,
            self.config,
            save_path,
            epoch,
            logs,
        )
        logger.info("Saved epoch visualization → %s", save_path)

        if self.config.viz_show_in_notebook:
            show_image_in_notebook(save_path)
