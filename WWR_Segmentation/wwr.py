"""Window-to-Wall Ratio (WWR) estimation module."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from WWR_Segmentation.config import Config
from WWR_Segmentation.dataset import has_test_set
from WWR_Segmentation.utils import list_image_files

logger = logging.getLogger("wwr_segmentation")


def compute_wwr_from_mask(
    mask: np.ndarray,
    window_class: int = 1,
    wall_class: int = 2,
) -> Dict[str, float]:
    """
    Compute Window-to-Wall Ratio from a segmentation mask.

    WWR = Window Pixels / Wall Pixels

    Args:
        mask: Integer label mask (H, W).
        window_class: Class index for windows.
        wall_class: Class index for walls.

    Returns:
        Dictionary with pixel counts and WWR value.
    """
    window_pixels = int(np.sum(mask == window_class))
    wall_pixels = int(np.sum(mask == wall_class))

    wwr = window_pixels / (wall_pixels + 1e-6)

    return {
        "window_pixels": window_pixels,
        "wall_pixels": wall_pixels,
        "wwr": float(wwr),
    }


def compute_wwr_batch(
    masks: np.ndarray,
    window_class: int = 1,
    wall_class: int = 2,
) -> List[Dict[str, float]]:
    """Compute WWR for a batch of masks."""
    return [
        compute_wwr_from_mask(masks[i], window_class, wall_class)
        for i in range(masks.shape[0])
    ]


def predict_and_compute_wwr(
    model: keras.Model,
    image: np.ndarray,
    config: Config,
) -> Dict:
    """
    Run inference on a single image and compute WWR.

    Args:
        model: Trained segmentation model.
        image: RGB image as float32 array (H, W, 3) in [0, 1] or uint8.
        config: Project configuration.

    Returns:
        Dictionary with predicted mask and WWR metrics.
    """
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    resized = tf.image.resize(image, config.image_size, method="bilinear")
    batch = tf.expand_dims(resized, 0)

    pred = model.predict(batch, verbose=0)
    pred_mask = np.argmax(pred[0], axis=-1)

    wwr_metrics = compute_wwr_from_mask(
        pred_mask, config.window_class, config.wall_class
    )
    wwr_metrics["predicted_mask"] = pred_mask
    return wwr_metrics


def process_directory(
    model: keras.Model,
    config: Config,
    images_dir: Optional[Path] = None,
    output_name: str = "wwr_results",
) -> pd.DataFrame:
    """
    Compute WWR for all images in a directory and export results.

    Args:
        model: Trained segmentation model.
        config: Project configuration.
        images_dir: Directory of RGB images (defaults to test set).
        output_name: Base filename for CSV/Excel exports.

    Returns:
        DataFrame with per-image WWR results.
    """
    images_dir = images_dir or config.test_images_dir
    image_paths = list_image_files(images_dir, config.image_extensions)

    results: List[Dict] = []
    for img_path in image_paths:
        img_bytes = tf.io.read_file(str(img_path))
        image = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        image = tf.cast(image, tf.float32).numpy() / 255.0

        metrics = predict_and_compute_wwr(model, image, config)
        results.append({
            "image": img_path.name,
            "window_pixels": metrics["window_pixels"],
            "wall_pixels": metrics["wall_pixels"],
            "wwr": metrics["wwr"],
        })
        logger.info("%s — WWR: %.4f (windows: %d, walls: %d)",
                    img_path.name, metrics["wwr"],
                    metrics["window_pixels"], metrics["wall_pixels"])

    df = pd.DataFrame(results)

    config.wwr_output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = config.wwr_output_dir / f"{output_name}.csv"
    excel_path = config.wwr_output_dir / f"{output_name}.xlsx"

    df.to_csv(csv_path, index=False)
    df.to_excel(excel_path, index=False)

    _save_wwr_summary(df, config)
    _save_wwr_visualization(df, config.wwr_output_dir / f"{output_name}_distribution.png")

    logger.info("WWR results exported to %s and %s", csv_path, excel_path)
    return df


def _save_wwr_summary(df: pd.DataFrame, config: Config) -> None:
    """Save a text summary of WWR statistics."""
    summary_path = config.wwr_output_dir / "wwr_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Window-to-Wall Ratio (WWR) Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Number of images: {len(df)}\n")
        f.write(f"Mean WWR:   {df['wwr'].mean():.4f}\n")
        f.write(f"Median WWR: {df['wwr'].median():.4f}\n")
        f.write(f"Std WWR:    {df['wwr'].std():.4f}\n")
        f.write(f"Min WWR:    {df['wwr'].min():.4f}\n")
        f.write(f"Max WWR:    {df['wwr'].max():.4f}\n")


def _save_wwr_visualization(df: pd.DataFrame, save_path: Path) -> None:
    """Plot WWR distribution histogram."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["wwr"], bins=20, edgecolor="black", alpha=0.7)
    ax.axvline(df["wwr"].mean(), color="red", linestyle="--", label=f"Mean: {df['wwr'].mean():.3f}")
    ax.set_xlabel("Window-to-Wall Ratio (WWR)")
    ax.set_ylabel("Count")
    ax.set_title("WWR Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_wwr_analysis(
    config: Config,
    model_path: Optional[str] = None,
    images_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load model and run WWR analysis on a directory of images."""
    if images_dir is None and not has_test_set(config):
        raise FileNotFoundError(
            "No test images available for WWR analysis. "
            "Upload the test set to Drive when ready, or pass images_dir explicitly."
        )

    path = model_path or str(config.best_model_path)
    model = keras.models.load_model(path, compile=False)

    img_dir = Path(images_dir) if images_dir else None
    return process_directory(model, config, images_dir=img_dir)
